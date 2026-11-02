"""Covering design C(v,k,t) via Max-SAT / WalkSAT stochastic local search."""
import itertools
import random
import time
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 125  # safety margin before 150s kill
    rng = random.Random(42)
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Precompute all k-subsets (potential blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    
    # Build incidence: for each k-subset, which t-subsets it covers
    # For v=19, k=7, t=4: C(19,7)=50388 blocks, C(19,4)=3876 t-subsets
    # Each block covers C(7,4)=35 t-subsets
    block_tsubs = []  # list of arrays of t-subset indices
    for block in all_ksubs:
        idxs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                idxs.append(idx)
        block_tsubs.append(np.array(idxs, dtype=np.int32))
    
    # Build reverse incidence: for each t-subset, which blocks cover it
    # This is needed for efficient clause violation checking
    tsub_blocks = defaultdict(list)
    for b_idx, tsub_arr in enumerate(block_tsubs):
        for t_idx in tsub_arr:
            tsub_blocks[t_idx].append(b_idx)
    tsub_blocks_arr = np.zeros(n_tsubs, dtype=object)
    for i in range(n_tsubs):
        tsub_blocks_arr[i] = np.array(tsub_blocks[i], dtype=np.int32)
    
    # Objective: minimize sum of selected blocks subject to all t-subsets covered
    # WalkSAT-style: start with a random feasible solution, then flip variables
    
    def get_coverage(selected):
        """Compute coverage counts for all t-subsets given a selection of blocks."""
        cov = np.zeros(n_tsubs, dtype=np.int32)
        for b_idx in selected:
            for t_idx in block_tsubs[b_idx]:
                cov[t_idx] += 1
        return cov
    
    def is_feasible(cov):
        return cov.min() >= 1
    
    # Phase 1: Construct an initial feasible solution via greedy
    # Use randomized greedy to get a starting covering
    def greedy_initial():
        """Build an initial covering using randomized greedy."""
        selected = []
        cov = np.zeros(n_tsubs, dtype=np.int32)
        remaining = set(range(n_tsubs))
        
        # Precompute gain for each block at each step would be expensive
        # Instead, use a simple greedy: repeatedly pick block covering most uncovered
        while remaining:
            best_gain = -1
            best_block = None
            # Sample a subset of blocks to speed up
            candidates = list(range(n_ksubs))
            rng.shuffle(candidates)
            # Look at top candidates
            for b_idx in candidates[:min(2000, n_ksubs)]:
                gain = 0
                for t_idx in block_tsubs[b_idx]:
                    if t_idx in remaining:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                elif gain == best_gain and best_block is not None:
                    if rng.random() < 0.3:
                        best_block = b_idx
            
            if best_block is None or best_gain <= 0:
                # Fallback: pick any block that covers something
                for b_idx in range(n_ksubs):
                    for t_idx in block_tsubs[b_idx]:
                        if t_idx in remaining:
                            best_block = b_idx
                            break
                    if best_block is not None:
                        break
            
            if best_block is None:
                break
                
            selected.append(best_block)
            for t_idx in block_tsubs[best_block]:
                if t_idx in remaining:
                    remaining.discard(t_idx)
        
        return selected
    
    # Get initial solution
    init_selected = greedy_initial()
    init_cov = get_coverage(init_selected)
    
    if not is_feasible(init_cov):
        # Fallback: just return the greedy result
        blocks = [list(all_ksubs[i]) for i in init_selected]
        return blocks
    
    best_selected = list(init_selected)
    best_count = len(init_selected)
    best_cov = init_cov.copy()
    
    # Phase 2: WalkSAT-style local search
    # Variables: which blocks are selected
    # Clauses: each t-subset must be covered by at least one selected block
    # Objective: minimize number of selected blocks
    
    def walk_sat_search(selected, cov, time_limit):
        """WalkSAT-style search to minimize blocks while maintaining feasibility."""
        selected_set = set(selected)
        n_sel = len(selected_set)
        
        # Track for each t-subset: how many selected blocks cover it
        # cov already has this
        
        best_local = list(selected_set)
        best_local_count = n_sel
        best_local_cov = cov.copy()
        
        stall_count = 0
        max_stall = 50
        
        while time.time() < time_limit:
            # Try to remove a block (minimize)
            removed = False
            # Shuffle order to avoid bias
            sel_list = list(selected_set)
            rng.shuffle(sel_list)
            
            for b_idx in sel_list:
                if time.time() > time_limit:
                    break
                # Try removing b_idx
                new_cov = cov.copy()
                for t_idx in block_tsubs[b_idx]:
                    new_cov[t_idx] -= 1
                
                if new_cov.min() >= 1:
                    # Successfully removed
                    selected_set.discard(b_idx)
                    cov = new_cov
                    n_sel -= 1
                    removed = True
                    if n_sel < best_local_count:
                        best_local_count = n_sel
                        best_local = list(selected_set)
                        best_local_cov = cov.copy()
                    break
            
            if removed:
                stall_count = 0
                continue
            
            # Try swapping: remove one, add one
            # Find t-subsets with minimum coverage (most vulnerable)
            min_cov = cov.min()
            if min_cov <= 1:
                # Find t-subsets with coverage 1
                vulnerable = np.where(cov == 1)[0]
                if len(vulnerable) > 0:
                    # Pick a random vulnerable t-subset
                    v_t = vulnerable[rng.randint(0, len(vulnerable)-1)]
                    # The block covering it
                    covering_blocks = [b for b in selected_set if v_t in block_tsubs[b]]
                    if covering_blocks:
                        # Try to swap: remove one covering block, add another
                        b_remove = covering_blocks[0]
                        # Find blocks that cover v_t but are not selected
                        all_covering_v = tsub_blocks_arr[v_t]
                        candidates_add = [b for b in all_covering_v if b not in selected_set]
                        
                        if candidates_add:
                            # Try each candidate add
                            rng.shuffle(candidates_add)
                            for b_add in candidates_add[:10]:
                                if time.time() > time_limit:
                                    break
                                # Remove b_remove, add b_add
                                new_selected = selected_set - {b_remove} | {b_add}
                                # Check if still feasible
                                new_cov = cov.copy()
                                for t_idx in block_tsubs[b_remove]:
                                    new_cov[t_idx] -= 1
                                for t_idx in block_tsubs[b_add]:
                                    new_cov[t_idx] += 1
                                
                                if new_cov.min() >= 1:
                                    # Check if this reduces total count or same count but better
                                    # Since we're removing and adding, count stays same
                                    # Only accept if it helps future removals (heuristic)
                                    # For now, accept with some probability
                                    if rng.random() < 0.1:
                                        selected_set = new_selected
                                        cov = new_cov
                                        break
            
            # Try adding a block that enables future removals (rarely)
            # This is less common; focus on removals
            
            stall_count += 1
            if stall_count > max_stall:
                # Restart with perturbation
                # Randomly flip a few variables
                for _ in range(rng.randint(1, 3)):
                    if selected_set:
                        b = rng.choice(list(selected_set))
                        if rng.random() < 0.5:
                            # Try to remove
                            new_cov = cov.copy()
                            for t_idx in block_tsubs[b]:
                                new_cov[t_idx] -= 1
                            if new_cov.min() >= 1:
                                selected_set.discard(b)
                                cov = new_cov
                                n_sel = len(selected_set)
                    else:
                        # Add a random block
                        b = rng.randint(0, n_ksubs-1)
                        selected_set.add(b)
                        for t_idx in block_tsubs[b]:
                            cov[t_idx] += 1
                        n_sel = len(selected_set)
                
                stall_count = 0
        
        return best_local, best_local_count, best_local_cov
    
    # Run WalkSAT with time budget
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Allocate time to WalkSAT
        walk_limit = time.time() + remaining_time * 0.7
        best_selected, best_count, best_cov = walk_sat_search(
            list(init_selected), init_cov.copy(), walk_limit
        )
    
    # Phase 3: Final deletion pass
    if time.time() < deadline - 5:
        selected_set = set(best_selected)
        cov = best_cov.copy()
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            sel_list = list(selected_set)
            rng.shuffle(sel_list)
            for b_idx in sel_list:
                if time.time() > deadline - 5:
                    break
                new_cov = cov.copy()
                for t_idx in block_tsubs[b_idx]:
                    new_cov[t_idx] -= 1
                if new_cov.min() >= 1:
                    selected_set.discard(b_idx)
                    cov = new_cov
                    improved = True
                    break
        
        if len(selected_set) < best_count:
            best_selected = list(selected_set)
            best_count = len(selected_set)
            best_cov = cov
    
    # Convert to list of blocks
    blocks = [list(all_ksubs[i]) for i in best_selected]
    return blocks
