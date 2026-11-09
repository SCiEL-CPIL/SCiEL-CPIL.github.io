import itertools
import random
import time
import numpy as np
from collections import Counter

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using fast greedy multi-start
    followed by local search refinement.
    """
    deadline = time.time() + 110.0  # Safety margin

    # Precompute t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute for each t-subset, the list of k-subsets that contain it
    tsub_to_ksubs = {}
    for ts in all_tsubs:
        ts_set = set(ts)
        remaining = [p for p in range(v) if p not in ts_set]
        ksubs = []
        for extra in itertools.combinations(remaining, k - t):
            block = tuple(sorted(ts + tuple(extra)))
            ksubs.append(block)
        tsub_to_ksubs[ts] = ksubs
    
    # Precompute for each k-subset, the list of t-subset indices it covers
    ksub_to_tsub_indices = {}
    
    def get_ksub_indices(block):
        block = tuple(block)
        if block not in ksub_to_tsub_indices:
            idxs = [tsub_to_idx[comb] for comb in itertools.combinations(block, t)]
            ksub_to_tsub_indices[block] = idxs
        return ksub_to_tsub_indices[block]
    
    def greedy_once(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = rng.choice(uncovered_indices)
            S = all_tsubs[start_idx]
            
            # Get precomputed k-subsets containing S
            candidates = tsub_to_ksubs[S]
            
            # Evaluate gain for each candidate
            best_block = None
            best_gain = -1
            
            # Shuffle for tie-breaking
            candidates = list(candidates)
            rng.shuffle(candidates)
            
            for B in candidates:
                idxs = get_ksub_indices(B)
                gain = np.sum(uncovered[idxs])
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = B
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            
            # Update uncovered mask
            idxs = get_ksub_indices(best_block)
            uncovered[idxs] = False
        
        return blocks

    def remove_redundant(blocks, max_iter=10):
        if not blocks:
            return blocks
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Count coverage of each t-subset
            cover_count = np.zeros(n_tsubs, dtype=np.int32)
            block_indices = []
            for b in blocks:
                idxs = get_ksub_indices(tuple(b))
                block_indices.append(idxs)
                cover_count[idxs] += 1
            
            valid_blocks = []
            removed_any = False
            
            for i, b in enumerate(blocks):
                if time.time() > deadline:
                    return blocks
                
                idxs = block_indices[i]
                if np.all(cover_count[idxs] > 1):
                    # Redundant: remove it
                    cover_count[idxs] -= 1
                    removed_any = True
                else:
                    valid_blocks.append(b)
            
            if not removed_any:
                break
            blocks = valid_blocks
        
        return blocks

    def validate_covering(blocks):
        """Check if blocks form a valid covering."""
        if not blocks:
            return n_tsubs == 0
        covered = np.zeros(n_tsubs, dtype=bool)
        for b in blocks:
            idxs = get_ksub_indices(tuple(b))
            covered[idxs] = True
        return np.all(covered)

    def local_search(blocks, rng, max_iters=50):
        """
        Local search: try to replace pairs of blocks with a single block
        that covers all t-subsets uniquely covered by the pair.
        """
        if not blocks:
            return blocks[:]
        
        best = blocks[:]
        best_len = len(best)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute coverage counts
            cover_count = np.zeros(n_tsubs, dtype=np.int32)
            for b in best:
                idxs = get_ksub_indices(tuple(b))
                cover_count[idxs] += 1
            
            # Find t-subsets covered exactly once
            unique_covered = np.where(cover_count == 1)[0]
            
            if len(unique_covered) == 0:
                break
            
            # Try to find two blocks whose union of unique t-subsets
            # can be covered by a single new block
            improved = False
            
            # Sample pairs of blocks
            n_blocks = len(best)
            for _ in range(20):  # Try 20 random pairs
                if time.time() > deadline:
                    break
                
                i, j = rng.sample(range(n_blocks), 2)
                b_i = best[i]
                b_j = best[j]
                
                # Get t-subsets uniquely covered by b_i or b_j
                idxs_i = get_ksub_indices(tuple(b_i))
                idxs_j = get_ksub_indices(tuple(b_j))
                
                unique_i = [idx for idx in idxs_i if cover_count[idx] == 1]
                unique_j = [idx for idx in idxs_j if cover_count[idx] == 1]
                
                if not unique_i and not unique_j:
                    continue
                
                # Find a block that covers all unique_i and unique_j t-subsets
                # This is a set cover problem - use greedy
                target_tsubs = set(unique_i) | set(unique_j)
                
                if not target_tsubs:
                    continue
                
                # Find candidate blocks that cover at least one target t-subset
                # and try to find one that covers all
                all_candidates = set()
                for tsub_idx in target_tsubs:
                    tsub = all_tsubs[tsub_idx]
                    for B in tsub_to_ksubs[tsub]:
                        all_candidates.add(B)
                
                # Try to find a single block that covers all target t-subsets
                best_new_block = None
                best_covered = -1
                
                for B in all_candidates:
                    idxs = get_ksub_indices(B)
                    covered_targets = sum(1 for idx in target_tsubs if idx in idxs)
                    if covered_targets == len(target_tsubs):
                        best_new_block = B
                        break
                    elif covered_targets > best_covered:
                        best_covered = covered_targets
                        best_new_block = B
                
                if best_new_block is not None and best_covered == len(target_tsubs):
                    # Create new solution without b_i and b_j, with best_new_block
                    new_blocks = [b for idx, b in enumerate(best) if idx != i and idx != j]
                    new_blocks.append(best_new_block)
                    
                    # Validate
                    if validate_covering(new_blocks) and len(new_blocks) < best_len:
                        best = new_blocks
                        best_len = len(best)
                        improved = True
                        break
            
            if not improved:
                # Try single block replacement
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    
                    # Pick a random block to replace
                    replace_idx = rng.randint(0, len(best) - 1)
                    old_block = best[replace_idx]
                    
                    # Find t-subsets uniquely covered by this block
                    idxs = get_ksub_indices(tuple(old_block))
                    unique = [idx for idx in idxs if cover_count[idx] == 1]
                    
                    if not unique:
                        continue
                    
                    # Find a block that covers all these unique t-subsets
                    target_tsubs = set(unique)
                    all_candidates = set()
                    for tsub_idx in target_tsubs:
                        tsub = all_tsubs[tsub_idx]
                        for B in tsub_to_ksubs[tsub]:
                            all_candidates.add(B)
                    
                    best_new_block = None
                    best_covered = -1
                    
                    for B in all_candidates:
                        if B == tuple(old_block):
                            continue
                        idxs = get_ksub_indices(B)
                        covered_targets = sum(1 for idx in target_tsubs if idx in idxs)
                        if covered_targets == len(target_tsubs):
                            best_new_block = B
                            break
                        elif covered_targets > best_covered:
                            best_covered = covered_targets
                            best_new_block = B
                    
                    if best_new_block is not None and best_covered == len(target_tsubs):
                        new_blocks = best[:]
                        new_blocks[replace_idx] = best_new_block
                        
                        # Check for duplicates
                        if len(set(tuple(b) for b in new_blocks)) == len(new_blocks):
                            if validate_covering(new_blocks) and len(new_blocks) < best_len:
                                best = new_blocks
                                best_len = len(best)
                                improved = True
                                break
            
            if not improved:
                break
        
        return best

    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multi-start greedy
    num_starts = 0
    
    # Run greedy starts until we have enough time for local search
    ls_time_budget = 40.0  # Reserve 40s for local search
    greedy_deadline = deadline - ls_time_budget
    
    while time.time() < greedy_deadline:
        seed = num_starts
        blocks = greedy_once(seed)
        
        if not blocks:
            num_starts += 1
            continue
            
        # Reduce
        reduced_blocks = remove_redundant(blocks)
        
        if len(reduced_blocks) < best_len:
            best_len = len(reduced_blocks)
            best_blocks = reduced_blocks[:]
            
        num_starts += 1
    
    if best_blocks is None:
        best_blocks = greedy_once(0)
        best_blocks = remove_redundant(best_blocks)
        best_len = len(best_blocks)
    
    # Phase 2: Local search refinement
    if time.time() < deadline - 5.0:
        rng = random.Random(12345)
        
        # Apply local search to the best solution
        ls_deadline = deadline - 5.0
        best_blocks = local_search(best_blocks, rng, max_iters=100)
        best_len = len(best_blocks)
        
        # Try local search on a few other good solutions if time permits
        if time.time() < deadline - 10.0:
            for _ in range(3):
                if time.time() > deadline - 10.0:
                    break
                
                seed = num_starts
                num_starts += 1
                blocks = greedy_once(seed)
                if blocks:
                    reduced = remove_redundant(blocks)
                    reduced = local_search(reduced, rng, max_iters=20)
                    if len(reduced) < best_len:
                        best_len = len(reduced)
                        best_blocks = reduced[:]
    
    return best_blocks
