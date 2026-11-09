import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    """
    Solve the covering design problem C(v, k, t) using a Max-SAT-inspired
    stochastic local search (WalkSAT-style) with greedy initialization.
    Each block is a Boolean variable; each t-subset generates a clause
    requiring at least one containing block. We minimize unsatisfied
    clauses plus a penalty on the number of active blocks.
    """
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin from 120s limit
    
    # Precompute all t-subsets (the universe to be covered)
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each potential block (k-subset), which t-subsets it covers
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_covered_t = [None] * num_blocks
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for i, b in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            t_to_blocks[idx].append(i)
        block_covered_t[i] = covered_indices
    
    # Numpy arrays for fast operations
    block_covered_t_np = [np.array(block_covered_t[i], dtype=np.int32) for i in range(num_blocks)]
    
    # Greedy initialization
    def greedy_init(seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = np.ones(num_t_subsets, dtype=bool)
        selected = set()
        
        while uncovered.any():
            if time.time() > deadline:
                break
            
            best_block = -1
            best_gain = 0
            
            # Sample candidates
            if num_blocks > 5000:
                sample_size = min(3000, num_blocks)
                candidates = random.sample(range(num_blocks), sample_size)
            else:
                candidates = range(num_blocks)
            
            for b_idx in candidates:
                if time.time() > deadline:
                    break
                covered = block_covered_t_np[b_idx]
                gain = np.count_nonzero(uncovered[covered])
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
            
            selected.add(best_block)
            uncovered[block_covered_t_np[best_block]] = False
        
        return list(selected)
    
    # WalkSAT-style local search for Max-SAT formulation
    def walk_sat_search(initial_blocks, max_iters=10000, penalty_weight=0.1):
        """
        Treat the problem as Max-SAT:
        - Variables: x_i = 1 if block i is selected
        - Clauses: For each t-subset s, clause (OR of x_i for i in t_to_blocks[s])
        - Objective: Minimize (# unsatisfied clauses) + penalty_weight * (# selected blocks)
        
        We use a WalkSAT-style approach:
        - Start from an initial assignment (selected blocks)
        - At each step, flip a variable to reduce objective
        - If stuck, randomly walk to escape local optima
        """
        if not initial_blocks:
            return []
        
        # State: set of selected block indices
        selected = set(initial_blocks)
        
        # Compute coverage
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_covered_t_np[b_idx]] += 1
        
        # Objective function
        def objective(sel_set, cc):
            # Number of uncovered t-subsets
            uncovered_count = np.count_nonzero(cc == 0)
            # Number of selected blocks
            num_sel = len(sel_set)
            return uncovered_count + penalty_weight * num_sel
        
        current_obj = objective(selected, cover_count)
        best_selected = set(selected)
        best_obj = current_obj
        
        # Precompute: for each block, its covered t-subsets as a set for fast lookup
        block_covered_sets = [set(block_covered_t[i]) for i in range(num_blocks)]
        
        # Build reverse index: for each t-subset, which blocks cover it
        # t_to_blocks is already built above
        
        # WalkSAT parameters
        temperature = 0.5  # Probability of random flip when no improving move
        flip_prob = 0.7    # Probability of flipping a block in/out
        
        for iteration in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            # Try to find an improving flip
            best_flip = None
            best_new_obj = current_obj
            best_delta = 0
            
            # Candidate flips: consider adding or removing blocks
            # For efficiency, sample a subset of candidates
            
            # 1) Try removing a block from selected
            if selected:
                # Sample some blocks to try removing
                sel_list = list(selected)
                if len(sel_list) > 50:
                    remove_candidates = random.sample(sel_list, 50)
                else:
                    remove_candidates = sel_list
                
                for b_idx in remove_candidates:
                    if time.time() > deadline - 2.0:
                        break
                    covered = block_covered_t_np[b_idx]
                    # If we remove this block, which t-subsets become uncovered?
                    newly_uncovered = np.count_nonzero(cover_count[covered] == 1)
                    delta = newly_uncovered  # Increase in uncovered
                    new_obj = current_obj + delta - penalty_weight
                    if new_obj < best_new_obj:
                        best_new_obj = new_obj
                        best_delta = delta
                        best_flip = ('remove', b_idx)
            
            # 2) Try adding a block not in selected
            # Focus on blocks that cover many uncovered t-subsets
            uncovered_mask = cover_count == 0
            if np.any(uncovered_mask):
                # Sample candidate blocks to add
                if num_blocks > 5000:
                    add_sample_size = min(2000, num_blocks)
                    add_candidates = random.sample(range(num_blocks), add_sample_size)
                else:
                    add_candidates = range(num_blocks)
                
                for b_idx in add_candidates:
                    if b_idx in selected:
                        continue
                    if time.time() > deadline - 2.0:
                        break
                    covered = block_covered_t_np[b_idx]
                    # How many currently uncovered t-subsets does this block cover?
                    newly_covered = np.count_nonzero(cover_count[covered] == 0)
                    delta = -newly_covered + penalty_weight  # Decrease in uncovered + penalty
                    new_obj = current_obj + delta
                    if new_obj < best_new_obj:
                        best_new_obj = new_obj
                        best_delta = delta
                        best_flip = ('add', b_idx)
            
            # 3) Try swapping: remove one block, add another
            if selected and np.any(uncovered_mask):
                sel_list = list(selected)
                if len(sel_list) > 20:
                    swap_remove_candidates = random.sample(sel_list, 20)
                else:
                    swap_remove_candidates = sel_list
                
                for b_remove in swap_remove_candidates:
                    if time.time() > deadline - 2.0:
                        break
                    covered_remove = block_covered_t_np[b_remove]
                    
                    # Sample blocks to add
                    if num_blocks > 5000:
                        swap_add_sample = min(1000, num_blocks)
                        swap_add_candidates = random.sample(range(num_blocks), swap_add_sample)
                    else:
                        swap_add_candidates = range(num_blocks)
                    
                    for b_add in swap_add_candidates:
                        if b_add in selected or b_add == b_remove:
                            continue
                        if time.time() > deadline - 2.0:
                            break
                        
                        covered_add = block_covered_t_np[b_add]
                        
                        # Net change in uncovered count
                        # t-subsets covered by remove but not add: may become uncovered
                        # t-subsets covered by add but not remove: may become covered
                        only_in_remove = np.setdiff1d(covered_remove, covered_add)
                        only_in_add = np.setdiff1d(covered_add, covered_remove)
                        
                        # Count t-subsets that will be uncovered after swap
                        will_uncover = 0
                        if len(only_in_remove) > 0:
                            will_uncover = np.count_nonzero(cover_count[only_in_remove] == 1)
                        
                        # Count t-subsets that will be covered after swap
                        will_cover = 0
                        if len(only_in_add) > 0:
                            will_cover = np.count_nonzero(cover_count[only_in_add] == 0)
                        
                        delta = will_uncover - will_cover  # Net change in uncovered
                        new_obj = current_obj + delta  # No change in penalty (same number of blocks)
                        if new_obj < best_new_obj:
                            best_new_obj = new_obj
                            best_delta = delta
                            best_flip = ('swap', b_remove, b_add)
            
            # Apply flip
            if best_flip is not None:
                op = best_flip[0]
                if op == 'remove':
                    b_idx = best_flip[1]
                    selected.discard(b_idx)
                    cover_count[block_covered_t_np[b_idx]] -= 1
                elif op == 'add':
                    b_idx = best_flip[1]
                    selected.add(b_idx)
                    cover_count[block_covered_t_np[b_idx]] += 1
                elif op == 'swap':
                    b_remove, b_add = best_flip[1], best_flip[2]
                    selected.discard(b_remove)
                    selected.add(b_add)
                    cover_count[block_covered_t_np[b_remove]] -= 1
                    cover_count[block_covered_t_np[b_add]] += 1
                
                current_obj = best_new_obj
                
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_selected = set(selected)
            else:
                # No improving move found: random walk to escape local optimum
                if random.random() < temperature:
                    # Randomly flip a variable
                    if selected and random.random() < flip_prob:
                        # Remove a random block
                        b_idx = random.choice(list(selected))
                        selected.discard(b_idx)
                        cover_count[block_covered_t_np[b_idx]] -= 1
                    else:
                        # Add a random block
                        b_idx = random.randint(0, num_blocks - 1)
                        if b_idx not in selected:
                            selected.add(b_idx)
                            cover_count[block_covered_t_np[b_idx]] += 1
                    
                    current_obj = objective(selected, cover_count)
                    
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_selected = set(selected)
        
        return list(best_selected)
    
    # Post-processing: remove redundant blocks
    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        blocks = list(blocks)
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in blocks:
            cover_count[block_covered_t_np[b_idx]] += 1
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    return blocks
                b_idx = blocks[i]
                covered = block_covered_t_np[b_idx]
                if np.all(cover_count[covered] > 1):
                    cover_count[covered] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        
        return blocks
    
    # Main search loop
    best_blocks = None
    best_len = float('inf')
    
    # Try multiple seeds with greedy init + WalkSAT
    seeds = [42, 123, 999, 0, 1, 2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 31, 37, 41, 43, 47]
    
    for seed in seeds:
        if time.time() > deadline - 30.0:
            break
        
        random.seed(seed)
        greedy_res = greedy_init(seed)
        if not greedy_res:
            continue
        
        # WalkSAT search
        ws_res = walk_sat_search(greedy_res, max_iters=5000)
        
        # Remove redundant blocks
        final_res = remove_redundant(ws_res)
        
        if len(final_res) < best_len:
            best_len = len(final_res)
            best_blocks = final_res
        
        if best_len <= 69:
            break
    
    # If we have time, try more random seeds
    if time.time() < deadline - 20.0:
        while time.time() < deadline - 8.0:
            seed = random.randint(0, 1000000)
            random.seed(seed)
            greedy_res = greedy_init(seed)
            if not greedy_res:
                continue
            
            ws_res = walk_sat_search(greedy_res, max_iters=3000)
            final_res = remove_redundant(ws_res)
            
            if len(final_res) < best_len:
                best_len = len(final_res)
                best_blocks = final_res
                
            if best_len <= 69:
                break
    
    if best_blocks is None:
        return []
    
    # Convert block indices back to the actual blocks (lists of points)
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    
    # Ensure distinctness and sorting
    for b in final_blocks:
        b.sort()
    
    return final_blocks
