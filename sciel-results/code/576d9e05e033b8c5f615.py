import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block index to list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_to_tsubs.append(subs)
        
    # Map t-subset index to list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_to_tsubs):
        for s in subs:
            tsub_to_blocks[s].append(b_idx)
            
    # Helper: Greedy construction
    def greedy_init():
        uncovered = np.ones(n_t_subs, dtype=np.int8)
        total_uncovered = n_t_subs
        solution_set = set()
        solution_list = []
        
        # For efficiency, we might just iterate blocks and pick best
        # But we need to update "uncovered count per block"
        # Let's use a simple approach: 
        # While uncovered > 0:
        #   Find block with max new coverage
        #   Add it
        
        # To make this fast, we can pre-calculate the size of each block's coverage (it's always C(k,t))
        # But the *new* coverage depends on current state.
        
        # Optimization: Instead of scanning all blocks every time (O(N_blocks * N_tsubs)),
        # we can maintain a count of how many uncovered subsets each block covers.
        # uncovered_count_per_block[b]
        
        uncovered_count_per_block = np.array([len(subs) for subs in block_to_tsubs], dtype=np.int32)
        
        while total_uncovered > 0:
            if time.time() > deadline:
                break
            # Find block with max uncovered count
            # np.argmax is fast
            best_b_idx = np.argmax(uncovered_count_per_block)
            if uncovered_count_per_block[best_b_idx] == 0:
                break # Should not happen if total_uncovered > 0 and valid problem
            
            solution_set.add(best_b_idx)
            solution_list.append(best_b_idx)
            
            # Update uncovered status and counts
            for s in block_to_tsubs[best_b_idx]:
                if uncovered[s]:
                    uncovered[s] = 0
                    total_uncovered -= 1
                    # Update counts for all blocks covering s
                    for b_idx in tsub_to_blocks[s]:
                        uncovered_count_per_block[b_idx] -= 1
                        
        return solution_list, solution_set

    # Helper: Local Search (Deletion and Swap)
    def local_search(solution_list, solution_set):
        # Initialize cover_count
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1
                
        improved = True
        while improved and time.time() < deadline - 1.0: # Reserve time
            improved = False
            
            # Pass 1: Deletion
            for i in range(len(solution_list)):
                b_idx = solution_list[i]
                redundant = True
                for s in block_to_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    solution_list.pop(i)
                    solution_set.remove(b_idx)
                    improved = True
                    break # Restart loop as indices changed
                    
            # Pass 2: Swap
            if not improved:
                for i in range(len(solution_list)):
                    b_idx = solution_list[i]
                    critical_subsets = []
                    for s in block_to_tsubs[b_idx]:
                        if cover_count[s] == 1:
                            critical_subsets.append(s)
                            
                    if not critical_subsets:
                        # Should have been deleted
                        for s in block_to_tsubs[b_idx]:
                            cover_count[s] -= 1
                        solution_list.pop(i)
                        solution_set.remove(b_idx)
                        improved = True
                        break
                        
                    # Find candidate block
                    # Intersect tsub_to_blocks for all critical subsets
                    if len(critical_subsets) == 1:
                        candidates = tsub_to_blocks[critical_subsets[0]]
                    else:
                        # Start with the smallest list for efficiency
                        min_idx = min(range(len(critical_subsets)), key=lambda j: len(tsub_to_blocks[critical_subsets[j]]))
                        candidates = set(tsub_to_blocks[critical_subsets[min_idx]])
                        for s in critical_subsets:
                            if s == critical_subsets[min_idx]:
                                continue
                            candidates.intersection_update(tsub_to_blocks[s])
                            if not candidates:
                                break
                    
                    if candidates:
                        # Find a candidate not in solution_set
                        best_candidate = -1
                        # Prefer candidates that cover many other subsets? 
                        # For now, just pick one. Randomization might help escape local minima.
                        # Let's pick randomly from candidates not in solution to diversify?
                        # Or just the first one.
                        for c in candidates:
                            if c not in solution_set:
                                best_candidate = c
                                break
                                
                        if best_candidate != -1:
                            # Perform Swap
                            # Decrement cover_count for b_idx
                            for s in block_to_tsubs[b_idx]:
                                cover_count[s] -= 1
                            # Increment cover_count for best_candidate
                            for s in block_to_tsubs[best_candidate]:
                                cover_count[s] += 1
                                
                            # Update solution
                            solution_list[i] = best_candidate
                            solution_set.remove(b_idx)
                            solution_set.add(best_candidate)
                            improved = True
                            break
                            
        return solution_list, solution_set

    # Helper: Perturbation and Repair
    def perturb_and_repair(solution_list, solution_set):
        # Randomly remove some blocks
        n_remove = random.randint(2, 5)
        if len(solution_list) <= n_remove:
            return greedy_init()
            
        indices_to_remove = random.sample(range(len(solution_list)), n_remove)
        # Sort indices to remove safely
        indices_to_remove.sort(reverse=True)
        
        # We need to update cover_count. 
        # It's easier to rebuild cover_count from scratch after perturbation? 
        # Or maintain it. Let's rebuild for simplicity and correctness, 
        # as the number of blocks is small (~150).
        
        # Actually, let's just compute the new state.
        # 1. Remove blocks
        new_solution_set = set(solution_set)
        for idx in indices_to_remove:
            b_idx = solution_list[idx]
            new_solution_set.remove(b_idx)
            
        new_solution_list = [b for i, b in enumerate(solution_list) if i not in indices_to_remove]
        
        # 2. Calculate uncovered subsets
        # We need to know which subsets are now uncovered.
        # A subset is uncovered if it was covered ONLY by the removed blocks.
        # So, cover_count[s] in the *original* solution minus the contributions of removed blocks.
        
        # Let's compute cover_count for the original solution first? 
        # We don't have it here. 
        # Alternative: 
        # Compute cover_count for the reduced set.
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in new_solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1
                
        # Find uncovered
        uncovered_mask = (cover_count == 0)
        uncovered_indices = np.where(uncovered_mask)[0]
        
        if len(uncovered_indices) == 0:
            return new_solution_list, new_solution_set
            
        # 3. Greedy Repair
        # We need to add blocks to cover uncovered_indices.
        # We can reuse the greedy logic but only on the uncovered set?
        # It's simpler to just run a greedy that focuses on uncovered.
        
        # Let's adapt the greedy to accept a set of uncovered subsets.
        # Or just run the standard greedy on the *current* state.
        
        # To be fast, we can just pick blocks that cover uncovered subsets.
        # Initialize a "to_cover" set
        to_cover = set(uncovered_indices.tolist())
        
        # We need to know how many subsets in `to_cover` each block covers.
        # We can compute this on the fly.
        
        while to_cover:
            if time.time() > deadline:
                break
            best_b_idx = -1
            max_cov = 0
            # Iterate all blocks? That's 50k blocks. 
            # 50k * (check intersection with to_cover) might be slow if to_cover is large.
            # But to_cover is likely small after removing 2-5 blocks.
            
            # Optimization: Only consider blocks that cover at least one uncovered subset.
            # We can get these from tsub_to_blocks for the first uncovered subset.
            # Then filter.
            
            if to_cover:
                first_uncovered = next(iter(to_cover))
                candidates = tsub_to_blocks[first_uncovered]
                
                for b_idx in candidates:
                    if b_idx in new_solution_set:
                        continue
                    # Count how many of to_cover it hits
                    cnt = 0
                    for s in block_to_tsubs[b_idx]:
                        if s in to_cover:
                            cnt += 1
                            if cnt > max_cov:
                                max_cov = cnt
                                best_b_idx = b_idx
                                # Early break if we covered all?
                                if cnt == len(to_cover):
                                    break
                    if max_cov == len(to_cover):
                        break
                if best_b_idx != -1:
                    new_solution_set.add(best_b_idx)
                    new_solution_list.append(best_b_idx)
                    # Update to_cover
                    for s in block_to_tsubs[best_b_idx]:
                        if s in to_cover:
                            to_cover.remove(s)
                else:
                    # Should not happen
                    break
                    
        return new_solution_list, new_solution_set

    # Main ILS Loop
    best_solution_list, best_solution_set = greedy_init()
    best_solution_list, best_solution_set = local_search(best_solution_list, best_solution_set)
    best_len = len(best_solution_list)
    current_solution_list = best_solution_list
    current_solution_set = best_solution_set
    
    random.seed(42)
    
    while time.time() < deadline - 5.0:
        # Perturb
        pert_list, pert_set = perturb_and_repair(current_solution_list, current_solution_set)
        
        # Local Search on perturbed
        opt_list, opt_set = local_search(pert_list, pert_set)
        
        # Update
        if len(opt_list) < best_len:
            best_len = len(opt_list)
            best_solution_list = opt_list
            best_solution_set = opt_set
            current_solution_list = opt_list
            current_solution_set = opt_set
        else:
            # Even if not better than best, accept as current if better than current?
            # Standard ILS: accept if better than current.
            if len(opt_list) < len(current_solution_list):
                current_solution_list = opt_list
                current_solution_set = opt_set
            # Or just keep current. Let's try accepting if better than current to explore more.
            # Actually, if it's not better than best, but better than current, it's a good move.
            # If it's worse than current, we might still want to jump to it? 
            # Usually, for minimization, we accept if new < current.
            # Let's stick to: if new < current, accept.
            
    # Convert block indices back to lists of points
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_solution_list]
    return final_blocks
