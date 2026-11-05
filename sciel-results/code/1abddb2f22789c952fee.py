import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill

    def time_exceeded():
        return time.time() - start_time > timeout

    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)

    all_subsets = list(itertools.combinations(points, t))
    N_subsets = len(all_subsets)

    # Precompute mappings
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    block_subsets = []
    blocks_covering_subset = [[] for _ in range(N_subsets)]
    
    for i, b in enumerate(all_blocks):
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)
        for s in subs:
            blocks_covering_subset[s].append(i)
    
    # Convert to numpy arrays for fast operations
    block_subset_arrays = [np.array(subs, dtype=np.int32) for subs in block_subsets]
    subset_block_arrays = [np.array(blocks, dtype=np.int32) for blocks in blocks_covering_subset]
    
    # Precompute potential (number of subsets each block covers)
    initial_potential = np.array([len(s) for s in block_subsets], dtype=np.int32)

    def run_greedy(randomize_ties=False, seed=None):
        if time_exceeded():
            return None

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        cover_count = np.zeros(N_subsets, dtype=np.int32)
        potential = initial_potential.copy()

        selected_blocks = []
        uncovered_count = N_subsets

        while uncovered_count > 0:
            if time_exceeded():
                return None

            if randomize_ties:
                # Add small random noise to break ties and diversify
                noise = rng.uniform(0, 0.1, size=N_blocks)
                scores = potential.astype(np.float64) + noise
                best_idx = int(np.argmax(scores))
            else:
                best_idx = int(np.argmax(potential))

            if potential[best_idx] == 0:
                break

            selected_blocks.append(best_idx)

            b_subs = block_subset_arrays[best_idx]
            for s in b_subs:
                old_count = cover_count[s]
                cover_count[s] += 1
                if old_count == 0:
                    uncovered_count -= 1
                    # Update potential for all blocks covering this subset
                    for b in subset_block_arrays[s]:
                        potential[b] -= 1

        return selected_blocks

    def local_search_vns(selected_blocks, time_budget=2.0):
        """Variable Neighborhood Search with 1, 2, 3-swap perturbations."""
        if time_exceeded():
            return selected_blocks

        local_start = time.time()
        
        cover_count = np.zeros(N_subsets, dtype=np.int32)
        for b_idx in selected_blocks:
            for s in block_subsets[b_idx]:
                cover_count[s] += 1

        selected_set = set(selected_blocks)
        selected_list = list(selected_blocks)

        def is_redundant(b_idx):
            for s in block_subsets[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True

        def try_remove(b_idx):
            nonlocal cover_count, selected_set, selected_list
            if b_idx not in selected_set:
                return False
            if is_redundant(b_idx):
                for s in block_subsets[b_idx]:
                    cover_count[s] -= 1
                selected_set.remove(b_idx)
                selected_list.remove(b_idx)
                return True
            return False

        def find_swap_candidate(b_idx, max_candidates=30):
            """Find a block that can replace b_idx while maintaining coverage."""
            if b_idx not in selected_set:
                return None
            
            # Find subsets only covered by b_idx
            unique_subs = []
            for s in block_subsets[b_idx]:
                if cover_count[s] == 1:
                    unique_subs.append(s)
            
            if not unique_subs:
                # No unique subsets, any swap is safe
                return None  # Will be handled as simple removal
            
            # Find blocks that cover all unique subsets
            if len(unique_subs) > 5:
                return None  # Too hard to satisfy
            
            candidate_blocks = set(subset_block_arrays[unique_subs[0]].tolist())
            for s in unique_subs[1:]:
                candidate_blocks &= set(subset_block_arrays[s].tolist())
            
            candidate_blocks -= selected_set
            candidate_blocks.discard(b_idx)
            
            if not candidate_blocks:
                return None
            
            # Score candidates by how many additional subsets they cover
            best_j = None
            best_score = -1
            for j_idx in list(candidate_blocks)[:max_candidates]:
                new_subs = 0
                for s in block_subsets[j_idx]:
                    if cover_count[s] == 0:
                        new_subs += 1
                if new_subs > best_score:
                    best_score = new_subs
                    best_j = j_idx
            
            return best_j

        def do_1swap_pass():
            nonlocal cover_count, selected_set, selected_list
            improved = True
            while improved and not time_exceeded() and (time.time() - local_start) < time_budget:
                improved = False
                indices = list(range(len(selected_list)))
                random.shuffle(indices)
                
                for i in indices:
                    if i >= len(selected_list):
                        continue
                    b_idx = selected_list[i]
                    
                    # Try removal first
                    if try_remove(b_idx):
                        improved = True
                        break
                    
                    # Try 1-swap
                    new_b = find_swap_candidate(b_idx)
                    if new_b is not None:
                        # Perform swap
                        for s in block_subsets[b_idx]:
                            cover_count[s] -= 1
                        selected_set.remove(b_idx)
                        selected_list.remove(b_idx)
                        
                        for s in block_subsets[new_b]:
                            cover_count[s] += 1
                        selected_set.add(new_b)
                        selected_list.append(new_b)
                        
                        improved = True
                        break

        def do_2swap_pass():
            nonlocal cover_count, selected_set, selected_list
            improved = True
            while improved and not time_exceeded() and (time.time() - local_start) < time_budget:
                improved = False
                if len(selected_list) < 2:
                    break
                
                indices = list(range(len(selected_list)))
                random.shuffle(indices)
                
                for i in indices:
                    if i >= len(selected_list):
                        continue
                    b1_idx = selected_list[i]
                    
                    if b1_idx not in selected_set:
                        continue
                    
                    # Try removing b1 first
                    if try_remove(b1_idx):
                        improved = True
                        break
                    
                    # Find subsets only covered by b1
                    unique_subs_b1 = []
                    for s in block_subsets[b1_idx]:
                        if cover_count[s] == 1:
                            unique_subs_b1.append(s)
                    
                    if not unique_subs_b1:
                        continue
                    
                    # Try to find a block that covers all unique subsets of b1
                    if len(unique_subs_b1) > 4:
                        continue
                    
                    candidate_blocks = set(subset_block_arrays[unique_subs_b1[0]].tolist())
                    for s in unique_subs_b1[1:]:
                        candidate_blocks &= set(subset_block_arrays[s].tolist())
                    candidate_blocks -= selected_set
                    candidate_blocks.discard(b1_idx)
                    
                    if not candidate_blocks:
                        continue
                    
                    # Try swapping b1 with a candidate
                    for new_b in list(candidate_blocks)[:15]:
                        # Check if this swap is valid
                        # Temporarily remove b1
                        temp_cover = cover_count.copy()
                        for s in block_subsets[b1_idx]:
                            temp_cover[s] -= 1
                        
                        # Check if all subsets are still covered
                        valid = True
                        for s in block_subsets[b1_idx]:
                            if temp_cover[s] == 0:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        # The swap is valid if new_b covers all unique subsets
                        # (which it does by construction)
                        
                        # Perform swap
                        for s in block_subsets[b1_idx]:
                            cover_count[s] -= 1
                        selected_set.remove(b1_idx)
                        selected_list.remove(b1_idx)
                        
                        for s in block_subsets[new_b]:
                            cover_count[s] += 1
                        selected_set.add(new_b)
                        selected_list.append(new_b)
                        
                        improved = True
                        break
                    
                    if improved:
                        break

        def do_3swap_pass():
            nonlocal cover_count, selected_set, selected_list
            improved = True
            while improved and not time_exceeded() and (time.time() - local_start) < time_budget:
                improved = False
                if len(selected_list) < 3:
                    break
                
                # Try to find 3 blocks that can be replaced by 2 or 3 better blocks
                # This is more complex, so we'll use a simpler heuristic:
                # Find blocks with many unique subsets and try to replace them
                
                indices = list(range(len(selected_list)))
                random.shuffle(indices)
                
                for i in indices[:10]:  # Limit work
                    if i >= len(selected_list):
                        continue
                    b_idx = selected_list[i]
                    
                    if b_idx not in selected_set:
                        continue
                    
                    # Try removal
                    if try_remove(b_idx):
                        improved = True
                        break
                    
                    # Try 1-swap as fallback
                    new_b = find_swap_candidate(b_idx, max_candidates=20)
                    if new_b is not None:
                        for s in block_subsets[b_idx]:
                            cover_count[s] -= 1
                        selected_set.remove(b_idx)
                        selected_list.remove(b_idx)
                        
                        for s in block_subsets[new_b]:
                            cover_count[s] += 1
                        selected_set.add(new_b)
                        selected_list.append(new_b)
                        
                        improved = True
                        break

        # Execute VNS passes
        do_1swap_pass()
        do_2swap_pass()
        do_3swap_pass()
        
        # Final cleanup: remove any redundant blocks
        changed = True
        while changed and not time_exceeded():
            changed = False
            for i in range(len(selected_list) - 1, -1, -1):
                if try_remove(selected_list[i]):
                    changed = True

        return selected_list

    best_blocks = None
    best_len = float('inf')

    # Deterministic greedy
    res = run_greedy(randomize_ties=False)
    if res:
        res = local_search_vns(res, time_budget=3.0)
        if res:
            best_blocks = res
            best_len = len(res)

    # Randomized restarts
    t_elapsed = time.time() - start_time
    if t_elapsed < timeout - 10:
        for i in range(40):
            if time.time() - start_time > timeout - 5:
                break
            res = run_greedy(randomize_ties=True, seed=100 + i * 7)
            if res:
                # Use less time for local search on later iterations
                time_budget = max(1.0, 3.0 - i * 0.05)
                res = local_search_vns(res, time_budget=time_budget)
                if res and len(res) < best_len:
                    best_len = len(res)
                    best_blocks = res

    if best_blocks is None:
        return []

    final_blocks = [list(all_blocks[i]) for i in best_blocks]
    return final_blocks
