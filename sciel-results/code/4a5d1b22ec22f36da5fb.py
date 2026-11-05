import time
import numpy as np
import random
import itertools
import sys

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Precompute block -> tsub indices mapping
    # This is memory efficient and fast to access
    block_to_tsubs = []
    # And tsub -> block indices mapping
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    
    for i, block in enumerate(all_kblocks):
        subs = [tsub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_to_tsubs.append(subs)
        for sub_idx in subs:
            tsub_to_blocks[sub_idx].append(i)
            
    # Convert to numpy arrays for faster processing? 
    # Lists of lists are okay for 50k blocks * 35 tsubs each.
    
    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    best_solution = None
    best_len = float('inf')
    
    rng = random.Random(42)

    # Phase 1: Heuristic Greedy
    # Strategy: Always pick a block that covers the "hardest" uncovered t-subset
    # "Hardest" = fewest number of *remaining* blocks that can cover it.
    # Since we don't remove blocks from the pool, just mark them used, 
    # the "remaining" count is total count - used count.
    
    def greedy_construction():
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        # remaining_cover_count[s] = number of blocks NOT yet selected that cover s
        # Initially, len(tsub_to_blocks[s])
        remaining_cover_count = np.array([len(lst) for lst in tsub_to_blocks], dtype=np.int32)
        
        selected_indices = []
        selected_set = set()
        
        while np.any(uncovered):
            if time.monotonic() > deadline - 5.0:
                break
                
            # Find uncovered tsubs with minimum remaining_cover_count
            # Mask uncovered
            unc_mask = uncovered == 1
            # We want min remaining_cover_count where unc_mask is True
            # Set covered ones to inf
            costs = np.where(unc_mask, remaining_cover_count, np.iinfo(np.int32).max)
            min_cost = np.min(costs)
            
            # Get all tsubs with this min cost
            candidates_tsubs = np.where(costs == min_cost)[0]
            
            if len(candidates_tsubs) == 0:
                break # Should not happen if uncovered exists and blocks exist
            
            # Pick a random one from these to break ties (or lexicographically first for determinism in search?)
            # Random is better for diversity in SA later? 
            # Let's pick one randomly
            target_tsub = candidates_tsubs[rng.randrange(len(candidates_tsubs))]
            
            # Find all blocks covering target_tsub
            candidate_blocks = tsub_to_blocks[target_tsub]
            
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidate_blocks:
                if b_idx in selected_set:
                    continue
                # Calculate gain: number of uncovered tsubs covered by b_idx
                gain = 0
                # Optimization: Only check if potential gain > best_gain
                # We can estimate max gain as min(len(block_to_tsubs[b_idx]), num_uncovered)
                # But just counting is fast enough (35 items)
                for s_idx in block_to_tsubs[b_idx]:
                    if uncovered[s_idx]:
                        gain += 1
                        if gain > best_gain:
                            # Early exit if we can't beat best? 
                            # Max possible remaining gain is 35 - (35 - best_gain) ... no
                            # If gain + (35 - current_count) <= best_gain, break? 
                            # Simpler: just count all. 35 is small.
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    # If best_gain is 35 (perfect block), we can't do better
                    if best_gain == 35:
                        break

            if best_block_idx == -1:
                # No valid block found? Should not happen
                break
                
            selected_indices.append(best_block_idx)
            selected_set.add(best_block_idx)
            
            # Update uncovered and remaining_cover_count
            for s_idx in block_to_tsubs[best_block_idx]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = 0
                remaining_cover_count[s_idx] -= 1
                
        return selected_indices

    # Phase 2: Local Search / Reduction
    def remove_redundant(indices):
        indices_set = set(indices)
        if not indices_set:
            return []
            
        # Count coverage
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            for s_idx in block_to_tsubs[idx]:
                counts[s_idx] += 1
                
        changed = True
        while changed:
            changed = False
            to_remove = []
            for idx in indices_set:
                # Check if all tsubs covered by idx have count > 1
                is_redundant = True
                for s_idx in block_to_tsubs[idx]:
                    if counts[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    to_remove.append(idx)
            
            for idx in to_remove:
                indices_set.remove(idx)
                for s_idx in block_to_tsubs[idx]:
                    counts[s_idx] -= 1
                changed = True
                
        return list(indices_set)

    def local_search(indices):
        indices_set = set(indices)
        if not indices_set:
            return []
            
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            for s_idx in block_to_tsubs[idx]:
                counts[s_idx] += 1
                
        improved = True
        while improved:
            improved = False
            
            # 1. Remove redundant
            new_indices = remove_redundant(list(indices_set))
            if len(new_indices) < len(indices_set):
                indices_set = set(new_indices)
                # Recount
                counts = np.zeros(num_tsubs, dtype=np.int32)
                for idx in indices_set:
                    for s_idx in block_to_tsubs[idx]:
                        counts[s_idx] += 1
                improved = True
                continue
            
            # 2. Try swaps
            # For each block, find critical tsubs
            # If critical tsubs exist, find a replacement block that covers them all
            idx_list = list(indices_set)
            rng.shuffle(idx_list)
            
            for b_idx in idx_list:
                if time.monotonic() > deadline - 10.0:
                    return list(indices_set)
                
                # Find critical tsubs for b_idx
                critical_tsubs = []
                for s_idx in block_to_tsubs[b_idx]:
                    if counts[s_idx] == 1:
                        critical_tsubs.append(s_idx)
                        
                if len(critical_tsubs) == 0:
                    # Redundant, should have been caught, but handle anyway
                    indices_set.remove(b_idx)
                    for s_idx in block_to_tsubs[b_idx]:
                        counts[s_idx] -= 1
                    improved = True
                    break
                
                # Find candidate replacement blocks
                # Must cover all critical_tsubs
                # Intersection of tsub_to_blocks[c] for c in critical_tsubs
                if len(critical_tsubs) > 10:
                    # Too many constraints, unlikely to find a single block
                    continue
                    
                # Start with blocks covering the first critical tsub
                candidate_set = set(tsub_to_blocks[critical_tsubs[0]])
                for c in critical_tsubs[1:]:
                    candidate_set.intersection_update(tsub_to_blocks[c])
                    
                # Remove current selected blocks
                candidate_set.difference_update(indices_set)
                
                if not candidate_set:
                    continue
                    
                # Pick a candidate that maximizes "quality"
                # Quality: number of tsubs covered by candidate that are currently under-covered (count == 1)
                # Or just random pick?
                # Let's pick the one that covers the most "weak" tsubs (count==1)
                best_cand = None
                best_score = -1
                
                # Limit candidates to avoid slow loop if many
                cand_list = list(candidate_set)
                if len(cand_list) > 50:
                    cand_list = rng.sample(cand_list, 50)
                
                for c_idx in cand_list:
                    score = 0
                    for s_idx in block_to_tsubs[c_idx]:
                        if counts[s_idx] == 1:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_cand = c_idx
                        
                if best_cand is not None:
                    # Perform swap
                    indices_set.remove(b_idx)
                    for s_idx in block_to_tsubs[b_idx]:
                        counts[s_idx] -= 1
                    indices_set.add(best_cand)
                    for s_idx in block_to_tsubs[best_cand]:
                        counts[s_idx] += 1
                    improved = True
                    break
                    
        return list(indices_set)

    # Phase 3: Simulated Annealing
    def simulated_annealing(initial_sol):
        current_set = set(initial_sol)
        current_counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in current_set:
            for s_idx in block_to_tsubs[idx]:
                current_counts[s_idx] += 1
                
        best_set = set(current_set)
        best_len = len(current_set)
        
        T = 10.0
        T_min = 0.1
        cooling = 0.995
        
        while T > T_min:
            if time.monotonic() > deadline - 5.0:
                break
                
            # Pick random block to remove
            if not current_set:
                break
            old_block = list(current_set)[rng.randrange(len(current_set))]
            
            # Find critical tsubs
            critical_tsubs = []
            for s_idx in block_to_tsubs[old_block]:
                if current_counts[s_idx] == 1:
                    critical_tsubs.append(s_idx)
            
            if len(critical_tsubs) == 0:
                # Remove it
                current_set.remove(old_block)
                for s_idx in block_to_tsubs[old_block]:
                    current_counts[s_idx] -= 1
                if len(current_set) < best_len:
                    best_len = len(current_set)
                    best_set = set(current_set)
                T *= cooling
                continue
            
            # Find replacement
            if len(critical_tsubs) > 10:
                T *= cooling
                continue
                
            candidate_set = set(tsub_to_blocks[critical_tsubs[0]])
            for c in critical_tsubs[1:]:
                candidate_set.intersection_update(tsub_to_blocks[c])
            candidate_set.difference_update(current_set)
            
            if not candidate_set:
                T *= cooling
                continue
                
            cand_list = list(candidate_set)
            new_block = cand_list[rng.randrange(len(cand_list))]
            
            # Perform swap
            current_set.remove(old_block)
            for s_idx in block_to_tsubs[old_block]:
                current_counts[s_idx] -= 1
            current_set.add(new_block)
            for s_idx in block_to_tsubs[new_block]:
                current_counts[s_idx] += 1
                
            # Check if better
            if len(current_set) < best_len:
                best_len = len(current_set)
                best_set = set(current_set)
                
            T *= cooling
            
        return list(best_set)

    # Execution
    # Run greedy multiple times with different random seeds or tie-breaking?
    # The greedy above uses random tie-breaking for "hardest" tsub.
    
    best_solution = None
    best_len = float('inf')
    
    # Run a few iterations of Greedy + Local Search
    for i in range(5):
        if time.monotonic() > deadline - 10.0:
            break
            
        sol = greedy_construction()
        sol = remove_redundant(sol)
        sol = local_search(sol)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
    # Run SA on the best solution found
    if best_solution and time.monotonic() < deadline - 10.0:
        sa_sol = simulated_annealing(best_solution)
        if len(sa_sol) < best_len:
            best_len = len(sa_sol)
            best_solution = sa_sol
            
    # Final polish
    if best_solution:
        best_solution = remove_redundant(best_solution)
        best_solution = local_search(best_solution)
        
    return get_blocks(best_solution)

# Test
if __name__ == "__main__":
    v, k, t = 19, 7, 4
    res = construct(v, k, t)
    print(f"Blocks: {len(res)}")
