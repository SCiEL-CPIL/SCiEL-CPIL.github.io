import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    # Precomputation specific to the problem instance
    # v=24, k=7, t=3
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute block_t_indices: for each block, the indices of the t-subsets it covers
    # Shape: (n_ks, C(k,t))
    t_count = len(list(itertools.combinations(range(k), t))) # C(7,3) = 35
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    
    # Precompute triple_to_blocks: for each t-subset, which blocks cover it
    triple_to_blocks = defaultdict(list)
    
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            idx = idx_map[sub]
            block_t_indices[i, j] = idx
            triple_to_blocks[idx].append(i)
            
    # Convert triple_to_blocks to numpy arrays for faster intersection if needed, 
    # or keep as lists. Lists are fine for small intersections.
    
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res

    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        # cover_freq helps in tie-breaking: prefer blocks covering rarer triples
        cover_freq = np.zeros(n_ts, dtype=np.int32)
        
        # Precompute gains? No, covered changes.
        # We can vectorize the gain calculation.
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 10.0:
                return None
            
            # Calculate gain for all blocks
            # covered[block_t_indices] is (n_ks, t_count) bool
            # gain[b] = number of False in covered[block_t_indices[b]]
            # i.e., t_count - sum(covered[block_t_indices[b]])
            
            # This is a large operation: 496k x 35.
            # Let's optimize: 
            # We only need to consider blocks that have at least 1 uncovered triple.
            # But calculating "has at least 1" is also O(n_ks * t_count).
            # Let's just do it. 496k * 35 = 17.3M ops. NumPy handles this in <10ms.
            
            block_covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - block_covered_counts
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Find blocks with max_gain
            max_gain_indices = np.where(gains == max_gain)[0]
            
            # Tie-breaking:
            # Score = max_gain + epsilon * (sum of 1/cover_freq for newly covered)
            # To keep it fast, we can just pick randomly from top K, 
            # or pick the one with highest "rarity score".
            
            # Let's pick randomly from the top 100 max_gain blocks to ensure diversity
            if len(max_gain_indices) > 100:
                candidates = random.sample(max_gain_indices.tolist(), 100)
            else:
                candidates = max_gain_indices.tolist()
                
            best_candidate = candidates[0]
            best_rarity_score = -1
            
            for cand in candidates:
                subs = block_t_indices[cand]
                newly = ~covered[subs]
                if np.any(newly):
                    # Rarity score: sum of 1/(1+freq) for newly covered
                    freqs = cover_freq[subs[newly]]
                    score = np.sum(1.0 / (1.0 + freqs))
                else:
                    score = 0.0
                    
                if score > best_rarity_score:
                    best_rarity_score = score
                    best_candidate = cand
            
            blocks_indices.append(best_candidate)
            
            subs_to_cover = block_t_indices[best_candidate]
            newly_covered = ~covered[subs_to_cover]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            cover_freq[subs_to_cover] += 1
            covered[subs_to_cover] = True
            
        return list(blocks_indices)

    def local_search(initial_blocks):
        if time.time() - start_time > time_limit - 5.0:
            return initial_blocks
            
        current_blocks = list(initial_blocks)
        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
            
        current_set = set(current_blocks)
        improved = True
        
        while improved:
            if time.time() - start_time > time_limit - 5.0:
                break
                
            improved = False
            
            # Phase 1: Remove redundant blocks
            i = 0
            while i < len(current_blocks):
                b_idx = current_blocks[i]
                subs = block_t_indices[b_idx]
                if np.all(coverage_count[subs] > 1):
                    # Remove
                    coverage_count[subs] -= 1
                    current_blocks.pop(i)
                    current_set.remove(b_idx)
                    improved = True
                    # Do not increment i, check next
                else:
                    i += 1
                    
            # Phase 2: Swap to enable removal or just improve structure
            # Try to swap a block B_out with B_in such that B_in covers the "unique" parts of B_out
            # and potentially allows other blocks to be removed?
            # Actually, a swap keeps count constant. We only care about swaps that allow a subsequent removal.
            # So, for each B_out, if we remove it, we have uncovered set U.
            # If we can find B_in covering U, we swap.
            # Then we retry Phase 1.
            
            # To avoid getting stuck in a loop of swaps that don't lead to removal,
            # we can limit the number of swaps or use a tabu list.
            # However, if a swap allows a removal, the count drops.
            # If a swap doesn't allow a removal, it might still be beneficial for future swaps.
            # Let's try a limited number of swaps per removal phase.
            
            swap_attempts = 0
            max_swaps_per_iteration = 50
            found_swap = False
            
            while swap_attempts < max_swaps_per_iteration and not found_swap:
                if time.time() - start_time > time_limit - 5.0:
                    break
                
                # Pick a random block to try swapping
                if not current_blocks:
                    break
                    
                b_out_idx = random.randint(0, len(current_blocks) - 1)
                b_out = current_blocks[b_out_idx]
                
                # Temporarily remove
                subs_out = block_t_indices[b_out]
                coverage_count[subs_out] -= 1
                
                # Find uncovered
                uncovered_mask = (coverage_count == 0)
                uncovered_indices = np.where(uncovered_mask)[0]
                
                if len(uncovered_indices) == 0:
                    # We should have removed this in Phase 1.
                    # This can happen if Phase 1 missed it? No, Phase 1 checks all.
                    # Unless coverage changed? No, we are in a loop.
                    # If this happens, just restore and break.
                    coverage_count[subs_out] += 1
                    break
                
                # Find B_in that covers all uncovered_indices
                # Intersection of triple_to_blocks for each uncovered index
                # Start with the smallest list to speed up
                lists = [triple_to_blocks[idx] for idx in uncovered_indices]
                lists.sort(key=len)
                
                if not lists:
                    coverage_count[subs_out] += 1
                    break
                    
                candidates = set(lists[0])
                for lst in lists[1:]:
                    candidates.intersection_update(lst)
                    if not candidates:
                        break
                
                # Remove current blocks from candidates
                candidates.difference_update(current_set)
                
                if candidates:
                    # Found a valid swap
                    b_in = random.choice(list(candidates))
                    
                    # Perform swap
                    # Remove b_out
                    current_blocks[b_out_idx] = b_in
                    current_set.remove(b_out)
                    current_set.add(b_in)
                    
                    # Update coverage
                    # We already decremented subs_out.
                    # Now increment subs_in
                    subs_in = block_t_indices[b_in]
                    coverage_count[subs_in] += 1
                    
                    # The swap is successful. 
                    # We don't know if it enables a removal yet.
                    # But we set found_swap = True to restart Phase 1.
                    found_swap = True
                    swap_attempts += 1
                    # It's possible the swap didn't help, but Phase 1 will check again.
                    # To prevent infinite swapping without progress, we rely on the fact 
                    # that random swaps eventually find a good configuration or we hit time limit.
                    # A more robust way is to only accept swap if it reduces "potential" 
                    # or just accept it and let Phase 1 try to remove.
                else:
                    # No swap found for this b_out
                    coverage_count[subs_out] += 1
                    swap_attempts += 1
            
            # If we found a swap, the loop continues (improved is technically not set by swap, 
            # but the state changed. We should set improved = True if a swap was made 
            # to allow another pass of Phase 1).
            if found_swap:
                improved = True
            else:
                # No removals, no swaps found
                improved = False
                
        return current_blocks

    best_solution = None
    best_score = float('inf')
    
    # Run multiple restarts
    restarts = 0
    while time.time() - start_time < time_limit - 15.0:
        seed = restarts
        greedy_blocks = run_greedy(seed)
        
        if greedy_blocks is None:
            break
            
        polished_blocks = local_search(greedy_blocks)
        
        if len(polished_blocks) < best_score:
            best_score = len(polished_blocks)
            best_solution = polished_blocks
            
        restarts += 1
        
        # If we hit the known best (76), we might want to keep trying for 75
        # But 76 is the current best known.
        
    if best_solution is None:
        # Fallback
        greedy_blocks = run_greedy(0)
        if greedy_blocks:
            best_solution = local_search(greedy_blocks)
        else:
            return []
            
    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
