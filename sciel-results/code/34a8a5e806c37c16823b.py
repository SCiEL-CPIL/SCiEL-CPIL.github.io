import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(all_t_subsets)
    
    # Map t-subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute block_subsets as lists of indices into all_t_subsets
    # Also subset_to_blocks for quick lookup
    block_subsets = [[] for _ in range(n_blocks)]
    subset_to_blocks = [[] for _ in range(n_t_subsets)]
    
    # Coverage matrix for greedy (n_blocks x n_t_subsets)
    # Using uint8 or bool. n_blocks=50388, n_t_subsets=3876. 
    # 50388 * 3876 * 1 byte ~ 195 MB. This is large but manageable.
    # Alternatively, we can just use the lists for greedy if we optimize the gain calc.
    # Let's try to avoid the huge matrix if possible, or use it only for greedy.
    # For local search, lists are faster.
    
    # Let's build the lists first.
    for i, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            block_subsets[i].append(idx)
            subset_to_blocks[idx].append(i)
            
    # Convert block_subsets to numpy arrays for faster indexing in local search?
    # block_subsets_np = [np.array(s, dtype=np.int32) for s in block_subsets]
    
    def greedy_construct():
        """
        Greedy construction using vectorized gain calculation.
        """
        in_block = np.zeros(n_blocks, dtype=bool)
        cover_count = np.zeros(n_t_subsets, dtype=np.int32)
        uncovered_count = n_t_subsets
        solution = []
        
        # We need a way to compute gains efficiently.
        # gain[i] = number of uncovered subsets in block_subsets[i]
        # If we don't have the matrix, we have to loop.
        # 50388 blocks * 35 subsets = 1.7M operations per step.
        # ~150 steps * 1.7M = 255M operations. In Python, this is too slow.
        # We MUST use the matrix or a faster method.
        
        # Let's build the matrix now. It's a one-time cost.
        coverage_matrix = np.zeros((n_blocks, n_t_subsets), dtype=bool)
        for i, subs in enumerate(block_subsets):
            if subs:
                coverage_matrix[i, subs] = True
                
        while uncovered_count > 0:
            # Calculate gain for all blocks
            uncovered_mask = (cover_count == 0)
            # gains[i] = sum(coverage_matrix[i] & uncovered_mask)
            gains = np.sum(coverage_matrix & uncovered_mask, axis=1)
            gains[in_block] = -1
            
            best_block = np.argmax(gains)
            max_gain = gains[best_block]
            
            if max_gain <= 0:
                break
                
            in_block[best_block] = True
            solution.append(best_block)
            
            # Update cover counts
            new_covers = coverage_matrix[best_block] & uncovered_mask
            cover_count += new_covers.astype(np.int32)
            uncovered_count -= np.sum(new_covers)
            
        return solution

    def remove_redundants(solution, cover_count):
        """
        Remove redundant blocks from solution.
        """
        changed = True
        while changed:
            changed = False
            for b in list(solution):
                subs = block_subsets[b]
                # Check if any subset in b is covered only once
                # If cover_count[s] == 1 for any s in subs, b is critical.
                # If no such s, b is redundant.
                is_critical = False
                for s_idx in subs:
                    if cover_count[s_idx] == 1:
                        is_critical = True
                        break
                if not is_critical:
                    # Remove b
                    solution.remove(b)
                    for s_idx in subs:
                        cover_count[s_idx] -= 1
                    changed = True
                    break # Restart loop to catch new redundants
                    
        return solution, cover_count

    # Initial Greedy
    sol = greedy_construct()
    cover_count = np.zeros(n_t_subsets, dtype=np.int32)
    for b in sol:
        cover_count += coverage_matrix[b].astype(np.int32)
        
    sol, cover_count = remove_redundants(sol, cover_count)
    
    best_solution = list(sol)
    best_count = len(best_solution)
    
    # Local Search
    # We use a simple swap-based local search with simulated annealing.
    # Move: Remove one block, add one valid block.
    # Objective: Minimize len(solution).
    # Secondary: Maximize "slack" or reduce "criticality".
    
    current_solution = list(best_solution)
    current_cover_count = cover_count.copy()
    current_len = len(current_solution)
    
    T = 1.0
    cooling_rate = 0.99
    min_T = 0.01
    
    # Precompute block_subsets as sets for faster intersection?
    # block_subsets_set = [set(s) for s in block_subsets]
    
    while time.time() < deadline:
        if not current_solution:
            break
            
        # Pick a random block to remove
        b_out_idx = random.randint(0, len(current_solution) - 1)
        b_out = current_solution[b_out_idx]
        
        # Find critical subsets for b_out
        criticals = []
        subs_out = block_subsets[b_out]
        for s_idx in subs_out:
            if current_cover_count[s_idx] == 1:
                criticals.append(s_idx)
                
        if not criticals:
            # b_out is redundant, remove it
            current_solution.pop(b_out_idx)
            for s_idx in subs_out:
                current_cover_count[s_idx] -= 1
            current_len -= 1
            # Remove other redundants
            current_solution, current_cover_count = remove_redundants(current_solution, current_cover_count)
            current_len = len(current_solution)
            
            if current_len < best_count:
                best_count = current_len
                best_solution = list(current_solution)
            continue
            
        # Find candidate b_in that covers all criticals
        # Candidates are blocks that cover criticals[0]
        candidates = subset_to_blocks[criticals[0]]
        
        best_candidate = -1
        best_score = -float('inf')
        
        # To speed up, we can limit the number of candidates we check or use a heuristic.
        # For each candidate, check if it covers all other criticals.
        # If yes, calculate a score.
        
        # Heuristic Score: 
        # We want to add a block that "helps" other blocks be removable.
        # A block is removable if it has no critical subsets.
        # Adding B_in increases cover count of subsets in B_in.
        # If a subset s in B_in was covered exactly once by B_x (x != B_out), 
        # then after adding B_in, s is covered by 2 blocks. 
        # This might allow B_x to be removed if B_x has no other critical subsets.
        # So, score = number of subsets in B_in that are currently covered exactly once 
        # by a block other than B_out.
        
        # To calculate this efficiently:
        # For each candidate B_in:
        #   1. Check if it covers all criticals.
        #   2. If yes, count subsets s in B_in such that cover_count[s] == 1 and B_in is not the only cover?
        #      Actually, if cover_count[s] == 1, the only cover is some block B_x.
        #      If we add B_in, cover_count[s] becomes 2.
        #      This is good if B_x is different from B_out.
        #      If B_x is B_out, it's in criticals, and we already know B_in covers it.
        #      So we just count subsets s in B_in where cover_count[s] == 1.
        #      Wait, if s is in criticals, cover_count[s] is 1 (by B_out). 
        #      Adding B_in makes it 2. This is necessary for validity.
        #      For other s in B_in, if cover_count[s] == 1 (by B_x != B_out), 
        #      adding B_in makes it 2. This is "slack" creation.
        
        found_valid = False
        best_score = -1
        
        # Limit candidates to first 100 or so to keep it fast?
        # Or iterate all?
        for b_in in candidates:
            if b_in in current_solution:
                continue
                
            # Check if b_in covers all criticals
            valid = True
            # Optimization: check criticals[1:] against b_in's subsets
            # We can use a set for b_in's subsets for O(1) lookup?
            # But we are in a loop.
            # Let's use the coverage_matrix for this check?
            # coverage_matrix[b_in, criticals] should be all True.
            if not np.all(coverage_matrix[b_in, criticals]):
                continue
                
            # Calculate score
            # Score = number of subsets in b_in that are currently covered exactly once
            # AND are not in criticals (since criticals are already forced)
            # Actually, just count all subsets in b_in with cover_count == 1.
            # This includes criticals. But criticals are good to cover.
            # Let's just use the count of cover_count == 1 in b_in.
            
            subs_in = block_subsets[b_in]
            # count = sum(1 for s in subs_in if current_cover_count[s] == 1)
            # Vectorized:
            count = np.sum(current_cover_count[subs_in] == 1)
            
            if count > best_score:
                best_score = count
                best_candidate = b_in
                found_valid = True
                
        if found_valid:
            # Perform swap
            current_solution.remove(b_out)
            for s_idx in subs_out:
                current_cover_count[s_idx] -= 1
                
            current_solution.append(best_candidate)
            subs_in = block_subsets[best_candidate]
            for s_idx in subs_in:
                current_cover_count[s_idx] += 1
                
            # Remove redundants
            current_solution, current_cover_count = remove_redundants(current_solution, current_cover_count)
            current_len = len(current_solution)
            
            if current_len < best_count:
                best_count = current_len
                best_solution = list(current_solution)
                
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T

    # Final polish
    best_solution, _ = remove_redundants(best_solution, cover_count)
    
    return [list(all_blocks[i]) for i in best_solution]
