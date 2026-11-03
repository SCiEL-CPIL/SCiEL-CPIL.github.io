import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety
    
    # For this specific task, q=3, n=6, radius=1
    # We assume the evaluator passes these, but we handle the general case if possible, 
    # though the problem is specific. The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
    # However, the seed program hardcodes the logic for n=6. 
    # If n or q change, the direct sum might not be optimal or valid.
    # But the task description says "Football pool problem for 6 matches... K_3(6,1)".
    # It's safer to assume q=3, n=6, radius=1 is the only case or we should handle general if possible.
    # Given the "Direct sum" seed, it implies a specific structure.
    # Let's assume the evaluator calls construct(3, 6, 1).
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to seed logic if parameters differ, to ensure validity
        import itertools
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
             # Brute force or simple heuristic for small n
             return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=6, radius=1
    total_words = q ** n
    # Map word (tuple of ints) to index 0..728
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute coverage masks for all 729 codewords
    # cover_mask[i] is a set (or bitmask) of words covered by codeword i
    # Using a list of sets might be slower than bitmasks, but Python ints are fast for bit ops.
    # Let's use integer bitmasks.
    cover_masks = [0] * total_words
    
    for i in range(total_words):
        w = idx_to_word(i)
        mask = 1 << i
        # Add neighbors
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                j = word_to_idx(new_w)
                mask |= (1 << j)
        cover_masks[i] = mask

    # Greedy Set Cover
    uncovered = (1 << total_words) - 1
    solution = []
    
    while uncovered:
        best_idx = -1
        max_covered = 0
        for i in range(total_words):
            if uncovered == 0:
                break
            # Count bits in (cover_masks[i] & uncovered)
            # Python 3.10+ has int.bit_count()
            cnt = (cover_masks[i] & uncovered).bit_count()
            if cnt > max_covered:
                max_covered = cnt
                best_idx = i
        if best_idx == -1:
            break
        solution.append(best_idx)
        uncovered &= ~cover_masks[best_idx]
        
        if time.time() > deadline:
            break

    # Local Search to reduce size
    # Maintain cover_count array
    cover_count = np.zeros(total_words, dtype=np.int32)
    for idx in solution:
        mask = cover_masks[idx]
        # Iterate set bits in mask to update counts
        # This is slow if done naively. 
        # Alternative: Just recompute validity from scratch? 
        # With 73 codewords, recomputing is 73 * 19 updates = 1387 ops. Very fast.
        
    def is_valid(sol):
        counts = [0] * total_words
        for idx in sol:
            mask = cover_masks[idx]
            # Update counts for bits in mask
            m = mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] += 1
                m ^= lsb
        return min(counts) > 0

    # Faster validity check using numpy?
    # Or just a simple loop since N is small.
    
    def try_remove(sol, idx_to_remove):
        new_sol = [x for x in sol if x != idx_to_remove]
        if len(new_sol) == len(sol):
            return None
        if is_valid(new_sol):
            return new_sol
        return None

    def local_search(sol):
        improved = True
        while improved:
            improved = False
            current_len = len(sol)
            # Try removing each codeword
            for i in range(len(sol)):
                if time.time() > deadline:
                    return sol
                new_sol = try_remove(sol, sol[i])
                if new_sol is not None:
                    sol = new_sol
                    improved = True
                    break # Restart loop with new sol
            
            if improved:
                continue
                
            # If no simple removal worked, try swaps
            # For each i in sol, for each j not in sol, try swap
            # This is O(N_sol * N_total * ValidityCost)
            # N_sol ~ 73, N_total = 729. 73 * 729 = 53k swaps.
            # ValidityCost ~ 1400 ops. 53k * 1400 = 74M ops. Might be too slow in Python if not optimized.
            
            # Optimization: Only try swaps that cover "deficient" words.
            # Compute deficiency: words with count 0 if we removed sol[i].
            
            for i in range(len(sol)):
                if time.time() > deadline:
                    return sol
                
                # Check if sol[i] is critical
                # Temporarily remove sol[i]
                sol_minus_i = [x for x in sol if x != sol[i]]
                
                # Find uncovered words
                counts = [0] * total_words
                for idx in sol_minus_i:
                    m = cover_masks[idx]
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        counts[bit_idx] += 1
                        m ^= lsb
                        
                deficient = [k for k in range(total_words) if counts[k] == 0]
                if not deficient:
                    # Should have been caught by try_remove
                    sol = sol_minus_i
                    improved = True
                    break
                
                # Find best j not in sol that covers most deficient words
                best_j = -1
                max_def_covered = 0
                sol_set = set(sol)
                
                for j in range(total_words):
                    if j in sol_set:
                        continue
                    # Count how many deficient words j covers
                    # deficient is a list. Check intersection with cover_masks[j]
                    # This is slow if deficient is large.
                    # But deficient should be small if sol is good.
                    
                    # Faster: Check if j covers any deficient word
                    # We can just count
                    cnt = 0
                    for d in deficient:
                        if (cover_masks[j] >> d) & 1:
                            cnt += 1
                            if cnt > max_def_covered:
                                max_def_covered = cnt
                                best_j = j
                    # Early termination if we found a perfect cover for all deficient?
                    # If max_def_covered == len(deficient), we can stop searching for j
                    if max_def_covered == len(deficient):
                         # We found a j that covers all deficient words.
                         # We still need to check if the resulting set allows removal of another word.
                         # But first, let's just see if this j is the best.
                         pass

                if best_j != -1 and max_def_covered > 0:
                    # Try swapping sol[i] with best_j
                    new_sol = sol_minus_i + [best_j]
                    if is_valid(new_sol):
                        # Now try to remove any word from new_sol
                        # This is a "Swap and Remove"
                        for k in range(len(new_sol)):
                            if time.time() > deadline:
                                break
                            rem_sol = try_remove(new_sol, new_sol[k])
                            if rem_sol is not None:
                                sol = rem_sol
                                improved = True
                                break
                        if improved:
                            break
                            
        return sol

    solution = local_search(solution)
    
    # Convert indices back to words
    return [idx_to_word(idx) for idx in solution]
