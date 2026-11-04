import numpy as np
import time
import itertools

def construct(q, n, radius):
    # The task specifies q=3, n=8, radius=1.
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute the coverage sets for every possible codeword index.
    # balls[i] is a list of indices of points covered by codeword i
    balls = np.empty((total_points, ball_size), dtype=np.int32)
    
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    # Precompute digits
    digits = np.zeros((total_points, n), dtype=np.int8)
    for idx in range(total_points):
        temp = idx
        for pos in range(n):
            digits[idx, pos] = temp % q
            temp //= q

    for idx in range(total_points):
        balls[idx, 0] = idx
        neighbor_idx = 1
        for pos in range(n):
            original_digit = digits[idx, pos]
            power = powers[pos]
            
            # +1
            delta = power
            if (original_digit + 1) % q == 0:
                delta = -2 * power
            balls[idx, neighbor_idx] = idx + delta
            neighbor_idx += 1
            
            # -1
            delta = -power
            if (original_digit - 1) % q == 2:
                delta = 2 * power
            balls[idx, neighbor_idx] = idx + delta
            neighbor_idx += 1

    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Greedy Set Cover
    is_covered = np.zeros(total_points, dtype=bool)
    code_set = set()
    code_list = []
    
    while not np.all(is_covered):
        if time.time() > deadline - 20.0:
            break
        # Calculate gain for each candidate
        # gain[i] = number of uncovered points in balls[i]
        # Vectorized: is_covered[balls] is (6561, 17). ~is_covered[balls] is bool.
        # Sum over axis 1.
        gains = np.sum(~is_covered[balls], axis=1)
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break # Should not happen if not all covered
        
        code_set.add(best_idx)
        code_list.append(best_idx)
        is_covered[balls[best_idx]] = True

    # 2. Pruning and Local Search
    # Maintain coverage_count to quickly identify unique points
    coverage_count = np.zeros(total_points, dtype=np.int32)
    for idx in code_list:
        coverage_count[balls[idx]] += 1

    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        
        # Shuffle code_list to avoid bias? Or just iterate.
        # We need to be careful with indices if we modify the list.
        # We will iterate over a copy or handle removals carefully.
        # Since we might remove items, we iterate in reverse or use a while loop with index.
        
        # Strategy: Try to remove or swap each codeword.
        # We process codewords in order of increasing unique count to prioritize "weak" ones.
        # Calculate unique counts
        unique_counts = np.array([np.sum(coverage_count[balls[idx]] == 1) for idx in code_list])
        
        # Get indices of codewords to try, sorted by unique count
        order = np.argsort(unique_counts)
        
        for pos in order:
            if time.time() > deadline - 5.0:
                break
            
            c_idx = code_list[pos]
            
            # Find points uniquely covered by c_idx
            ball_c = balls[c_idx]
            unique_mask = (coverage_count[ball_c] == 1)
            unique_points = ball_c[unique_mask]
            
            if len(unique_points) == 0:
                # c_idx is redundant, remove it
                coverage_count[ball_c] -= 1
                code_set.remove(c_idx)
                code_list.pop(pos)
                improved = True
                # After removal, unique counts change. Break and restart the loop to recalculate.
                break
            else:
                # c_idx is not redundant. Try to swap it.
                # We need to find a p not in code_set such that Ball(p) covers all unique_points.
                # If such p exists, swap c_idx with p.
                # Then check if this swap makes any other codeword redundant.
                
                # Optimization: If len(unique_points) > 3, it's unlikely to be covered by a single ball
                # (max intersection of balls is small). But let's check up to 4 or 5.
                if len(unique_points) > 4:
                    continue
                
                found_swap = False
                # Iterate through all possible candidates p
                # To speed up, we can filter candidates that cover the first unique point.
                # Candidates must be in Ball(unique_points[0]).
                first_point = unique_points[0]
                candidates = balls[first_point] # Points p such that first_point in Ball(p)
                
                for p in candidates:
                    if p in code_set:
                        continue
                    # Check if p covers all unique_points
                    # unique_points are in Ball(p) if they are in balls[p]
                    # Since |unique_points| is small, we can check membership.
                    # Convert balls[p] to a set for fast lookup? 
                    # Or just check if all up are in balls[p].
                    # balls[p] is an array of 17 ints.
                    if all(up in balls[p] for up in unique_points):
                        # Valid swap found
                        # Perform swap
                        code_set.remove(c_idx)
                        code_set.add(p)
                        code_list[pos] = p
                        
                        coverage_count[ball_c] -= 1
                        coverage_count[balls[p]] += 1
                        
                        # Check if this swap allows removing another codeword
                        # Recalculate unique counts for all codewords? 
                        # Or just check if any codeword now has 0 unique points.
                        # We can check all codewords in code_list.
                        removal_found = False
                        for j in range(len(code_list)):
                            if j == pos:
                                continue
                            cj = code_list[j]
                            if np.sum(coverage_count[balls[cj]] == 1) == 0:
                                # Remove cj
                                coverage_count[balls[cj]] -= 1
                                code_set.remove(cj)
                                code_list.pop(j)
                                removal_found = True
                                break
                        
                        improved = True
                        found_swap = True
                        break # Break candidate loop, restart main loop to recalculate unique counts
                
                if found_swap:
                    break # Restart main loop

    # Final Pruning pass (in case we exited loop early)
    if len(code_list) > 0:
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in code_list:
            coverage_count[balls[idx]] += 1
        
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            for i in range(len(code_list)):
                if np.sum(coverage_count[balls[code_list[i]]] == 1) == 0:
                    coverage_count[balls[code_list[i]]] -= 1
                    code_list.pop(i)
                    changed = True
                    break

    # Convert indices to words
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    result = [idx_to_word(idx) for idx in code_list]
    return result
