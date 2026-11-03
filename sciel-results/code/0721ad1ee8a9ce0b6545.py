import numpy as np
import itertools
import random
import time
import heapq

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other params, though task is specific
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    M = 3**n
    ball_size = 1 + 2*n # 17
    
    # Precompute balls as lists of indices
    # balls[i] contains i and its 16 neighbors
    balls = [None] * M
    # Precompute neighbors for faster access
    # We can compute neighbors on the fly or precompute.
    # Precomputing 6561 * 17 ints is cheap.
    
    # Helper to convert index to digits and back
    # But we can compute neighbors directly from index using arithmetic
    
    for i in range(M):
        ball = [i]
        # For each position, change digit to +1 and -1
        x = i
        # We need digits to know what to change
        # Or we can just use arithmetic:
        # pos 0: i + 1, i - 1 (with wrap)
        # pos 1: i + 3, i - 3
        # ...
        # pos k: i + 3^k, i - 3^k
        # Wrap around: if digit becomes 3, subtract 3^(k+1). If -1, add 3^(k+1).
        
        # Let's stick to digit conversion for safety and speed (it's fast enough)
        digits = [0]*n
        temp = i
        for j in range(n):
            digits[j] = temp % 3
            temp //= 3
        
        for pos in range(n):
            # +1
            new_digit = (digits[pos] + 1) % 3
            diff = new_digit - digits[pos]
            if diff != 0:
                # If diff is 1, add 3^pos. If diff is -2 (2->0), subtract 2*3^pos.
                # Actually: new_val = i + diff * (3**pos)
                # But if wrap occurred, the diff logic above handles it if we use the digit values?
                # No, `diff` is calculated from digits.
                # Example: digit 2, new 0. diff = -2.
                # i + (-2)*3^pos. Correct.
                # Example: digit 0, new 1. diff = 1.
                # i + 1*3^pos. Correct.
                ball.append(i + diff * (3**pos))
                
            # -1
            new_digit = (digits[pos] - 1) % 3
            diff = new_digit - digits[pos]
            if diff != 0:
                ball.append(i + diff * (3**pos))
        
        balls[i] = ball
        
    # Precompute coverers: for each point p, list of codewords c such that p in balls[c]
    coverers = [[] for _ in range(M)]
    for c in range(M):
        for p in balls[c]:
            coverers[p].append(c)
            
    # 1. Greedy Set Cover
    code = set()
    uncovered = np.ones(M, dtype=bool)
    gains = np.full(M, ball_size, dtype=np.int32)
    
    # Use a heap for max gain
    # Heap entries: (-gain, codeword)
    heap = [(-ball_size, c) for c in range(M)]
    heapq.heapify(heap)
    
    uncovered_count = M
    
    while uncovered_count > 0:
        valid_c = None
        while heap:
            neg_g, c = heapq.heappop(heap)
            if c in code:
                continue
            if -neg_g != gains[c]:
                continue
            if gains[c] <= 0:
                continue
            valid_c = c
            break
            
        if valid_c is None:
            break
            
        code.add(valid_c)
        
        # Update uncovered and gains
        for p in balls[valid_c]:
            if uncovered[p]:
                uncovered[p] = False
                uncovered_count -= 1
                # Decrease gain for all codewords covering p
                # This is the expensive part: O(17 * avg_degree)
                # avg_degree of coverers[p] is roughly (M * ball_size) / M = ball_size = 17.
                # So 17 * 17 = 289 updates per selected codeword.
                # Total K * 289. K~500. 500*289 = 144,500 operations. Very fast.
                for c_prime in coverers[p]:
                    if c_prime not in code:
                        gains[c_prime] -= 1
                        heapq.heappush(heap, (-gains[c_prime], c_prime))
                        
    # 2. Local Search
    start_time = time.time()
    time_limit = 115
    
    # Convert code to list for iteration
    # Use NumPy for counts
    count = np.zeros(M, dtype=np.int32)
    for c in code:
        for p in balls[c]:
            count[p] += 1
            
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Phase 1: Remove redundant
        # A codeword c is redundant if for all p in balls[c], count[p] > 1
        to_remove = []
        for c in list(code):
            redundant = True
            for p in balls[c]:
                if count[p] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(c)
                
        if to_remove:
            for c in to_remove:
                code.remove(c)
                for p in balls[c]:
                    count[p] -= 1
            improved = True
            continue
        
        # Phase 2: Swap to enable removals
        # For each c, find unique points U_c
        # If U_c is not empty, find c' covering U_c
        # If found, swap c -> c'
        
        code_list = list(code)
        random.shuffle(code_list)
        
        swapped = False
        for c in code_list:
            if (time.time() - start_time) >= time_limit:
                break
            
            unique_points = [p for p in balls[c] if count[p] == 1]
            
            if not unique_points:
                # Should have been caught in Phase 1, but maybe race condition?
                # Or maybe we just removed it? No, we continue if to_remove.
                # If unique_points is empty, it is redundant.
                code.remove(c)
                for p in balls[c]:
                    count[p] -= 1
                improved = True
                swapped = True
                break
            
            # Find candidates for c'
            # c' must cover all points in unique_points
            candidates = set(coverers[unique_points[0]])
            for p in unique_points[1:]:
                candidates.intersection_update(coverers[p])
            
            # Remove current code
            candidates -= code
            
            if candidates:
                # Pick a random candidate? Or one that covers most other points?
                # To minimize size, we just need *any* valid swap.
                # The swap keeps size constant. The hope is that the new structure
                # allows a removal in the next iteration.
                # To be more effective, pick c' that maximizes "redundancy potential"?
                # i.e., c' covers many points that are already covered by >1.
                # This might make those points "more redundant"? No.
                # Let's just pick one randomly or the one with highest degree.
                
                # Optimization: Limit candidates to avoid huge intersections?
                # unique_points is usually small (1-3).
                # Intersection of 3 lists of size ~17 is small.
                
                cand_list = list(candidates)
                random.shuffle(cand_list)
                
                # Heuristic: Pick c' that covers the most points with count > 1
                # This might help?
                # Actually, just picking one is fine for ILS.
                c_new = cand_list[0]
                
                # Perform Swap
                code.remove(c)
                for p in balls[c]:
                    count[p] -= 1
                    
                code.add(c_new)
                for p in balls[c_new]:
                    count[p] += 1
                    
                improved = True
                swapped = True
                break
                
        if not swapped:
            # No improvement found, break or perturb?
            # To escape local minima, we can perturb:
            # Remove k random codewords, re-add greedily.
            if (time.time() - start_time) < time_limit:
                # Perturbation
                k = 5
                to_perturb = random.sample(list(code), k)
                for c in to_perturb:
                    code.remove(c)
                    for p in balls[c]:
                        count[p] -= 1
                
                # Re-add greedily
                # Recalculate gains?
                # Just run a few steps of greedy
                # For simplicity, re-run the whole greedy on the remaining uncovered?
                # No, just add random codewords until covered?
                # Better: Re-run the "Greedy" logic on the current state.
                
                # Update gains for the current state
                # This is expensive.
                # Alternative: Just add random codewords that cover the most uncovered points.
                
                # Let's just break and rely on the initial greedy being good.
                # Or, do a "Random Restart" of the whole process?
                # That might be too slow.
                
                # Let's try a simple perturbation:
                # Add a random codeword, then run Local Search again.
                # This increases size, but might lead to a smaller local minimum.
                
                # Actually, if no swap is found, we are in a local minimum for 1-move.
                # We can try 2-move swaps, but that's complex.
                # Let's just break.
                break

    return [list(map(int, [ (c // (3**i)) % 3 for i in range(n) ])) for c in code]
