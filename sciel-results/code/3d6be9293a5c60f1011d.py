import numpy as np
import itertools
import random
import time
import heapq

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other params
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    M = 3**n
    ball_size = 1 + 2*n
    
    # Precompute balls
    balls = [None] * M
    powers = [3**i for i in range(n)]
    
    for i in range(M):
        ball = [i]
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
                ball.append(i + diff * powers[pos])
            # -1
            new_digit = (digits[pos] - 1) % 3
            diff = new_digit - digits[pos]
            if diff != 0:
                ball.append(i + diff * powers[pos])
        balls[i] = ball
        
    # Precompute coverers
    coverers = [[] for _ in range(M)]
    for c in range(M):
        for p in balls[c]:
            coverers[p].append(c)
            
    start_time = time.time()
    time_limit = 115
    
    def greedy_init():
        code = set()
        uncovered = np.ones(M, dtype=bool)
        uncovered_count = M
        gains = np.full(M, ball_size, dtype=np.int32)
        
        # Max-heap by gain
        heap = [(-gains[c], c) for c in range(M)]
        heapq.heapify(heap)
        
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
            for p in balls[valid_c]:
                if uncovered[p]:
                    uncovered[p] = False
                    uncovered_count -= 1
                    for c_prime in coverers[p]:
                        if c_prime not in code:
                            gains[c_prime] -= 1
                            heapq.heappush(heap, (-gains[c_prime], c_prime))
        return code

    def local_search(code, time_limit):
        count = np.zeros(M, dtype=np.int32)
        for c in code:
            for p in balls[c]:
                count[p] += 1
                
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # 1. Remove redundant
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
            
            # 2. 1-Swap
            code_list = list(code)
            random.shuffle(code_list)
            swapped = False
            for c in code_list:
                if (time.time() - start_time) >= time_limit:
                    break
                unique_points = [p for p in balls[c] if count[p] == 1]
                if not unique_points:
                    # Should be caught by redundancy check, but just in case
                    code.remove(c)
                    for p in balls[c]:
                        count[p] -= 1
                    improved = True
                    swapped = True
                    break
                
                candidates = set(coverers[unique_points[0]])
                for p in unique_points[1:]:
                    candidates.intersection_update(coverers[p])
                candidates -= code
                
                if candidates:
                    # Pick random candidate
                    c_new = random.choice(list(candidates))
                    code.remove(c)
                    for p in balls[c]:
                        count[p] -= 1
                    code.add(c_new)
                    for p in balls[c_new]:
                        count[p] += 1
                    improved = True
                    swapped = True
                    break
            if swapped:
                continue
                
            # 3. Perturbation if stuck
            if (time.time() - start_time) < time_limit - 5:
                # Remove 3-5 random codewords
                k = random.randint(3, 5)
                to_perturb = random.sample(list(code), min(k, len(code)))
                for c in to_perturb:
                    code.remove(c)
                    for p in balls[c]:
                        count[p] -= 1
                
                # Refill greedily
                # Recalculate gains for uncovered points?
                # Simple greedy refill
                uncovered = np.zeros(M, dtype=bool)
                for p in range(M):
                    if count[p] == 0:
                        uncovered[p] = True
                
                if np.any(uncovered):
                    # Use a simplified greedy: pick codeword covering most uncovered
                    # This is O(K * M) which is slow if done naively.
                    # We'll just add random codewords from coverers of uncovered points
                    # until covered.
                    while np.any(uncovered):
                        # Pick a random uncovered point
                        unc_idx = np.where(uncovered)[0]
                        if len(unc_idx) == 0: break
                        p = unc_idx[0]
                        # Pick a random codeword covering p
                        c_new = random.choice(coverers[p])
                        if c_new not in code:
                            code.add(c_new)
                            for pp in balls[c_new]:
                                count[pp] += 1
                                if count[pp] == 1:
                                    uncovered[pp] = False
                            
        return code

    # Main Loop: ILS
    best_code = greedy_init()
    best_size = len(best_code)
    
    # Initial Local Search
    best_code = local_search(best_code, time_limit)
    best_size = len(best_code)
    
    while (time.time() - start_time) < time_limit:
        # Perturb
        current_code = set(best_code)
        k = random.randint(5, 10)
        if len(current_code) > k:
            to_remove = random.sample(list(current_code), k)
            for c in to_remove:
                current_code.remove(c)
        
        # Refill
        # Calculate uncovered
        count = np.zeros(M, dtype=np.int32)
        for c in current_code:
            for p in balls[c]:
                count[p] += 1
        
        uncovered = count == 0
        if np.any(uncovered):
            # Greedy refill
            # To speed up, we can just add random coverers of uncovered points
            unc_points = np.where(uncovered)[0]
            random.shuffle(unc_points)
            for p in unc_points:
                if not uncovered[p]: continue
                c_new = random.choice(coverers[p])
                if c_new not in current_code:
                    current_code.add(c_new)
                    for pp in balls[c_new]:
                        count[pp] += 1
                        if count[pp] == 1:
                            uncovered[pp] = False
                            
        # Local Search
        current_code = local_search(current_code, time_limit)
        
        if len(current_code) < best_size:
            best_code = current_code
            best_size = len(current_code)
            
    return [list(map(int, [ (c // powers[i]) % 3 for i in range(n) ])) for c in best_code]
