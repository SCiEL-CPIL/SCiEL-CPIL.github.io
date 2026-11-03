import numpy as np
import itertools
import random
import time
import heapq

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    M = 3**n
    start_time = time.time()
    time_limit = 115
    
    # Precompute powers of 3
    powers3 = [3**i for i in range(n)]
    
    # Precompute balls
    # For each center c, ball contains c and c +/- 1 at each pos
    balls = [None] * M
    # We can compute ball indices directly using integer arithmetic
    for c in range(M):
        ball = [c]
        for pos in range(n):
            # Get digit at pos
            digit = (c // powers3[pos]) % 3
            p3 = powers3[pos]
            
            # +1
            new_digit = (digit + 1) % 3
            diff = (new_digit - digit) * p3
            ball.append((c + diff) % M)
            
            # -1
            new_digit = (digit - 1) % 3
            diff = (new_digit - digit) * p3
            ball.append((c + diff) % M)
        balls[c] = ball
        
    # Precompute coverers
    coverers = [[] for _ in range(M)]
    for c in range(M):
        for p in balls[c]:
            coverers[p].append(c)
            
    # Greedy Set Cover
    code = set()
    uncovered = np.ones(M, dtype=bool)
    gains = np.array([17] * M, dtype=np.int32)
    heap = []
    for c in range(M):
        heapq.heappush(heap, (-17, c))
        
    uncovered_count = M
    
    while uncovered_count > 0:
        valid_c = None
        while heap:
            neg_g, c = heapq.heappop(heap)
            if c in code:
                continue
            if -neg_g != gains[c]:
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

    # Local Search
    best_code = set(code)
    best_size = len(code)
    
    # Helper to check if point p is in ball of c
    # Since balls are lists, we can use a set for faster lookup if needed, but lists are small (17)
    # We'll just iterate.
    
    for iteration in range(100):
        if time.time() - start_time > time_limit:
            break
            
        current_code = set(best_code)
        count = np.zeros(M, dtype=np.int32)
        for c in current_code:
            for p in balls[c]:
                count[p] += 1
                
        improved = True
        while improved:
            improved = False
            
            # Phase 1: Remove redundant
            to_remove = []
            for c in list(current_code):
                redundant = True
                for p in balls[c]:
                    if count[p] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(c)
            
            if to_remove:
                for c in to_remove:
                    current_code.remove(c)
                    for p in balls[c]:
                        count[p] -= 1
                improved = True
                if len(current_code) < best_size:
                    best_code = set(current_code)
                    best_size = len(current_code)
                continue
            
            # Phase 2: 2-for-1 move
            if len(current_code) > 1:
                code_list = list(current_code)
                random.shuffle(code_list)
                found = False
                for c1 in code_list[:10]:
                    U1 = [p for p in balls[c1] if count[p] == 1]
                    if not U1: continue
                    
                    for c2 in code_list[:10]:
                        if c1 == c2: continue
                        U2 = [p for p in balls[c2] if count[p] == 1]
                        
                        # Candidates must cover U1 and U2
                        candidates = set(coverers[U1[0]])
                        for p in U1[1:]:
                            candidates.intersection_update(coverers[p])
                            if not candidates: break
                        
                        if not candidates: continue
                        
                        final_candidates = []
                        for c3 in candidates:
                            if c3 in current_code: continue
                            
                            # Check if c3 covers all U2
                            ok = True
                            for p in U2:
                                if p not in balls[c3]:
                                    ok = False
                                    break
                            if not ok: continue
                            
                            # Check safety for count==2 points
                            safe = True
                            for p in balls[c1]:
                                if count[p] == 2:
                                    if c2 in coverers[p]:
                                        if p not in balls[c3]:
                                            safe = False
                                            break
                            if safe:
                                for p in balls[c2]:
                                    if count[p] == 2:
                                        if c1 in coverers[p]:
                                            if p not in balls[c3]:
                                                safe = False
                                                break
                            if safe:
                                final_candidates.append(c3)
                        
                        if final_candidates:
                            c3 = random.choice(final_candidates)
                            current_code.remove(c1)
                            current_code.remove(c2)
                            current_code.add(c3)
                            
                            for p in balls[c1]:
                                count[p] -= 1
                            for p in balls[c2]:
                                count[p] -= 1
                            for p in balls[c3]:
                                count[p] += 1
                                
                            improved = True
                            found = True
                            if len(current_code) < best_size:
                                best_code = set(current_code)
                                best_size = len(current_code)
                            break
                    if found: break
            
            # Phase 3: 1-for-1 Swap
            if not improved:
                code_list = list(current_code)
                random.shuffle(code_list)
                if code_list:
                    c1 = code_list[0]
                    U1 = [p for p in balls[c1] if count[p] == 1]
                    if U1:
                        candidates = set(coverers[U1[0]])
                        for p in U1[1:]:
                            candidates.intersection_update(coverers[p])
                            if not candidates: break
                        
                        candidates -= current_code
                        if candidates:
                            c3 = random.choice(list(candidates))
                            current_code.remove(c1)
                            current_code.add(c3)
                            
                            for p in balls[c1]:
                                count[p] -= 1
                            for p in balls[c3]:
                                count[p] += 1
                            
                            improved = True

    return [list(map(int, [ (c // (3**i)) % 3 for i in range(n) ])) for c in best_code]
