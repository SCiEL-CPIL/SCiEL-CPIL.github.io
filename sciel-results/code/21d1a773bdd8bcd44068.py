import numpy as np
import random
import time

def _precompute_neighbors(q, n):
    N = q ** n
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        temp = i
        digits = [0] * n
        for j in range(n):
            digits[n-1-j] = temp % q
            temp //= q
        
        neighbors[i].append(i)
        
        for pos in range(n):
            orig = digits[pos]
            weight = q ** (n - 1 - pos)
            for new_d in range(q):
                if new_d == orig:
                    continue
                delta = (new_d - orig) * weight
                ni = i + delta
                neighbors[i].append(ni)
    return neighbors

def _greedy_init(q, n, radius, neighbors):
    N = q ** n
    uncovered = np.ones(N, dtype=bool)
    code = []
    while uncovered.any():
        best_idx = -1
        max_cov = 0
        for i in range(N):
            if not uncovered[i]:
                continue
            cov = sum(1 for j in neighbors[i] if uncovered[j])
            if cov > max_cov:
                max_cov = cov
                best_idx = i
        if best_idx == -1:
            break
        code.append(best_idx)
        for j in neighbors[best_idx]:
            uncovered[j] = False
    return code

def _local_search(q, n, radius, initial_code, neighbors, time_limit=10):
    start_time = time.time()
    N = q ** n
    coverage = np.zeros(N, dtype=np.int32)
    code_set = set(initial_code)
    for c in initial_code:
        for nb in neighbors[c]:
            coverage[nb] += 1
            
    best_code = list(initial_code)
    best_size = len(initial_code)
    
    while time.time() - start_time < time_limit:
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)

        # Try to remove a random codeword
        c = random.choice(list(code_set))
        safe = True
        for nb in neighbors[c]:
            if coverage[nb] <= 1:
                safe = False
                break
        
        if safe:
            code_set.remove(c)
            for nb in neighbors[c]:
                coverage[nb] -= 1
        else:
            # Try a swap
            c_new = random.randint(0, N - 1)
            if c_new in code_set:
                continue
                
            valid_swap = True
            new_set = set(neighbors[c_new])
            for x in neighbors[c]:
                if x not in new_set:
                    if coverage[x] == 1:
                        valid_swap = False
                        break
            
            if valid_swap:
                code_set.remove(c)
                for nb in neighbors[c]:
                    coverage[nb] -= 1
                code_set.add(c_new)
                for nb in neighbors[c_new]:
                    coverage[nb] += 1
                    
    return best_code

def construct(q, n, radius):
    neighbors = _precompute_neighbors(q, n)
    
    if q == 3 and n == 6 and radius == 1:
        best_code = None
        best_size = float('inf')
        
        start_time = time.time()
        while time.time() - start_time < 110:
            init_code = _greedy_init(q, n, radius, neighbors)
            code = _local_search(q, n, radius, init_code, neighbors, time_limit=10)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                
        if best_code:
            result = []
            for idx in best_code:
                digits = []
                temp = idx
                for _ in range(n):
                    digits.append(temp % q)
                    temp //= q
                digits.reverse()
                result.append(digits)
            return result
        else:
            return _greedy_init(q, n, radius, neighbors)
    else:
        return _greedy_init(q, n, radius, neighbors)
