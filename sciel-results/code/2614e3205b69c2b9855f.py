import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    space_size = q ** n
    
    # Precompute digits
    digits = np.zeros((space_size, n), dtype=np.int8)
    for i in range(space_size):
        val = i
        for k in range(n - 1, -1, -1):
            digits[i, k] = val % q
            val //= q
            
    # Precompute N (neighbors matrix)
    diff = digits[:, np.newaxis, :] != digits[np.newaxis, :, :]
    dist = np.sum(diff, axis=2)
    N = (dist <= radius).astype(np.int8)
    
    def greedy_set_cover(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            gains = np.sum(N[:, :] & uncovered[np.newaxis, :], axis=1)
            max_gain = np.max(gains)
            candidates = np.where(gains == max_gain)[0]
            w = candidates[rng.integers(len(candidates))]
            
            code.append(w)
            uncovered &= ~N[w, :]
            
        return code

    def optimize(code, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        code_set = set(code)
        counts = np.sum(N[code], axis=0)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Deletion
            for i in range(len(code) - 1, -1, -1):
                if time.time() > deadline:
                    break
                w = code[i]
                if np.min(counts[N[w] == 1]) >= 2:
                    counts -= N[w]
                    code.pop(i)
                    code_set.remove(w)
                    improved = True
            
            if not improved:
                # Swaps
                for i in range(len(code)):
                    if time.time() > deadline:
                        break
                    w = code[i]
                    base = counts - N[w]
                    required = np.where(base == 0)[0]
                    
                    if len(required) == 0:
                        continue
                        
                    in_code_mask = np.zeros(space_size, dtype=bool)
                    for c in code:
                        in_code_mask[c] = True
                        
                    candidate_mask = np.all(N[:, required] == 1, axis=1)
                    candidates = np.where(candidate_mask & ~in_code_mask)[0]
                    
                    if len(candidates) > 0:
                        k = candidates[rng.integers(len(candidates))]
                        counts = base + N[k]
                        code[i] = k
                        code_set.remove(w)
                        code_set.add(k)
                        improved = True
                        break
                        
        return code, counts

    best_code = None
    best_size = float('inf')
    
    max_restarts = 20
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        code = greedy_set_cover(random_seed=r)
        code, counts = optimize(code, random_seed=r)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
    if best_code is None:
        code = greedy_set_cover()
        best_code = code
        
    result = []
    for idx in best_code:
        w = [0] * n
        val = idx
        for k in range(n - 1, -1, -1):
            w[k] = val % q
            val //= q
        result.append(w)
        
    return result
