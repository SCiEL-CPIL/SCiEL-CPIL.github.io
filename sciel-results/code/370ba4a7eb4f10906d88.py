import numpy as np
import random
import time
from itertools import combinations

def construct():
    def is_paley_valid(p):
        if p < 5:
            return False, None
        if p < 2:
            return False, None
        for i in range(2, int(p**0.5) + 1):
            if p % i == 0:
                return False, None
        if p % 4 != 3:
            return False, None
            
        squares = {x * x % p for x in range(1, p)}
        in_masks = [0] * p
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    orientation.append(0)
                    in_masks[j] |= (1 << i)
                else:
                    orientation.append(1)
                    in_masks[i] |= (1 << j)
        
        for a in range(p):
            for b in range(a+1, p):
                mask_ab = in_masks[a] & in_masks[b]
                if mask_ab == 0:
                    return False, None
                for c in range(b+1, p):
                    mask_abc = mask_ab & in_masks[c]
                    if mask_abc == 0:
                        return False, None
                    for d in range(c+1, p):
                        if (mask_abc & in_masks[d]) == 0:
                            return False, None
        
        return True, orientation

    primes_to_try = [31, 43, 59, 67]
    
    for p in primes_to_try:
        valid, orient = is_paley_valid(p)
        if valid:
            return {'n': p, 'orientation': orient}
            
    def run_heuristic(n, time_limit=110.0):
        start_time = time.time()
        num_edges = n * (n - 1) // 2
        edge_to_ij = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        def orient_to_in_masks(orient):
            in_masks = [0] * n
            for idx, (i, j) in enumerate(edge_to_ij):
                if orient[idx] == 0:
                    in_masks[j] |= (1 << i)
                else:
                    in_masks[i] |= (1 << j)
            return in_masks

        def compute_fitness(in_masks):
            uncovered = 0
            for a in range(n):
                ma = in_masks[a]
                for b in range(a+1, n):
                    mab = ma & in_masks[b]
                    if mab == 0:
                        uncovered += (n-2)*(n-3)//2
                        continue
                    for c in range(b+1, n):
                        mabc = mab & in_masks[c]
                        if mabc == 0:
                            uncovered += (n-3)
                            continue
                        for d in range(c+1, n):
                            if (mabc & in_masks[d]) == 0:
                                uncovered += 1
            return uncovered

        def local_search(orient, time_limit_inner=0.5):
            in_masks = orient_to_in_masks(orient)
            current_fit = compute_fitness(in_masks)
            if current_fit == 0:
                return orient, current_fit
            
            last_improve = time.time()
            while time.time() - start_time < time_limit and time.time() - last_improve < time_limit_inner:
                improved = False
                for _ in range(20):
                    flip_idx = random.randint(0, num_edges - 1)
                    i, j = edge_to_ij[flip_idx]
                    old_bit = orient[flip_idx]
                    new_bit = 1 - old_bit
                    orient[flip_idx] = new_bit
                    
                    if old_bit == 0:
                        in_masks[j] &= ~(1 << i)
                        in_masks[i] |= (1 << j)
                    else:
                        in_masks[i] &= ~(1 << j)
                        in_masks[j] |= (1 << i)
                    
                    new_fit = compute_fitness(in_masks)
                    if new_fit < current_fit:
                        current_fit = new_fit
                        last_improve = time.time()
                        improved = True
                        if current_fit == 0:
                            return orient, current_fit
                    else:
                        orient[flip_idx] = old_bit
                        if old_bit == 0:
                            in_masks[j] |= (1 << i)
                            in_masks[i] &= ~(1 << j)
                        else:
                            in_masks[i] |= (1 << j)
                            in_masks[j] &= ~(1 << i)
                
                if not improved:
                    break
            return orient, current_fit

        best_orient = None
        best_fit = float('inf')
        
        for _ in range(20):
            if time.time() - start_time >= time_limit:
                break
            orient = [random.randint(0, 1) for _ in range(num_edges)]
            orient, fit = local_search(orient, time_limit_inner=1.0)
            if fit < best_fit:
                best_fit = fit
                best_orient = orient[:]
                if best_fit == 0:
                    break
        
        return best_orient, best_fit

    for n in range(31, 40):
        if time.time() > time.time() + 0:  # just to keep structure
            pass
        ind, fit = run_heuristic(n, time_limit=30.0)
        if fit == 0:
            return {'n': n, 'orientation': ind}
    
    p = 83
    squares = {x * x % p for x in range(1, p)}
    return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}

# EVOLVE-BLOCK-END
