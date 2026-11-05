import numpy as np
import random

def _square_nonzero_count(exponents, coeffs):
    """Compute number of nonzero terms in P^2 given sparse representation."""
    max_deg = 2 * max(exponents)
    poly = [0] * (max_deg + 1)
    for e, c in zip(exponents, coeffs):
        poly[e] = c
    sq = np.convolve(poly, poly)
    return int(np.count_nonzero(sq))

def construct():
    best_count = float('inf')
    best_exps = None
    best_coeffs = None
    
    def try_candidate(exps, coeffs):
        nonlocal best_count, best_exps, best_coeffs
        if len(exps) != 12 or len(coeffs) != 12:
            return
        if max(exps) > 40:
            return
        if any(c == 0 for c in coeffs):
            return
        if any(c < -64 or c > 64 for c in coeffs):
            return
        if len(set(exps)) != 12:
            return
        count = _square_nonzero_count(exps, coeffs)
        if count < best_count:
            best_count = count
            best_exps = list(exps)
            best_coeffs = list(coeffs)
    
    # Structured candidates
    patterns_12 = [
        [(-1)**i for i in range(12)],
        [1]*12,
        [1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
        [1,-1,-1,1,1,-1,-1,1,1,-1,-1,1],
        [1,1,1,1,1,1,1,1,1,1,1,-1],
        [1,-1,1,-1,1,-1,1,-1,1,-1,1,1],
        [2,-2,2,-2,2,-2,2,-2,2,-2,2,-2],
        [1,2,-2,-1,-1,2,1,-2,2,-1,1,-1],
        [1,-2,3,-4,5,-6,7,-8,9,-10,11,-12],
    ]
    
    # Consecutive exponents
    for pat in patterns_12:
        try_candidate(list(range(12)), pat)
    
    # Step 3
    for pat in patterns_12:
        try_candidate(list(range(0, 36, 3)), pat)
    
    # Step 2
    for pat in patterns_12:
        try_candidate(list(range(0, 24, 2)), pat)
    
    # Other steps
    for step in [4, 5, 6, 7, 8]:
        for start in range(0, 4):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                for pat in patterns_12[:4]:
                    try_candidate(exps, pat)
    
    # Random search
    random.seed(12345)
    for _ in range(20000):
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-2,-1,1,2]) for _ in range(12)]
        try_candidate(exps, coeffs)
    
    # Try to improve best found by local search
    if best_exps is not None:
        improved = True
        while improved:
            improved = False
            for i in range(12):
                for new_c in [-64,-20,-5,-2,-1,1,2,5,20,64]:
                    if new_c == 0:
                        continue
                    new_coeffs = list(best_coeffs)
                    new_coeffs[i] = new_c
                    try_candidate(best_exps, new_coeffs)
                    if best_exps != None and best_count < float('inf'):
                        # Check if we actually improved
                        pass
            # Also try perturbing exponents
            for i in range(12):
                for delta in [-2,-1,1,2]:
                    new_exps = list(best_exps)
                    old_e = new_exps[i]
                    new_e = old_e + delta
                    if new_e < 0 or new_e > 40:
                        continue
                    if new_e in new_exps:
                        continue
                    new_exps[i] = new_e
                    new_exps.sort()
                    # Reassign coeffs in sorted order - this is tricky
                    # Skip for simplicity in local search
                    pass
    
    # Build result
    result = [0] * 41
    for e, c in zip(best_exps, best_coeffs):
        result[e] = c
    
    # Trim trailing zeros but ensure last coefficient is nonzero
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    # Ensure nonzero final coefficient
    if result[-1] == 0:
        result[-1] = best_coeffs[-1]  # This should already be nonzero
    
    return result
