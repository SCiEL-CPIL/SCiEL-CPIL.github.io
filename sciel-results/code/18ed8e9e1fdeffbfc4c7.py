import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit (or 60s if strict)
    
    best_count = float('inf')
    best_coeffs = [1] * 12
    best_exponents = list(range(12))
    
    def count_nonzero_square(exponents, coeffs):
        """
        Count nonzero terms in the square of the polynomial defined by exponents and coeffs.
        """
        n = len(exponents)
        if n == 0:
            return 0
        max_deg = max(exponents)
        square = [0] * (2 * max_deg + 1)
        
        for i in range(n):
            ci = coeffs[i]
            ei = exponents[i]
            if ci == 0:
                continue
            square[2 * ei] += ci * ci
            for j in range(i + 1, n):
                cj = coeffs[j]
                ej = exponents[j]
                if cj == 0:
                    continue
                prod = 2 * ci * cj
                square[ei + ej] += prod
        
        return sum(1 for x in square if x != 0)

    def update_best(exponents, coeffs, count):
        nonlocal best_count, best_coeffs, best_exponents
        if count < best_count:
            best_count = count
            best_exponents = exponents[:]
            best_coeffs = coeffs[:]
            if best_count == 12:
                return True
        return False

    # Phase 1: Systematic search over arithmetic progressions with various coefficient patterns
    # For AP with step k, exponents are [start, start+k, start+2k, ..., start+11k]
    # Need start + 11k <= 40, so k <= 3 when start=0
    
    # Try all valid (start, k) pairs for 12-term AP
    for k in range(1, 4):  # step 1, 2, 3
        max_start = 40 - 11 * k
        for start in range(0, max_start + 1):
            exponents = [start + i * k for i in range(12)]
            
            # For each AP, try various coefficient patterns
            # Key insight: for AP with step k, terms at positions i and j contribute to degree (i+j)*k + 2*start
            # Cancellation occurs when multiple pairs (i,j) with i+j = const have coefficients that sum to zero
            
            # Try all 2^12 sign patterns (coefficients in {-1, 1})
            for bits in range(1 << 12):
                if time.time() > deadline:
                    return _build_result(best_exponents, best_coeffs)
                coeffs = [1 if (bits >> i) & 1 else -1 for i in range(12)]
                count = count_nonzero_square(exponents, coeffs)
                if update_best(exponents, coeffs, count):
                    return _build_result(best_exponents, best_coeffs)
    
    # Phase 2: Try coefficients in {-2, -1, 1, 2} for promising APs
    # Focus on k=1,2,3 with start=0 (most constrained, most cancellation potential)
    for k in [1, 2, 3]:
        max_start = 40 - 11 * k
        for start in range(0, min(max_start + 1, 5)):  # Limit starts to keep search manageable
            exponents = [start + i * k for i in range(12)]
            
            # Try combinations of coefficients from {-2, -1, 1, 2}
            # Use randomized search with local improvement
            random.seed(42 + k * 100 + start)
            
            for _ in range(2000):
                if time.time() > deadline:
                    return _build_result(best_exponents, best_coeffs)
                
                coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                count = count_nonzero_square(exponents, coeffs)
                
                if count < best_count:
                    # Local improvement
                    improved = True
                    while improved and time.time() < deadline:
                        improved = False
                        for i in range(12):
                            old_val = coeffs[i]
                            for new_val in [-2, -1, 1, 2]:
                                if new_val == old_val:
                                    continue
                                coeffs[i] = new_val
                                new_count = count_nonzero_square(exponents, coeffs)
                                if new_count < count:
                                    count = new_count
                                    improved = True
                                else:
                                    coeffs[i] = old_val
                            if time.time() > deadline:
                                break
                        if time.time() > deadline:
                            break
                    
                    update_best(exponents, coeffs, count)
                    if best_count == 12:
                        return _build_result(best_exponents, best_coeffs)
    
    # Phase 3: Random search with non-AP structures
    random.seed(12345)
    current_exponents = best_exponents[:]
    current_coeffs = best_coeffs[:]
    current_count = best_count
    
    coeff_values = [-2, -1, 1, 2]
    
    while time.time() < deadline:
        # Random restart or local move
        if random.random() < 0.3:
            # Random restart
            exps = random.sample(range(41), 12)
            exps.sort()
            coeffs = [random.choice(coeff_values) for _ in range(12)]
            current_exponents = exps
            current_coeffs = coeffs
            current_count = count_nonzero_square(current_exponents, current_coeffs)
            update_best(current_exponents, current_coeffs, current_count)
        else:
            # Local move: change one coefficient
            idx = random.randint(0, 11)
            old_val = current_coeffs[idx]
            new_val = random.choice(coeff_values)
            if new_val != old_val:
                current_coeffs[idx] = new_val
                new_count = count_nonzero_square(current_exponents, current_coeffs)
                if new_count < current_count:
                    current_count = new_count
                    update_best(current_exponents, current_coeffs, new_count)
                elif random.random() < 0.1:
                    current_count = new_count
                else:
                    current_coeffs[idx] = old_val
    
    return _build_result(best_exponents, best_coeffs)

def _build_result(exponents, coeffs):
    max_deg = max(exponents) if exponents else 0
    result = [0] * (max_deg + 1)
    for e, c in zip(exponents, coeffs):
        result[e] = c
    if len(result) > 41:
        result = result[:41]
    if result and result[-1] == 0:
        for i in range(len(result) - 1, -1, -1):
            if result[i] != 0:
                result = result[:i+1]
                break
    if not result:
        result = [1]
    return result
