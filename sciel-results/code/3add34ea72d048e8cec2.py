import time
import random

def construct():
    start_time = time.time()
    timeout = 55.0
    
    best_count = float('inf')
    best_coeffs = [1]*12
    
    # Phase 1: Brute force signs for 12 consecutive terms (0..11)
    # c_i in {-1, 1}
    for bits in range(1 << 12):
        if time.time() - start_time > timeout:
            break
        coeffs = [1 if (bits >> i) & 1 else -1 for i in range(12)]
        square = [0]*23
        for i in range(12):
            ci = coeffs[i]
            for j in range(12):
                square[i+j] += ci * coeffs[j]
        count = sum(1 for x in square if x != 0)
        if count < best_count:
            best_count = count
            best_coeffs = coeffs

    # Phase 2: Random search for coefficients in {-2, -1, 1, 2} with exponents 0..11
    # This explores a larger coefficient space on the densest exponent set.
    random.seed(42)
    while time.time() - start_time < timeout:
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        square = [0]*23
        for i in range(12):
            ci = coeffs[i]
            for j in range(12):
                square[i+j] += ci * coeffs[j]
        count = sum(1 for x in square if x != 0)
        if count < best_count:
            best_count = count
            best_coeffs = coeffs
            if best_count == 12:
                return best_coeffs

    # Phase 3: Random search with random exponents (dense) and coefficients
    # If we didn't find something better, try varying exponents too.
    # We will sample 12 exponents from 0..15 to keep it dense.
    while time.time() - start_time < timeout:
        exponents = random.sample(range(16), 12)
        exponents.sort()
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        
        max_deg = exponents[-1]
        square = [0] * (2 * max_deg + 1)
        for i in range(12):
            ei = exponents[i]
            ci = coeffs[i]
            for j in range(12):
                ej = exponents[j]
                cj = coeffs[j]
                square[ei + ej] += ci * cj
                
        count = sum(1 for x in square if x != 0)
        if count < best_count:
            best_count = count
            # Construct full list
            full_coeffs = [0] * (max_deg + 1)
            for e, c in zip(exponents, coeffs):
                full_coeffs[e] = c
            best_coeffs = full_coeffs
            if best_count == 12:
                return best_coeffs

    return best_coeffs
