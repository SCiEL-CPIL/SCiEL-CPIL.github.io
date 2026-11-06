import time
import random

def _square_count(coeffs):
    """Calculate the number of non-zero terms in the square of the polynomial."""
    n = len(coeffs)
    sq = [0] * (2 * n - 1)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        ci = coeffs[i]
        for j in range(n):
            if coeffs[j] == 0:
                continue
            sq[i + j] += ci * coeffs[j]
    count = 0
    for v in sq:
        if v != 0:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_poly = [1] * 12
    best_count = _square_count(best_poly)
    
    # Initial candidates
    candidates = [
        [1] * 12,
        [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1], # Thue-Morse like
        [1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1],    # Repeated 4-pattern
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1],      # One flip
        [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1], # Half flip
    ]
    
    for c in candidates:
        cnt = _square_count(c)
        if cnt < best_count:
            best_count = cnt
            best_poly = list(c)
    
    rng = random.Random(123)
    
    # Local Search
    current_poly = list(best_poly)
    current_count = best_count
    
    while time.time() < deadline:
        # Random move: flip a random coefficient
        i = rng.randint(0, 11)
        current_poly[i] *= -1
        new_count = _square_count(current_poly)
        
        if new_count < current_count:
            current_count = new_count
            if new_count < best_count:
                best_count = new_count
                best_poly = list(current_poly)
                if best_count <= 12:
                    break
        else:
            # Revert
            current_poly[i] *= -1
            
            # Occasionally, to escape local minima, try a random restart
            # or a larger perturbation
            if rng.random() < 0.01:
                # Perturb 2 random coefficients
                i1 = rng.randint(0, 11)
                i2 = rng.randint(0, 11)
                current_poly[i1] *= -1
                current_poly[i2] *= -1
                perturbed_count = _square_count(current_poly)
                if perturbed_count < current_count:
                    current_count = perturbed_count
                    if current_count < best_count:
                        best_count = current_count
                        best_poly = list(current_poly)
                else:
                    # Revert perturbation
                    current_poly[i1] *= -1
                    current_poly[i2] *= -1
                    
    return best_poly
