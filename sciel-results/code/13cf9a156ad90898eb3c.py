import time
import numpy as np
from itertools import combinations

DEADLINE = 85.0  # safety margin before 90s kill

def _square_nonzero_count(coeffs, exponents):
    """Compute number of nonzero terms in the square of sum c_i * x^{e_i}."""
    n = len(coeffs)
    # Build the full coefficient array
    max_deg = 2 * exponents[-1]
    sq = np.zeros(max_deg + 1, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            sq[exponents[i] + exponents[j]] += coeffs[i] * coeffs[j]
    return np.count_nonzero(sq)

def _square_nonzero_count_fast(coeffs, exponents):
    """Faster version using numpy."""
    n = len(coeffs)
    max_deg = 2 * exponents[-1]
    sq = np.zeros(max_deg + 1, dtype=np.int64)
    # Use outer product for speed
    c = np.array(coeffs, dtype=np.int64)
    e = np.array(exponents, dtype=np.int64)
    sums = e[:, None] + e[None, :]
    prods = c[:, None] * c[None, :]
    np.add.at(sq, sums.ravel(), prods.ravel())
    return np.count_nonzero(sq)

def _count_nonzero_list(coeff_list):
    """Count nonzero entries in a coefficient list."""
    return sum(1 for x in coeff_list if x != 0)

def _build_coeff_list(coeffs, exponents):
    """Build coefficient list in ascending degree."""
    max_deg = 2 * exponents[-1]
    result = [0] * (max_deg + 1)
    for c, e in zip(coeffs, exponents):
        result[e] = c
    return result

def construct():
    start_time = time.time()
    
    # We search over exponent sets and coefficient assignments
    # Strategy: simulated annealing on (exponents, coefficients)
    
    n = 12
    max_degree = 40  # since length <= 41 means degree <= 40
    
    # Initial state: random exponents and small coefficients
    best_score = 78  # worst case: all sums distinct
    best_state = None
    
    # Start with a reasonable initial guess
    # Try exponents that are powers of 2 (Sidon-like) but then perturb
    def random_state():
        exponents = sorted(np.random.choice(range(1, max_degree + 1), size=n, replace=False).tolist())
        # Coefficients: small integers, nonzero
        coeffs = np.random.choice([-3, -2, -1, 1, 2, 3], size=n).tolist()
        # Ensure no zero coefficients
        coeffs = [c if c != 0 else 1 for c in coeffs]
        return exponents, coeffs
    
    def evaluate(exponents, coeffs):
        return _square_nonzero_count_fast(coeffs, exponents)
    
    # Initialize
    exponents, coeffs = random_state()
    current_score = evaluate(exponents, coeffs)
    best_score = current_score
    best_exponents = list(exponents)
    best_coeffs = list(coeffs)
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    steps = 0
    while time.time() - start_time < DEADLINE and T > T_min:
        steps += 1
        
        # Generate a move
        move_type = np.random.random()
        
        if move_type < 0.4:
            # Change one coefficient
            idx = np.random.randint(n)
            old_c = coeffs[idx]
            new_c = np.random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            if new_c == old_c:
                continue
            coeffs[idx] = new_c
            new_score = evaluate(exponents, coeffs)
            delta = new_score - current_score
            if delta <= 0 or np.random.random() < np.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_exponents = list(exponents)
                    best_coeffs = list(coeffs)
            else:
                coeffs[idx] = old_c
        elif move_type < 0.7:
            # Change one exponent
            idx = np.random.randint(n)
            old_e = exponents[idx]
            new_e = np.random.randint(1, max_degree + 1)
            # Check no collision with other exponents
            if new_e in exponents:
                continue
            exponents[idx] = new_e
            exponents.sort()
            new_score = evaluate(exponents, coeffs)
            delta = new_score - current_score
            if delta <= 0 or np.random.random() < np.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_exponents = list(exponents)
                    best_coeffs = list(coeffs)
            else:
                exponents[idx] = old_e
                exponents.sort()
        elif move_type < 0.85:
            # Swap two exponents with random values
            i, j = np.random.choice(n, 2, replace=False)
            old_ei, old_ej = exponents[i], exponents[j]
            new_ei = np.random.randint(1, max_degree + 1)
            new_ej = np.random.randint(1, max_degree + 1)
            # Check for collisions
            temp = list(exponents)
            temp[i] = new_ei
            temp[j] = new_ej
            if len(set(temp)) < n:
                continue
            exponents = sorted(temp)
            new_score = evaluate(exponents, coeffs)
            delta = new_score - current_score
            if delta <= 0 or np.random.random() < np.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_exponents = list(exponents)
                    best_coeffs = list(coeffs)
            else:
                exponents = sorted(list(exponents))
                exponents[i], exponents[j] = old_ei, old_ej
                exponents = sorted(exponents)
        else:
            # Change two coefficients
            i, j = np.random.choice(n, 2, replace=False)
            old_ci, old_cj = coeffs[i], coeffs[j]
            new_ci = np.random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            new_cj = np.random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            coeffs[i], coeffs[j] = new_ci, new_cj
            new_score = evaluate(exponents, coeffs)
            delta = new_score - current_score
            if delta <= 0 or np.random.random() < np.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_exponents = list(exponents)
                    best_coeffs = list(coeffs)
            else:
                coeffs[i], coeffs[j] = old_ci, old_cj
        
        T *= cooling
    
    # Final polish: try local search on best state
    exponents, coeffs = list(best_exponents), list(best_coeffs)
    current_score = best_score
    
    # Local search: try all single changes
    for _ in range(200):  # limited iterations
        improved = False
        # Try changing each coefficient
        for i in range(n):
            old_c = coeffs[i]
            for new_c in range(-5, 6):
                if new_c == 0 or new_c == old_c:
                    continue
                coeffs[i] = new_c
                s = evaluate(exponents, coeffs)
                if s < current_score:
                    current_score = s
                    improved = True
                    break
                elif s == current_score:
                    # Keep it (might help later)
                    pass
                else:
                    break
            coeffs[i] = old_c if not improved else coeffs[i]
        
        # Try changing each exponent
        for i in range(n):
            old_e = exponents[i]
            for new_e in range(1, max_degree + 1):
                if new_e in exponents:
                    continue
                exponents[i] = new_e
                exponents.sort()
                s = evaluate(exponents, coeffs)
                if s < current_score:
                    current_score = s
                    improved = True
                    break
                exponents[i] = old_e
                exponents.sort()
            if improved:
                break
        
        if not improved:
            break
    
    # Build the final coefficient list
    max_deg = 2 * max(best_exponents)
    result = [0] * (max_deg + 1)
    for c, e in zip(best_coeffs, best_exponents):
        result[e] = c
    
    # Verify: exactly 12 nonzero terms
    nz = sum(1 for x in result if x != 0)
    if nz != 12:
        # Fallback
        return [1] * 12
    
    # Verify: length <= 41
    if len(result) > 41:
        # Trim or fallback
        return [1] * 12
    
    # Verify: entries in [-64, 64]
    if any(abs(x) > 64 for x in result):
        return [1] * 12
    
    # Verify: nonzero final coefficient
    if result[-1] == 0:
        return [1] * 12
    
    return result
