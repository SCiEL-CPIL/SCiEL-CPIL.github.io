import time
import random
import numpy as np
from itertools import combinations

def _count_nonzero(coeffs):
    """Count nonzero entries in a coefficient list."""
    return sum(1 for c in coeffs if c != 0)

def _square_coeff_list(poly):
    """
    Given poly as a list of (exponent, coeff) pairs,
    return the coefficient list of poly^2 in ascending degree.
    """
    # Find max degree
    max_deg = max(exp for exp, _ in poly) if poly else 0
    sq_deg = 2 * max_deg
    result = [0] * (sq_deg + 1)
    for i in range(len(poly)):
        e1, c1 = poly[i]
        for j in range(len(poly)):
            e2, c2 = poly[j]
            result[e1 + e2] += c1 * c2
    # Trim trailing zeros but keep at least 1 element
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result

def _square_nonzero_count(poly):
    """Return the number of nonzero terms in the square of poly."""
    sq = _square_coeff_list(poly)
    return _count_nonzero(sq)

def _validate(poly):
    """Check if poly meets all constraints."""
    if len(poly) != 12:
        return False
    # Check exactly 12 nonzero terms (all should be nonzero)
    if any(c == 0 for _, c in poly):
        return False
    # Check coefficients in [-64, 64]
    if any(c < -64 or c > 64 for _, c in poly):
        return False
    # Check exponents are non-negative integers
    if any(e < 0 or not isinstance(e, int) for e, _ in poly):
        return False
    # Check degree <= 40 (length <= 41 means max degree <= 40)
    if any(e > 40 for e, _ in poly):
        return False
    # Check nonzero final coefficient (the square's highest degree term must be nonzero)
    max_deg = max(e for e, _ in poly)
    # The highest degree term in the square is 2*max_deg with coeff c^2 where c is the coeff of x^max_deg
    # Since c != 0, c^2 != 0, so this is automatically satisfied
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_poly = None
    best_score = float('inf')
    
    # Initial candidate: the seed
    seed_poly = [(i, 1) for i in range(12)]
    seed_score = _square_nonzero_count(seed_poly)
    if _validate(seed_poly):
        best_poly = seed_poly
        best_score = seed_score
    
    # Try structured families
    
    # Family 1: Arithmetic progression with signs
    for step in range(1, 5):
        for start in range(0, 41 - 11 * step):
            for sign_pattern in [[1] * 12, [1, -1] * 6, [-1, 1] * 6]:
                poly = [(start + i * step, sign_pattern[i]) for i in range(12)]
                if _validate(poly):
                    score = _square_nonzero_count(poly)
                    if score < best_score:
                        best_score = score
                        best_poly = poly
                if time.time() > deadline:
                    return [0] * (max(e for e, _ in best_poly) + 1) if best_poly else [1] * 12
    
    # Family 2: Two groups with disjoint exponent ranges
    # P = A(x) + x^k * B(x) where A and B are sparse
    # Try to make A^2, B^2, and AB all sparse
    
    # Random search with local optimization
    random.seed(42)
    
    def _local_search(poly, max_iter=200):
        """Local search: try swapping exponents and adjusting coefficients."""
        best = list(poly)
        best_score = _square_nonzero_count(best)
        current = list(poly)
        current_score = _square_nonzero_count(current)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Try changing one exponent
            for i in range(12):
                for new_exp in range(0, 41):
                    if new_exp == current[i][0]:
                        continue
                    # Check no duplicate exponents
                    if any(current[j][0] == new_exp for j in range(12) if j != i):
                        continue
                    trial = list(current)
                    trial[i] = (new_exp, trial[i][1])
                    trial.sort(key=lambda x: x[0])
                    if _validate(trial):
                        score = _square_nonzero_count(trial)
                        if score < current_score:
                            current = trial
                            current_score = score
                            improved = True
                            break
                if improved:
                    break
            
            # Try changing one coefficient
            if not improved:
                for i in range(12):
                    for new_coeff in range(-64, 65):
                        if new_coeff == 0 or new_coeff == current[i][1]:
                            continue
                        trial = list(current)
                        trial[i] = (trial[i][0], new_coeff)
                        if _validate(trial):
                            score = _square_nonzero_count(trial)
                            if score < current_score:
                                current = trial
                                current_score = score
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                break
            
            if current_score < best_score:
                best = list(current)
                best_score = current_score
        
        return best, best_score
    
    # Run random restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate random initial polynomial
        # Use random exponents and small coefficients
        exponents = sorted(random.sample(range(0, 41), 12))
        coefficients = [random.choice([-2, -1, 1, 2, 3, 5]) for _ in range(12)]
        poly = list(zip(exponents, coefficients))
        
        if not _validate(poly):
            continue
        
        score = _square_nonzero_count(poly)
        if score < best_score:
            best_score = score
            best_poly = poly
        
        # Local search
        local_poly, local_score = _local_search(poly, max_iter=100)
        if local_score < best_score:
            best_score = local_score
            best_poly = local_poly
        
        if best_score <= 12:
            break
    
    # Convert best_poly to coefficient list
    if best_poly is None:
        best_poly = seed_poly
    
    max_deg = max(e for e, _ in best_poly)
    coeff_list = [0] * (max_deg + 1)
    for exp, coeff in best_poly:
        coeff_list[exp] = coeff
    
    return coeff_list
