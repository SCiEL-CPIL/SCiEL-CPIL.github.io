"""
Strategy: Structured Exponent Search + Focused Coefficient Optimization
Instead of randomizing both exponents and coefficients, we first identify exponent
sets that have a small sumset (E+E). A small sumset implies many collisions in the
square, providing more opportunities for cancellation. We then focus the search
budget on finding coefficients for these specific structures.
"""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _square_count(coeffs_list):
    """
    Calculate the number of non-zero terms in the square of a polynomial
    represented as a list of coefficients (index = degree).
    """
    if not coeffs_list:
        return 0
    # Use numpy convolve for speed
    sq = np.convolve(coeffs_list, coeffs_list)
    return np.count_nonzero(sq)

def _sumset_size(exponents):
    """
    Calculate the number of distinct sums in E+E.
    """
    sums = set()
    n = len(exponents)
    for i in range(n):
        for j in range(i, n):
            sums.add(exponents[i] + exponents[j])
    return len(sums)

def _generate_exponent_candidates():
    """
    Generate a diverse set of exponent sets with 12 elements,
    prioritizing those with small sumsets.
    """
    candidates = []
    
    # 1. Arithmetic Progressions (APs)
    # APs have very small sumsets.
    for start in range(0, 5):
        for step in range(1, 6):
            exps = [start + i * step for i in range(12)]
            if max(exps) < 41:
                candidates.append(exps)
                
    # 2. "Lattice" like structures
    # e.g., {0, 1, 2, 3} + {0, 10, 20, 30} -> 16 terms? No, we need 12.
    # Let's try subsets of grids.
    # Grid 3x4: 0..2, 0..30 step 10?
    # Exps: i + 10*j for i in 0..2, j in 0..3 -> 12 terms.
    # Max exp: 2 + 30 = 32.
    grid_exps = []
    for i in range(3):
        for j in range(4):
            grid_exps.append(i + 10 * j)
    candidates.append(sorted(grid_exps))
    
    # Another grid: 4x3
    grid_exps2 = []
    for i in range(4):
        for j in range(3):
            grid_exps2.append(i + 15 * j)
    candidates.append(sorted(grid_exps2))
    
    # 3. Random sets filtered by sumset size
    # We want sumset size to be small. Theoretical min for 12 distinct integers is 23 (AP).
    # If we allow duplicates in exponents, we can't (poly terms must be distinct).
    # So min sumset size is 23.
    # Let's generate random sets and keep those with sumset size <= 25.
    for _ in range(2000):
        exps = sorted(random.sample(range(0, 41), 12))
        ss_size = _sumset_size(exps)
        if ss_size <= 26:
            candidates.append(exps)
            
    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    return unique_candidates

def _optimize_coefficients(exps, time_limit=2.0):
    """
    Given a fixed set of exponents, find coefficients in {-2, -1, 1, 2}
    that minimize the square's sparsity.
    Uses Simulated Annealing.
    """
    n = len(exps)
    max_exp = max(exps)
    
    # Initial random assignment
    coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(n)]
    
    def get_poly(coeffs):
        poly = [0] * (max_exp + 1)
        for i, e in enumerate(exps):
            poly[e] = coeffs[i]
        return poly
    
    current_poly = get_poly(coeffs)
    current_score = _square_count(current_poly)
    best_poly = current_poly[:]
    best_score = current_score
    
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        # Pick a random index to change
        idx = random.randint(0, n - 1)
        old_val = coeffs[idx]
        new_val = random.choice([-2, -1, 1, 2])
        
        if new_val == old_val:
            continue
            
        coeffs[idx] = new_val
        new_poly = get_poly(coeffs)
        new_score = _square_count(new_poly)
        
        delta = new_score - current_score
        
        if delta < 0 or (temperature > 0.1 and random.random() < np.exp(-delta / temperature)):
            current_poly = new_poly
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_poly = new_poly[:]
        else:
            coeffs[idx] = old_val
            
        temperature *= cooling_rate
        
    return best_poly, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Generate exponent candidates
    exp_candidates = _generate_exponent_candidates()
    
    if not exp_candidates:
        # Fallback
        return [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
        
    # Sort candidates by sumset size (smaller is better)
    exp_candidates.sort(key=lambda e: _sumset_size(e))
    
    best_poly = None
    best_score = 1000
    
    # 2. Optimize coefficients for each candidate
    # We prioritize candidates with smaller sumsets.
    for exps in exp_candidates:
        if time.time() > deadline - 5.0:
            break
            
        # Validate exponents
        if len(exps) != 12:
            continue
        if max(exps) > 40:
            continue
            
        # Run optimization
        # Allocate more time to better candidates?
        # For now, fixed time per candidate.
        # If we have 50 candidates and 80s, we have 1.6s each.
        # Let's use 1.0s per candidate to be safe, and allow more for top ones.
        time_alloc = 1.0
        if _sumset_size(exps) <= 24:
            time_alloc = 2.0
            
        cand_deadline = time.time() + time_alloc
        poly, score = _optimize_coefficients(exps, time_limit=time_alloc)
        
        # Validate result
        if len(poly) > 41 or poly[-1] == 0:
            continue
        if sum(1 for x in poly if x != 0) != 12:
            continue
        if any(abs(x) > 64 for x in poly):
            continue
            
        if score < best_score:
            best_score = score
            best_poly = poly[:]
            
        # Early stopping if we hit the theoretical lower bound
        # Min non-zero terms in square of 12-term poly:
        # If P is a sum of 12 monomials, P^2 has at least 12 diagonal terms.
        # If all off-diagonal terms cancel, we have 12.
        # Is 12 possible?
        # (1+x+...+x^11)^2 has 23 terms.
        # If we use signs, can we cancel all off-diagonals?
        # Only if the sumset structure allows it.
        # For AP, off-diagonals are distinct sums 1..21.
        # Can we make c_i c_j = 0 for all i!=j? No, c_i != 0.
        # Can we make c_i c_j + c_k c_l = 0?
        # Yes, if the sumset has collisions.
        # In AP, sums are unique for each pair? No, 1+4 = 2+3.
        # So collisions exist.
        # If we find 12, it's likely optimal.
        if best_score <= 14:
            break

    if best_poly is None:
        # Fallback: simple AP with alternating signs
        return [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
        
    return best_poly
# EVOLVE-BLOCK-END
