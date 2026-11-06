"""Sparse squares of twelve-term integer polynomials - algebraic construction strategy."""
import time
import random
import numpy as np
from collections import defaultdict

def _square_nonzero_count(coeffs_list, exps):
    """Count nonzero terms in the square of a polynomial given as (coeffs, exponents)."""
    if len(coeffs_list) == 0:
        return 0
    min_e = min(exps)
    max_e = max(exps)
    length = 2 * (max_e - min_e) + 1
    poly = np.zeros(length, dtype=np.int64)
    for c, e in zip(coeffs_list, exps):
        poly[e - min_e] = c
    sq = np.convolve(poly, poly)
    return int(np.count_nonzero(sq))

def _square_nonzero_count_fast(coeffs_dict):
    """Fast count using dict for sparse representation."""
    res = defaultdict(int)
    items = list(coeffs_dict.items())
    n = len(items)
    for i in range(n):
        c1, e1 = items[i]
        for j in range(i, n):
            c2, e2 = items[j]
            e_sum = e1 + e2
            if i == j:
                val = c1 * c1
            else:
                val = 2 * c1 * c2
            if val != 0:
                res[e_sum] += val
    return sum(1 for v in res.values() if v != 0)

def _generate_cyclotomic_exponents():
    """Generate exponent sets based on cyclotomic polynomial structures.
    
    Cyclotomic polynomials have roots that are roots of unity, creating
    systematic cancellation patterns when squared.
    """
    exponent_sets = []
    
    # Set 1: Consecutive (baseline)
    exponent_sets.append(list(range(12)))
    
    # Set 2: Two clusters with gap (mimics product of two sparse polys)
    # P(x) = A(x) * B(x) where A has 6 terms, B has 2 terms
    # Exponents: {0,1,2,3,4,5} + {10, 11} -> 12 terms
    exponent_sets.append([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15])
    
    # Set 3: Symmetric around center (palindromic structure)
    # x^5 * (x^-5 + x^-4 + x^-3 + x^-2 + x^-1 + 1 + x + x^2 + x^3 + x^4 + x^5 + x^6)
    exponent_sets.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    # Set 4: Geometric progression (powers of 2) - creates many distinct sums
    # But we want FEW sums, so use small gaps
    exponent_sets.append([0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13])
    
    # Set 5: Two interleaved arithmetic progressions
    # Even: 0,2,4,6,8,10 and Odd: 1,3,5,7,9,11
    exponent_sets.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    # Set 6: Clustered - 3 groups of 4
    exponent_sets.append([0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23])
    
    # Set 7: Near-geometric with small base
    exponent_sets.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20])
    
    # Set 8: Two clusters of 6 with moderate gap
    exponent_sets.append([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
    
    # Set 9: Arithmetic with step 1 but shifted
    exponent_sets.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    
    # Set 10: Mixed - dense core with sparse tails
    exponent_sets.append([0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 25, 26])
    
    return exponent_sets

def _generate_structured_exponents():
    """Generate more diverse exponent structures designed for cancellation."""
    sets = []
    
    # Palindromic exponent sets (P(x) = x^k * P(1/x) structure)
    # These create symmetric cancellation in P^2
    for center in range(0, 12):
        for spread in range(1, 8):
            exps = []
            # Build symmetric set around center
            used = set()
            for offset in range(spread + 1):
                e1 = center - offset
                e2 = center + offset
                if 0 <= e1 <= 40 and e1 not in used:
                    exps.append(e1)
                    used.add(e1)
                if 0 <= e2 <= 40 and e2 not in used and len(exps) < 12:
                    exps.append(e2)
                    used.add(e2)
            if len(exps) == 12:
                exps.sort()
                sets.append(exps)
    
    # Arithmetic progressions with various steps
    for start in range(0, 5):
        for step in range(1, 4):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                sets.append(exps)
    
    # Two arithmetic progressions (product structure)
    for a_start in range(0, 4):
        for a_step in range(1, 3):
            for b_start in range(5, 15):
                for b_step in range(1, 3):
                    exps_a = [a_start + i * a_step for i in range(6)]
                    exps_b = [b_start + i * b_step for i in range(6)]
                    exps = sorted(exps_a + exps_b)
                    if len(exps) == 12 and max(exps) <= 40 and len(set(exps)) == 12:
                        sets.append(exps)
    
    # Remove duplicates
    unique_sets = []
    seen = set()
    for s in sets:
        t = tuple(s)
        if t not in seen:
            seen.add(t)
            unique_sets.append(s)
    
    return unique_sets[:50]  # Limit to 50 sets

def _local_search(coeffs, exps, max_iters=2000, timeout=5.0):
    """Local search: flip signs or change coefficient values to minimize nonzero count."""
    start_time = time.time()
    current = list(coeffs)
    current_score = _square_nonzero_count_fast(dict(zip(exps, current)))
    best = list(current)
    best_score = current_score
    
    coeff_options = [-2, -1, 1, 2]
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout:
            break
        if current_score <= 12:  # Can't do better than 12 (minimum possible)
            break
            
        # Pick a random position to modify
        pos = random.randint(0, 11)
        old_val = current[pos]
        
        # Try a different value
        new_val = random.choice([v for v in coeff_options if v != old_val])
        current[pos] = new_val
        
        new_score = _square_nonzero_count_fast(dict(zip(exps, current)))
        
        if new_score <= current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = list(current)
        else:
            # Revert
            current[pos] = old_val
            
    return best, best_score

def _simulated_annealing(coeffs, exps, max_iters=5000, timeout=8.0):
    """Simulated annealing for coefficient optimization."""
    start_time = time.time()
    current = list(coeffs)
    current_score = _square_nonzero_count_fast(dict(zip(exps, current)))
    best = list(current)
    best_score = current_score
    
    coeff_options = [-2, -1, 1, 2]
    T = 1.0
    cooling = 0.995
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout:
            break
        if current_score <= 12:
            break
            
        pos = random.randint(0, 11)
        old_val = current[pos]
        new_val = random.choice([v for v in coeff_options if v != old_val])
        current[pos] = new_val
        
        new_score = _square_nonzero_count_fast(dict(zip(exps, current)))
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = list(current)
        else:
            current[pos] = old_val
            
        T *= cooling
        
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    best_score = 1000
    best_result = None
    
    # Phase 1: Generate diverse exponent structures
    exp_sets_1 = _generate_cyclotomic_exponents()
    exp_sets_2 = _generate_structured_exponents()
    all_exp_sets = exp_sets_1 + exp_sets_2
    
    # Deduplicate
    unique_exp_sets = []
    seen = set()
    for es in all_exp_sets:
        t = tuple(es)
        if t not in seen:
            seen.add(t)
            unique_exp_sets.append(es)
    
    random.shuffle(unique_exp_sets)
    
    # Phase 2: For each exponent set, run local search with multiple restarts
    for exps in unique_exp_sets:
        if time.time() > deadline - 10.0:
            break
            
        # Multiple restarts for each exponent set
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
                
            # Initialize with random coefficients
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            
            # Run local search
            result, score = _local_search(coeffs, exps, max_iters=1500, timeout=3.0)
            
            if score < best_score:
                best_score = score
                # Construct the full coefficient list
                max_deg = max(exps)
                result_list = [0] * (max_deg + 1)
                for c, e in zip(result, exps):
                    result_list[e] = c
                best_result = result_list
                
                # Early termination if we hit the theoretical minimum
                if best_score <= 12:
                    return best_result
        
        # If local search didn't find something great, try simulated annealing
        if time.time() > deadline - 15.0:
            continue
            
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        result, score = _simulated_annealing(coeffs, exps, max_iters=3000, timeout=5.0)
        
        if score < best_score:
            best_score = score
            max_deg = max(exps)
            result_list = [0] * (max_deg + 1)
            for c, e in zip(result, exps):
                result_list[e] = c
            best_result = result_list
            
            if best_score <= 12:
                return best_result
    
    # Phase 3: Focused search on the best exponent set found
    if best_result is not None and time.time() < deadline - 15.0:
        # Extract exponents from best result
        nonzero_positions = [i for i, v in enumerate(best_result) if v != 0]
        if len(nonzero_positions) == 12:
            best_exps = nonzero_positions
            best_coeffs = [best_result[i] for i in nonzero_positions]
            
            # Intensive local search on this structure
            for _ in range(5):
                if time.time() > deadline - 5.0:
                    break
                perturbed = list(best_coeffs)
                # Perturb a few coefficients
                for _ in range(2):
                    pos = random.randint(0, 11)
                    perturbed[pos] = random.choice([-2, -1, 1, 2])
                
                result, score = _local_search(perturbed, best_exps, max_iters=3000, timeout=4.0)
                if score < best_score:
                    best_score = score
                    result_list = [0] * (max(best_exps) + 1)
                    for c, e in zip(result, best_exps):
                        result_list[e] = c
                    best_result = result_list
                    if best_score <= 12:
                        return best_result
    
    # Fallback: return a valid 12-term polynomial
    if best_result is None:
        # Simple fallback: x^0 + x^1 + ... + x^11
        return [1] * 12
    
    # Validate: ensure exactly 12 nonzero terms, length <= 41, nonzero final coeff
    # Trim trailing zeros
    while len(best_result) > 1 and best_result[-1] == 0:
        best_result.pop()
    
    # Ensure nonzero final coefficient
    if best_result and best_result[-1] == 0:
        best_result[-1] = 1
    
    # Ensure length <= 41
    if len(best_result) > 41:
        best_result = best_result[:41]
        # Ensure last is nonzero
        if best_result[-1] == 0:
            best_result[-1] = 1
    
    # Ensure exactly 12 nonzero terms
    nz_count = sum(1 for x in best_result if x != 0)
    if nz_count != 12:
        # This shouldn't happen, but fallback
        return [1] * 12
    
    return best_result
