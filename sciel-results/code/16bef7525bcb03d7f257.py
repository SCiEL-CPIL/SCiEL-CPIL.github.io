# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _count_nonzero_square(exps, coeffs):
    """
    Calculate the number of non-zero terms in the square of the polynomial
    defined by exponents 'exps' and coefficients 'coeffs'.
    """
    if not exps:
        return 0
    
    # Determine the maximum degree to size the array
    max_deg = max(exps)
    # The square will have degree 2 * max_deg
    # Length of convolution result is 2 * max_deg + 1
    arr = np.zeros(2 * max_deg + 1, dtype=np.int64)
    
    for e, c in zip(exps, coeffs):
        if c != 0:
            arr[e] += c
            
    # Compute square
    sq = np.convolve(arr, arr, mode='full')
    
    # Count non-zeros
    return int(np.count_nonzero(sq))

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    # Parameters
    n_terms = 12
    max_degree = 40 # Length <= 41 means indices 0..40
    
    # Initialize best solution
    best_score = float('inf')
    best_exps = None
    best_coeffs = None
    
    # Helper to generate a random valid state
    def random_state():
        exps = random.sample(range(max_degree + 1), n_terms)
        coeffs = [random.choice([-1, 1]) for _ in range(n_terms)]
        # Ensure non-zero coefficients (already done by choice)
        return exps, coeffs
    
    # Initial random search to find a good starting point
    for _ in range(100):
        if time.time() - start_time > time_limit:
            break
        exps, coeffs = random_state()
        score = _count_nonzero_square(exps, coeffs)
        if score < best_score:
            best_score = score
            best_exps = exps[:]
            best_coeffs = coeffs[:]
            
    # Simulated Annealing
    # Start temperature
    T = 10.0
    cooling_rate = 0.999
    
    current_exps = best_exps[:]
    current_coeffs = best_coeffs[:]
    current_score = best_score
    
    # We want to minimize score.
    # Lower bound for 12 terms: 
    # If we use distinct exponents such that all pair sums are distinct, 
    # the square has 12^2 = 144 terms? No.
    # Square of sum x^e_i: sum c_i^2 x^{2e_i} + sum_{i!=j} c_i c_j x^{e_i+e_j}.
    # If all e_i + e_j are distinct and no e_i + e_j = 2e_k, then 12 + 12*11 = 144 terms.
    # We want to minimize this by creating collisions.
    # The theoretical minimum is 12 (if the square is also 12 terms, e.g. if it's a perfect square of a sparse poly, but that's hard).
    # Actually, if P(x) has 12 terms, P(x)^2 can have as few as 12 terms if P(x) is a monomial? No, P has 12 terms.
    # Example: (1 + x + ... + x^11)^2 has 23 terms.
    # Let's just search.
    
    while time.time() - start_time < time_limit:
        if T < 0.01:
            break
            
        # Generate a neighbor
        move_type = random.randint(0, 3)
        
        if move_type == 0:
            # Change one coefficient
            i = random.randint(0, n_terms - 1)
            new_coeffs = current_coeffs[:]
            # Pick a new non-zero coefficient
            new_val = random.choice([-1, 1, 2, -2, 3, -3, 4, -4])
            if new_val == new_coeffs[i]:
                continue
            new_coeffs[i] = new_val
            new_exps = current_exps
            new_score = _count_nonzero_square(new_exps, new_coeffs)
            
        elif move_type == 1:
            # Move one exponent
            i = random.randint(0, n_terms - 1)
            old_e = current_exps[i]
            # Try to move to a random valid exponent
            # Ensure uniqueness
            used = set(current_exps)
            used.discard(old_e)
            # Pick a random available exponent
            available = [e for e in range(max_degree + 1) if e not in used]
            if not available:
                continue
            new_e = random.choice(available)
            
            new_exps = current_exps[:]
            new_exps[i] = new_e
            new_coeffs = current_coeffs
            new_score = _count_nonzero_square(new_exps, new_coeffs)
            
        elif move_type == 2:
            # Swap two exponents
            i, j = random.sample(range(n_terms), 2)
            new_exps = current_exps[:]
            new_exps[i], new_exps[j] = new_exps[j], new_exps[i]
            new_coeffs = current_coeffs
            new_score = _count_nonzero_square(new_exps, new_coeffs)
            
        else:
            # Change two coefficients
            i, j = random.sample(range(n_terms), 2)
            new_coeffs = current_coeffs[:]
            new_coeffs[i] = random.choice([-1, 1, 2, -2])
            new_coeffs[j] = random.choice([-1, 1, 2, -2])
            new_exps = current_exps
            new_score = _count_nonzero_square(new_exps, new_coeffs)

        # Acceptance criterion
        delta = new_score - current_score
        
        if delta < 0:
            # Improvement
            current_exps = new_exps
            current_coeffs = new_coeffs
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_exps = current_exps[:]
                best_coeffs = current_coeffs[:]
        else:
            # Worse or equal
            # Accept with probability exp(-delta / T)
            if random.random() < np.exp(-delta / T):
                current_exps = new_exps
                current_coeffs = new_coeffs
                current_score = new_score
        
        T *= cooling_rate

    # Final validation and construction of result
    # Ensure best_exps and best_coeffs are valid
    if best_exps is None or best_coeffs is None:
        # Fallback
        best_exps = list(range(12))
        best_coeffs = [1] * 12
        best_score = _count_nonzero_square(best_exps, best_coeffs)

    # Construct the coefficient list
    # Length must be <= 41
    # Nonzero final coefficient
    # Exactly 12 nonzero terms
    
    max_deg = max(best_exps)
    result = [0] * (max_deg + 1)
    
    for e, c in zip(best_exps, best_coeffs):
        result[e] = c
        
    # Check constraints
    if len(result) > 41:
        # This shouldn't happen as max_deg <= 40
        result = result[:41]
        
    # Ensure nonzero final coefficient
    if result[-1] == 0:
        # Find last nonzero
        last_nonzero = -1
        for i in range(len(result) - 1, -1, -1):
            if result[i] != 0:
                last_nonzero = i
                break
        if last_nonzero != -1:
            result = result[:last_nonzero + 1]
        if result[-1] == 0:
            # Should not happen if we tracked max_deg correctly, but safety
            result[-1] = 1
            
    # Ensure exactly 12 nonzero terms
    # Our construction guarantees this if we don't have duplicates in best_exps.
    # random.sample ensures unique exponents.
    # Coeffs are non-zero.
    # So count should be 12.
    
    return result

# EVOLVE-BLOCK-END
