import itertools
import numpy as np
import random
import time

def _square_coefficients(coeffs):
    """Compute exact integer square of polynomial given by coefficient list (ascending degree)."""
    n = len(coeffs)
    # Trim trailing zeros for efficiency
    while n > 0 and coeffs[n-1] == 0:
        n -= 1
    if n == 0:
        return [0]
    result = [0] * (2 * n - 1)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        for j in range(n):
            if coeffs[j] == 0:
                continue
            result[i + j] += coeffs[i] * coeffs[j]
    # Trim trailing zeros
    while len(result) > 0 and result[-1] == 0:
        result.pop()
    if not result:
        result = [0]
    return result

def _count_nonzero(coeffs):
    return sum(1 for c in coeffs if c != 0)

def _valid_candidate(coeffs):
    """Check all constraints for a candidate polynomial."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    for c in coeffs:
        if not isinstance(c, int) or c < -64 or c > 64:
            return False
    if _count_nonzero(coeffs) != 12:
        return False
    return True

def _square_count_from_sparse(exps, coefs):
    """
    Compute number of non-zero terms in the square of a sparse polynomial.
    exps: list of exponents
    coefs: list of coefficients
    Returns count of non-zero coefficients in the square.
    """
    # Max degree is 40 + 40 = 80
    sq_coeffs = [0] * 81
    
    n = len(exps)
    for i in range(n):
        for j in range(n):
            deg = exps[i] + exps[j]
            if deg <= 80:
                sq_coeffs[deg] += coefs[i] * coefs[j]
        
    return sum(1 for c in sq_coeffs if c != 0)

def construct():
    # Phase 1: Search best +/- 1 pattern with consecutive exponents 0..11
    best_count = float('inf')
    best_exps = list(range(12))
    best_coefs = [1] * 12
    
    # Fix leading coefficient to 1
    for mask in range(2048):
        coefs = [1] * 12
        for i in range(11):
            if (mask >> i) & 1:
                coefs[i] = -1
        # exps are 0..11
        count = _square_count_from_sparse(best_exps, coefs)
        if count < best_count:
            best_count = count
            best_coefs = coefs[:]
    
    # Phase 2: Local Search with sparse exponents and varied coefficients
    # Initialize
    cur_exps = best_exps[:]
    cur_coefs = best_coefs[:]
    cur_count = best_count
    
    # Time budget: 55 seconds
    start_time = time.time()
    max_time = 55.0
    
    # Use a fixed seed for reproducibility in development, but we want to explore
    # random.seed(42)
    
    while time.time() - start_time < max_time:
        # Generate a neighbor
        new_exps = cur_exps[:]
        new_coefs = cur_coefs[:]
        
        # Mutation strategy
        r = random.random()
        if r < 0.5:
            # Change a coefficient
            idx = random.randint(0, 11)
            # Pick a new non-zero coefficient from {-2, -1, 1, 2}
            # Avoid current value to force change
            choices = [-2, -1, 1, 2]
            if new_coefs[idx] in choices:
                choices.remove(new_coefs[idx])
            new_coefs[idx] = random.choice(choices)
        elif r < 0.8:
            # Shift an exponent
            idx = random.randint(0, 11)
            delta = random.choice([-2, -1, 1, 2])
            new_exp = new_exps[idx] + delta
            if 0 <= new_exp <= 40:
                # Check for collision
                if new_exp not in new_exps:
                    new_exps[idx] = new_exp
                else:
                    # Collision, skip this mutation or try another
                    continue
        else:
            # Randomize one exponent completely
            idx = random.randint(0, 11)
            new_exp = random.randint(0, 40)
            if new_exp not in new_exps:
                new_exps[idx] = new_exp
            else:
                continue
        
        # Evaluate
        new_count = _square_count_from_sparse(new_exps, new_coefs)
        
        if new_count < cur_count:
            cur_exps = new_exps
            cur_coefs = new_coefs
            cur_count = new_count
            # Update global best
            if cur_count < best_count:
                best_count = cur_count
                best_exps = cur_exps[:]
                best_coefs = cur_coefs[:]
        
        # Optional: If stuck, do a larger jump?
        # For now, simple hill climbing.
        
    # Convert best_exps/best_coefs to coefficient list
    # Sort by exponent
    pairs = sorted(zip(best_exps, best_coefs))
    max_exp = max(best_exps)
    result_coeffs = [0] * (max_exp + 1)
    for exp, coef in pairs:
        result_coeffs[exp] = coef
        
    # Validate
    if _valid_candidate(result_coeffs):
        return result_coeffs
    else:
        # Fallback to best +/- 1 consecutive
        fallback = [0] * 12
        for i in range(12):
            fallback[i] = best_coefs[i] # This is wrong, best_coefs from phase 1 might not match best_exps from phase 2 if we didn't update
        # Actually, best_exps and best_coefs are kept in sync.
        # If validation fails (shouldn't), return a known valid one
        return [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]

