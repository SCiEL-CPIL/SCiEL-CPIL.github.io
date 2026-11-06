"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _square_nonzero_count(coeffs):
    """Compute the number of nonzero terms in the exact square of a polynomial."""
    n = len(coeffs)
    # Square coefficients: result[i] = sum_{j+k=i} coeffs[j]*coeffs[k]
    # Max degree is 2*(n-1), so result has length 2*n - 1
    sq = [0] * (2 * n - 1)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        ci = coeffs[i]
        for j in range(i, n):
            if coeffs[j] == 0:
                continue
            ci_j = ci * coeffs[j]
            if i == j:
                sq[2 * i] += ci_j
            else:
                sq[i + j] += ci_j
                sq[i + j] += ci_j  # Wait, this is wrong for cross terms
    # Let me redo this properly
    sq = [0] * (2 * n - 1)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        for j in range(i, n):
            if coeffs[j] == 0:
                continue
                pass
    # Proper convolution
    sq = [0] * (2 * n - 1)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        for j in range(n):
            if coeffs[j] == 0:
                continue
            sq[i + j] += coeffs[i] * coeffs[j]
    
    count = sum(1 for x in sq if x != 0)
    return count

def _valid_candidate(coeffs):
    """Check if coeffs is a valid candidate."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0:
        return False
    if coeffs[-1] == 0:
        return False
    nonzero = sum(1 for c in coeffs if c != 0)
    if nonzero != 12:
        return False
    if any(c < -64 or c > 64 for c in coeffs):
        return False
    return True

def _try_polynomial(coeffs, deadline):
    """Try a polynomial and return (nonzero_count, coeffs) or None if timeout."""
    if time.time() > deadline:
        return None
    if not _valid_candidate(coeffs):
        return None
    nz = _square_nonzero_count(coeffs)
    return (nz, coeffs)

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s limit
    
    best_count = None
    best_coeffs = None
    
    # Strategy: Try structured polynomials that are likely to have sparse squares
    # Key insight: if P(x) = sum a_i x^{b_i}, then P(x)^2 has terms at degrees 2b_i and b_i+b_j
    # To minimize nonzero terms in square, we want many cancellations
    
    # Approach 1: Try polynomials with coefficients ±1, ±2, etc. at strategic positions
    # Approach 2: Use the fact that (a+b+c+...+l)^2 can have cancellations
    
    # Let's try systematic search with small coefficient sets
    # Start with known good structures
    
    # Try polynomials of form: coefficients at positions that create cancellation
    # For example, if we have terms at positions that are "additively independent"
    
    # First, try some hand-crafted candidates
    candidates = []
    
    # Candidate 1: All ones (baseline) - square has 12+11+...+1 = 78 terms (no cancellation)
    # We need better
    
    # Try sparse structures with potential cancellation
    # Key: if P(x) = x^a - x^b + x^c - ... , cancellations can occur
    
    # Let's do a targeted search
    # Try degree up to 40, with 12 nonzero terms
    
    # Heuristic: try polynomials where coefficients are small integers
    # and positions are chosen to maximize overlap in square
    
    # Strategy: Random search with structured initialization
    random.seed(42)
    
    best = (float('inf'), None)
    
    # Try some structured patterns first
    # Pattern: alternating signs to encourage cancellation
    for trial in range(20000):
        if time.time() > deadline:
            break
        
        # Generate a random 12-term polynomial
        # Choose 12 distinct positions from [0, 40]
        positions = random.sample(range(41), 12)
        positions.sort()
        
        # Choose coefficients from a small set
        coeff_set = [-2, -1, 1, 2]
        coeffs = [0] * 41
        for pos in positions:
            coeffs[pos] = random.choice(coeff_set)
        
        # Ensure last coefficient is nonzero
        if coeffs[-1] == 0:
            # Move one term to position 40
            pos_to_move = random.choice(positions[:-1]) if len(positions) > 1 else positions[0]
            coeffs[pos_to_move] = 0
            coeffs[40] = random.choice(coeff_set)
            positions.remove(pos_to_move)
            positions.append(40)
            positions.sort()
        
        nz = _square_nonzero_count(coeffs)
        if nz < best[0]:
            best = (nz, coeffs[:])
            if nz <= 12:  # Can't do better than 12 (since P has 12 terms, P^2 has at least... well, could be less)
                # Actually minimum could be 1 if P is constant, but we need 12 terms
                # The square of a 12-term polynomial can have as few as... let's think
                # If all cross terms cancel, we'd have just the 12 diagonal terms
                # So minimum is 12
                pass
    
    if best[1] is not None:
        return best[1]
    
    # Fallback: return the baseline
    return [1] * 12
# EVOLVE-BLOCK-END
