"""
Strategy: Targeted Local Search for Coefficient Cancellation.
Instead of random polynomial generation, we fix the degree structure to an
Arithmetic Progression (0..11) which minimizes the number of distinct sums
(23 distinct degrees in the square). We then perform a local search on the
coefficients to maximize the number of zero coefficients in the square.
We also try other degree structures that might allow more cancellations.
"""
import random
import sys
import time

# EVOLVE-BLOCK-START

_DEADLINE = time.time() + 85.0  # Safety margin before 90s kill

def _is_deadline():
    return time.time() >= _DEADLINE

def _square_count(coeffs):
    """Calculate the number of nonzero terms in the exact integer square."""
    n = len(coeffs)
    if n == 0:
        return 0
    # Use a dictionary for sparse square computation to be faster for high degrees
    # But for n=12, list is fine.
    # Let's use a dict for clarity and speed if degrees are sparse.
    # Actually, the input 'coeffs' is a list in ascending degree.
    # Let's convert to dict of nonzeros.
    nz = {i: c for i, c in enumerate(coeffs) if c != 0}
    if not nz:
        return 0
    
    sq = {}
    for i, a in nz.items():
        for j, b in nz.items():
            if i <= j: # Avoid double counting? No, square is symmetric.
                # We need to sum all pairs.
                # Let's just do all pairs.
                pass
    
    # Full convolution
    sq = [0] * (2 * n - 1)
    for i, a in nz.items():
        for j, b in nz.items():
            sq[i + j] += a * b
            
    count = 0
    for v in sq:
        if v != 0:
            count += 1
    return count

def _normalize(poly):
    """Check validity and normalize."""
    if not poly:
        return None
    # Trim trailing zeros
    while len(poly) > 1 and poly[-1] == 0:
        poly = poly[:-1]
    if len(poly) > 41:
        return None
    if poly[-1] == 0:
        return None
    nz = sum(1 for c in poly if c != 0)
    if nz != 12:
        return None
    for c in poly:
        if not (-64 <= c <= 64):
            return None
    return poly

def _local_search(degrees, initial_coeffs, rng, max_steps=10000):
    """
    Perform local search to minimize nonzero terms in the square.
    degrees: sorted list of 12 degrees.
    initial_coeffs: list of 12 integers corresponding to degrees.
    """
    current_coeffs = list(initial_coeffs)
    current_sq_count = _square_count_from_degrees(degrees, current_coeffs)
    
    best_coeffs = list(current_coeffs)
    best_count = current_sq_count
    
    for step in range(max_steps):
        if _is_deadline():
            break
        
        # Pick a random index to modify
        idx = rng.randint(0, 11)
        old_val = current_coeffs[idx]
        
        # Try a small random change
        # Bias towards changes that might cause cancellation
        # Change to a random value in [-64, 64]
        new_val = rng.randint(-64, 64)
        if new_val == old_val:
            continue
            
        current_coeffs[idx] = new_val
        
        # Calculate new square count
        # Optimization: Only update the square terms affected by idx
        # But for simplicity and correctness first, recalculate.
        # 12 terms -> 144 products. Recalculating is fast.
        new_sq_count = _square_count_from_degrees(degrees, current_coeffs)
        
        if new_sq_count < best_count:
            best_count = new_sq_count
            best_coeffs = list(current_coeffs)
        else:
            # Revert if not better (greedy hill climbing)
            # Or accept with small probability for simulated annealing?
            # Greedy is safer to guarantee improvement or stability.
            current_coeffs[idx] = old_val
            
    return best_coeffs, best_count

def _square_count_from_degrees(degrees, coeffs):
    """Calculate square count for given degrees and coeffs."""
    # Create a dense list for convolution? 
    # Max degree is 80. List of 81 is fine.
    max_deg = max(degrees)
    sq = [0] * (2 * max_deg + 1)
    for i in range(12):
        a = coeffs[i]
        if a == 0: continue
        d_i = degrees[i]
        for j in range(12):
            b = coeffs[j]
            if b == 0: continue
            d_j = degrees[j]
            sq[d_i + d_j] += a * b
            
    count = 0
    for v in sq:
        if v != 0:
            count += 1
    return count

def _search():
    rng = random.Random(42)
    best_poly = None
    best_score = 1000 # Max possible is 21 (for 12 terms, max distinct sums is 12*12/2=78, but min is 23. Wait.
    # Min distinct sums for 12 terms is 23 (AP).
    # Max distinct sums is 78.
    # So score is between 1 and 78.
    # Current best is 1.0 (seed). We don't know the seed value.
    # Assume seed value is around 20-30.
    
    # Strategy 1: Arithmetic Progression 0..11
    # This minimizes the support of the square (23 terms).
    # We try to cancel as many as possible.
    
    degrees_ap = list(range(12))
    
    # Initialize with random coefficients
    for trial in range(20):
        if _is_deadline():
            break
        init_coeffs = [rng.randint(-64, 64) for _ in range(12)]
        # Ensure nonzero
        for i in range(12):
            if init_coeffs[i] == 0:
                init_coeffs[i] = 1
                
        coeffs, score = _local_search(degrees_ap, init_coeffs, rng, max_steps=5000)
        
        # Convert to polynomial list
        poly = [0] * (degrees_ap[-1] + 1)
        for d, c in zip(degrees_ap, coeffs):
            poly[d] = c
            
        norm = _normalize(poly)
        if norm:
            # Recalculate score on normalized poly to be safe
            final_score = _square_count(norm)
            if final_score < best_score:
                best_score = final_score
                best_poly = norm
                
    # Strategy 2: Other degree structures
    # Try to find degree sets that allow more cancellation.
    # E.g., symmetric degrees?
    # Or degrees that are powers of 2? (No overlap in sums, so 78 terms, bad)
    # AP is best for overlap.
    
    # Strategy 3: Specific patterns
    # Try alternating signs 1, -1, 1, -1...
    # P = 1 - x + x^2 - x^3 ...
    # P^2 = (1 - x + x^2 ...)^2
    # (1-x)^2 = 1 - 2x + x^2
    # (1-x+x^2)^2 = 1 - 2x + 3x^2 - 2x^3 + x^4 (5 terms)
    # Generally, alternating signs don't reduce the number of terms much for high degrees.
    
    # Strategy 4: Simulated Annealing on AP degrees
    # Allow some uphill moves to escape local minima.
    for trial in range(10):
        if _is_deadline():
            break
        init_coeffs = [rng.randint(-64, 64) for _ in range(12)]
        for i in range(12):
            if init_coeffs[i] == 0:
                init_coeffs[i] = 1
                
        current_coeffs = list(init_coeffs)
        current_score = _square_count_from_degrees(degrees_ap, current_coeffs)
        best_coeffs = list(current_coeffs)
        best_score_local = current_score
        
        T = 10.0
        for step in range(20000):
            if _is_deadline():
                break
            idx = rng.randint(0, 11)
            old_val = current_coeffs[idx]
            new_val = rng.randint(-64, 64)
            if new_val == old_val:
                continue
                
            current_coeffs[idx] = new_val
            new_score = _square_count_from_degrees(degrees_ap, current_coeffs)
            
            delta = new_score - current_score
            if delta < 0 or rng.random() < (1.0 / (1.0 + exp(delta / T))):
                # Accept
                current_score = new_score
                if new_score < best_score_local:
                    best_score_local = new_score
                    best_coeffs = list(current_coeffs)
            else:
                # Reject
                current_coeffs[idx] = old_val
                
            T *= 0.999 # Cool down
            
        poly = [0] * (degrees_ap[-1] + 1)
        for d, c in zip(degrees_ap, best_coeffs):
            poly[d] = c
        norm = _normalize(poly)
        if norm:
            final_score = _square_count(norm)
            if final_score < best_score:
                best_score = final_score
                best_poly = norm

    if best_poly is not None:
        return best_poly

    # Fallback
    fallback = [0] * 12
    for i in range(12):
        fallback[i] = 1
    return fallback

# Helper for SA
def exp(x):
    import math
    return math.exp(x)

# EVOLVE-BLOCK-END

def construct():
    """Return coefficient list in ascending degree, length<=41, nonzero final
    coefficient, entries integers in [-64,64], exactly 12 nonzero terms.
    Minimize nonzero terms in the EXACT integer square."""
    result = _search()
    if result is None:
        result = [1] * 12
    return result
