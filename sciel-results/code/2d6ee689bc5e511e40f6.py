"""Bootstrap SEARCH program for sparse squares of 12-term integer polynomials."""
import numpy as np
import random
import sys
import time

# EVOLVE-BLOCK-START

_DEADLINE = time.time() + 85.0  # 5s margin before 90s kill

def _is_deadline():
    return time.time() >= _DEADLINE

def _square_count_fast(coeffs):
    """
    Compute the number of nonzero terms in the square of the polynomial
    defined by coeffs (list of ints, ascending degree).
    Uses numpy convolution for speed.
    """
    if not coeffs:
        return 0
    # Ensure it's a numpy array for fast convolution
    arr = np.array(coeffs, dtype=np.int64)
    # Convolution of arr with itself
    sq = np.convolve(arr, arr, mode='full')
    # Count non-zeros
    return int(np.count_nonzero(sq))

def _normalize(poly):
    """
    Validate and normalize the polynomial.
    Returns None if invalid.
    """
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

def _random_poly(rng):
    """Generate a random valid 12-term polynomial."""
    # Choose 12 distinct degrees in [0, 40]
    degrees = sorted(rng.sample(range(0, 41), 12))
    poly = [0] * 41
    for d in degrees:
        c = rng.randint(1, 64)
        if rng.random() < 0.5:
            c = -c
        poly[d] = c
    # Trim to length of max degree + 1
    last_nz = max(d for d in degrees if poly[d] != 0)
    return poly[:last_nz+1]

def _local_search(start_poly, rng, max_steps=10000):
    """
    Hill climbing: try to reduce the square term count by modifying single coefficients.
    """
    current = list(start_poly)
    current_score = _square_count_fast(current)
    
    # If current is invalid, fix it or return None
    norm = _normalize(current)
    if norm is None:
        return None
    
    # We operate on the normalized list, but we might change degrees.
    # To allow degree changes, we should operate on a fixed-size buffer or re-normalize often.
    # Let's operate on a list of (degree, coeff) pairs for flexibility, 
    # but for speed, we stick to array-based if degrees are fixed, 
    # or use a sparse representation if we move degrees.
    
    # Given the constraint of 12 terms, moving degrees is hard with arrays.
    # Let's stick to coefficient value changes first, then maybe degree swaps.
    
    best_poly = list(current)
    best_score = current_score
    steps = 0
    
    while steps < max_steps and not _is_deadline():
        improved = False
        # Try changing each of the 12 nonzero terms
        nz_indices = [i for i, c in enumerate(current) if c != 0]
        
        for idx in nz_indices:
            if _is_deadline():
                break
            orig_val = current[idx]
            
            # Try a few new values
            for _ in range(5):
                new_val = rng.randint(-64, 64)
                if new_val == 0:
                    continue # Must remain nonzero to keep 12 terms? 
                             # Actually, if we set to 0, we have 11 terms. Invalid.
                             # So we must keep it nonzero.
                if new_val == orig_val:
                    continue
                
                current[idx] = new_val
                score = _square_count_fast(current)
                
                if score < best_score:
                    best_score = score
                    best_poly = list(current)
                    improved = True
                    # Don't break, keep searching for even better? 
                    # Or accept and move to next index?
                    # Standard hill climbing: accept improvement, continue loop.
                else:
                    current[idx] = orig_val # Revert if not better
            
            if improved and _is_deadline():
                break
                
        # If no improvement in a full pass, try swapping degrees?
        # Swapping degrees is more complex. Let's try a random degree swap for one term.
        if not improved:
            # Pick two random nonzero indices
            if len(nz_indices) >= 2:
                i1, i2 = rng.sample(nz_indices, 2)
                d1, d2 = i1, i2
                # Swap degrees: move coeff at d1 to d2 and vice versa?
                # This requires rebuilding the array.
                # Let's just try a random perturbation of one degree.
                idx_to_move = rng.choice(nz_indices)
                val = current[idx_to_move]
                current[idx_to_move] = 0
                
                # Find a new empty spot
                empty_spots = [i for i, c in enumerate(current) if c == 0]
                if empty_spots:
                    new_deg = rng.choice(empty_spots)
                    current[new_deg] = val
                    norm_check = _normalize(current)
                    if norm_check is not None:
                        score = _square_count_fast(norm_check)
                        if score < best_score:
                            best_score = score
                            best_poly = list(norm_check)
                            improved = True
                        else:
                            # Revert
                            current[new_deg] = 0
                            current[idx_to_move] = val
                    else:
                        # Revert
                        current[new_deg] = 0
                        current[idx_to_move] = val
                else:
                    # No empty spots, revert
                    current[idx_to_move] = val
            else:
                break # Can't move if < 2 terms? But we have 12.
                
        steps += 1
        if not improved and steps > 50:
            # Stuck, break to restart
            break
            
    return best_poly

def _search():
    best_poly = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    # Phase 1: Multiple random restarts with local search
    for restart in range(100):
        if _is_deadline():
            break
            
        start = _random_poly(rng)
        norm = _normalize(start)
        if norm is None:
            continue
            
        # Run local search
        result = _local_search(norm, rng, max_steps=5000)
        
        if result is not None:
            score = _square_count_fast(result)
            if score < best_score:
                best_score = score
                best_poly = result
                # If we hit a very low score, we can stop or continue
                if best_score <= 15: # Heuristic lower bound guess
                    # Try to squeeze a bit more?
                    pass
    
    # Phase 2: Specific structured candidates
    # Try polynomials with symmetry or specific patterns that might have been missed
    # e.g., alternating signs
    for pattern in ["alternating", "positive"]:
        if _is_deadline():
            break
        poly = [0] * 41
        degrees = sorted(rng.sample(range(0, 41), 12))
        for i, d in enumerate(degrees):
            if pattern == "alternating":
                poly[d] = 1 if i % 2 == 0 else -1
            else:
                poly[d] = 1
        norm = _normalize(poly)
        if norm:
            score = _square_count_fast(norm)
            if score < best_score:
                best_score = score
                best_poly = norm
                
    if best_poly is not None:
        return best_poly
        
    # Fallback
    fallback = [0] * 41
    for i in range(12):
        fallback[i] = 1
    return _normalize(fallback)

# EVOLVE-BLOCK-END

def construct():
    """Return coefficient list in ascending degree, length<=41, nonzero final
    coefficient, entries integers in [-64,64], exactly 12 nonzero terms.
    Minimize nonzero terms in the EXACT integer square."""
    result = _search()
    if result is None:
        # Ultimate fallback
        result = [1] * 12
        # Ensure valid
        result = _normalize(result)
        if result is None:
             # Last resort valid poly
             result = [0]*41
             for i in range(12):
                 result[i] = 1
             result = _normalize(result)
    return result
