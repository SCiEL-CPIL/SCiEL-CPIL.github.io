"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def count_nonzero(arr):
    return sum(1 for x in arr if x != 0)

def compute_square_nonzero(poly):
    square = np.convolve(poly, poly)
    return count_nonzero(square)

def generate_random_candidate():
    """Generates a random polynomial with exactly 12 non-zero terms."""
    poly = [0] * 41
    # Choose 12 distinct exponents
    exponents = random.sample(range(0, 41), 12)
    for e in exponents:
        # Random small coefficients
        poly[e] = random.choice([-3, -2, -1, 1, 2, 3])
    return poly

def local_search(poly, time_limit):
    """
    Performs local search by changing values or positions of non-zero terms.
    Returns the best polynomial found.
    """
    current_poly = poly[:]
    current_nz = compute_square_nonzero(current_poly)
    
    # Identify non-zero indices
    nz_indices = [i for i, c in enumerate(current_poly) if c != 0]
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Randomly pick a move type
        if random.random() < 0.5:
            # Move 1: Change value of a random non-zero term
            idx = random.choice(nz_indices)
            old_val = current_poly[idx]
            # Try a few new values
            new_val = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            if new_val == old_val:
                continue
            current_poly[idx] = new_val
            new_nz = compute_square_nonzero(current_poly)
            if new_nz < current_nz:
                current_nz = new_nz
                improved = True
            else:
                current_poly[idx] = old_val
        else:
            # Move 2: Move a non-zero term to a new random position
            idx = random.choice(nz_indices)
            old_val = current_poly[idx]
            
            # Find a random empty position
            empty_indices = [i for i in range(41) if current_poly[i] == 0]
            if not empty_indices:
                continue
            new_idx = random.choice(empty_indices)
            
            # Perform move
            current_poly[idx] = 0
            current_poly[new_idx] = old_val
            
            new_nz = compute_square_nonzero(current_poly)
            if new_nz < current_nz:
                current_nz = new_nz
                improved = True
                # Update nz_indices
                nz_indices.remove(idx)
                nz_indices.append(new_idx)
            else:
                # Revert
                current_poly[new_idx] = 0
                current_poly[idx] = old_val
    
    return current_poly, current_nz

def construct():
    start_time = time.time()
    time_limit = start_time + 55
    best_poly = None
    best_nonzero = float('inf')
    
    # Run multiple random restarts
    while time.time() < time_limit:
        # Generate a random starting point
        # Bias towards clustered exponents to increase collision probability in sums
        if random.random() < 0.3:
            # Clustered
            base = random.randint(0, 20)
            exponents = sorted(random.sample(range(base, min(base+25, 41)), 12))
            poly = [0]*41
            for e in exponents:
                poly[e] = random.choice([-1, 1])
        else:
            poly = generate_random_candidate()
            
        # Local search
        local_poly, local_nz = local_search(poly, time_limit)
        
        if local_nz < best_nonzero:
            best_nonzero = local_nz
            best_poly = local_poly[:]
            
            # Early exit if we found a very good solution
            # The minimum number of non-zero terms in the square of a 12-term poly
            # is at least 1 (if it's a square of a monomial, but we have 12 terms).
            # Actually, (sum c_i x^i)^2 has at least 1 term.
            # Can we get very low?
            # If best_nonzero <= 10, it's likely near optimal for this problem size.
            if best_nonzero <= 12:
                break

    # Fallback
    if best_poly is None:
        best_poly = generate_random_candidate()
        
    # Post-processing: Ensure constraints
    # 1. Length <= 41 (already satisfied)
    # 2. Nonzero final coefficient
    # 3. Exactly 12 nonzero terms (maintained by local search, but verify)
    
    # Trim trailing zeros to minimize length, but ensure final coeff is non-zero
    # The problem says "length<=41, nonzero final coefficient".
    # It doesn't strictly require minimal length, just <=41.
    # However, "exactly 12 nonzero terms" is a hard constraint.
    
    # Check non-zero count
    nz_count = sum(1 for x in best_poly if x != 0)
    if nz_count != 12:
        # This should not happen with our local search, but fix if it does
        # Just return a valid random one
        best_poly = generate_random_candidate()
        
    # Ensure final coefficient is non-zero
    # If the highest degree term is 0, we might have trailing zeros.
    # The constraint is "nonzero final coefficient".
    # If we trim, the new final coefficient must be non-zero.
    # If we don't trim, the final coefficient (index 40) must be non-zero?
    # "Return coefficient list in ascending degree, length<=41, nonzero final coefficient"
    # Usually this implies the last element of the list returned is non-zero.
    
    # Let's trim trailing zeros.
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
        
    # If we popped everything or the last one is 0 (shouldn't happen if we stop at non-zero),
    # but if the list was all zeros (impossible with 12 terms), handle it.
    if not best_poly or best_poly[-1] == 0:
        # Force the last element to be non-zero
        # Find the last non-zero index
        last_nz = -1
        for i in range(len(best_poly)-1, -1, -1):
            if best_poly[i] != 0:
                last_nz = i
                break
        if last_nz == -1:
             # All zero? Impossible.
             best_poly = [0]*41
             best_poly[0] = 1
             best_poly[1] = 1
             # ... fill 12 terms
             for i in range(2, 12):
                 best_poly[i] = 1
        else:
             # Trim to last_nz
             best_poly = best_poly[:last_nz+1]
             
    # Ensure length is <= 41
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
        while len(best_poly) > 1 and best_poly[-1] == 0:
            best_poly.pop()
            
    return best_poly
# EVOLVE-BLOCK-END
