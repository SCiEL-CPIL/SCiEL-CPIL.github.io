import numpy as np
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Helper to count non-zeros in square
    def get_score(poly):
        # poly is a numpy array of length 41
        sq = np.convolve(poly, poly)
        return np.count_nonzero(sq)
    
    # Helper to check validity
    def is_valid(poly):
        if len(poly) != 41:
            return False
        if poly[-1] == 0:
            return False
        nz = np.count_nonzero(poly)
        if nz != 12:
            return False
        # Check bounds
        non_zero_vals = poly[poly != 0]
        if np.any(non_zero_vals < -64) or np.any(non_zero_vals > 64):
            return False
        return True

    # Generate initial valid polynomial
    def generate_initial():
        while True:
            # Pick 12 indices, one must be 40
            idxs = [40]
            other_idxs = random.sample(range(0, 40), 11)
            idxs.extend(other_idxs)
            random.shuffle(idxs)
            
            # Pick 12 values
            vals = [random.randint(-64, 64) for _ in range(12)]
            # Ensure non-zero
            for i in range(12):
                if vals[i] == 0:
                    vals[i] = 1
            
            poly = np.zeros(41, dtype=int)
            for i, v in zip(idxs, vals):
                poly[i] = v
            
            if is_valid(poly):
                return poly

    # Generate neighbor
    def generate_neighbor(poly):
        nz_indices = np.nonzero(poly)[0]
        
        # Move types:
        # 1. Change value of one non-zero term
        # 2. Move one non-zero term to a new index
        # 3. Swap values of two non-zero terms
        
        move_type = random.random()
        
        new_poly = poly.copy()
        
        if move_type < 0.5:
            # Change value
            i = random.choice(nz_indices)
            new_val = random.randint(-64, 64)
            if new_val == 0:
                new_val = 1
            new_poly[i] = new_val
            
        elif move_type < 0.8:
            # Move term
            i = random.choice(nz_indices)
            # Pick a new index not in nz_indices
            zero_indices = np.nonzero(poly == 0)[0]
            if len(zero_indices) == 0:
                return poly # No move possible
            
            # If we move the term at 40, we must ensure 40 stays non-zero
            # If we move a term to 40, 40 becomes non-zero.
            
            j = random.choice(zero_indices)
            
            val = poly[i]
            new_poly[i] = 0
            new_poly[j] = val
            
            # Check validity constraints:
            # 1. Exactly 12 non-zeros (satisfied by move)
            # 2. poly[-1] != 0
            if new_poly[-1] == 0:
                # We lost the term at 40.
                # We must move another term to 40.
                # Find another non-zero term k != i
                # And move it to 40?
                # But 40 is currently 0 (since we moved the term at 40 away).
                # So we can just move term at k to 40.
                remaining_nz = np.nonzero(new_poly)[0]
                if len(remaining_nz) > 0:
                    k = random.choice(remaining_nz)
                    val_k = new_poly[k]
                    new_poly[k] = 0
                    new_poly[40] = val_k
                else:
                    # Should not happen
                    return poly
                    
        else:
            # Swap values
            if len(nz_indices) < 2:
                return poly
            i, j = random.sample(list(nz_indices), 2)
            new_poly[i], new_poly[j] = new_poly[j], new_poly[i]
        
        # Final validity check
        if not is_valid(new_poly):
             # Try to fix or reject
             # If invalid, return original or a fixed version
             # Simple fix: if poly[-1] == 0, swap with another non-zero
             if new_poly[-1] == 0:
                 nz = np.nonzero(new_poly)[0]
                 if len(nz) > 0:
                     k = random.choice(nz)
                     new_poly[40], new_poly[k] = new_poly[k], new_poly[40]
             
             if not is_valid(new_poly):
                 return poly
        
        return new_poly

    # Main Loop
    current_poly = generate_initial()
    current_score = get_score(current_poly)
    best_poly = current_poly.copy()
    best_score = current_score
    
    T = 1.0
    min_T = 0.01
    
    while time.time() < deadline:
        neighbor = generate_neighbor(current_poly)
        neighbor_score = get_score(neighbor)
        
        delta = neighbor_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_poly = neighbor
            current_score = neighbor_score
            
            if current_score < best_score:
                best_score = current_score
                best_poly = current_poly.copy()
                
        T = max(min_T, T * 0.999)
        
        # Occasionally restart if stuck?
        # For now, simple SA.
        
    # Convert to list
    # Ensure length <= 41 and last coeff non-zero
    # The poly is already length 41 and last coeff non-zero.
    # We can trim trailing zeros? No, last coeff is non-zero.
    # But we can trim leading zeros to make it shorter?
    # "Return coefficient list in ascending degree, length<=41"
    # It doesn't say length MUST be 41.
    # Trimming leading zeros (low degree) is not allowed if they are part of the polynomial?
    # "coefficient list in ascending degree" usually means $c_0, c_1, \dots$.
    # If $c_0=0$, is it included?
    # Usually, a polynomial is represented by its coefficients.
    # If I return [0, 1, 2], it's $1 + 2x$.
    # If I return [1, 2], it's $1 + 2x$.
    # The constraint "exactly 12 nonzero terms" applies to the list.
    # If I trim leading zeros, the number of non-zeros stays 12.
    # The length decreases.
    # This is valid.
    
    # Find first non-zero
    first_nz = np.nonzero(best_poly)[0][0]
    result_list = best_poly[first_nz:].tolist()
    
    # Check if last element is non-zero (it is, by construction)
    # Check length <= 41 (it is)
    # Check 12 non-zeros (it is)
    
    return result_list
