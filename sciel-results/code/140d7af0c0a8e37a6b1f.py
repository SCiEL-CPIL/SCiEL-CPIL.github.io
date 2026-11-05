import numpy as np
import time
import random

def _count_nonzero_square(coeffs):
    """
    Compute the number of non-zero terms in the square of the polynomial 
    defined by coeff_list.
    """
    if not coeffs:
        return 0
    # np.convolve computes the full convolution
    sq = np.convolve(coeffs, coeffs)
    return np.count_nonzero(sq)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_score = 100
    best_coeffs = None
    
    # We will search for polynomials with 12 non-zero terms.
    # Coefficients from {-2, -1, 1, 2}
    # Exponents from 0..40
    
    # Strategy: Randomized Hill Climbing / Simulated Annealing
    # But simple random search with many iterations might be sufficient 
    # if we are smart about the candidate generation.
    
    # However, the space is huge. 
    # Let's try to exploit structure.
    # Many good sparse squares come from "self-similar" or structured exponents.
    
    # Let's do a mix of random restarts and local optimization.
    
    def get_random_candidate():
        exps = sorted(random.sample(range(41), 12))
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        # Build the full list
        full = [0] * 41
        for i in range(12):
            full[exps[i]] = coeffs[i]
        # Trim trailing zeros to satisfy length <= 41 and nonzero final coeff
        # But we need to return a list. The problem says "Return coefficient list in ascending degree, length<=41, nonzero final coefficient".
        # If the highest exponent is < 40, we should trim.
        while len(full) > 0 and full[-1] == 0:
            full.pop()
        if not full or full[-1] == 0:
            # This shouldn't happen if we pick exps, but just in case
            return None
        # Check constraints
        if len(full) > 41:
            return None
        nz = sum(1 for c in full if c != 0)
        if nz != 12:
            return None
        if any(c < -64 or c > 64 for c in full):
            return None
        return full

    def local_search(current_coeffs, max_steps=50):
        # current_coeffs is a list
        exps = [i for i, c in enumerate(current_coeffs) if c != 0]
        vals = [c for c in current_coeffs if c != 0]
        
        curr_score = _count_nonzero_square(current_coeffs)
        best_local_score = curr_score
        best_local_coeffs = list(current_coeffs)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            # Try changing one coefficient
            new_vals = list(vals)
            idx = random.randint(0, 11)
            old_val = new_vals[idx]
            new_val = random.choice([-2, -1, 1, 2])
            if new_val == old_val:
                continue
            new_vals[idx] = new_val
            
            # Reconstruct list
            new_full = [0] * len(best_local_coeffs) # or max(exp)+1
            # Need to be careful with length if we change exponents, but here we only change values.
            # The exponents are fixed in this step.
            for i in range(12):
                new_full[exps[i]] = new_vals[i]
            
            # Trim? No, length is fixed by max exp.
            # But if the last coefficient becomes 0, the length requirement "nonzero final coefficient" might be violated if we don't trim?
            # The problem says "nonzero final coefficient". If the highest degree term becomes 0, the list length should technically be shorter, or the "final coefficient" of the *returned* list must be non-zero.
            # If I return a list with trailing zeros, it might be invalid or padded.
            # Let's enforce: if the highest exponent's coeff is 0, we must trim or pick a different structure.
            # In local search, if we change the coeff of the highest exp to 0, it's invalid.
            if new_vals[-1] == 0: # assuming exps are sorted, last is highest
                continue
            
            new_full_trimmed = new_full
            # We don't need to trim if we just check score, but we need to return valid later.
            # Score calculation works on the array.
            
            new_score = _count_nonzero_square(new_full)
            if new_score < best_local_score:
                best_local_score = new_score
                best_local_coeffs = list(new_full_trimmed)
                vals = new_vals
                curr_score = new_score
            else:
                # Accept worse moves with some probability? No, keep it simple: only accept improvements or equal.
                # Or use SA. For now, greedy.
                pass
        
        return best_local_coeffs, best_local_score

    # Initial random search
    best_initial = None
    best_initial_score = 100
    
    iterations = 0
    while time.time() < deadline - 5:
        candidate = get_random_candidate()
        if candidate is None:
            continue
        
        score = _count_nonzero_square(candidate)
        if score < best_initial_score:
            best_initial_score = score
            best_initial = candidate
            if score <= 13: # Very good
                break
        
        iterations += 1
        if iterations % 1000 == 0 and best_initial is not None:
             # Try local search on the best found so far
             local_best, local_score = local_search(best_initial, max_steps=100)
             if local_score < best_initial_score:
                 best_initial_score = local_score
                 best_initial = local_best
                 if best_initial_score <= 13:
                     break
        if iterations > 50000:
             # Do a final local search phase
             break

    # If we didn't find a good one in random, do extensive local search from several random starts
    if best_initial_score > 13:
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            candidate = get_random_candidate()
            if candidate is None:
                continue
            local_best, local_score = local_search(candidate, max_steps=200)
            if local_score < best_initial_score:
                best_initial_score = local_score
                best_initial = local_best

    if best_initial is None:
        # Fallback
        return [1] * 12

    # Ensure the returned list is valid
    # Trim trailing zeros
    while len(best_initial) > 0 and best_initial[-1] == 0:
        best_initial.pop()
    
    return best_initial
