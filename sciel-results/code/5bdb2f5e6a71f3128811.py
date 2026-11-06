import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    def get_square_nonzero_count(exponents, coeffs):
        # exponents: list of 12 ints
        # coeffs: list of 12 ints
        max_deg = max(exponents) * 2
        square_coeffs = [0] * (max_deg + 1)
        
        # Compute square
        for i in range(12):
            ei = exponents[i]
            ci = coeffs[i]
            for j in range(12):
                ej = exponents[j]
                cj = coeffs[j]
                square_coeffs[ei + ej] += ci * cj
                
        return sum(1 for c in square_coeffs if c != 0)

    def random_candidate():
        # Generate 12 distinct exponents
        exponents = random.sample(range(0, 41), 12)
        exponents.sort()
        # Generate 12 coefficients
        coeffs = [random.randint(-64, 64) for _ in range(12)]
        # Ensure not all zero (though random makes it unlikely, and we check nonzero final)
        if coeffs[-1] == 0:
            coeffs[-1] = 1
        return exponents, coeffs

    best_exponents, best_coeffs = None, None
    best_score = float('inf')
    
    # Initial candidate from seed logic
    init_exp = list(range(12))
    init_coeff = [1] * 12
    init_score = get_square_nonzero_count(init_exp, init_coeff)
    best_exponents, best_coeffs = init_exp, init_coeff
    best_score = init_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    
    current_exponents, current_coeffs = init_exp[:], init_coeff[:]
    current_score = init_score
    
    while time.time() < deadline:
        # Propose move
        move_type = random.choice(['coeff', 'exponent'])
        
        new_exponents = current_exponents[:]
        new_coeffs = current_coeffs[:]
        
        if move_type == 'coeff':
            idx = random.randint(0, 11)
            new_coeffs[idx] = random.randint(-64, 64)
        else:
            # Change one exponent
            idx = random.randint(0, 11)
            old_exp = new_exponents[idx]
            # Try to shift or swap
            if random.random() < 0.5:
                # Shift
                shift = random.randint(-3, 3)
                new_exp = old_exp + shift
                if 0 <= new_exp <= 40 and new_exp not in new_exponents:
                    new_exponents[idx] = new_exp
                else:
                    continue
            else:
                # Swap with another
                jdx = random.randint(0, 11)
                if jdx != idx:
                    new_exponents[idx], new_exponents[jdx] = new_exponents[jdx], new_exponents[idx]
                    # Ensure sorted? Not strictly necessary for calculation, but good for consistency
                    new_exponents.sort()
                    # Note: coeffs must follow exponents if we sort exponents.
                    # This is tricky. Let's stick to shifting or swapping indices in the list
                    # and keeping coeffs associated with the original index?
                    # Actually, P(x) is defined by the set of (exp, coeff) pairs.
                    # Swapping exponents changes the polynomial.
                    # Let's just swap the exponents in the list. The coeffs stay at their indices.
                    # This effectively changes which coefficient is attached to which exponent.
        
        # Evaluate
        new_score = get_square_nonzero_count(new_exponents, new_coeffs)
        
        # Accept
        if new_score < current_score:
            current_exponents = new_exponents
            current_coeffs = new_coeffs
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_exponents = current_exponents[:]
                best_coeffs = current_coeffs[:]
        else:
            delta = new_score - current_score
            if random.random() < np.exp(-delta / T):
                current_exponents = new_exponents
                current_coeffs = new_coeffs
                current_score = new_score
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart with random candidate if stuck?
            if random.random() < 0.1:
                current_exponents, current_coeffs = random_candidate()
                current_score = get_square_nonzero_count(current_exponents, current_coeffs)
                T = 10.0

    # Format output
    # Return coefficient list in ascending degree
    # Length <= 41
    # Nonzero final coefficient
    # Entries integers in [-64, 64]
    # Exactly 12 nonzero terms in P
    
    max_deg = max(best_exponents)
    result = [0] * (max_deg + 1)
    for e, c in zip(best_exponents, best_coeffs):
        result[e] = c
        
    # Ensure final coefficient is nonzero
    if result[-1] == 0:
        # Find a nonzero coeff and move it to the end?
        # Or just force one?
        # The task says "nonzero final coefficient".
        # If our search resulted in a trailing zero, we should fix it.
        # But we ensured coeffs[-1] != 0 in random_candidate?
        # And we didn't explicitly force it in SA.
        # Let's check.
        if result[-1] == 0:
            # Find last nonzero
            for i in range(len(result)-1, -1, -1):
                if result[i] != 0:
                    # Move this term to the end?
                    # This changes the polynomial.
                    # Better to just set a new exponent to max_deg and coeff to 1?
                    # But we must maintain 12 terms.
                    # Let's just pick a random nonzero coeff and set its exponent to max_deg.
                    nz_indices = [i for i, c in enumerate(best_coeffs) if c != 0]
                    if nz_indices:
                        idx = nz_indices[0]
                        best_exponents[idx] = max_deg
                        # Recalculate
                        result = [0] * (max_deg + 1)
                        for e, c in zip(best_exponents, best_coeffs):
                            result[e] = c
                    else:
                        # All zero? Impossible.
                        result[-1] = 1

    return result

# EVOLVE-BLOCK-START
# The code above is the implementation.
# I will wrap it in the required format.
# EVOLVE-BLOCK-END
