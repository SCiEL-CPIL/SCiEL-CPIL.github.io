import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Parameters for the search
    MAX_DEGREE = 40
    N_TERMS = 12
    COEFF_RANGE = 20  # Coefficients in [-20, 20]
    
    # Helper to evaluate the square's sparsity
    def count_nonzero_square(coeffs):
        """
        coeffs: dense list of length MAX_DEGREE+1
        Returns count of non-zero coefficients in coeffs * coeffs
        """
        # Using np.convolve is often faster for small arrays in Python due to C backend
        # But creating the np array has overhead. 
        # For length 41, manual loop might be comparable or faster if optimized.
        # Let's try a manual loop with early exit or just simple accumulation.
        sq = [0] * (2 * MAX_DEGREE + 1)
        # Optimization: only iterate over non-zero terms
        nz_indices = [i for i, c in enumerate(coeffs) if c != 0]
        if not nz_indices:
            return 0
        for i in nz_indices:
            ci = coeffs[i]
            for j in nz_indices:
                sq[i + j] += ci * coeffs[j]
        return sum(1 for x in sq if x != 0)

    def random_polynomial():
        """
        Generate a random polynomial with exactly N_TERMS non-zero terms.
        Degrees are unique. Coefficients are non-zero.
        """
        degrees = random.sample(range(MAX_DEGREE + 1), N_TERMS)
        coeffs = [0] * (MAX_DEGREE + 1)
        for d in degrees:
            coeffs[d] = random.randint(1, COEFF_RANGE) * (1 if random.random() > 0.5 else -1)
        return coeffs

    best_coeffs = None
    best_score = float('inf')
    
    # Simulated Annealing
    # State: list of (degree, coeff) or dense array?
    # Dense array is easier for evaluation, but mutation is easier on sparse representation.
    # Let's stick to dense array for evaluation speed, but track the set of active indices.
    
    def mutate(coeffs, active_indices):
        """
        Mutate the polynomial. 
        Returns new_coeffs, new_active_indices
        """
        new_coeffs = coeffs[:]
        new_active = set(active_indices)
        
        # Mutation types:
        # 1. Change value of an existing term
        # 2. Change degree of an existing term (if slot is free)
        # 3. Remove a term and add a new one (maintain count)
        
        r = random.random()
        if r < 0.5:
            # Change value
            idx = random.choice(list(new_active))
            new_coeffs[idx] = random.randint(1, COEFF_RANGE) * (1 if random.random() > 0.5 else -1)
        elif r < 0.8:
            # Change degree
            old_idx = random.choice(list(new_active))
            new_idx = random.randint(0, MAX_DEGREE)
            if new_idx != old_idx and new_idx not in new_active:
                new_coeffs[old_idx] = 0
                new_coeffs[new_idx] = coeffs[old_idx]
                new_active.remove(old_idx)
                new_active.add(new_idx)
            else:
                # If collision or same, just change value slightly or keep
                new_coeffs[old_idx] = random.randint(1, COEFF_RANGE) * (1 if random.random() > 0.5 else -1)
        else:
            # Swap: Remove one, add new at random degree
            old_idx = random.choice(list(new_active))
            new_idx = random.randint(0, MAX_DEGREE)
            if new_idx not in new_active:
                new_coeffs[old_idx] = 0
                new_coeffs[new_idx] = random.randint(1, COEFF_RANGE) * (1 if random.random() > 0.5 else -1)
                new_active.remove(old_idx)
                new_active.add(new_idx)
            else:
                # Collision, just change value of old
                new_coeffs[old_idx] = random.randint(1, COEFF_RANGE) * (1 if random.random() > 0.5 else -1)
                
        return new_coeffs, new_active

    # Initial population
    current_coeffs = random_polynomial()
    current_active = set(i for i, c in enumerate(current_coeffs) if c != 0)
    current_score = count_nonzero_square(current_coeffs)
    
    if current_score < best_score:
        best_score = current_score
        best_coeffs = current_coeffs[:]

    T = 100.0 # Initial temperature
    T_min = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        new_coeffs, new_active = mutate(current_coeffs, current_active)
        new_score = count_nonzero_square(new_coeffs)
        
        if new_score < current_score:
            current_coeffs = new_coeffs
            current_active = new_active
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_coeffs[:]
                if best_score == 12: # Theoretical minimum? 12 terms in P, square has at least 12 terms?
                    # Actually, if P has 12 terms, P^2 has at least 12 terms? 
                    # Not necessarily, but 12 is very low. 
                    # Let's keep going to see if we can get lower.
                    pass
        else:
            # Accept with probability exp(-(new_score - current_score) / T)
            delta = new_score - current_score
            if delta < 0:
                current_coeffs = new_coeffs
                current_active = new_active
                current_score = new_score
            else:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_coeffs = new_coeffs
                    current_active = new_active
                    current_score = new_score
        
        T *= cooling_rate
        if T < T_min:
            # Restart with a new random state or reheat?
            # Reheat to escape local minima
            T = T_min * 5
            # Or restart
            # current_coeffs = random_polynomial()
            # current_active = set(i for i, c in enumerate(current_coeffs) if c != 0)
            # current_score = count_nonzero_square(current_coeffs)
            # if current_score < best_score:
            #     best_score = current_score
            #     best_coeffs = current_coeffs[:]

    # Ensure final coefficient is non-zero
    # The polynomial is represented as a list of length MAX_DEGREE+1.
    # We need to return a list in ascending degree, length <= 41.
    # The task says "nonzero final coefficient".
    # If best_coeffs[-1] is 0, we might have a trailing zero.
    # We should trim trailing zeros.
    
    result = best_coeffs[:]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
        
    # Check constraints
    # 1. Length <= 41: Yes, max 41.
    # 2. Nonzero final coefficient: Yes, after trimming.
    # 3. Entries integers in [-64, 64]: Yes, we used [-20, 20].
    # 4. Exactly 12 nonzero terms: 
    #    We maintained this in the search, but let's verify.
    #    If we removed a term and didn't add one, we'd have 11. 
    #    My mutation logic:
    #    - Change value: count stays 12.
    #    - Change degree: count stays 12.
    #    - Swap: count stays 12.
    #    So it should be exactly 12.
    
    # Verify exactly 12 non-zero terms
    nz_count = sum(1 for c in result if c != 0)
    if nz_count != 12:
        # Fallback: if something went wrong, generate a valid one
        # This should not happen, but just in case.
        result = random_polynomial()
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        
    return result
