import numpy as np
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    num_terms = 12
    max_degree = 40
    coeff_range = list(range(-64, 65))
    coeff_range.remove(0)
    
    best_score = float('inf')
    best_P = None
    
    def get_score(exponents, coeffs):
        # p is length 41
        p = np.zeros(max_degree + 1, dtype=np.int64)
        for e, c in zip(exponents, coeffs):
            p[e] = c
        q = np.convolve(p, p)
        return np.count_nonzero(q)

    def random_valid_P():
        exponents = random.sample(range(max_degree + 1), num_terms)
        exponents.sort()
        coeffs = [random.choice(coeff_range) for _ in range(num_terms)]
        return exponents, coeffs

    # Run multiple restarts if time permits
    restarts = 0
    max_restarts = 100
    
    while time.time() < deadline and restarts < max_restarts:
        current_exponents, current_coeffs = random_valid_P()
        current_score = get_score(current_exponents, current_coeffs)
        
        if current_score < best_score:
            best_score = current_score
            best_P = (current_exponents[:], current_coeffs[:])
            # Reset T on new best? Or just continue.
            # Let's do a local SA run for a fixed time or iterations
            # Simpler: Just do a long SA loop that restarts internally if stuck?
            # No, let's just do one long SA with cooling.
            # But if we start bad, we might get stuck.
            # Better: Loop over restarts, each doing a short SA.
            
            # Actually, the current structure is a bit mixed.
            # Let's restructure:
            # Outer loop: Restarts
            # Inner loop: SA steps for this restart
            
            T = 1.0
            T_min = 0.01
            alpha = 0.999
            iterations = 10000 # Fixed steps per restart? Or time-based?
            # Time-based is safer.
            restart_deadline = time.time() + 5.0 # 5s per restart
            
            while time.time() < restart_deadline and time.time() < deadline:
                new_exponents = current_exponents[:]
                new_coeffs = current_coeffs[:]
                
                move_type = random.randint(0, 3)
                if move_type == 0:
                    idx = random.randint(0, num_terms - 1)
                    new_coeffs[idx] = random.choice(coeff_range)
                elif move_type == 1:
                    idx = random.randint(0, num_terms - 1)
                    new_e = random.randint(0, max_degree)
                    if new_e not in new_exponents:
                        new_exponents[idx] = new_e
                        new_exponents.sort()
                    else:
                        continue
                elif move_type == 2:
                    i, j = random.sample(range(num_terms), 2)
                    new_coeffs[i], new_coeffs[j] = new_coeffs[j], new_coeffs[i]
                else:
                    i, j = random.sample(range(num_terms), 2)
                    new_exponents[i], new_exponents[j] = new_exponents[j], new_exponents[i]
                    new_exponents.sort()
                    # Note: sorting exponents changes the pairing if we don't sort coeffs.
                    # Wait. If I have pairs (e1, c1), (e2, c2).
                    # Swap e1, e2 -> (e2, c1), (e1, c2).
                    # Sort exponents: e1 < e2.
                    # New list: e1, e2.
                    # Corresponding coeffs? 
                    # The pair (e2, c1) is now at index 1 (if e2 > e1).
                    # The pair (e1, c2) is now at index 0.
                    # So coeffs at index 0 is c2, index 1 is c1.
                    # This means swapping exponents AND sorting effectively swaps coeffs?
                    # Let's trace:
                    # Start: E=[1,2], C=[10, 20]. P = 10x + 20x^2.
                    # Swap E: E=[2,1]. P is still 10x^2 + 20x? No, the move is "change the polynomial".
                    # The move "Swap Exponents" implies $P' = c_1 x^{e_2} + c_2 x^{e_1}$.
                    # So $P' = 10 x^2 + 20 x^1$.
                    # To represent this in sorted E, C:
                    # E' = [1, 2].
                    # C' must be [20, 10].
                    # So if I swap E[i], E[j] and sort, I MUST also swap C[i], C[j] to maintain the "pairing" logic?
                    # No.
                    # If I define the state as (Sorted E, C aligned to Sorted E).
                    # Move: Swap E[i] and E[j].
                    # This changes the polynomial to $c_i x^{e_j} + c_j x^{e_i} + \dots$
                    # To store this in sorted E, I need to find where $e_i$ and $e_j$ land.
                    # If $e_i < e_j$, they were at i, j.
                    # After swap, values are $e_j$ at i, $e_i$ at j.
                    # Sorted E will have $e_i$ at some position $k$ and $e_j$ at $l$.
                    # The coefficient for $e_i$ is $c_j$.
                    # The coefficient for $e_j$ is $c_i$.
                    # So yes, effectively, swapping exponents in an unsorted view and then sorting is equivalent to 
                    # swapping the coefficients corresponding to those exponents?
                    # Let's verify.
                    # State: E=[1,2], C=[10,20].
                    # Swap E[0], E[1] -> Unsorted E=[2,1], C=[10,20].
                    # Polynomial: $10 x^2 + 20 x^1$.
                    # Sort E: [1,2].
                    # We need C to be [20, 10] to represent $20 x^1 + 10 x^2$.
                    # So, if I swap E[i] and E[j], and then sort E, I must ALSO swap C[i] and C[j]?
                    # If I swap C[i] and C[j] as well:
                    # Unsorted: E=[2,1], C=[20,10].
                    # Poly: $20 x^2 + 10 x^1$.
                    # Sort E: [1,2]. C aligned: [10, 20].
                    # This is the ORIGINAL polynomial.
                    # So swapping E and C simultaneously is an identity move (if sorted).
                    # Therefore, the move "Swap Exponents" (keeping C fixed) is a valid non-trivial move.
                    # It results in a permutation of C relative to the sorted E.
                    # Specifically, it swaps the coefficients assigned to the two exponents.
                    # So, implementation:
                    # 1. Swap E[i], E[j].
                    # 2. Sort E.
                    # 3. Do NOT swap C.
                    # This correctly models $c_i x^{e_j} + c_j x^{e_i}$.
                    
                new_score = get_score(new_exponents, new_coeffs)
                
                delta = new_score - current_score
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    current_exponents = new_exponents
                    current_coeffs = new_coeffs
                    current_score = new_score
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_P = (current_exponents[:], current_coeffs[:])
                        
                T *= alpha
                if T < T_min:
                    T = T_min
        restarts += 1

    if best_P is None:
        return [1] * 12
        
    exponents, coeffs = best_P
    result = [0] * (max_degree + 1)
    for e, c in zip(exponents, coeffs):
        result[e] = c
        
    while len(result) > 1 and result[-1] == 0:
        result.pop()
        
    if len(result) > 41:
        result = result[:41]
        
    return result
