import time
import random
import sys

def count_nonzero_square(coeffs):
    """
    Compute the number of nonzero terms in the square of the polynomial
    represented by coeffs.
    """
    n = len(coeffs)
    # Square convolution
    # coeffs[i] * coeffs[j] contributes to index i+j
    # Max index is 2*(n-1)
    square = [0] * (2 * n - 1)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        for j in range(n):
            if coeffs[j] == 0:
                continue
            square[i + j] += coeffs[i] * coeffs[j]
    
    # Count non-zeros
    return sum(1 for c in square if c != 0)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initial candidate: [1]*12
    best_coeffs = [1] * 12
    best_score = count_nonzero_square(best_coeffs) # Should be 23
    
    # We want to minimize best_score.
    
    # Search strategy: Random restarts with local search (Hill Climbing)
    # Coefficients are integers in [-64, 64].
    # We can restrict search to small integers like {-2, -1, 0, 1, 2} initially, 
    # then expand if needed, but sparse squares often use small coeffs.
    
    def random_candidate():
        # Ensure exactly 12 nonzero terms
        # Pick 12 distinct indices from 0 to 40 (max degree 40 for length 41)
        # But length <= 41.
        # Let's fix the support size to 12.
        # Randomly choose 12 positions.
        positions = random.sample(range(41), 12)
        positions.sort()
        coeffs = [0] * 41
        for p in positions:
            # Random coefficient from small set
            coeffs[p] = random.choice([-1, 1, -2, 2, -3, 3])
        # Ensure final coefficient is nonzero (positions[-1] is max, so coeffs[positions[-1]] is nonzero)
        # The constraint "nonzero final coefficient" means coeffs[-1] != 0 if we return the full list?
        # "Return coefficient list... nonzero final coefficient".
        # Usually this means the last element of the returned list is non-zero.
        # If we return a list of length 41, coeffs[40] must be non-zero.
        # So one of the 12 positions MUST be 40.
        
        # Let's enforce position 40 is included.
        other_positions = random.sample(range(40), 11)
        positions = sorted(other_positions + [40])
        coeffs = [0] * 41
        for p in positions:
            coeffs[p] = random.choice([-1, 1, -2, 2])
            if coeffs[p] == 0: # safeguard
                coeffs[p] = 1
        return coeffs

    def local_search(coeffs, current_score):
        nonlocal best_coeffs, best_score
        curr_coeffs = coeffs[:]
        curr_score = current_score
        no_improve_count = 0
        max_no_improve = 200
        
        while no_improve_count < max_no_improve:
            if time.time() > deadline:
                break
            
            # Pick a random nonzero term to mutate
            nonzero_indices = [i for i, c in enumerate(curr_coeffs) if c != 0]
            if not nonzero_indices:
                break
            idx = random.choice(nonzero_indices)
            old_val = curr_coeffs[idx]
            
            # Mutate value
            # Try small changes
            new_val = old_val + random.choice([-2, -1, 1, 2])
            # Keep within bounds
            if new_val < -64: new_val = -64
            if new_val > 64: new_val = 64
            
            # If new_val is 0, we lose a term. 
            # We must maintain EXACTLY 12 nonzero terms.
            # So if we set a term to 0, we must set another 0 term to non-zero.
            # Or we just change the value to a different non-zero.
            
            if new_val == 0:
                # Find a zero index to activate
                zero_indices = [i for i, c in enumerate(curr_coeffs) if c == 0]
                if not zero_indices:
                    continue
                new_idx = random.choice(zero_indices)
                
                # Save state
                temp_coeffs = curr_coeffs[:]
                temp_coeffs[idx] = 0
                temp_coeffs[new_idx] = random.choice([-1, 1, -2, 2])
                
                new_score = count_nonzero_square(temp_coeffs)
                
                if new_score < curr_score:
                    curr_coeffs = temp_coeffs
                    curr_score = new_score
                    no_improve_count = 0
                    if curr_score < best_score:
                        best_score = curr_score
                        best_coeffs = curr_coeffs[:]
                elif new_score == curr_score and random.random() < 0.1:
                    # Accept equal with low prob to escape
                    curr_coeffs = temp_coeffs
                    no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                # Just change value
                temp_coeffs = curr_coeffs[:]
                temp_coeffs[idx] = new_val
                new_score = count_nonzero_square(temp_coeffs)
                
                if new_score < curr_score:
                    curr_coeffs = temp_coeffs
                    curr_score = new_score
                    no_improve_count = 0
                    if curr_score < best_score:
                        best_score = curr_score
                        best_coeffs = curr_coeffs[:]
                elif new_score == curr_score and random.random() < 0.1:
                    curr_coeffs = temp_coeffs
                    no_improve_count += 1
                else:
                    no_improve_count += 1
                    
        return curr_score

    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        c = random_candidate()
        s = count_nonzero_square(c)
        if s < best_score:
            best_score = s
            best_coeffs = c[:]
        
        # Local search
        local_search(c, s)
        
        # If we found a very good score, we might stop early or continue to find better
        if best_score <= 11:
             # 11 is a strong target. 12 is also good.
             # Let's keep searching a bit more to see if 10 is possible? 
             # But 11 is likely the minimum for 12 terms?
             # Actually, for 1 term, square has 1 term.
             # For 2 terms $a+b$, square has 3 terms (unless $2ab=0$).
             # For 3 terms, min 5?
             # General bound: $2k-1$.
             # But cancellations can reduce this.
             # If we hit 11, it's a big jump from 23.
             pass

    # Ensure valid format
    # Trim trailing zeros? "length<=41, nonzero final coefficient".
    # If best_coeffs[40] is 0, we should trim?
    # "nonzero final coefficient" implies the last element of the returned list is non-zero.
    # If we trim, the length changes.
    # Let's trim trailing zeros to ensure the last coefficient is non-zero and length is minimal?
    # The task says "length<=41". It doesn't say "exactly 41".
    # "nonzero final coefficient" means `result[-1] != 0`.
    
    # Find last non-zero index
    last_nonzero = 0
    for i in range(len(best_coeffs) - 1, -1, -1):
        if best_coeffs[i] != 0:
            last_nonzero = i
            break
            
    # Trim to last_nonzero + 1
    final_coeffs = best_coeffs[:last_nonzero + 1]
    
    # Verify constraints
    # 1. Length <= 41
    assert len(final_coeffs) <= 41
    # 2. Nonzero final coefficient
    assert final_coeffs[-1] != 0
    # 3. Entries integers in [-64, 64]
    for c in final_coeffs:
        assert -64 <= c <= 64
    # 4. Exactly 12 nonzero terms
    nz_count = sum(1 for c in final_coeffs if c != 0)
    if nz_count != 12:
        # Fallback to seed if something went wrong
        return [1] * 12
        
    return final_coeffs

