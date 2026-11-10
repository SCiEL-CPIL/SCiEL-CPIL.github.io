import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin

    N_TERMS = 12
    MAX_DEG = 40  # Length 41

    def evaluate(coeffs):
        """
        coeffs: list of 41 ints
        Returns: int count of non-zero terms in coeffs * coeffs
        """
        # Filter out trailing zeros for efficiency? No, np.convolve handles it.
        # But we can trim to max non-zero index.
        # For simplicity and speed, just use np.convolve.
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    def count_nonzero(coeffs):
        return sum(1 for c in coeffs if c != 0)

    def validate(coeffs):
        if len(coeffs) != MAX_DEG + 1:
            return False
        if coeffs[-1] == 0:
            # The task says "nonzero final coefficient". 
            # This usually means the highest degree term in the list must be non-zero?
            # "Return coefficient list in ascending degree, length<=41, nonzero final coefficient"
            # This implies the last element of the returned list must be non-zero.
            # If I have a polynomial of degree 10, I should return a list of length 11?
            # Or length 41 with trailing zeros?
            # "length<=41". If I return length 11, the final coefficient is coeffs[10].
            # If I return length 41, coeffs[40] must be non-zero?
            # Usually "final coefficient" means the coefficient of the highest power present.
            # If the list has trailing zeros, the "final coefficient" of the *list* is 0.
            # So I should trim the list to the highest non-zero degree + 1.
            pass
        # Let's re-read carefully: "Return coefficient list in ascending degree, length<=41, nonzero final coefficient"
        # This strongly suggests that if the highest non-zero term is at degree $D$, the list length should be $D+1$.
        # And $coeffs[D] \neq 0$.
        
        # My internal representation is length 41.
        # I must trim before returning.
        
        if count_nonzero(coeffs) != N_TERMS:
            return False
        if not all(-64 <= c <= 64 for c in coeffs):
            return False
        # Check integers
        if not all(isinstance(c, int) for c in coeffs):
            return False
        return True

    def trim_and_validate(coeffs):
        """
        Trims trailing zeros and validates.
        Returns (trimmed_list, is_valid)
        """
        last_nz = -1
        for i in range(len(coeffs) - 1, -1, -1):
            if coeffs[i] != 0:
                last_nz = i
                break
        
        if last_nz == -1:
            return None, False
            
        trimmed = coeffs[:last_nz + 1]
        
        if len(trimmed) > 41:
            return None, False
        if trimmed[-1] == 0:
            return None, False
        if count_nonzero(trimmed) != N_TERMS:
            return None, False
        if not all(-64 <= c <= 64 for c in trimmed):
            return None, False
        if not all(isinstance(c, int) for c in trimmed):
            return None, False
        return trimmed, True

    best_score = float('inf')
    best_poly = None
    best_trimmed = None

    random.seed(42)

    def run_sa(num_restarts=20, iters_per_restart=20000):
        nonlocal best_score, best_poly, best_trimmed
        
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize
            # Strategy: Start with consecutive exponents 0..11 with random signs
            curr = [0] * (MAX_DEG + 1)
            # Place 12 terms at 0..11
            for i in range(N_TERMS):
                curr[i] = random.choice([-1, 1])
            
            curr_score = evaluate(curr)
            
            if curr_score < best_score:
                best_score = curr_score
                best_poly = curr.copy()
                trimmed, valid = trim_and_validate(curr)
                if valid:
                    best_trimmed = trimmed
                    
            T = 5.0
            T_min = 0.1
            cooling_rate = 0.9995
            stuck_count = 0
            max_stuck = 1000
            
            for i in range(iters_per_restart):
                if time.time() > deadline - 2:
                    return
                
                # Choose move type
                move_type = random.random()
                
                if move_type < 0.6:
                    # Change value of a non-zero coeff
                    nz_indices = [j for j in range(MAX_DEG + 1) if curr[j] != 0]
                    idx = random.choice(nz_indices)
                    old_val = curr[idx]
                    
                    # Choose new value
                    if random.random() < 0.7:
                        new_val = random.choice([-3, -2, -1, 1, 2, 3])
                    else:
                        new_val = random.randint(-64, 64)
                    
                    if new_val == 0:
                        new_val = 1
                    if new_val == old_val:
                        continue
                        
                    curr[idx] = new_val
                    new_score = evaluate(curr)
                    
                    delta = new_score - curr_score
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        curr_score = new_score
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = curr.copy()
                            trimmed, valid = trim_and_validate(curr)
                            if valid:
                                best_trimmed = trimmed
                            stuck_count = 0
                    else:
                        curr[idx] = old_val
                        stuck_count += 1
                        
                elif move_type < 0.8:
                    # Move a non-zero coeff to a new index
                    nz_indices = [j for j in range(MAX_DEG + 1) if curr[j] != 0]
                    idx = random.choice(nz_indices)
                    val = curr[idx]
                    
                    # Choose new index
                    new_idx = random.randint(0, MAX_DEG)
                    if new_idx == idx:
                        continue
                    if curr[new_idx] != 0:
                        # Can't move into an occupied slot
                        continue
                        
                    curr[idx] = 0
                    curr[new_idx] = val
                    new_score = evaluate(curr)
                    
                    delta = new_score - curr_score
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        curr_score = new_score
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = curr.copy()
                            trimmed, valid = trim_and_validate(curr)
                            if valid:
                                best_trimmed = trimmed
                            stuck_count = 0
                    else:
                        curr[new_idx] = 0
                        curr[idx] = val
                        stuck_count += 1
                        
                else:
                    # Swap values of two non-zero coeffs
                    nz_indices = [j for j in range(MAX_DEG + 1) if curr[j] != 0]
                    if len(nz_indices) < 2:
                        continue
                    idx1, idx2 = random.sample(nz_indices, 2)
                    if curr[idx1] == curr[idx2]:
                        continue
                    
                    v1, v2 = curr[idx1], curr[idx2]
                    curr[idx1], curr[idx2] = v2, v1
                    new_score = evaluate(curr)
                    
                    delta = new_score - curr_score
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        curr_score = new_score
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = curr.copy()
                            trimmed, valid = trim_and_validate(curr)
                            if valid:
                                best_trimmed = trimmed
                            stuck_count = 0
                    else:
                        curr[idx1], curr[idx2] = v1, v2
                        stuck_count += 1

                T *= cooling_rate
                if T < T_min:
                    T = T_min

    # Phase 1: Deterministic Seeds
    # Seed 1: Consecutive +/-1
    s1 = [0] * (MAX_DEG + 1)
    for i in range(12):
        s1[i] = 1 if i % 2 == 0 else -1
    score = evaluate(s1)
    if score < best_score:
        best_score = score
        best_poly = s1.copy()
        trimmed, valid = trim_and_validate(s1)
        if valid:
            best_trimmed = trimmed

    # Seed 2: Arithmetic progression with step 2 (even powers)
    s2 = [0] * (MAX_DEG + 1)
    for i in range(12):
        idx = i * 2
        if idx <= MAX_DEG:
            s2[idx] = 1 if i % 2 == 0 else -1
    score = evaluate(s2)
    if score < best_score:
        best_score = score
        best_poly = s2.copy()
        trimmed, valid = trim_and_validate(s2)
        if valid:
            best_trimmed = trimmed

    # Run SA
    run_sa(num_restarts=15, iters_per_restart=15000)

    # Phase 3: Local Search Polish on best_poly
    if best_poly is not None and time.time() < deadline - 5:
        curr = best_poly.copy()
        curr_score = best_score
        
        no_improve = 0
        max_no_improve = 3000
        
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            move_type = random.random()
            improved = False
            
            if move_type < 0.6:
                nz_indices = [j for j in range(MAX_DEG + 1) if curr[j] != 0]
                idx = random.choice(nz_indices)
                old_val = curr[idx]
                
                candidates = set()
                for dv in [-2, -1, 1, 2]:
                    nv = old_val + dv
                    if -64 <= nv <= 64 and nv != 0:
                        candidates.add(nv)
                candidates.add(-old_val)
                candidates.add(random.choice([-5, -3, -2, -1, 1, 2, 3, 5]))
                
                for new_val in candidates:
                    if new_val == old_val:
                        continue
                    curr[idx] = new_val
                    new_score = evaluate(curr)
                    
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = curr.copy()
                            trimmed, valid = trim_and_validate(curr)
                            if valid:
                                best_trimmed = trimmed
                        no_improve = 0
                        break
                if not improved:
                    curr[idx] = old_val
                    no_improve += 1

            elif move_type < 0.8:
                nz_indices = [j for j in range(MAX_DEG + 1) if curr[j] != 0]
                idx = random.choice(nz_indices)
                val = curr[idx]
                
                # Try moving to nearby indices
                candidates = set()
                for d in [-2, -1, 1, 2]:
                    ni = idx + d
                    if 0 <= ni <= MAX_DEG and curr[ni] == 0:
                        candidates.add(ni)
                # Also a random jump
                candidates.add(random.randint(0, MAX_DEG))
                
                for new_idx in candidates:
                    if new_idx == idx or curr[new_idx] != 0:
                        continue
                    curr[idx] = 0
                    curr[new_idx] = val
                    new_score = evaluate(curr)
                    
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = curr.copy()
                            trimmed, valid = trim_and_validate(curr)
                            if valid:
                                best_trimmed = trimmed
                        no_improve = 0
                        break
                    else:
                        curr[new_idx] = 0
                        curr[idx] = val
                
                if not improved:
                    no_improve += 1
            else:
                no_improve += 1

    if best_trimmed is None:
        # Fallback
        fallback = [1] * 12
        best_trimmed = fallback

    return best_trimmed
