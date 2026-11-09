"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _square_coeffs(c):
    """Return coefficient list of c*c (convolution), trimmed to remove trailing zeros."""
    c_arr = np.array(c, dtype=np.int64)
    result = np.convolve(c_arr, c_arr)
    nz = np.nonzero(result)[0]
    if len(nz) == 0:
        return [0]
    last = nz[-1]
    return result[:last+1].tolist()

def _nnz(c):
    """Count non-zero elements in a list or numpy array."""
    if isinstance(c, np.ndarray):
        return int(np.count_nonzero(c))
    return sum(1 for x in c if x != 0)

def _count_nonzero_terms(poly):
    """Count nonzero terms in polynomial (list of coeffs)."""
    return sum(1 for x in poly if x != 0)

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s limit
    
    # Helper to evaluate a polynomial
    def evaluate(poly):
        if any(abs(x) > 64 for x in poly):
            return float('inf')
        if _count_nonzero_terms(poly) != 12:
            return float('inf')
        sq = np.convolve(np.array(poly, dtype=np.int64), np.array(poly, dtype=np.int64))
        return int(np.count_nonzero(sq))

    # Initialize best
    best_sq_nnz = float('inf')
    best_poly = [1] * 12
    
    # Update best if current is better
    def update_best(poly):
        nonlocal best_sq_nnz, best_poly
        score = evaluate(poly)
        if score < best_sq_nnz:
            best_sq_nnz = score
            best_poly = poly[:]
            return True
        return False

    # --- Phase 1: Simulated Annealing on different families ---
    
    # Family 1: General 12-term polynomials
    # We represent the polynomial by 12 coefficients.
    # Constraint: all must be non-zero.
    
    def sa_general():
        random.seed(int(time.time() * 1000) % 10000) # Dynamic seed for variety
        # Start with a random valid polynomial
        current = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        for i in range(12):
            if current[i] == 0: current[i] = 1
            
        current_score = evaluate(current)
        best_local = current[:]
        best_local_score = current_score
        
        start_temp = 10.0
        end_temp = 0.01
        steps = 150000 # Adjust based on time
        
        for i in range(steps):
            if time.time() > deadline - 5.0: # Check time periodically
                break
                
            temp = start_temp * (end_temp / start_temp) ** (i / steps)
            
            idx = random.randint(0, 11)
            old_val = current[idx]
            
            # Propose change: random value in [-10, 10] excluding 0 and old_val
            new_val = random.randint(-10, 10)
            if new_val == 0 or new_val == old_val:
                continue
                
            current[idx] = new_val
            new_score = evaluate(current)
            
            delta = new_score - current_score
            if delta < 0:
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local = current[:]
            else:
                # Accept with probability exp(-delta / temp)
                if random.random() < np.exp(-delta / temp):
                    current_score = new_score
                else:
                    current[idx] = old_val
        
        return best_local, best_local_score

    # Family 2: Palindromic (c_i = c_{11-i})
    # 6 free variables: v[0..5]. Poly: v0, v1, v2, v3, v4, v5, v5, v4, v3, v2, v1, v0
    def get_palindromic(v):
        return [v[0], v[1], v[2], v[3], v[4], v[5], v[5], v[4], v[3], v[2], v[1], v[0]]

    def sa_palindromic():
        random.seed(int(time.time() * 1000) % 10000 + 1)
        current_v = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(6)]
        for i in range(6):
            if current_v[i] == 0: current_v[i] = 1
            
        current = get_palindromic(current_v)
        current_score = evaluate(current)
        best_local = current[:]
        best_local_score = current_score
        
        start_temp = 10.0
        end_temp = 0.01
        steps = 150000
        
        for i in range(steps):
            if time.time() > deadline - 5.0:
                break
                
            temp = start_temp * (end_temp / start_temp) ** (i / steps)
            
            idx = random.randint(0, 5)
            old_val = current_v[idx]
            
            new_val = random.randint(-10, 10)
            if new_val == 0 or new_val == old_val:
                continue
                
            current_v[idx] = new_val
            new_poly = get_palindromic(current_v)
            new_score = evaluate(new_poly)
            
            delta = new_score - current_score
            if delta < 0:
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local = new_poly[:]
            else:
                if random.random() < np.exp(-delta / temp):
                    current_score = new_score
                    # Update best_local only if we accept and it's better? 
                    # Actually, we track best_local separately.
                else:
                    current_v[idx] = old_val
        
        return best_local, best_local_score

    # Family 3: Anti-palindromic (c_i = -c_{11-i})
    # 6 free variables: v[0..5]. Poly: v0, v1, v2, v3, v4, v5, -v5, -v4, -v3, -v2, -v1, -v0
    def get_anti_palindromic(v):
        return [v[0], v[1], v[2], v[3], v[4], v[5], -v[5], -v[4], -v[3], -v[2], -v[1], -v[0]]

    def sa_anti_palindromic():
        random.seed(int(time.time() * 1000) % 10000 + 2)
        current_v = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(6)]
        for i in range(6):
            if current_v[i] == 0: current_v[i] = 1
            
        current = get_anti_palindromic(current_v)
        current_score = evaluate(current)
        best_local = current[:]
        best_local_score = current_score
        
        start_temp = 10.0
        end_temp = 0.01
        steps = 150000
        
        for i in range(steps):
            if time.time() > deadline - 5.0:
                break
                
            temp = start_temp * (end_temp / start_temp) ** (i / steps)
            
            idx = random.randint(0, 5)
            old_val = current_v[idx]
            
            new_val = random.randint(-10, 10)
            if new_val == 0 or new_val == old_val:
                continue
                
            current_v[idx] = new_val
            new_poly = get_anti_palindromic(current_v)
            new_score = evaluate(new_poly)
            
            delta = new_score - current_score
            if delta < 0:
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local = new_poly[:]
            else:
                if random.random() < np.exp(-delta / temp):
                    current_score = new_score
                else:
                    current_v[idx] = old_val
        
        return best_local, best_local_score

    # Run SA for each family
    # We can run them in parallel-ish by just calling them sequentially with time checks,
    # or split the time budget. Let's give each ~20s.
    
    # Run 1: General
    if time.time() < deadline - 25.0:
        poly, score = sa_general()
        update_best(poly)

    # Run 2: Palindromic
    if time.time() < deadline - 20.0:
        poly, score = sa_palindromic()
        update_best(poly)

    # Run 3: Anti-palindromic
    if time.time() < deadline - 15.0:
        poly, score = sa_anti_palindromic()
        update_best(poly)
        
    # Run 4: Another General run with different seed
    if time.time() < deadline - 10.0:
        random.seed(12345)
        poly, score = sa_general()
        update_best(poly)

    # Run 5: Another Palindromic run
    if time.time() < deadline - 5.0:
        random.seed(54321)
        poly, score = sa_palindromic()
        update_best(poly)

    # --- Phase 2: Local Search (Polishing) ---
    # Try to improve the best_poly found so far by changing one coefficient at a time.
    
    if time.time() < deadline - 2.0:
        current_poly = best_poly[:]
        current_score = best_sq_nnz
        
        improved = True
        max_iterations = 1000 # Limit iterations to prevent infinite loops
        iterations = 0
        
        while improved and time.time() < deadline - 2.0 and iterations < max_iterations:
            improved = False
            iterations += 1
            for i in range(12):
                old_val = current_poly[i]
                # Try small changes: +/- 1, +/- 2, +/- 3
                for delta in [-3, -2, -1, 1, 2, 3]:
                    new_val = old_val + delta
                    if new_val == 0:
                        continue
                    if abs(new_val) > 64:
                        continue
                    
                    current_poly[i] = new_val
                    new_score = evaluate(current_poly)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        # Update global best immediately
                        update_best(current_poly)
                    else:
                        current_poly[i] = old_val
                
                # If we improved, we might want to restart the loop or continue?
                # Continuing is fine, but restarting ensures we don't miss other single-step improvements
                # in the new neighborhood. However, simple hill climbing is usually sufficient.
                # Let's break out of the inner loop if we found an improvement to re-evaluate neighbors
                # in the new state? No, standard hill climbing checks all neighbors.
                # But if we change i, the score changes. The other indices' neighbors might be different.
                # So we should restart the outer loop if we improved.
                if improved:
                    break
        
        # Final check
        if time.time() < deadline - 1.0:
             update_best(current_poly)

    # Ensure output is valid
    # Trim trailing zeros? No, the problem says "coefficient list in ascending degree".
    # If the leading coefficient is 0, it's not a valid polynomial of that degree, but the list length is <= 41.
    # "nonzero final coefficient" implies the last element in the list must be non-zero.
    # So we should trim trailing zeros.
    
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
        
    if len(best_poly) == 0 or best_poly[-1] == 0:
        best_poly = [1]*12
        
    # Verify exactly 12 nonzero terms
    if _count_nonzero_terms(best_poly) != 12:
        # Fallback to a known valid 12-term poly
        best_poly = [1]*12
        
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
        
    return best_poly

# EVOLVE-BLOCK-END
