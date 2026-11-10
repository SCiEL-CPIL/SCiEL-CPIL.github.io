import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin: 90s limit, keep 5s buffer

    def count_nonzero(poly):
        # poly is a list or array of ints
        return sum(1 for x in poly if x != 0)

    def evaluate(coeffs):
        """
        Fast evaluation of the number of non-zero terms in the square.
        coeffs: list of 12 ints.
        Returns: int count of non-zero terms in coeffs * coeffs.
        """
        # Using numpy convolve is generally fast for small n, but let's try manual 
        # to avoid overhead if possible. 
        # Actually, for n=12, np.convolve is very fast.
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    def validate(coeffs):
        if len(coeffs) != 12:
            return False
        if coeffs[-1] == 0:
            return False
        if coeffs[0] == 0:
            return False
        if not all(isinstance(c, int) for c in coeffs):
            return False
        if not all(-64 <= c <= 64 for c in coeffs):
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    random.seed(42)

    # Precompute pair contributions for fast algebraic solving
    # sq[k] = sum_{i+j=k} c[i]*c[j]
    # We need this for try_zero_out
    
    def compute_square(coeffs):
        """Compute full square coefficients as a list of 23 ints."""
        sq = [0] * 23
        for i in range(12):
            ci = coeffs[i]
            if ci == 0:
                continue
            for j in range(i, 12): # Optimization: exploit symmetry? 
                # Actually, convolve handles it. For algebraic solving, we need exact terms.
                # Let's just do the full sum for correctness in the solver
                pass
        # Re-do full sum for correctness in solver context
        sq = [0] * 23
        for i in range(12):
            ci = coeffs[i]
            if ci == 0:
                continue
            for j in range(12):
                cj = coeffs[j]
                if cj == 0:
                    continue
                sq[i + j] += ci * cj
        return sq

    def count_nz(sq):
        return sum(1 for x in sq if x != 0)

    def try_zero_out(coeffs, k, max_iter=100):
        """
        Try to zero out sq[k] by modifying exactly one coefficient c[i].
        Returns (new_coeffs, new_score) if successful and better, else (None, None).
        """
        sq = compute_square(coeffs)
        if sq[k] == 0:
            return None, None
        
        current_score = count_nz(sq)
        
        # For each i that can affect sq[k]
        for i in range(12):
            j = k - i
            if j < 0 or j >= 12:
                continue
            
            # sq[k] = A*c[i]^2 + B*c[i] + C
            A = 1 if (2 * i == k) else 0
            B = 0
            C = 0
            
            # Calculate B and C by excluding terms involving i
            for l in range(12):
                m = k - l
                if m < 0 or m >= 12:
                    continue
                if l == i or m == i:
                    continue
                # If l == m, term is c[l]^2. If l != m, term is c[l]*c[m].
                # But in the sum sq[k] = sum c[l]c[m], if l!=m, both (l,m) and (m,l) are in the sum?
                # Yes, convolve sums all pairs.
                # So if l != m, we add c[l]*c[m] once here? No, the loop over l covers all l.
                # If we iterate l from 0..11, and m=k-l is valid:
                # The term c[l]*c[m] is added.
                # If l != m, the pair (m,l) is also added when l=m.
                # So C is just the sum of all c[l]*c[m] where l!=i and m!=i.
                C += coeffs[l] * coeffs[m]
            
            # B: Coefficient of c[i] in sq[k]
            # Terms involving c[i]: c[i]*c[j] where j=k-i.
            # If i != j: The terms are c[i]*c[j] (when l=i, m=j) and c[j]*c[i] (when l=j, m=i).
            # So B = 2 * coeffs[j].
            # If i == j: The term is c[i]*c[i] = c[i]^2. This is the A term. B=0.
            if i != j:
                B = 2 * coeffs[j]
            else:
                B = 0
            
            # Solve A*c[i]^2 + B*c[i] + C = 0
            new_val = None
            if A == 0:
                # Linear: B*c[i] + C = 0
                if B != 0:
                    if C % B == 0:
                        candidate = -C // B
                        if -64 <= candidate <= 64 and candidate != 0:
                            new_val = candidate
            else:
                # Quadratic: c[i]^2 + B*c[i] + C = 0
                disc = B * B - 4 * C
                if disc >= 0:
                    sqrt_disc = int(np.sqrt(disc))
                    if sqrt_disc * sqrt_disc == disc:
                        for sign in [1, -1]:
                            num = -B + sign * sqrt_disc
                            den = 2
                            if num % den == 0:
                                candidate = num // den
                                if -64 <= candidate <= 64 and candidate != 0:
                                    # Prefer the one that is not the current value if possible, 
                                    # or just take the first valid one
                                    if new_val is None or candidate != coeffs[i]:
                                        new_val = candidate
            
            if new_val is not None and new_val != coeffs[i]:
                new_coeffs = coeffs.copy()
                new_coeffs[i] = new_val
                new_sq = compute_square(new_coeffs)
                new_score = count_nz(new_sq)
                if new_score < current_score:
                    return new_coeffs, new_score
        
        return None, None

    def run_sa(num_restarts=10, iters_per_restart=10000):
        nonlocal best_score, best_poly
        
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize
            curr = [random.choice([-1, 1]) for _ in range(12)]
            curr_score = evaluate(curr)
            
            if curr_score < best_score:
                best_score = curr_score
                best_poly = curr.copy()
                
            T = 5.0
            T_min = 0.1
            cooling_rate = 0.9995
            stuck_count = 0
            max_stuck = 500
            
            for i in range(iters_per_restart):
                if time.time() > deadline - 2:
                    return
                
                # Occasionally try an algebraic zeroing move if stuck
                if stuck_count > max_stuck and random.random() < 0.1:
                    sq = compute_square(curr)
                    nonzero_ks = [k for k in range(23) if sq[k] != 0]
                    if nonzero_ks:
                        k = random.choice(nonzero_ks)
                        res = try_zero_out(curr, k)
                        if res[0] is not None:
                            curr = res[0]
                            curr_score = res[1]
                            if curr_score < best_score:
                                best_score = curr_score
                                best_poly = curr.copy()
                            stuck_count = 0
                            continue

                # Standard SA move
                idx = random.randint(0, 11)
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
                        stuck_count = 0
                    else:
                        stuck_count += 1
                else:
                    curr[idx] = old_val
                    stuck_count += 1
                    
                T *= cooling_rate
                if T < T_min:
                    T = T_min

    # Phase 1: Deterministic Seeds
    seeds = [
        [1]*12,
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1], # Invalid, will be fixed
        [2, 1, -1, -2, 1, 2, -1, -2, 1, 2, -1, 1],
    ]
    for s in seeds:
        # Ensure valid
        if len(s) != 12: continue
        valid_s = [x if x != 0 else 1 for x in s]
        if all(-64 <= x <= 64 for x in valid_s):
            score = evaluate(valid_s)
            if score < best_score:
                best_score = score
                best_poly = valid_s.copy()

    # If no valid best found yet, use random
    if best_poly is None:
        best_poly = [1]*12
        best_score = evaluate(best_poly)

    # Phase 2: Simulated Annealing
    run_sa(num_restarts=15, iters_per_restart=15000)

    # Phase 3: Local Search Polish
    if best_poly is not None and time.time() < deadline - 5:
        curr = best_poly.copy()
        curr_score = best_score
        
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            idx = random.randint(0, 11)
            old_val = curr[idx]
            
            candidates = set()
            for dv in [-2, -1, 1, 2]:
                nv = old_val + dv
                if -64 <= nv <= 64 and nv != 0:
                    candidates.add(nv)
            candidates.add(-old_val)
            candidates.add(random.choice([-5, -3, -2, -1, 1, 2, 3, 5]))
            
            improved = False
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
                    no_improve = 0
                    break
            if not improved:
                curr[idx] = old_val
                no_improve += 1

    if best_poly is None:
        best_poly = [1] * 12

    if not validate(best_poly):
        best_poly = [1] * 12

    return best_poly
