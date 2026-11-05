import time
import itertools
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    def count_square_terms_sparse(d):
        """Count nonzero terms in the exact integer square of sparse poly d={exp:coeff}."""
        if not d:
            return 0
        items = list(d.items())
        n = len(items)
        sq = {}
        for i in range(n):
            ei, ci = items[i]
            for j in range(i, n):
                ej, cj = items[j]
                key = ei + ej
                if i == j:
                    val = ci * ci
                else:
                    val = 2 * ci * cj
                sq[key] = sq.get(key, 0) + val
        return sum(1 for v in sq.values() if v != 0)

    def to_list(d):
        """Convert sparse dict to dense coefficient list (ascending degree)."""
        if not d:
            return [0]
        max_exp = max(d.keys())
        lst = [0] * (max_exp + 1)
        for e, c in d.items():
            lst[e] = c
        # Remove trailing zeros
        while len(lst) > 1 and lst[-1] == 0:
            lst.pop()
        return lst

    def is_valid(d):
        """Check if sparse dict represents a valid polynomial per task constraints."""
        if not d:
            return False
        if len(d) != 12:
            return False
        max_exp = max(d.keys())
        if max_exp > 40:
            return False
        for c in d.values():
            if c < -64 or c > 64:
                return False
        # Check final coefficient is nonzero
        if d[max_exp] == 0:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    def update_best(d):
        nonlocal best_score, best_poly
        if not is_valid(d):
            return
        score = count_square_terms_sparse(d)
        if score < best_score:
            best_score = score
            best_poly = to_list(d)

    # ====================================================================
    # PHASE 1: Arithmetic progression with step 2, optimized coefficients
    # Exponents: 0, 2, 4, ..., 22 (12 terms)
    # This gives a structured sumset where many pairs overlap.
    # Try small coefficient sets first, then local search.
    # ====================================================================
    
    exps_ap2 = list(range(0, 24, 2))  # 0,2,4,...,22
    
    # Try all combinations of coefficients from {-2,-1,1,2}
    # 4^12 = 16.7M - too many for full search, do targeted patterns
    
    # Pattern: alternating signs with magnitude 1
    d = {exps_ap2[i]: (1 if i % 2 == 0 else -1) for i in range(12)}
    update_best(d)
    
    # Pattern: all +1
    d = {exps_ap2[i]: 1 for i in range(12)}
    update_best(d)
    
    # Pattern: geometric-like with small coeffs
    for mag in [1, 2]:
        d = {exps_ap2[i]: (mag if i % 2 == 0 else -mag) for i in range(12)}
        update_best(d)
    
    # ====================================================================
    # PHASE 2: Local hill climbing from good starting points
    # ====================================================================
    
    def hill_climb(start_d, max_iters=50000, T_start=2.0, cooling=0.999, T_min=0.01):
        """Simulated annealing / hill climbing on coefficient values."""
        nonlocal best_score, best_poly
        
        d = dict(start_d)
        current_score = count_square_terms_sparse(d)
        
        if current_score < best_score:
            best_score = current_score
            best_poly = to_list(d)
        
        T = T_start
        iters = 0
        while T > T_min and iters < max_iters:
            if time.time() > deadline:
                break
            iters += 1
            
            # Pick a random exponent to change
            exp = random.choice(list(d.keys()))
            old_val = d[exp]
            
            # Try a new value
            new_val = random.randint(-64, 64)
            if new_val == 0:
                new_val = random.choice([-1, 1])
            
            d[exp] = new_val
            
            # Check validity
            if len(d) != 12 or any(c == 0 for c in d.values()):
                d[exp] = old_val
                T *= cooling
                continue
            
            new_score = count_square_terms_sparse(d)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_poly = to_list(d)
            else:
                # Metropolis acceptance
                delta = new_score - current_score
                if random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    d[exp] = old_val
            
            T *= cooling
        
        return best_score

    # Start from several initial configurations
    starts = []
    
    # AP step 2, alternating signs
    starts.append({exps_ap2[i]: (1 if i % 2 == 0 else -1) for i in range(12)})
    # AP step 2, all ones
    starts.append({exps_ap2[i]: 1 for i in range(12)})
    # Consecutive exponents, alternating signs
    starts.append({i: (1 if i % 2 == 0 else -1) for i in range(12)})
    # Consecutive exponents, all ones
    starts.append({i: 1 for i in range(12)})
    
    for s in starts:
        if time.time() > deadline:
            break
        hill_climb(s, max_iters=30000)

    # ====================================================================
    # PHASE 3: Structured search with arithmetic progression step 1
    # but with sign patterns designed to maximize cancellation
    # ====================================================================
    
    # For consecutive exponents 0..11, the square has support {0..22} = 23 terms
    # We want to cancel as many as possible.
    # Key insight: if c_i = (-1)^i, then P(x) = sum (-1)^i x^i = (1-x^12)/(1+x)
    # P(x)^2 = (1-x^12)^2 / (1+x)^2 = (1 - 2x^12 + x^24) / (1+2x+x^2)
    # This doesn't directly help, but let's search systematically.
    
    # Try all 2^12 sign patterns for exponents 0..11
    # 4096 patterns - very fast
    for signs in itertools.product([-1, 1], repeat=12):
        if time.time() > deadline:
            break
        d = {i: signs[i] for i in range(12)}
        score = count_square_terms_sparse(d)
        if score < best_score:
            best_score = score
            best_poly = to_list(d)
    
    # ====================================================================
    # PHASE 4: Two-cluster structure with optimization
    # P(x) = A(x) + x^M * B(x) where A and B are low-degree polys
    # Choose M so that cross terms create structured cancellation
    # ====================================================================
    
    if time.time() < deadline - 5.0:
        # Try M values that create good overlap structure
        for M in [5, 6, 7, 8, 9, 10]:
            if time.time() > deadline:
                break
            
            # A: 6 terms at exponents 0..5
            # B: 6 terms at exponents M..M+5
            # Try small coefficient sets for A and B
            
            # A with alternating signs, B with same pattern
            for sign_pattern in [(-1)**i for i in range(6)]:
                for mag_a in [1, 2]:
                    for mag_b in [1, 2]:
                        d = {}
                        for i in range(6):
                            d[i] = sign_pattern[i] * mag_a
                        for j in range(6):
                            d[M+j] = sign_pattern[j] * mag_b
                        if len(d) == 12:
                            update_best(d)

    # ====================================================================
    # PHASE 5: Random restarts with local search
    # ====================================================================
    
    if time.time() < deadline - 3.0:
        for _ in range(200):  # Up to 200 random restarts
            if time.time() > deadline:
                break
            
            # Generate random valid polynomial
            # Pick 12 distinct exponents from 0..40
            exps = random.sample(range(0, 41), 12)
            exps.sort()
            
            # Random small coefficients
            d = {e: random.choice([-2, -1, 1, 2]) for e in exps}
            
            # Quick local search
            hill_climb(d, max_iters=5000, T_start=1.0, cooling=0.995, T_min=0.01)

    # ====================================================================
    # FINAL: Validate and return
    # ====================================================================
    
    if best_poly is None:
        # Fallback: 12 consecutive terms with coefficient 1
        best_poly = [1] * 12
    else:
        # Ensure final coefficient is nonzero
        while len(best_poly) > 1 and best_poly[-1] == 0:
            best_poly.pop()
        # Ensure length <= 41
        if len(best_poly) > 41:
            best_poly = [1] * 12
        # Ensure exactly 12 nonzero terms
        nz = sum(1 for c in best_poly if c != 0)
        if nz != 12:
            best_poly = [1] * 12

    return best_poly
