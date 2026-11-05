import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    def count_square_terms(d):
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
        return True

    def to_list(d):
        """Convert sparse dict to dense coefficient list (ascending degree)."""
        if not d:
            return [0]
        max_exp = max(d.keys())
        lst = [0] * (max_exp + 1)
        for e, c in d.items():
            lst[e] = c
        return lst

    best_score = float('inf')
    best_poly = None

    def update_best(d, score):
        nonlocal best_score, best_poly
        if score < best_score:
            best_score = score
            best_poly = to_list(d)

    # ====================================================================
    # PHASE 1: Structural Construction via Q(x^d)
    # Strategy: P(x) = Q(x^d). If we choose Q such that Q^2 has few terms,
    # and d is large enough to separate cross-terms, P^2 will have few terms.
    # Specifically, if Q has 12 terms, Q^2 has at most 24 terms (if no overlap).
    # But we want to MINIMIZE terms in P^2.
    # If we pick Q with 12 terms all with same sign, Q^2 has 13 terms (0..22) if consecutive.
    # P(x) = Q(x^d) => P^2 = Q(x^d)^2.
    # If Q has consecutive exponents 0..11, Q^2 has exponents 0..22 (13 terms).
    # P^2 will have exponents 0, d, 2d, ..., 22d. All distinct.
    # So P^2 has 13 nonzero terms.
    # This is a guaranteed 13-term solution.
    # ====================================================================
    
    # Try Q with consecutive exponents 0..11, coeffs all 1.
    d_Q = {i: 1 for i in range(12)}
    # Q^2 has terms at 0, 1, ..., 22. 13 terms.
    # P(x) = Q(x^d). P has 12 terms at 0, d, ..., 11d.
    # P^2 has terms at 0, d, ..., 22d. 13 terms.
    # We need 12 terms in P. This works.
    # We need to minimize terms in P^2. 13 is a good baseline.
    # Can we do better?
    # If Q has 12 terms, can Q^2 have < 13 terms?
    # Minimum terms in square of 12-term poly is 1 (if it's a monomial, but it has 12 terms).
    # Actually, if Q = x^a + x^b + ... + x^l, Q^2 has 13 terms if consecutive.
    # If Q has gaps, Q^2 has more terms (up to 24).
    # So consecutive exponents in Q minimize the spread, but the number of terms in Q^2
    # depends on overlaps.
    # For Q with 12 terms, min terms in Q^2 is likely 13 (consecutive).
    # So P^2 will have 13 terms.
    
    # Let's construct this specific P.
    d_val = 3 # Must be such that max_exp <= 40. 11*3 = 33 <= 40.
    d_P = {i * d_val: 1 for i in range(12)}
    score_P = count_square_terms(d_P)
    update_best(d_P, score_P)
    
    # Try to find a Q such that Q^2 has fewer than 13 terms?
    # Impossible for 12 distinct terms? 
    # If Q = (x^a + ... + x^l)^2, the number of terms is the number of distinct sums i+j.
    # For a set S of size 12, |S+S| >= 2*12 - 1 = 23? No, that's for arithmetic progressions?
    # For consecutive integers, |S+S| = 23? No, 0..11 + 0..11 = 0..22 (23 terms).
    # Wait. 0+0=0, 11+11=22. 23 terms.
    # My previous calculation of 13 was wrong.
    # Let's re-verify.
    # Q = 1 + x + ... + x^11.
    # Q^2 = (1 + x + ... + x^11)^2.
    # Coeff of x^k in Q^2 is min(k+1, 23-k, 12).
    # All coeffs are nonzero.
    # So Q^2 has 23 terms.
    # So P^2 has 23 terms.
    # This is worse than the current best (which is likely ~14-16 from random search).
    
    # So the "guaranteed 12 terms" in the prompt idea was based on a misunderstanding.
    # "force all cross-terms ... to land on distinct degrees ... guaranteeing a square with exactly 12 non-zero terms"
    # This is impossible. If P has 12 terms, P^2 has at least 12 terms (the squares of the terms).
    # But cross terms add more.
    # Unless cross terms cancel.
    
    # So Phase 1 should just be a fast random search to get a decent baseline quickly,
    # then use Simulated Annealing to improve.
    
    # ====================================================================
    # PHASE 2: Simulated Annealing / Stochastic Local Search
    # ====================================================================
    
    # Initialize with a random valid polynomial
    def random_poly():
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        return {exps[i]: coeffs[i] for i in range(12)}
    
    current_d = random_poly()
    current_score = count_square_terms(current_d)
    update_best(current_d, current_score)
    
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Generate neighbor
        new_d = dict(current_d)
        move_type = random.randint(0, 2)
        
        if move_type == 0:
            # Change a coefficient
            key = random.choice(list(new_d.keys()))
            old_c = new_d[key]
            new_c = old_c + random.choice([-2, -1, 1, 2])
            if new_c < -64 or new_c > 64:
                continue
            if new_c == 0:
                del new_d[key]
                # Need to add another term to keep 12 terms?
                # Or just allow variable number of terms?
                # Task requires EXACTLY 12 nonzero terms.
                # So we must maintain 12 terms.
                # If we delete, we must add a new one.
                new_exp = random.choice([e for e in range(41) if e not in new_d])
                new_d[new_exp] = random.choice([-2, -1, 1, 2])
            else:
                new_d[key] = new_c
                
        elif move_type == 1:
            # Change an exponent
            key = random.choice(list(new_d.keys()))
            c = new_d[key]
            del new_d[key]
            new_exp = random.choice([e for e in range(41) if e not in new_d])
            new_d[new_exp] = c
            
        else:
            # Swap two exponents
            keys = list(new_d.keys())
            k1, k2 = random.sample(keys, 2)
            c1, c2 = new_d[k1], new_d[k2]
            new_d[k1] = c2
            new_d[k2] = c1
            
        if len(new_d) != 12:
            continue
            
        new_score = count_square_terms(new_d)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_d = new_d
            current_score = new_score
            update_best(current_d, current_score)
            
        T *= cooling
        if T < 0.01:
            T = 0.01 # Keep some randomness

    # ====================================================================
    # FINAL: Return best found, or fallback
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
