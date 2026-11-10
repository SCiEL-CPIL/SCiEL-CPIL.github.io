import numpy as np
import time
import random
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    def count_nonzero(poly):
        return int(np.count_nonzero(poly))
    
    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)
    
    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)
    
    best_score = float('inf')
    best_poly = [1] * 12  # baseline
    
    def try_poly(coeffs):
        nonlocal best_score, best_poly
        if len(coeffs) != 12:
            return
        if sum(1 for c in coeffs if c != 0) != 12:
            return
        if any(c < -64 or c > 64 for c in coeffs):
            return
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_poly = list(coeffs)
    
    # Try baseline
    try_poly([1] * 12)
    
    # === CONSTRAINT SATISFACTION APPROACH ===
    # The square has 23 coefficients (degrees 0..22).
    # sq[k] = sum_{i+j=k} p[i]*p[j]
    # We want to choose p[0..11] (all nonzero, in [-64,64]) to minimize nonzero sq entries.
    # Strategy: pick a target support S for the square (which degrees are nonzero),
    # then solve for p coefficients that make all other degrees zero.
    
    # For a given target support, the zero constraints are linear equations in p.
    # But they're actually quadratic (products of p's). However, we can use
    # the fact that if we fix some p's, the remaining constraints become simpler.
    
    # Key observation: if p is palindromic (p[i] = p[11-i]), then the square is also
    # palindromic, so we only need to worry about degrees 0..11 for the square.
    # More generally, let's try structured approaches.
    
    # Approach 1: Try to find p such that p^2 has exactly k nonzero terms for small k.
    # The minimum possible is... let's think. p has 12 terms. p^2 has at most 23 terms.
    # Can we get p^2 to have very few terms?
    
    # If p = a*(x^n - 1) for some n, p^2 = a^2*(x^{2n} - 2x^n + 1) has 3 terms.
    # But p must have exactly 12 nonzero terms.
    
    # Let's try: p = sum of 12 monomials with coefficients chosen so cross-terms cancel.
    
    # For the square coefficient at degree k: sq[k] = sum_{i=0}^{min(k,11)} p[i]*p[k-i]
    # (where p[i]=0 for i<0 or i>11)
    
    # Let's do a systematic search over the "shape" of the square.
    # Try all subsets of {0,...,22} of size k for the nonzero positions of the square,
    # for small k, and check if there exist integer p's satisfying the constraints.
    
    # Actually, let's be smarter. Let's parameterize p by its values and use
    # the zero-constraint equations to guide the search.
    
    # For degree 0: sq[0] = p[0]^2 (always nonzero if p[0]!=0)
    # For degree 1: sq[1] = 2*p[0]*p[1]
    # For degree 2: sq[2] = p[1]^2 + 2*p[0]*p[2]
    # etc.
    
    # If we want sq[1] = 0, we need p[0]*p[1] = 0, but both must be nonzero.
    # So sq[1] can NEVER be zero! Similarly, sq[0] = p[0]^2 > 0 always.
    
    # Wait, that's for integer coefficients. p[0] and p[1] are both nonzero integers,
    # so 2*p[0]*p[1] is never zero. So degree 1 of the square is ALWAYS nonzero.
    # Similarly, degree 0 is always nonzero.
    # Degree 22 = p[11]^2 is always nonzero.
    # Degree 21 = 2*p[10]*p[11] is always nonzero.
    
    # So at minimum, the square has nonzero terms at degrees 0, 1, 21, 22. That's 4 terms.
    # Can we make all others zero?
    
    # Degree 2: p[1]^2 + 2*p[0]*p[2] = 0 => p[2] = -p[1]^2/(2*p[0])
    # For this to be integer, 2*p[0] must divide p[1]^2.
    
    # Let's try a backtracking approach: build p[0], p[1], ..., p[11] one by one,
    # and at each step, check which square coefficients are already determined
    # and enforce that they're zero if we want them to be.
    
    # Actually, the square coefficient at degree k depends on p[0..k] and p[k-0..k-11].
    # Once we've set p[0..m], the square coefficients at degrees 0..m are fully determined
    # (for degree k <= m, sq[k] = sum_{i=0}^{k} p[i]*p[k-i], and all indices are <= m).
    
    # So we can do: choose p[0], then p[1], ..., and at each step m, the square
    # coefficients sq[0..m] are determined. We can choose p[m] to try to make
    # sq[m] = 0 (if possible) or not.
    
    # sq[m] = p[m]^2 + 2*sum_{i=1}^{floor(m/2)} p[i]*p[m-i] + (if m even, p[m/2]^2 already counted)
    # Wait, let me be more careful.
    # sq[m] = sum_{i=0}^{m} p[i]*p[m-i] where p[j]=0 for j>11
    # = p[0]*p[m] + p[1]*p[m-1] + ... + p[m]*p[0]
    # = 2*sum_{i=0}^{floor((m-1)/2)} p[i]*p[m-i] + (if m even) p[m/2]^2
    # But p[m] appears in p[0]*p[m] and p[m]*p[0], so:
    # sq[m] = 2*p[0]*p[m] + 2*sum_{i=1}^{floor((m-1)/2)} p[i]*p[m-i] + (if m even, p[m/2]^2)
    
    # For m <= 11: sq[m] = 2*p[0]*p[m] + 2*sum_{i=1}^{floor((m-1)/2)} p[i]*p[m-i] + (m even ? p[m/2]^2 : 0)
    # To make sq[m]=0: 2*p[0]*p[m] = -(2*sum_{i=1}^{floor((m-1)/2)} p[i]*p[m-i] + (m even ? p[m/2]^2 : 0))
    # So p[m] = -(sum_{i=1}^{floor((m-1)/2)} p[i]*p[m-i] + (m even ? p[m/2]^2/2 : 0)) / p[0]
    
    # This gives a unique value for p[m] (if p[0] != 0) that makes sq[m]=0.
    # We can check if it's an integer in [-64,64] and nonzero.
    
    # This is great! We can do a tree search:
    # - Choose p[0] (nonzero, in [-64,64])
    # - Choose p[1] (nonzero, in [-64,64])
    # - For m = 2,3,...,11: either set p[m] to the "canceling" value (if valid) to make sq[m]=0,
    #   or choose p[m] freely (nonzero, in [-64,64]) to make sq[m]!=0.
    
    # But we also need to handle degrees 12..22 which involve p[12..22]=0.
    # For m >= 12: sq[m] = sum_{i=m-11}^{11} p[i]*p[m-i] (since p[j]=0 for j<0 or j>11)
    # These don't involve p[m] (since m>11), so they're determined once p[0..11] are set.
    
    # So the strategy is:
    # 1. Choose p[0], p[1]
    # 2. For m=2..11, decide whether to cancel sq[m] or not
    # 3. After setting all p[0..11], check sq[12..22] and count total nonzeros
    
    # Let's implement this!
    
    def backtrack_search():
        nonlocal best_score, best_poly
        
        def sq_coeff(m, p):
            """Compute square coefficient at degree m given p[0..min(m,11)]"""
            total = 0
            for i in range(max(0, m-11), min(m, 11) + 1):
                j = m - i
                if 0 <= j <= 11:
                    total += p[i] * p[j]
            return total
        
        def count_sq_nonzero(p):
            """Count nonzero coefficients in p^2"""
            count = 0
            for m in range(23):
                if sq_coeff(m, p) != 0:
                    count += 1
            return count
        
        def rec(p, m):
            nonlocal best_score, best_poly
            if time.time() > deadline:
                return
            if m > 11:
                # All p's set, evaluate
                score = count_sq_nonzero(p)
                if score < best_score:
                    best_score = score
                    best_poly = list(p)
                return
            
            if m == 0:
                # Choose p[0]
                for v in range(1, 10):  # By symmetry, p[0] > 0 (we can flip signs)
                    p[0] = v
                    rec(p, 1)
                return
            
            if m == 1:
                # Choose p[1]
                for v in range(-9, 10):
                    if v == 0:
                        continue
                    p[1] = v
                    rec(p, 2)
                return
            
            # For m >= 2: decide cancel or free
            # First, try the canceling value
            # sq[m] = 2*p[0]*p[m] + 2*sum_{i=1}^{floor((m-1)/2)} p[i]*p[m-i] + (m even ? p[m/2]^2 : 0)
            # Cancel: p[m] = -(2*sum_{i=1}^{floor((m-1)/2)} p[i]*p[m-i] + (m even ? p[m/2]^2 : 0)) / (2*p[0])
            
            cross_sum = 0
            for i in range(1, m):
                j = m - i
                if 0 <= j <= 11 and i != j:
                    cross_sum += p[i] * p[j]
                elif i == j and 0 <= i <= 11:
                    cross_sum += p[i] * p[j]  # p[m/2]^2 when m even and i=j=m/2
            # Wait, I need to be more careful.
            # sq[m] = sum_{i=0}^{m} p[i]*p[m-i] where p[j]=0 for j>11
            # Terms involving p[m]: i=0 gives p[0]*p[m], i=m gives p[m]*p[0]
            # Other terms: i=1..m-1, j=m-i, both in [0,11]
            other_sum = 0
            for i in range(1, m):
                j = m - i
                if 0 <= i <= 11 and 0 <= j <= 11:
                    other_sum += p[i] * p[j]
            # sq[m] = 2*p[0]*p[m] + other_sum
            # Cancel: p[m] = -other_sum / (2*p[0])
            
            if other_sum % (2 * p[0]) == 0:
                cancel_val = -other_sum // (2 * p[0])
                if cancel_val != 0 and -64 <= cancel_val <= 64:
                    p[m] = cancel_val
                    rec(p, m + 1)
            
            # Also try some free values
            # To limit branching, try a few strategic values
            free_vals = set()
            # Values related to existing coefficients
            for i in range(m):
                free_vals.add(p[i])
                free_vals.add(-p[i])
                free_vals.add(2 * p[i])
                free_vals.add(-2 * p[i])
            free_vals.add(1)
            free_vals.add(-1)
            free_vals.add(2)
            free_vals.add(-2)
            free_vals.add(3)
            free_vals.add(-3)
            
            for v in free_vals:
                if v == 0 or not (-64 <= v <= 64):
                    continue
                if v == cancel_val if other_sum % (2*p[0]) == 0 else True:
                    continue  # Already tried cancel
                p[m] = v
                rec(p, m + 1)
        
        p = [0] * 12
        rec(p, 0)
    
    # Run the backtracking search
    backtrack_search()
    
    # Also try with p[0] = -1 (sign flip symmetry)
    # Actually, if p works, -p works too, and p^2 = (-p)^2, so no new solutions.
    # But we restricted p[0] > 0, which is fine.
    
    # Additional: try palindromic polynomials specifically
    # p[i] = p[11-i] for all i
    # Then the square is also palindromic, and we only need to set p[0..5]
    
    def palindromic_search():
        nonlocal best_score, best_poly
        
        def sq_coeff_pal(m, p_half):
            """p[i] = p_half[min(i, 11-i)] for i in 0..11"""
            p = [p_half[min(i, 11-i)] for i in range(12)]
            total = 0
            for i in range(max(0, m-11), min(m, 11) + 1):
                j = m - i
                if 0 <= j <= 11:
                    total += p[i] * p[j]
            return total
        
        def count_sq_nonzero_pal(p_half):
            count = 0
            for m in range(23):
                if sq_coeff_pal(m, p_half) != 0:
                    count += 1
            return count
        
        def rec_pal(p_half, m):
            nonlocal best_score, best_poly
            if time.time() > deadline:
                return
            if m > 5:
                # All half-coeffs set
                p = [p_half[min(i, 11-i)] for i in range(12)]
                if all(p[i] != 0 for i in range(12)):
                    score = count_sq_nonzero_pal(p_half)
                    if score < best_score:
                        best_score = score
                        best_poly = list(p)
                return
            
            if m == 0:
                for v in range(1, 10):
                    p_half[0] = v
                    rec_pal(p_half, 1)
                return
            
            # For m >= 1, try a range of values
            for v in range(-8, 9):
                if v == 0:
                    continue
                p_half[m] = v
                rec_pal(p_half, m + 1)
        
        p_half = [0] * 6
        rec_pal(p_half, 0)
    
    palindromic_search()
    
    # Now do the existing local search to polish
    random.seed(42)
    
    def local_search(start, max_iters=3000):
        curr = list(start)
        curr_score = evaluate(curr)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            pos = random.randint(0, 11)
            old_val = curr[pos]
            
            candidate_vals = set()
            for other_pos in range(12):
                if other_pos != pos:
                    candidate_vals.add(curr[other_pos])
                    candidate_vals.add(-curr[other_pos])
                    candidate_vals.add(2 * curr[other_pos])
                    candidate_vals.add(-2 * curr[other_pos])
                    candidate_vals.add(3 * curr[other_pos])
                    candidate_vals.add(-3 * curr[other_pos])
            
            for _ in range(8):
                candidate_vals.add(random.randint(-64, 64))
            
            candidate_vals = [v for v in candidate_vals if -64 <= v <= 64 and v != 0 and v != old_val]
            
            if candidate_vals:
                new_val = random.choice(candidate_vals)
                curr[pos] = new_val
                new_score = evaluate(curr)
                if new_score < curr_score:
                    curr_score = new_score
                    improved = True
                    try_poly(curr)
                else:
                    curr[pos] = old_val
            
            if random.random() < 0.25:
                i, j = random.sample(range(12), 2)
                a, b = curr[i], curr[j]
                curr[i], curr[j] = b, a
                new_score = evaluate(curr)
                if new_score < curr_score:
                    curr_score = new_score
                    improved = True
                    try_poly(curr)
                else:
                    curr[i], curr[j] = a, b
            
            if random.random() < 0.15:
                pos = random.randint(0, 11)
                old_val = curr[pos]
                new_val = -old_val
                if new_val != 0:
                    curr[pos] = new_val
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        try_poly(curr)
                    else:
                        curr[pos] = old_val
            
            if random.random() < 0.1:
                i, j = 11 - random.randint(0, 5), random.randint(0, 5)
                if i != j:
                    a, b = curr[i], curr[j]
                    curr[i] = b
                    curr[j] = a
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        try_poly(curr)
                    else:
                        curr[i], curr[j] = a, b
            
            if not improved and random.random() < 0.05:
                for _ in range(random.randint(1, 3)):
                    p = random.randint(0, 11)
                    curr[p] = random.randint(-16, 16)
                    if curr[p] == 0:
                        curr[p] = 1
        
        return curr, curr_score
    
    # Multiple restarts
    for restart in range(800):
        if time.time() > deadline:
            break
        
        if restart < 15:
            start = list(best_poly)
        elif random.random() < 0.4:
            start = list(best_poly)
            for _ in range(random.randint(1, 4)):
                p = random.randint(0, 11)
                start[p] = max(-64, min(64, start[p] + random.randint(-10, 10)))
                if start[p] == 0:
                    start[p] = random.choice([-1, 1])
        else:
            pattern_type = random.randint(0, 5)
            if pattern_type == 0:
                start = [random.choice([-1, 1]) * random.randint(1, 6) for _ in range(12)]
            elif pattern_type == 1:
                base = random.randint(1, 4)
                start = [base * random.choice([-1, 1]) * (1 + (i % 3)) for i in range(12)]
            elif pattern_type == 2:
                start = [random.randint(-4, 4) for _ in range(12)]
                for i in range(12):
                    if start[i] == 0:
                        start[i] = random.choice([-1, 1])
            elif pattern_type == 3:
                start = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
            elif pattern_type == 4:
                half = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(6)]
                start = half + half[::-1]
                for i in range(12):
                    if start[i] == 0:
                        start[i] = random.choice([-1, 1])
            else:
                start = [random.choice([-1, 1]) * (1 + (i % 2)) * random.randint(1, 4) for i in range(12)]
                for i in range(12):
                    if start[i] == 0:
                        start[i] = 1
        
        result, score = local_search(start, max_iters=1200)
        if score < best_score:
            best_score = score
            best_poly = result
    
    # Final refinement
    if time.time() < deadline:
        for pos in range(12):
            if time.time() > deadline:
                break
            old_val = best_poly[pos]
            for new_val in range(-64, 65):
                if new_val == old_val or new_val == 0:
                    continue
                best_poly[pos] = new_val
                new_score = evaluate(best_poly)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_poly[pos] = old_val
            best_poly[pos] = old_val
    
    if time.time() < deadline:
        for pos in range(12):
            if time.time() > deadline:
                break
            old_val = best_poly[pos]
            for new_val in range(-64, 65):
                if new_val == old_val or new_val == 0:
                    continue
                best_poly[pos] = new_val
                new_score = evaluate(best_poly)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_poly[pos] = old_val
            best_poly[pos] = old_val
    
    return best_poly
