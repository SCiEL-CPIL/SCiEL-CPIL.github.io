import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    best_score = float('inf')
    best_poly = None
    
    def update_best(poly_dict, score):
        nonlocal best_score, best_poly
        if score < best_score:
            best_score = score
            # Convert dict to list
            max_exp = max(poly_dict.keys())
            lst = [0] * (max_exp + 1)
            for e, c in poly_dict.items():
                lst[e] = c
            best_poly = lst
    
    def get_score(d):
        """Exact count of nonzero terms in square."""
        square = {}
        items = list(d.items())
        for i in range(len(items)):
            ei, ci = items[i]
            for j in range(i, len(items)):
                ej, cj = items[j]
                key = ei + ej
                if i == j:
                    val = ci * ci
                else:
                    val = 2 * ci * cj
                square[key] = square.get(key, 0) + val
        return sum(1 for v in square.values() if v != 0)

    # ====================================================================
    # STRATEGY: Systematic Constraint-Driven Search on Arithmetic Progressions
    #
    # Insight: If exponents are $e_i = a + i \cdot d$, the square's degrees
    # are $2a + (i+j)d$. The number of distinct degrees is at most $2N-1$.
    # We want to choose coefficients $c_i$ such that sums of products cancel.
    #
    # We focus on $d=1$ (consecutive exponents) as it maximizes overlap
    # potential for cancellation. We use backtracking to assign coefficients
    # $c_0, \dots, c_{11}$ such that $S_k = \sum c_i c_{k-i} = 0$ for many $k$.
    #
    # Pruning: We can compute partial sums as we assign coefficients.
    # ====================================================================
    
    # Try different starting exponents and step sizes
    # But primarily focus on d=1 for densest packing
    base_exps = [
        list(range(12)), # 0..11
        list(range(1, 13)), # 1..12
        list(range(2, 14)), # 2..13
        # Maybe some with small gaps? e.g., 0,1,2,3,4,5,6,7,8,9,10,11 is best for overlap
    ]
    
    # Coefficient range to search
    # Small integers are most likely to produce exact cancellations
    coeff_choices = [-3, -2, -1, 1, 2, 3]
    
    # We will use a recursive backtracking solver for the coefficients
    # given a fixed set of exponents.
    
    def solve_for_exponents(exps, coeff_set):
        """
        Find coefficients for given exponents that minimize square terms.
        Uses backtracking with pruning based on partial square sums.
        """
        n = len(exps)
        if n != 12:
            return
        
        # Precompute the mapping from index pairs to degree
        # deg[i][j] = exps[i] + exps[j]
        # We want to minimize the number of non-zero degrees in the sum.
        
        # State: coeffs[0..k-1] are assigned.
        # We can track the current partial square contributions.
        # However, full tracking is expensive. 
        # Instead, we rely on the fact that for d=1, S_k depends only on c_i for i in range.
        
        # Let's use a simpler heuristic: Local Search with strong initialization
        # based on known algebraic identities.
        
        # Identity: (1+x+x^2+...+x^{n-1})^2 has coefficients 1,2,...,n,...,2,1.
        # Identity: (1-x+x^2-...+x^{n-1})^2 = (1+x^2+x^4+...+x^{2(n-1)}) - 2(1-x^2+...)? No.
        # (1-x)^2 = 1-2x+x^2.
        # (1-x+x^2)^2 = 1 - 2x + 3x^2 - 2x^3 + x^4.
        
        # Let's try to find coefficients that make S_1, S_2, ... zero.
        # S_1 = 2 c_0 c_1 = 0 => c_0=0 or c_1=0. But we need 12 nonzeros.
        # So S_1 cannot be zero if c_0, c_1 != 0.
        # S_2 = c_0^2 + 2 c_1 c_2? No, S_2 = c_0 c_2 + c_1 c_1 + c_2 c_0 = 2 c_0 c_2 + c_1^2.
        # We want 2 c_0 c_2 + c_1^2 = 0.
        # If c_0=1, c_1=2, then 2(1)c_2 + 4 = 0 => c_2 = -2.
        # If c_0=1, c_1=1, then 2 c_2 + 1 = 0 => no integer solution.
        # So c_1 must be even if c_0 is odd? Or c_0 even?
        
        # Let's try a specific sequence:
        # c_0 = 1
        # c_1 = 2 -> S_1 = 4 (nonzero)
        # c_2 = -2 -> S_2 = 2(1)(-2) + 2^2 = -4+4=0. Good.
        # S_3 = 2 c_0 c_3 + 2 c_1 c_2 = 2(1)c_3 + 2(2)(-2) = 2 c_3 - 8.
        # Want S_3=0 => c_3 = 4.
        # S_4 = 2 c_0 c_4 + 2 c_1 c_3 + c_2^2 = 2(1)c_4 + 2(2)(4) + (-2)^2 = 2 c_4 + 16 + 4 = 2 c_4 + 20.
        # Want S_4=0 => c_4 = -10.
        # S_5 = 2 c_0 c_5 + 2 c_1 c_4 + 2 c_2 c_3 = 2(1)c_5 + 2(2)(-10) + 2(-2)(4) = 2 c_5 - 40 - 16 = 2 c_5 - 56.
        # Want S_5=0 => c_5 = 28.
        # S_6 = 2 c_0 c_6 + 2 c_1 c_5 + 2 c_2 c_4 + c_3^2
        # = 2(1)c_6 + 2(2)(28) + 2(-2)(-10) + 4^2
        # = 2 c_6 + 112 + 40 + 16 = 2 c_6 + 168.
        # Want S_6=0 => c_6 = -84. (Out of bounds [-64, 64])
        
        # The coefficients grow too fast. This suggests that zeroing out ALL middle terms
        # is impossible with bounded coefficients.
        
        # However, we don't need ALL to be zero. We just need to MINIMIZE the count.
        # The "greedy" zeroing above failed due to bounds.
        
        # Let's try smaller coefficients.
        # c_0 = 1
        # c_1 = 1 -> S_1 = 2
        # c_2 = ? S_2 = 2 c_2 + 1. No integer zero.
        
        # What if we allow S_1 to be nonzero?
        # The minimum number of nonzero terms in the square of a 12-term polynomial
        # is likely around 12-15 if we can cancel enough.
        
        # Let's use a randomized local search but with a smarter move:
        # Change two coefficients at once to satisfy a quadratic constraint.
        
        def local_search(exps, max_iters=20000):
            # Initialize with random small integers
            curr = {}
            for e in exps:
                curr[e] = random.choice([-2, -1, 1, 2])
            
            curr_score = get_score(curr)
            if curr_score < best_score:
                update_best(curr, curr_score)
            
            T = 5.0
            for it in range(max_iters):
                if time.time() > deadline:
                    return
                
                # Pick two indices to modify
                i1 = random.randint(0, 11)
                i2 = random.randint(0, 11)
                if i1 == i2:
                    i2 = (i2 + 1) % 12
                
                e1 = exps[i1]
                e2 = exps[i2]
                
                old_v1 = curr[e1]
                old_v2 = curr[e2]
                
                new_v1 = random.choice([-3, -2, -1, 1, 2, 3])
                new_v2 = random.choice([-3, -2, -1, 1, 2, 3])
                
                if new_v1 == old_v1 and new_v2 == old_v2:
                    continue
                
                curr[e1] = new_v1
                curr[e2] = new_v2
                
                new_score = get_score(curr)
                
                if new_score < curr_score:
                    curr_score = new_score
                    if curr_score < best_score:
                        update_best(curr, curr_score)
                else:
                    delta = new_score - curr_score
                    if random.random() < pow(2.718, -delta / max(T, 0.1)):
                        curr_score = new_score
                    else:
                        curr[e1] = old_v1
                        curr[e2] = old_v2
                
                T *= 0.999
            
            return curr_score

        # Run local search for this exponent set
        local_search(exps)
    
    # Run for a few different exponent configurations
    # 1. Consecutive 0..11
    if time.time() < deadline:
        solve_for_exponents(list(range(12)), coeff_choices)
        
    # 2. Consecutive 1..12
    if time.time() < deadline:
        solve_for_exponents(list(range(1, 13)), coeff_choices)
        
    # 3. Maybe a "block" structure: 0,1,2,3, 10,11,12,13, 20,21,22,23
    # This separates the polynomial into 3 clusters.
    # Cross terms between clusters are distinct degrees.
    # Intra-cluster terms are dense.
    # This might not be better than dense.
    
    # 4. Symmetric exponents? e.g., 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 is already symmetric in support.
    
    # Let's try a specific algebraic construction:
    # P(x) = (1+x+x^2+x^3+x^4+x^5)^2 - something? No, P must have 12 terms.
    
    # Try: P(x) = 1 + x + x^2 + x^3 + x^4 + x^5 - x^6 - x^7 - x^8 - x^9 - x^10 - x^11
    # This is the alternating block pattern.
    d = {i: (1 if i < 6 else -1) for i in range(12)}
    score = get_score(d)
    update_best(d, score)
    
    # Try: P(x) = 1 + x - x^2 - x^3 + x^4 + x^5 - x^6 - x^7 + x^8 + x^9 - x^10 - x^11
    d = {i: (1 if (i // 2) % 2 == 0 else -1) for i in range(12)}
    score = get_score(d)
    update_best(d, score)
    
    # Try: P(x) = 1 + 2x - 2x^2 + 4x^3 - 10x^4 + 28x^5 - 84x^6 ... (The growing sequence)
    # We saw c_6 = -84 is out of bounds. But maybe we can stop the zeroing earlier?
    # If we just use the first few terms of that sequence and randomize the rest?
    seq = [1, 2, -2, 4, -10, 28]
    d = {}
    for i, c in enumerate(seq):
        d[i] = c
    for i in range(6, 12):
        d[i] = random.choice([-3, -2, -1, 1, 2, 3])
    score = get_score(d)
    update_best(d, score)
    
    # Let's do a more targeted search on the "growing" sequence idea
    # but with smaller initial values to keep coefficients bounded.
    # Try c_0=1, c_1=1. S_1=2.
    # c_2=1. S_2=2+1=3.
    # c_3=-1. S_3=2(1)(-1)+2(1)(1)=0.
    # c_4=1. S_4=2(1)(1)+2(1)(-1)+1^2 = 2-2+1=1.
    # c_5=-1. S_5=2(1)(-1)+2(1)(1)+2(-1)(-1) = -2+2+2=2.
    # This doesn't seem to cancel well.
    
    # Let's rely on the local search to find the best pattern.
    # To improve local search, we can use a "restart" strategy with different seeds.
    
    random.seed(12345)
    for restart in range(5):
        if time.time() > deadline:
            break
        solve_for_exponents(list(range(12)), coeff_choices)
        
    # Fallback
    if best_poly is None:
        best_poly = [1] * 12
    
    # Final validation
    final_dict = {i: best_poly[i] for i in range(len(best_poly)) if best_poly[i] != 0}
    if len(final_dict) != 12:
        # This shouldn't happen if update_best is correct, but just in case
        best_poly = [1] * 12
    else:
        for c in final_dict.values():
            if not (-64 <= c <= 64):
                best_poly = [1] * 12
                break
        max_exp = max(final_dict.keys())
        if max_exp > 40:
            best_poly = [1] * 12

    return best_poly
