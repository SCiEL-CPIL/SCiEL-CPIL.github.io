import time
import numpy as np
import itertools
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Fallback: (1 + x + ... + x^11)^2 has 21 non-zero terms
    fallback = [1] * 12
    fallback_score = 21
    
    best_score = fallback_score
    best_candidate = fallback

    def compute_score(exps, coeffs):
        """
        Computes the number of non-zero terms in P(x)^2.
        exps: list of 12 exponents
        coeffs: list of 12 coefficients
        """
        if not exps or not coeffs:
            return float('inf')
        
        max_deg = max(exps)
        if max_deg > 40:
            return float('inf')
        if any(abs(c) > 64 for c in coeffs):
            return float('inf')
        if any(c == 0 for c in coeffs):
            return float('inf')
        if len(set(exps)) != 12:
            return float('inf')
            
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coeffs):
            p[e] = c
            
        q = np.convolve(p, p)
        count = np.count_nonzero(q)
        return count

    def _update_best(exps, coeffs):
        """Update the global best candidate dense list."""
        nonlocal best_score, best_candidate
        score = compute_score(exps, coeffs)
        if score < best_score:
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if len(dense) <= 41 and dense[-1] != 0:
                best_score = score
                best_candidate = dense

    # Strategy: Model P(x) = A(x) * B(x) * C(x) where A, B, C are 4-term polynomials
    # with disjoint exponent supports (after appropriate shifts).
    # P(x)^2 = A(x)^2 * B(x)^2 * C(x)^2
    # We want to minimize non-zero terms in this product.
    
    # Let A(x) = sum_{i=0}^{3} a_i x^{e_i}
    # Let B(x) = sum_{j=0}^{3} b_j x^{f_j}
    # Let C(x) = sum_{k=0}^{3} c_k x^{g_k}
    
    # P(x) has 12 terms: {e_i + f_j + g_k} for all i,j,k.
    # We need these 12 exponents to be distinct.
    # This is possible if the sets {e_i}, {f_j}, {g_k} are chosen such that
    # the sums e_i + f_j + g_k are all distinct.
    # A simple way is to use arithmetic progressions with large enough gaps.
    # e.g., e_i = i, f_j = 10*j, g_k = 100*k.
    # Then exponents are i + 10j + 100k.
    # Max exponent = 3 + 30 + 300 = 333 > 40. Too big.
    
    # We need tighter packing.
    # Let e_i = i (0,1,2,3)
    # Let f_j = 4*j (0,4,8,12)
    # Let g_k = 16*k (0,16,32,48) -> Max 3+12+48=63 > 40.
    
    # Let's try:
    # e_i = i (0,1,2,3)
    # f_j = 4*j (0,4,8,12)
    # g_k = 13*k (0,13,26,39) -> Max 3+12+39 = 54 > 40.
    
    # Let's try smaller bases.
    # We need 12 distinct sums from {0..40}.
    # Let's just search for valid (e,f,g) tuples first.
    
    def find_valid_bases():
        """
        Find sets of 4 exponents for A, B, C such that all 12 sums are distinct and <= 40.
        Returns list of ((e0,e1,e2,e3), (f0,f1,f2,f3), (g0,g1,g2,g3))
        """
        bases = []
        # To keep it fast, we fix A to be {0,1,2,3} and search B, C.
        # This reduces symmetry.
        A = (0, 1, 2, 3)
        
        # B must be 4 distinct integers.
        # C must be 4 distinct integers.
        # Sums A_i + B_j + C_k must be distinct.
        
        # Let's iterate over possible B and C from a small range.
        # Since we can shift the whole polynomial, we can fix B_0 = 0.
        # Similarly C_0 = 0.
        
        # B = {0, b1, b2, b3} with 0 < b1 < b2 < b3
        # C = {0, c1, c2, c3} with 0 < c1 < c2 < c3
        
        # Max sum: 3 + b3 + c3 <= 40 => b3 + c3 <= 37.
        
        # Iterate b1, b2, b3
        for b1 in range(1, 10):
            for b2 in range(b1+1, 15):
                for b3 in range(b2+1, 30):
                    B = (0, b1, b2, b3)
                    # Iterate c1, c2, c3
                    # c3 <= 37 - b3
                    max_c3 = 37 - b3
                    if max_c3 < 3: continue
                    for c1 in range(1, 10):
                        for c2 in range(c1+1, 15):
                            for c3 in range(c2+1, max_c3+1):
                                C = (0, c1, c2, c3)
                                
                                # Check distinctness
                                sums = set()
                                valid = True
                                for ai in A:
                                    for bj in B:
                                        for ck in C:
                                            s = ai + bj + ck
                                            if s in sums:
                                                valid = False
                                                break
                                            sums.add(s)
                                        if not valid: break
                                    if not valid: break
                                if valid:
                                    bases.append((A, B, C))
        return bases

    # Pre-compute some valid bases
    # This might take a moment, but we can limit the search or do it lazily.
    # Let's just generate a few specific ones manually or with a quick loop.
    
    # Example: A={0,1,2,3}, B={0,4,8,12}, C={0,13,26,39} -> Max 54 (Invalid)
    # Example: A={0,1,2,3}, B={0,4,8,12}, C={0,10,20,30} -> 
    # Sums:
    # 0+0+0=0, 0+0+10=10, 0+0+20=20, 0+0+30=30
    # 0+4+0=4, 0+4+10=14, 0+4+20=24, 0+4+30=34
    # 0+8+0=8, 0+8+10=18, 0+8+20=28, 0+8+30=38
    # 0+12+0=12, 0+12+10=22, 0+12+20=32, 0+12+30=42 (>40) -> Invalid
    
    # Try A={0,1,2,3}, B={0,3,6,9}, C={0,10,20,30}
    # Max: 3+9+30 = 42 (>40)
    
    # Try A={0,1,2,3}, B={0,2,4,6}, C={0,8,16,24}
    # Max: 3+6+24 = 33.
    # Sums:
    # 0+0+0=0, 0+0+8=8, 0+0+16=16, 0+0+24=24
    # 0+2+0=2, 0+2+8=10, 0+2+16=18, 0+2+24=26
    # 0+4+0=4, 0+4+8=12, 0+4+16=20, 0+4+24=28
    # 0+6+0=6, 0+6+8=14, 0+6+16=22, 0+6+24=30
    # 1+0+0=1, 1+0+8=9, 1+0+16=17, 1+0+24=25
    # 1+2+0=3, 1+2+8=11, 1+2+16=19, 1+2+24=27
    # 1+4+0=5, 1+4+8=13, 1+4+16=21, 1+4+24=29
    # 1+6+0=7, 1+6+8=15, 1+6+16=23, 1+6+24=31
    # 2+0+0=2 (Collision with 0+2+0) -> Invalid
    
    # We need a valid base.
    # Let's use the find_valid_bases function but limit it.
    
    valid_bases = []
    # Quick search for a few valid bases
    A = (0, 1, 2, 3)
    # Try specific structured bases
    for b_step in [4, 5, 6]:
        for c_step in [10, 12, 15]:
            B = tuple([b_step * i for i in range(4)])
            C = tuple([c_step * i for i in range(4)])
            if max(B) + max(C) + max(A) > 40:
                continue
            sums = set()
            valid = True
            for ai in A:
                for bj in B:
                    for ck in C:
                        s = ai + bj + ck
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid: break
                if not valid: break
            if valid:
                valid_bases.append((A, B, C))
                
    # If no structured bases found, try random ones
    if not valid_bases:
        random.seed(42)
        for _ in range(1000):
            B = sorted(random.sample(range(1, 20), 3))
            B = (0,) + tuple(B)
            C = sorted(random.sample(range(1, 30), 3))
            C = (0,) + tuple(C)
            if max(B) + max(C) + 3 > 40:
                continue
            sums = set()
            valid = True
            for ai in A:
                for bj in B:
                    for ck in C:
                        s = ai + bj + ck
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid: break
                if not valid: break
            if valid:
                valid_bases.append((A, B, C))
                if len(valid_bases) >= 5:
                    break

    # Now, for each valid base (A, B, C), we have exponents fixed as {a_i + b_j + c_k}.
    # The coefficients are products a_i * b_j * c_k.
    # We want to choose a_i, b_j, c_k (integers in [-64,64], non-zero)
    # to minimize non-zero terms in P(x)^2.
    
    # P(x) = sum_{i,j,k} a_i b_j c_k x^{a_i+b_j+c_k}
    # Let p_{ijk} = a_i b_j c_k.
    # P(x)^2 = sum_{(i,j,k), (i',j',k')} p_{ijk} p_{i'j'k'} x^{a_i+b_j+c_k + a_{i'}+b_{j'}+c_{k'}}
    
    # This is complex to optimize directly.
    # However, note that P(x) = A(x)B(x)C(x).
    # P(x)^2 = A(x)^2 B(x)^2 C(x)^2.
    # If we can make A(x)^2, B(x)^2, C(x)^2 sparse, their product might be sparse?
    # Not necessarily.
    
    # Alternative: Just do local search on the 12 coefficients of P(x) directly,
    # but constrained to the form p_{ijk} = a_i b_j c_k.
    # This is a non-linear constraint.
    
    # Simpler: Just use the exponents from a valid base, and do local search
    # on the 12 coefficients freely. The structure of exponents might help
    # cancellation patterns.
    
    for A, B, C in valid_bases:
        if time.time() > deadline - 10:
            break
            
        exps = []
        for ai in A:
            for bj in B:
                for ck in C:
                    exps.append(ai + bj + ck)
                    
        # Initialize coeffs with 1
        coeffs = [1] * 12
        
        # Local search on coeffs
        current_score = compute_score(exps, coeffs)
        _update_best(exps, coeffs)
        
        COEFF_VALUES = [-3, -2, -1, 1, 2, 3]
        
        no_improve = 0
        while no_improve < 50 and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                for new_c in COEFF_VALUES:
                    if new_c == coeffs[i]:
                        continue
                    old_c = coeffs[i]
                    coeffs[i] = new_c
                    new_score = compute_score(exps, coeffs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        _update_best(exps, coeffs)
                    else:
                        coeffs[i] = old_c
            if not improved:
                no_improve += 1
            else:
                no_improve = 0

    return best_candidate

# EVOLVE-BLOCK-END
