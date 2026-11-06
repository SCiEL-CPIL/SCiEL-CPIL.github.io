import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    # Fallback: (1 + x + ... + x^11)^2 has 21 non-zero terms
    fallback = [1] * 12
    fallback_score = 21

    best_score = fallback_score
    best_candidate = fallback

    def compute_square_nonzeros(exps, coeffs):
        """
        Compute number of nonzero terms in P(x)^2 where P has terms at exps with coeffs.
        Returns (count, square_array)
        """
        n = len(exps)
        max_deg = max(exps) * 2
        square = np.zeros(max_deg + 1, dtype=np.int64)
        
        for i in range(n):
            ei = exps[i]
            ci = coeffs[i]
            square[2 * ei] += ci * ci
            for j in range(i + 1, n):
                ej = exps[j]
                cj = coeffs[j]
                square[ei + ej] += 2 * ci * cj
        
        return int(np.count_nonzero(square))

    def try_candidate(exps, coeffs):
        """
        Check if candidate is valid and update best if better.
        """
        nonlocal best_score, best_candidate
        
        # Validation
        if len(exps) != 12 or len(coeffs) != 12:
            return
        if len(set(exps)) != 12:
            return
        if max(exps) > 40:
            return
        if any(abs(c) > 64 for c in coeffs):
            return
        if any(c == 0 for c in coeffs):
            return
        
        score = compute_square_nonzeros(exps, coeffs)
        
        if score < best_score:
            best_score = score
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if dense[-1] != 0:
                best_candidate = dense

    def exhaustive_sign_search(exps):
        """
        Exhaustive search over all 2048 sign patterns for a given exponent set.
        """
        if time.time() > deadline:
            return
        
        n = len(exps)
        best_local = float('inf')
        best_local_signs = [1] * n
        
        for mask in range(1 << n):
            signs = []
            for i in range(n):
                signs.append(1 if (mask >> i) & 1 else -1)
            
            score = compute_square_nonzeros(exps, signs)
            if score < best_local:
                best_local = score
                best_local_signs = signs[:]
            
            if time.time() > deadline:
                return
        
        if best_local < best_score:
            try_candidate(exps, best_local_signs)

    def optimize_signs_local(exps, max_iter=500):
        """
        Local search for best sign assignment with multi-flip moves.
        """
        if time.time() > deadline:
            return
        
        n = len(exps)
        best_local_score = float('inf')
        best_local_signs = [1] * n

        for restart in range(12):
            if time.time() > deadline:
                break
            
            if restart == 0:
                signs = [1] * n
            else:
                signs = [1 if random.random() > 0.5 else -1 for _ in range(n)]
            
            current_score = compute_square_nonzeros(exps, signs)
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                
                improved = False
                
                # Single flip moves
                for i in range(n):
                    old = signs[i]
                    signs[i] = -signs[i]
                    new_score = compute_square_nonzeros(exps, signs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        signs[i] = old
                
                # Double flip moves
                if not improved:
                    for i in range(n):
                        for j in range(i + 1, n):
                            old_i, old_j = signs[i], signs[j]
                            signs[i] = -signs[i]
                            signs[j] = -signs[j]
                            new_score = compute_square_nonzeros(exps, signs)
                            if new_score < current_score:
                                current_score = new_score
                                improved = True
                            else:
                                signs[i], signs[j] = old_i, old_j
                            if improved:
                                break
                        if improved:
                            break
                
                if not improved:
                    break
            
            if current_score < best_local_score:
                best_local_score = current_score
                best_local_signs = signs[:]
        
        if best_local_score < best_score:
            try_candidate(exps, best_local_signs)

    def try_three_factor_product():
        """
        Model P = A + B + C where A, B, C are 4-term polynomials with disjoint
        exponent supports. The square P^2 = A^2 + B^2 + C^2 + 2AB + 2AC + 2BC.
        We choose exponent sets to minimize collisions between cross terms.
        """
        if time.time() > deadline:
            return
        
        # Try various structured three-group decompositions
        # Group sizes: 4, 4, 4 with well-separated exponent ranges
        
        # Pattern 1: Three blocks of 4 consecutive exponents
        for gap1 in range(1, 8):
            for gap2 in range(1, 8):
                for start in range(0, 5):
                    a_exps = [start + i for i in range(4)]
                    b_start = start + 4 + gap1
                    b_exps = [b_start + i for i in range(4)]
                    c_start = b_start + 4 + gap2
                    c_exps = [c_start + i for i in range(4)]
                    
                    all_exps = sorted(a_exps + b_exps + c_exps)
                    if len(set(all_exps)) != 12 or max(all_exps) > 40:
                        continue
                    
                    # Exhaustive sign search
                    exhaustive_sign_search(all_exps)
                    if time.time() > deadline:
                        return
        
        # Pattern 2: Three interleaved groups with different strides
        for d1 in [1, 2, 3]:
            for d2 in [1, 2, 3]:
                for d3 in [1, 2, 3]:
                    for off1 in range(0, 4):
                        for off2 in range(0, 4):
                            a_exps = [off1 + i * d1 for i in range(4)]
                            b_exps = [off2 + i * d2 for i in range(4)]
                            c_exps = [off1 + off2 + i * d3 for i in range(4)]
                            
                            all_exps = sorted(a_exps + b_exps + c_exps)
                            if len(set(all_exps)) != 12 or max(all_exps) > 40:
                                continue
                            
                            exhaustive_sign_search(all_exps)
                            if time.time() > deadline:
                                return
        
        # Pattern 3: One low group, one mid group, one high group with gaps
        for a_start in range(0, 6):
            for b_start in range(8, 16):
                for c_start in range(20, 30):
                    a_exps = [a_start + i for i in range(4)]
                    b_exps = [b_start + i for i in range(4)]
                    c_exps = [c_start + i for i in range(4)]
                    
                    all_exps = sorted(a_exps + b_exps + c_exps)
                    if len(set(all_exps)) != 12 or max(all_exps) > 40:
                        continue
                    
                    exhaustive_sign_search(all_exps)
                    if time.time() > deadline:
                        return

    def try_geometric_patterns():
        """
        Try geometric and other structured exponent patterns.
        """
        if time.time() > deadline:
            return
        
        # Geometric sequences
        for base in [2, 3]:
            exps = [base ** i for i in range(12)]
            if max(exps) <= 40:
                exhaustive_sign_search(exps)
        
        # Quadratic: e_i = i^2
        exps = [i * i for i in range(12)]
        if max(exps) <= 40:
            exhaustive_sign_search(exps)
        
        # Mixed patterns
        for start in range(0, 5):
            exps1 = [start + i for i in range(6)]
            exps2 = [start + 10 + i * i for i in range(6)]
            all_exps = sorted(exps1 + exps2)
            if len(set(all_exps)) == 12 and max(all_exps) <= 40:
                exhaustive_sign_search(all_exps)
        
        # Symmetric around center
        for center in range(10, 25):
            for offsets in [[1,2,3,4,5,6], [1,2,4,5,7,8], [1,3,4,6,7,9], [2,3,5,6,8,10]]:
                exps = [center - o for o in offsets] + [center + o for o in offsets]
                if len(set(exps)) == 12 and min(exps) >= 0 and max(exps) <= 40:
                    exhaustive_sign_search(sorted(exps))

    def try_AP_patterns():
        """
        Systematic arithmetic progression search with exhaustive sign optimization.
        """
        if time.time() > deadline:
            return
        
        seen = set()
        for start in range(0, 12):
            for step in range(1, 6):
                exps = [start + i * step for i in range(12)]
                if max(exps) > 40:
                    continue
                key = tuple(exps)
                if key in seen:
                    continue
                seen.add(key)
                
                if time.time() > deadline:
                    return
                
                exhaustive_sign_search(exps)

    def try_random_exponents():
        """
        Random exponent generation with local sign optimization.
        """
        if time.time() > deadline:
            return
        
        random.seed(12345)
        attempts = 0
        while time.time() < deadline and attempts < 200:
            attempts += 1
            
            # Generate 12 distinct random exponents in [0, 40]
            exps = random.sample(range(0, 41), 12)
            exps.sort()
            
            if time.time() > deadline:
                return
            
            # Quick local optimization
            optimize_signs_local(exps, max_iter=200)

    def try_coefficient_optimization():
        """
        For promising exponent sets, optimize both signs and small integer coefficients.
        """
        if time.time() > deadline:
            return
        
        # Re-run on best structures with coefficient variation
        seen = set()
        for start in range(0, 8):
            for step in range(1, 4):
                exps = [start + i * step for i in range(12)]
                if max(exps) > 40:
                    continue
                key = tuple(exps)
                if key in seen:
                    continue
                seen.add(key)
                
                if time.time() > deadline:
                    return
                
                # Try various coefficient patterns
                best_local = float('inf')
                best_local_coeffs = [1] * 12
                
                for _ in range(30):
                    coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
                    score = compute_square_nonzeros(exps, coeffs)
                    if score < best_local:
                        best_local = score
                        best_local_coeffs = coeffs[:]
                
                if best_local < best_score:
                    try_candidate(exps, best_local_coeffs)

    # --- Main search strategy ---
    
    # Phase 1: Three-factor product structure (key structural change)
    try_three_factor_product()
    
    if time.time() > deadline:
        return best_candidate
    
    # Phase 2: Exhaustive sign search on AP patterns
    try_AP_patterns()
    
    if time.time() > deadline:
        return best_candidate
    
    # Phase 3: Geometric and symmetric patterns
    try_geometric_patterns()
    
    if time.time() > deadline:
        return best_candidate
    
    # Phase 4: Random exponent exploration
    try_random_exponents()
    
    if time.time() > deadline:
        return best_candidate
    
    # Phase 5: Coefficient optimization on best structures
    try_coefficient_optimization()
    
    return best_candidate

# EVOLVE-BLOCK-END
