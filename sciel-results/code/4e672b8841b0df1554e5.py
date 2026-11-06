import time
import random
import numpy as np
from itertools import product as iproduct

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    best_score = float('inf')
    best_candidate = None

    def square_nonzero_count(pairs):
        """pairs: list of (exp, coeff) tuples. Returns (nonzero_count, valid)."""
        if len(pairs) != 12:
            return float('inf'), False
        max_deg = max(e for e, c in pairs)
        if max_deg > 40:
            return float('inf'), False
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in pairs:
            if c == 0:
                return float('inf'), False
            if abs(c) > 64:
                return float('inf'), False
            p[e] = c
        square = np.convolve(p, p)
        return int(np.count_nonzero(square)), True

    def update_best(pairs, score):
        nonlocal best_score, best_candidate
        if score < best_score:
            best_score = score
            max_deg = max(e for e, c in pairs)
            dense = [0] * (max_deg + 1)
            for e, c in pairs:
                dense[e] = int(c)
            if len(dense) <= 41 and dense[-1] != 0:
                best_candidate = dense

    def product_of_three_four_term(exps1, exps2, exps3, coeffs1, coeffs2, coeffs3):
        """Build the 12-term polynomial as product of three 4-term factors."""
        p1 = np.zeros(41, dtype=np.int64)
        p2 = np.zeros(41, dtype=np.int64)
        p3 = np.zeros(41, dtype=np.int64)
        for e, c in zip(exps1, coeffs1):
            p1[e] = c
        for e, c in zip(exps2, coeffs2):
            p2[e] = c
        for e, c in zip(exps3, coeffs3):
            p3[e] = c
        prod = np.convolve(np.convolve(p1, p2), p3)
        pairs = [(i, int(c)) for i, c in enumerate(prod) if c != 0]
        return pairs

    def optimize_product_structure(exps1, exps2, exps3, max_iters=2000):
        """Optimize coefficients for a fixed product-of-three structure."""
        nonlocal best_score, best_candidate
        current_best = float('inf')
        current_best_pairs = None
        
        # Try all sign combinations for the 12 coefficients
        # But that's 2^12 = 4096, which is manageable
        # Actually, let's be smarter: the product structure means
        # coeff of term i = c1[a] * c2[b] * c3[c] for some a,b,c
        # So we can enumerate sign patterns of the 4+4+4 = 12 base coeffs
        
        all_signs = list(iproduct([-1, 1], repeat=12))
        
        for signs in all_signs:
            if time.time() > deadline - 5:
                break
            c1 = list(signs[0:4])
            c2 = list(signs[4:8])
            c3 = list(signs[8:12])
            pairs = product_of_three_four_term(exps1, exps2, exps3, c1, c2, c3)
            if len(pairs) == 12:
                score, valid = square_nonzero_count(pairs)
                if valid and score < current_best:
                    current_best = score
                    current_best_pairs = pairs[:]
        
        if current_best_pairs is not None:
            update_best(current_best_pairs, current_best)
        
        return current_best, current_best_pairs

    def optimize_exponents_for_product(max_restarts=30):
        """Optimize the exponent choices for the three 4-term factors."""
        nonlocal best_score, best_candidate
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Choose 12 distinct exponents, split into three groups of 4
            all_exps = sorted(random.sample(range(0, 41), 12))
            
            # Try different splits
            for split in range(3):
                if time.time() > deadline - 5:
                    break
                if split == 0:
                    exps1 = all_exps[0:4]
                    exps2 = all_exps[4:8]
                    exps3 = all_exps[8:12]
                elif split == 1:
                    exps1 = all_exps[0:4]
                    exps2 = all_exps[4:8]
                    exps3 = all_exps[8:12]
                else:
                    exps1 = all_exps[0:4]
                    exps2 = all_exps[4:8]
                    exps3 = all_exps[8:12]
                
                _, _ = optimize_product_structure(exps1, exps2, exps3, max_iters=100)

    def local_search_on_pairs(pairs, max_iters=3000):
        """Local search on a 12-term polynomial."""
        current_pairs = pairs[:]
        current_score, valid = square_nonzero_count(current_pairs)
        if not valid:
            return
        update_best(current_pairs, current_score)
        
        temp = 1.0
        cooling = 0.9995
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            new_pairs = current_pairs[:]
            move_type = random.random()
            
            if move_type < 0.6:
                idx = random.randint(0, 11)
                old_exp, old_coeff = new_pairs[idx]
                new_coeff = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                if new_coeff == old_coeff:
                    continue
                new_pairs[idx] = (old_exp, new_coeff)
            else:
                idx = random.randint(0, 11)
                old_exp, old_coeff = new_pairs[idx]
                delta = random.choice([-3, -2, -1, 1, 2, 3])
                new_exp = old_exp + delta
                if new_exp < 0 or new_exp > 40:
                    continue
                other_exps = [e for j, (e, c) in enumerate(new_pairs) if j != idx]
                if new_exp in other_exps:
                    continue
                new_pairs[idx] = (new_exp, old_coeff)
                
            new_score, valid = square_nonzero_count(new_pairs)
            if not valid:
                continue
            
            if new_score < current_score:
                current_pairs = new_pairs
                current_score = new_score
                update_best(current_pairs, current_score)
            else:
                delta_e = new_score - current_score
                if random.random() < np.exp(-delta_e / temp):
                    current_pairs = new_pairs
                    current_score = new_score
            
            temp *= cooling

    # --- Phase 1: Product-of-three structure search ---
    optimize_exponents_for_product(max_restarts=50)

    # --- Phase 2: Structured initial candidates + local search ---
    candidates = []
    
    # Arithmetic progressions
    for start in range(0, 10):
        for step in range(1, 5):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                coeffs = [1] * 12
                candidates.append(list(zip(exps, coeffs)))
                coeffs_alt = [1 if i % 2 == 0 else -1 for i in range(12)]
                candidates.append(list(zip(exps, coeffs_alt)))
    
    # Random starts
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-1, 1]) for _ in range(12)]
        candidates.append(list(zip(exps, coeffs)))
    
    # A+B decomposition candidates
    for _ in range(30):
        if time.time() > deadline - 10:
            break
        all_exps = sorted(random.sample(range(0, 41), 12))
        a_exps = all_exps[:6]
        b_exps = all_exps[6:]
        for sign_pattern in range(4):
            a_coeffs = []
            b_coeffs = []
            for i in range(6):
                if sign_pattern == 0:
                    a_coeffs.append(1)
                    b_coeffs.append(1)
                elif sign_pattern == 1:
                    a_coeffs.append(1 if i % 2 == 0 else -1)
                    b_coeffs.append(1 if i % 2 == 0 else -1)
                elif sign_pattern == 2:
                    a_coeffs.append(1)
                    b_coeffs.append(-1)
                else:
                    a_coeffs.append(1 if i < 3 else -1)
                    b_coeffs.append(1 if i < 3 else -1)
            candidates.append(list(zip(a_exps, a_coeffs)) + list(zip(b_exps, b_coeffs)))

    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    # Run local search on each candidate
    for c in unique_candidates:
        if time.time() > deadline:
            break
        local_search_on_pairs(c)

    # --- Phase 3: Additional product-of-three with varied splits ---
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        all_exps = sorted(random.sample(range(0, 41), 12))
        # Interleaved splits
        exps1 = all_exps[0:4]
        exps2 = all_exps[4:8]
        exps3 = all_exps[8:12]
        _, _ = optimize_product_structure(exps1, exps2, exps3, max_iters=100)

    # If no valid candidate found, return a fallback
    if best_candidate is None:
        best_candidate = [1] * 12

    return best_candidate
