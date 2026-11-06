import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    best_score = float('inf')
    best_candidate = None

    def compute_score(pairs):
        """
        pairs: list of 12 (exp, coeff) tuples.
        Returns number of non-zero terms in the square.
        """
        if len(pairs) != 12:
            return float('inf')
        
        max_deg = max(e for e, c in pairs)
        if max_deg > 40:
            return float('inf')
            
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in pairs:
            if c == 0:
                return float('inf') # Must be exactly 12 non-zero terms
            if abs(c) > 64:
                return float('inf')
            p[e] = c
        
        square = np.convolve(p, p)
        return int(np.count_nonzero(square))

    def update_best(pairs, score):
        nonlocal best_score, best_candidate
        if score < best_score:
            best_score = score
            max_deg = max(e for e, c in pairs)
            dense = [0] * (max_deg + 1)
            for e, c in pairs:
                dense[e] = c
            if len(dense) <= 41 and dense[-1] != 0:
                best_candidate = dense

    def optimize(pairs, max_iters=2000):
        """
        Local search / Simulated Annealing on a list of 12 (exp, coeff) tuples.
        """
        current_pairs = pairs[:]
        current_score = compute_score(current_pairs)
        update_best(current_pairs, current_score)
        
        temp = 1.0
        cooling = 0.9995
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            new_pairs = current_pairs[:]
            move_type = random.random()
            
            if move_type < 0.6:
                # Change coefficient
                idx = random.randint(0, 11)
                old_exp, old_coeff = new_pairs[idx]
                new_coeff = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                if new_coeff == old_coeff:
                    continue
                new_pairs[idx] = (old_exp, new_coeff)
            else:
                # Change exponent
                idx = random.randint(0, 11)
                old_exp, old_coeff = new_pairs[idx]
                delta = random.choice([-3, -2, -1, 1, 2, 3])
                new_exp = old_exp + delta
                if new_exp < 0 or new_exp > 40:
                    continue
                # Check collision
                other_exps = [e for j, (e, c) in enumerate(new_pairs) if j != idx]
                if new_exp in other_exps:
                    continue
                new_pairs[idx] = (new_exp, old_coeff)
                
            new_score = compute_score(new_pairs)
            
            if new_score < current_score:
                current_pairs = new_pairs
                current_score = new_score
                update_best(current_pairs, current_score)
            else:
                # Simulated annealing acceptance
                delta_e = new_score - current_score
                if random.random() < np.exp(-delta_e / temp):
                    current_pairs = new_pairs
                    current_score = new_score
            
            temp *= cooling

    def optimize_ab_decomposition(max_iters=3000):
        """
        Optimize P = A + B where A and B each have 6 terms.
        Enforce structural constraints to minimize cross-term collisions.
        """
        nonlocal best_score, best_candidate
        
        best_ab_score = float('inf')
        best_ab_pairs = None
        
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            # Generate A and B with disjoint exponent sets
            all_exps = sorted(random.sample(range(0, 41), 12))
            
            # Split into two groups of 6
            a_exps = all_exps[:6]
            b_exps = all_exps[6:]
            
            # Try different coefficient patterns
            for pattern in range(8):
                if time.time() > deadline - 5:
                    break
                
                a_coeffs = []
                b_coeffs = []
                
                for i in range(6):
                    if pattern % 2 == 0:
                        # All +1
                        a_coeffs.append(1)
                        b_coeffs.append(1)
                    else:
                        # Alternating signs
                        a_coeffs.append(1 if i % 2 == 0 else -1)
                        b_coeffs.append(1 if i % 2 == 0 else -1)
                
                pairs = list(zip(a_exps, a_coeffs)) + list(zip(b_exps, b_coeffs))
                
                # Optimize this A+B structure
                current_pairs = pairs[:]
                current_score = compute_score(current_pairs)
                
                if current_score < best_ab_score:
                    best_ab_score = current_score
                    best_ab_pairs = current_pairs[:]
                
                # Local search on A+B structure
                temp = 0.5
                cooling = 0.999
                
                for _ in range(max_iters // 50):
                    if time.time() > deadline - 5:
                        break
                    
                    new_pairs = current_pairs[:]
                    move_type = random.random()
                    
                    if move_type < 0.5:
                        # Modify A coefficients
                        idx = random.randint(0, 5)
                        old_exp, old_coeff = new_pairs[idx]
                        new_coeff = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                        if new_coeff != old_coeff:
                            new_pairs[idx] = (old_exp, new_coeff)
                    elif move_type < 0.8:
                        # Modify B coefficients
                        idx = random.randint(6, 11)
                        old_exp, old_coeff = new_pairs[idx]
                        new_coeff = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                        if new_coeff != old_coeff:
                            new_pairs[idx] = (old_exp, new_coeff)
                    else:
                        # Move an exponent within A or B
                        if random.random() < 0.5:
                            # Move in A
                            idx = random.randint(0, 5)
                            old_exp, old_coeff = new_pairs[idx]
                            delta = random.choice([-2, -1, 1, 2])
                            new_exp = old_exp + delta
                            if 0 <= new_exp <= 40:
                                other_a_exps = [e for j in range(6) if j != idx]
                                other_b_exps = [e for j in range(6, 12)]
                                if new_exp not in other_a_exps and new_exp not in other_b_exps:
                                    new_pairs[idx] = (new_exp, old_coeff)
                        else:
                            # Move in B
                            idx = random.randint(6, 11)
                            old_exp, old_coeff = new_pairs[idx]
                            delta = random.choice([-2, -1, 1, 2])
                            new_exp = old_exp + delta
                            if 0 <= new_exp <= 40:
                                other_b_exps = [e for j in range(6, 12) if j != idx]
                                other_a_exps = [e for j in range(6)]
                                if new_exp not in other_b_exps and new_exp not in other_a_exps:
                                    new_pairs[idx] = (new_exp, old_coeff)
                    
                    new_score = compute_score(new_pairs)
                    
                    if new_score < current_score:
                        current_pairs = new_pairs
                        current_score = new_score
                        if current_score < best_ab_score:
                            best_ab_score = current_score
                            best_ab_pairs = current_pairs[:]
                    else:
                        delta_e = new_score - current_score
                        if random.random() < np.exp(-delta_e / temp):
                            current_pairs = new_pairs
                            current_score = new_score
                    
                    temp *= cooling
        
        if best_ab_pairs is not None:
            update_best(best_ab_pairs, best_ab_score)

    # --- Generate Initial Candidates ---
    
    candidates = []
    
    # 1. Random starts
    for _ in range(50):
        if time.time() > deadline - 10: break
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-1, 1]) for _ in range(12)]
        candidates.append(list(zip(exps, coeffs)))
        
    # 2. Structured starts: Arithmetic Progressions
    for start in range(0, 10):
        for step in range(1, 5):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                coeffs = [1] * 12
                candidates.append(list(zip(exps, coeffs)))
                # Also try alternating signs
                coeffs_alt = [1 if i % 2 == 0 else -1 for i in range(12)]
                candidates.append(list(zip(exps, coeffs_alt)))
                
    # 3. Product of small polynomials
    # A (3 terms) * B (4 terms)
    for _ in range(50):
        if time.time() > deadline - 10: break
        # Random A
        a_exps = sorted(random.sample(range(0, 20), 3))
        a_coeffs = [random.choice([-1, 1]) for _ in range(3)]
        # Random B
        b_exps = sorted(random.sample(range(0, 20), 4))
        b_coeffs = [random.choice([-1, 1]) for _ in range(4)]
        
        # Compute product
        p_a = np.zeros(41, dtype=np.int64)
        for e, c in zip(a_exps, a_coeffs):
            p_a[e] = c
        p_b = np.zeros(41, dtype=np.int64)
        for e, c in zip(b_exps, b_coeffs):
            p_b[e] = c
            
        p_prod = np.convolve(p_a, p_b)
        # Extract non-zero terms
        prod_pairs = [(i, int(c)) for i, c in enumerate(p_prod) if c != 0]
        
        if len(prod_pairs) == 12:
            candidates.append(prod_pairs)
        elif len(prod_pairs) > 12:
            # Truncate to 12? No, must be exactly 12.
            # Pick 12 random non-zero terms?
            # This might break the "product" structure but gives a valid 12-term poly.
            sampled = random.sample(prod_pairs, 12)
            candidates.append(sampled)
            
    # 4. A+B decomposition candidates with disjoint exponent sets
    for _ in range(30):
        if time.time() > deadline - 10: break
        all_exps = sorted(random.sample(range(0, 41), 12))
        a_exps = all_exps[:6]
        b_exps = all_exps[6:]
        
        # Try different sign patterns
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

    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    # Run optimization on each candidate
    for c in unique_candidates:
        if time.time() > deadline:
            break
        optimize(c)
    
    # Run A+B decomposition optimization
    optimize_ab_decomposition()

    # If no valid candidate found, return a fallback
    if best_candidate is None:
        # Fallback: 1 + x + ... + x^11
        best_candidate = [1] * 12

    return best_candidate
