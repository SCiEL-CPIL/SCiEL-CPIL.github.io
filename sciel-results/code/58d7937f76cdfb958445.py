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
                return float('inf')
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

    def compute_score_AB(A_pairs, B_pairs):
        """
        Compute score for P = A + B where A and B have disjoint exponent sets.
        A^2 + 2AB + B^2
        """
        all_pairs = A_pairs + B_pairs
        if len(all_pairs) != 12:
            return float('inf')
        
        # Check disjointness
        A_exps = set(e for e, c in A_pairs)
        B_exps = set(e for e, c in B_pairs)
        if A_exps & B_exps:
            return float('inf')
        
        max_deg = max(e for e, c in all_pairs)
        if max_deg > 40:
            return float('inf')
            
        p_a = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in A_pairs:
            if c == 0:
                return float('inf')
            if abs(c) > 64:
                return float('inf')
            p_a[e] = c
            
        p_b = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in B_pairs:
            if c == 0:
                return float('inf')
            if abs(c) > 64:
                return float('inf')
            p_b[e] = c
            
        square = np.convolve(p_a, p_a) + 2 * np.convolve(p_a, p_b) + np.convolve(p_b, p_b)
        return int(np.count_nonzero(square))

    def optimize_AB(A_pairs, B_pairs, max_iters=3000):
        """
        Optimize P = A + B where A and B are 6-term polynomials with disjoint exponents.
        """
        current_A = A_pairs[:]
        current_B = B_pairs[:]
        current_score = compute_score_AB(current_A, current_B)
        update_best(current_A + current_B, current_score)
        
        temp = 1.0
        cooling = 0.9997
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            new_A = current_A[:]
            new_B = current_B[:]
            move_type = random.random()
            
            if move_type < 0.4:
                # Change coefficient in A
                idx = random.randint(0, 5)
                old_exp, old_coeff = new_A[idx]
                new_coeff = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                if new_coeff == old_coeff:
                    continue
                new_A[idx] = (old_exp, new_coeff)
            elif move_type < 0.8:
                # Change coefficient in B
                idx = random.randint(0, 5)
                old_exp, old_coeff = new_B[idx]
                new_coeff = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                if new_coeff == old_coeff:
                    continue
                new_B[idx] = (old_exp, new_coeff)
            else:
                # Change exponent in A or B
                if random.random() < 0.5:
                    idx = random.randint(0, 5)
                    old_exp, old_coeff = new_A[idx]
                    delta = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                    new_exp = old_exp + delta
                    if new_exp < 0 or new_exp > 40:
                        continue
                    # Check collision with all other exponents
                    other_exps = set(e for j, (e, c) in enumerate(new_A) if j != idx)
                    other_exps.update(e for e, c in new_B)
                    if new_exp in other_exps:
                        continue
                    new_A[idx] = (new_exp, old_coeff)
                else:
                    idx = random.randint(0, 5)
                    old_exp, old_coeff = new_B[idx]
                    delta = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                    new_exp = old_exp + delta
                    if new_exp < 0 or new_exp > 40:
                        continue
                    # Check collision with all other exponents
                    other_exps = set(e for j, (e, c) in enumerate(new_B) if j != idx)
                    other_exps.update(e for e, c in new_A)
                    if new_exp in other_exps:
                        continue
                    new_B[idx] = (new_exp, old_coeff)
                
            new_score = compute_score_AB(new_A, new_B)
            
            if new_score < current_score:
                current_A = new_A
                current_B = new_B
                current_score = new_score
                update_best(current_A + current_B, current_score)
            else:
                # Simulated annealing acceptance
                delta_e = new_score - current_score
                if delta_e == 0:
                    current_A = new_A
                    current_B = new_B
                elif random.random() < np.exp(-delta_e / temp):
                    current_A = new_A
                    current_B = new_B
                    current_score = new_score
            
            temp *= cooling

    # --- Generate Initial A+B Candidates ---
    
    AB_candidates = []
    
    # Strategy 1: Separate A and B into low and high degree ranges
    # A uses exponents [0, 20], B uses exponents [21, 40]
    for _ in range(100):
        if time.time() > deadline - 15:
            break
        A_exps = sorted(random.sample(range(0, 21), 6))
        B_exps = sorted(random.sample(range(21, 41), 6))
        A_coeffs = [random.choice([-1, 1]) for _ in range(6)]
        B_coeffs = [random.choice([-1, 1]) for _ in range(6)]
        A_pairs = list(zip(A_exps, A_coeffs))
        B_pairs = list(zip(B_exps, B_coeffs))
        AB_candidates.append((A_pairs, B_pairs))
    
    # Strategy 2: Interleaved but disjoint - A gets even-ish, B gets odd-ish positions
    for _ in range(100):
        if time.time() > deadline - 15:
            break
        # Sample 12 distinct exponents, split into two groups
        all_exps = sorted(random.sample(range(0, 41), 12))
        # Alternate assignment
        A_exps = [all_exps[i] for i in range(0, 12, 2)]
        B_exps = [all_exps[i] for i in range(1, 12, 2)]
        A_coeffs = [random.choice([-1, 1]) for _ in range(6)]
        B_coeffs = [random.choice([-1, 1]) for _ in range(6)]
        A_pairs = list(zip(A_exps, A_coeffs))
        B_pairs = list(zip(B_exps, B_coeffs))
        AB_candidates.append((A_pairs, B_pairs))
    
    # Strategy 3: Arithmetic progressions for A and B with different steps
    for start_a in range(0, 15, 3):
        for step_a in [2, 3, 4]:
            A_exps = [start_a + i * step_a for i in range(6)]
            if max(A_exps) > 40:
                continue
            for start_b in range(0, 41, 5):
                for step_b in [2, 3, 4, 5]:
                    B_exps = [start_b + i * step_b for i in range(6)]
                    if max(B_exps) > 40:
                        continue
                    # Check disjointness
                    if set(A_exps) & set(B_exps):
                        continue
                    A_coeffs = [1] * 6
                    B_coeffs = [1] * 6
                    A_pairs = list(zip(A_exps, A_coeffs))
                    B_pairs = list(zip(B_exps, B_coeffs))
                    AB_candidates.append((A_pairs, B_pairs))
                    # Also try alternating signs
                    A_coeffs_alt = [1 if i % 2 == 0 else -1 for i in range(6)]
                    B_coeffs_alt = [1 if i % 2 == 0 else -1 for i in range(6)]
                    A_pairs_alt = list(zip(A_exps, A_coeffs_alt))
                    B_pairs_alt = list(zip(B_exps, B_coeffs_alt))
                    AB_candidates.append((A_pairs_alt, B_pairs_alt))
    
    # Strategy 4: A is dense low-degree, B is sparse high-degree
    for _ in range(50):
        if time.time() > deadline - 15:
            break
        A_exps = sorted(random.sample(range(0, 15), 6))
        B_exps = sorted(random.sample(range(15, 41), 6))
        A_coeffs = [random.choice([-1, 1]) for _ in range(6)]
        B_coeffs = [random.choice([-1, 1]) for _ in range(6)]
        A_pairs = list(zip(A_exps, A_coeffs))
        B_pairs = list(zip(B_exps, B_coeffs))
        AB_candidates.append((A_pairs, B_pairs))
    
    # Deduplicate
    seen = set()
    unique_AB = []
    for A, B in AB_candidates:
        key = (tuple(sorted(A)), tuple(sorted(B)))
        if key not in seen:
            seen.add(key)
            unique_AB.append((A, B))

    # Run optimization on each A+B candidate
    for A, B in unique_AB:
        if time.time() > deadline - 5:
            break
        optimize_AB(A, B)

    # If no valid candidate found, return a fallback
    if best_candidate is None:
        # Fallback: 1 + x + ... + x^11
        best_candidate = [1] * 12

    return best_candidate
