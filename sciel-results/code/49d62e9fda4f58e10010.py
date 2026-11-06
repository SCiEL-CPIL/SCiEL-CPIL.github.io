import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

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

    def optimize(pairs, max_iters=5000):
        """
        Local search / Simulated Annealing on a list of 12 (exp, coeff) tuples.
        """
        current_pairs = pairs[:]
        current_score = compute_score(current_pairs)
        update_best(current_pairs, current_score)
        
        temp = 2.0
        cooling = 0.999
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            new_pairs = current_pairs[:]
            move_type = random.random()
            
            if move_type < 0.5:
                # Change coefficient
                idx = random.randint(0, 11)
                old_exp, old_coeff = new_pairs[idx]
                # Try small changes or random
                if random.random() < 0.5:
                    new_coeff = old_coeff + random.choice([-2, -1, 1, 2])
                else:
                    new_coeff = random.randint(-64, 64)
                
                if new_coeff == old_coeff or abs(new_coeff) > 64 or new_coeff == 0:
                    continue
                new_pairs[idx] = (old_exp, new_coeff)
            elif move_type < 0.8:
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
            else:
                # Swap two terms' exponents? Or just random restart of one term
                idx = random.randint(0, 11)
                new_exp = random.randint(0, 40)
                old_exp, old_coeff = new_pairs[idx]
                if new_exp == old_exp:
                    continue
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

    # --- Generate Initial Candidates ---
    
    candidates = []
    
    # 1. Random starts with arithmetic progression structure
    for _ in range(100):
        if time.time() > deadline - 10: break
        start = random.randint(0, 10)
        step = random.randint(1, 3)
        exps = [start + i * step for i in range(12)]
        if max(exps) > 40:
            continue
        coeffs = [random.randint(-64, 64) for _ in range(12)]
        # Ensure no zero coeffs
        for i in range(12):
            if coeffs[i] == 0:
                coeffs[i] = 1
        candidates.append(list(zip(exps, coeffs)))

    # 2. Pure random starts
    for _ in range(50):
        if time.time() > deadline - 10: break
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.randint(-64, 64) for _ in range(12)]
        for i in range(12):
            if coeffs[i] == 0:
                coeffs[i] = 1
        candidates.append(list(zip(exps, coeffs)))

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

    # If no valid candidate found, return a fallback
    if best_candidate is None:
        # Fallback: 1 + x + ... + x^11
        best_candidate = [1] * 12

    return best_candidate
