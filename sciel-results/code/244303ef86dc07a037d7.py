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

    def generate_sum_structure_candidates(max_restarts=100):
        """
        Generate candidates based on P = A + B where A, B are 6-term polynomials.
        We generate E_A, E_B disjoint, form E_P = E_A U E_B, and run local search.
        """
        nonlocal best_score, best_candidate
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Strategy: Choose E_A and E_B to be structured to encourage cancellation
            # in P^2 = A^2 + B^2 + 2AB.
            
            # Approach 1: Arithmetic progressions with overlapping sum sets
            if restart % 2 == 0:
                # Try to align 2*E_A with E_A + E_B
                # If E_A = {a, a+d, a+2d, a+3d, a+4d, a+5d}
                # 2*E_A = {2a, 2a+d, ..., 2a+10d}
                # If E_B = {b, b+d, b+2d, b+3d, b+4d, b+5d}
                # E_A + E_B = {a+b, a+b+d, ..., a+b+10d}
                # If we choose b = a, then E_A and E_B are not disjoint.
                # We need disjoint E_A, E_B.
                # Let E_A = {0, 1, 2, 3, 4, 5} + offset_A
                # Let E_B = {6, 7, 8, 9, 10, 11} + offset_B
                # This doesn't encourage overlap in squares.
                
                # Better: Interleaved or shifted APs
                base_a = random.randint(0, 5)
                base_b = random.randint(0, 5)
                step_a = random.choice([1, 2, 3])
                step_b = random.choice([1, 2, 3])
                
                e_a = [base_a + i * step_a for i in range(6)]
                e_b = [base_b + i * step_b for i in range(6)]
                
                # Ensure disjoint
                if len(set(e_a).intersection(set(e_b))) > 0:
                    continue
                    
                e_p = sorted(e_a + e_b)
                if max(e_p) > 40:
                    continue
                
                # Initial coefficients: Try alternating signs to promote cancellation
                coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
                pairs = list(zip(e_p, coeffs))
                local_search_on_pairs(pairs, max_iters=500)
            
            # Approach 2: Random disjoint sets
            else:
                all_exps = sorted(random.sample(range(0, 41), 12))
                e_a = all_exps[:6]
                e_b = all_exps[6:]
                coeffs = [random.choice([-1, 1]) for _ in range(12)]
                pairs = list(zip(all_exps, coeffs))
                local_search_on_pairs(pairs, max_iters=500)

    # --- Phase 1: Sum-structure search (A+B model) ---
    generate_sum_structure_candidates(max_restarts=150)

    # --- Phase 2: Additional random restarts with local search ---
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-1, 1]) for _ in range(12)]
        pairs = list(zip(exps, coeffs))
        local_search_on_pairs(pairs, max_iters=1000)

    # If no valid candidate found, return a fallback
    if best_candidate is None:
        best_candidate = [1] * 12

    return best_candidate
