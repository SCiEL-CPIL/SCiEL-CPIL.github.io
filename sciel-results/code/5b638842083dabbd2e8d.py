import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit

    best_score = 21
    best_candidate = [1] * 12  # fallback: (1+x+...+x^11)^2 has 21 nonzero terms

    def compute_square_nonzeros(exps, coeffs):
        """Compute number of nonzero terms in P(x)^2."""
        square = {}
        n = len(exps)
        for i in range(n):
            ei = exps[i]
            ci = coeffs[i]
            key = 2 * ei
            square[key] = square.get(key, 0) + ci * ci
            for j in range(i + 1, n):
                ej = exps[j]
                cj = coeffs[j]
                key = ei + ej
                square[key] = square.get(key, 0) + 2 * ci * cj
        count = 0
        for val in square.values():
            if val != 0:
                count += 1
        return count

    def compute_square_nonzeros_fast(exps, coeffs):
        """Compute number of nonzero terms in P(x)^2 using list-based approach."""
        max_deg = max(exps) * 2
        sq = [0] * (max_deg + 1)
        n = len(exps)
        for i in range(n):
            ei = exps[i]
            ci = coeffs[i]
            sq[2 * ei] += ci * ci
            for j in range(i + 1, n):
                ej = exps[j]
                cj = coeffs[j]
                sq[ei + ej] += 2 * ci * cj
        return sum(1 for v in sq if v != 0)

    def try_candidate(exps, coeffs):
        nonlocal best_score, best_candidate
        if len(exps) != 12 or len(coeffs) != 12:
            return
        if len(set(exps)) != 12:
            return
        if max(exps) > 40 or min(exps) < 0:
            return
        if any(abs(c) > 64 or c == 0 for c in coeffs):
            return
        score = compute_square_nonzeros_fast(exps, coeffs)
        if score < best_score:
            best_score = score
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if dense[-1] != 0:
                best_candidate = dense

    def local_search(exps, coeffs, max_iters=50):
        """Local search: flip one coefficient at a time."""
        best = list(coeffs)
        best_score_local = compute_square_nonzeros_fast(exps, best)
        small_vals = [-3, -2, -1, 1, 2, 3, 4, 5]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            for i in range(12):
                old = best[i]
                for new_val in small_vals:
                    if new_val == old:
                        continue
                    best[i] = new_val
                    s = compute_square_nonzeros_fast(exps, best)
                    if s < best_score_local:
                        best_score_local = s
                        improved = True
                    else:
                        best[i] = old
            if not improved:
                break
        return best, best_score_local

    def two_part_search():
        """Search using P = A + B where A and B are 6-term polynomials."""
        nonlocal best_score, best_candidate
        
        # Generate pairs of 6-term exponent sets
        random.seed(42)
        
        # Strategy: A uses low exponents, B uses high exponents
        # Try to make A^2 and B^2 have few terms, and cross terms don't collide
        for _ in range(5000):
            if time.time() > deadline:
                break
            
            # Generate A exponents (6 terms)
            a_start = random.randint(0, 8)
            a_step = random.choice([1, 2, 3])
            a_exps = sorted([a_start + i * a_step for i in range(6)])
            if max(a_exps) > 25:
                continue
            
            # Generate B exponents (6 terms), offset to avoid collisions
            b_start = random.randint(15, 35)
            b_step = random.choice([1, 2, 3])
            b_exps = sorted([b_start + i * b_step for i in range(6)])
            if max(b_exps) > 40:
                continue
            
            all_exps = sorted(a_exps + b_exps)
            if len(set(all_exps)) != 12:
                continue
            
            # Try various coefficient patterns
            patterns = [
                ([1]*6, [1]*6),
                ([1]*6, [-1]*6),
                ([1,-1,1,-1,1,-1], [1,-1,1,-1,1,-1]),
                ([1,-1,1,-1,1,-1], [-1,1,-1,1,-1,1]),
                ([1,1,1,1,1,1], [-1,-1,-1,-1,-1,-1]),
            ]
            
            for pa, pb in patterns:
                coeffs = pa + pb
                # Map to sorted exponents
                exp_coeff = dict(zip(a_exps, pa))
                for e, c in zip(b_exps, pb):
                    if e in exp_coeff:
                        exp_coeff[e] += c
                    else:
                        exp_coeff[e] = c
                
                if len(exp_coeff) != 12:
                    continue
                if any(abs(c) > 64 or c == 0 for c in exp_coeff.values()):
                    continue
                
                final_exps = sorted(exp_coeff.keys())
                final_coeffs = [exp_coeff[e] for e in final_exps]
                
                score = compute_square_nonzeros_fast(final_exps, final_coeffs)
                if score < best_score:
                    best_score = score
                    max_deg = max(final_exps)
                    dense = [0] * (max_deg + 1)
                    for e, c in zip(final_exps, final_coeffs):
                        dense[e] = c
                    if dense[-1] != 0:
                        best_candidate = dense

    def structured_search():
        """Search over structured exponent sets with optimized coefficients."""
        nonlocal best_score, best_candidate
        
        candidate_exps = []
        
        # 1. Arithmetic progressions
        for start in range(0, 12):
            for step in range(1, 5):
                exps = [start + i * step for i in range(12)]
                if max(exps) <= 40:
                    candidate_exps.append(tuple(exps))
        
        # 2. Symmetric: e_i + e_{11-i} = D
        for D in range(20, 41):
            for d1 in range(1, 5):
                for d2 in range(1, 5):
                    exps_left = [d1 * i for i in range(6)]
                    exps_right = [D - d2 * i for i in range(6)]
                    exps = sorted(exps_left + exps_right)
                    if len(set(exps)) == 12 and max(exps) <= 40 and min(exps) >= 0:
                        candidate_exps.append(tuple(exps))
        
        # 3. Two interleaved APs
        for s1 in range(0, 6):
            for d1 in range(1, 4):
                for s2 in range(5, 20):
                    for d2 in range(1, 4):
                        exps1 = [s1 + i * d1 for i in range(6)]
                        exps2 = [s2 + i * d2 for i in range(6)]
                        exps = sorted(exps1 + exps2)
                        if len(set(exps)) == 12 and max(exps) <= 40 and min(exps) >= 0:
                            candidate_exps.append(tuple(exps))
        
        # 4. Random sets
        random.seed(123)
        for _ in range(1000):
            exps = sorted(random.sample(range(0, 41), 12))
            candidate_exps.append(tuple(exps))
        
        # Deduplicate
        seen = set()
        unique_exps = []
        for exps in candidate_exps:
            if exps not in seen:
                seen.add(exps)
                unique_exps.append(list(exps))
        
        # Sort: prioritize smaller span (more collisions possible)
        unique_exps.sort(key=lambda e: max(e) - min(e))
        
        small_coeffs = [-2, -1, 1, 2]
        signs = [-1, 1]
        
        for exps in unique_exps:
            if time.time() > deadline:
                break
            
            # Try all-ones, all-alternating signs
            try_candidate(exps, [1] * 12)
            try_candidate(exps, [1 if i % 2 == 0 else -1 for i in range(12)])
            
            # Random coefficient search
            n_tries = 30 if (max(exps) - min(exps)) < 20 else 15
            for _ in range(n_tries):
                if time.time() > deadline:
                    break
                if random.random() < 0.4:
                    coeffs = [random.choice(signs) for _ in range(12)]
                else:
                    coeffs = [random.choice(small_coeffs) for _ in range(12)]
                try_candidate(exps, coeffs)
            
            # Local search
            if time.time() < deadline - 3:
                best_local, best_local_score = local_search(exps, [1]*12, max_iters=20)
                if best_local_score < best_score:
                    try_candidate(exps, best_local)

    # Run structured search first (more systematic)
    structured_search()
    
    # Then two-part search if time remains
    if time.time() < deadline - 5:
        two_part_search()
    
    # Final local search on best candidate found
    if time.time() < deadline - 2:
        if len(best_candidate) > 0:
            # Extract exponents and coeffs from best_candidate
            exps = [i for i, c in enumerate(best_candidate) if c != 0]
            coeffs = [c for c in best_candidate if c != 0]
            if len(exps) == 12:
                improved, _ = local_search(exps, coeffs, max_iters=30)
                try_candidate(exps, improved)
    
    return best_candidate

# EVOLVE-BLOCK-END
