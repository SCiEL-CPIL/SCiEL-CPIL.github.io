import time
import random
import math
from itertools import combinations, product

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    best_score = 21
    best_candidate = [1] * 12  # fallback

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
        return sum(1 for val in square.values() if val != 0)

    def get_dense(exps, coeffs):
        max_deg = max(exps)
        dense = [0] * (max_deg + 1)
        for e, c in zip(exps, coeffs):
            dense[e] = c
        if dense[-1] != 0:
            return dense
        return None

    def try_update(exps, coeffs):
        nonlocal best_score, best_candidate
        if len(exps) != 12 or len(coeffs) != 12:
            return
        if len(set(exps)) != 12:
            return
        if max(exps) > 40 or min(exps) < 0:
            return
        if any(abs(c) > 64 or c == 0 for c in coeffs):
            return
        score = compute_square_nonzeros(exps, coeffs)
        if score < best_score:
            best_score = score
            cand = get_dense(exps, coeffs)
            if cand:
                best_candidate = cand

    def optimize_signs(exps, coeff_pool):
        """For fixed exponents, optimize coefficients from a pool to minimize square nonzeros."""
        n = len(exps)
        best_local = None
        best_local_score = 999
        
        # If pool is small enough, do exhaustive or near-exhaustive search
        pool_size = len(coeff_pool)
        if pool_size <= 2:
            # Exhaustive: 2^12 = 4096
            for signs in product(coeff_pool, repeat=n):
                score = compute_square_nonzeros(exps, list(signs))
                if score < best_local_score:
                    best_local_score = score
                    best_local = list(signs)
        elif pool_size <= 4:
            # 4^12 is too big; use local search
            # Start with random assignment, then hill climb
            for restart in range(200):
                coeffs = [random.choice(coeff_pool) for _ in range(n)]
                score = compute_square_nonzeros(exps, coeffs)
                improved = True
                while improved:
                    improved = False
                    for i in range(n):
                        old_c = coeffs[i]
                        for new_c in coeff_pool:
                            if new_c == old_c:
                                continue
                            coeffs[i] = new_c
                            new_score = compute_square_nonzeros(exps, coeffs)
                            if new_score < score:
                                score = new_score
                                improved = True
                            else:
                                coeffs[i] = old_c
                if score < best_local_score:
                    best_local_score = score
                    best_local = list(coeffs)
        else:
            # Larger pool: more aggressive local search
            for restart in range(100):
                coeffs = [random.choice(coeff_pool) for _ in range(n)]
                score = compute_square_nonzeros(exps, coeffs)
                improved = True
                while improved:
                    improved = False
                    for i in range(n):
                        old_c = coeffs[i]
                        for new_c in coeff_pool:
                            if new_c == old_c:
                                continue
                            coeffs[i] = new_c
                            new_score = compute_square_nonzeros(exps, coeffs)
                            if new_score < score:
                                score = new_score
                                improved = True
                            else:
                                coeffs[i] = old_c
                if score < best_local_score:
                    best_local_score = score
                    best_local = list(coeffs)
        
        return best_local, best_local_score

    def optimize_exponents_and_signs(exps, coeff_pool):
        """Optimize both exponents (via local moves) and signs jointly."""
        n = len(exps)
        exps = list(exps)
        best_local_score = 999
        best_local_exps = list(exps)
        best_local_coeffs = None
        
        # First optimize signs for the given exponents
        coeffs, score = optimize_signs(exps, coeff_pool)
        if coeffs and score < best_local_score:
            best_local_score = score
            best_local_exps = list(exps)
            best_local_coeffs = list(coeffs)
        
        # Now do local search on exponents
        for iteration in range(2000):
            # Pick a random exponent to move
            i = random.randint(0, n - 1)
            old_exp = exps[i]
            # Try small shifts
            for delta in [-2, -1, 1, 2]:
                new_exp = old_exp + delta
                if new_exp < 0 or new_exp > 40:
                    continue
                # Check uniqueness
                if new_exp in exps and exps.index(new_exp) != i:
                    continue
                exps[i] = new_exp
                # Re-optimize signs (quick)
                c, s = optimize_signs(exps, coeff_pool)
                if c and s < best_local_score:
                    best_local_score = s
                    best_local_exps = list(exps)
                    best_local_coeffs = list(c)
                    exps = list(c and best_local_exps)  # keep current
                else:
                    exps[i] = old_exp
        
        return best_local_exps, best_local_coeffs, best_local_score

    # Generate candidate exponent sets
    candidate_exps = []
    
    # 1. Arithmetic progressions
    for start in range(0, 10):
        for step in range(1, 5):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                candidate_exps.append(sorted(exps))
    
    # 2. Symmetric sets
    for _ in range(200):
        left = sorted(random.sample(range(0, 21), 6))
        right = sorted(40 - x for x in left)
        exps = sorted(left + right)
        if len(set(exps)) == 12:
            candidate_exps.append(exps)
    
    # 3. Random sets
    for _ in range(300):
        exps = sorted(random.sample(range(0, 41), 12))
        candidate_exps.append(exps)
    
    # 4. Factor-based: P = A * B where A has 4 terms, B has 3 terms
    # A = sum a_i x^{p_i}, B = sum b_j x^{q_j}
    # P has up to 12 terms at p_i + q_j
    for _ in range(500):
        # Random 4-term and 3-term exponents
        p_exps = sorted(random.sample(range(0, 25), 4))
        q_exps = sorted(random.sample(range(0, 25), 3))
        p_set = set()
        p_terms = []
        for pi in p_exps:
            for qj in q_exps:
                e = pi + qj
                if e not in p_set and e <= 40:
                    p_set.add(e)
                    p_terms.append(e)
        if len(p_terms) == 12:
            candidate_exps.append(sorted(p_terms))
    
    # 5. Structured: pairs that create sum collisions
    # If e_i + e_j = e_k + e_l, we can get cancellation
    for _ in range(300):
        # Build from pairs
        base = random.sample(range(0, 41), 6)
        exps = []
        for b in base:
            partner = random.choice([b + d for d in [-3, -2, -1, 1, 2, 3] if 0 <= b + d <= 40])
            exps.append(b)
            exps.append(partner)
        exps = sorted(set(exps))
        if len(exps) == 12:
            candidate_exps.append(exps)

    # Deduplicate
    candidate_exps = [list(s) for s in set(tuple(e) for e in candidate_exps)]
    random.shuffle(candidate_exps)

    # Process candidates
    coeff_pools = [
        [-1, 1],
        [-1, 1, -2, 2],
        [-1, 1, -3, 3],
        [-2, -1, 1, 2],
        [-3, -2, -1, 1, 2, 3],
    ]
    
    # Quick pass: just optimize signs for each candidate
    for exps in candidate_exps:
        if time.time() > deadline - 10:
            break
        for pool in coeff_pools[:2]:  # Start with smaller pools
            coeffs, score = optimize_signs(exps, pool)
            if coeffs and score < best_score:
                try_update(exps, coeffs)
    
    # Deeper pass on best candidates
    # Find the top candidates by quick score
    scored_candidates = []
    for exps in candidate_exps[:200]:
        if time.time() > deadline - 15:
            break
        coeffs, score = optimize_signs(exps, [-1, 1])
        scored_candidates.append((score, exps))
    
    scored_candidates.sort()
    
    # Deep optimize the top ones
    for score, exps in scored_candidates[:30]:
        if time.time() > deadline - 5:
            break
        # Try with larger coefficient pools
        for pool in coeff_pools:
            if time.time() > deadline - 3:
                break
            coeffs, s = optimize_signs(exps, pool)
            if coeffs and s < best_score:
                try_update(exps, coeffs)
        
        # Joint exponent+sign optimization
        if time.time() > deadline - 3:
            break
        e2, c2, s2 = optimize_exponents_and_signs(exps, [-1, 1])
        if c2 and s2 < best_score:
            try_update(e2, c2)
    
    # Final local polish on best candidate
    if best_candidate and best_score < 21:
        best_exps = [i for i, c in enumerate(best_candidate) if c != 0]
        best_coeffs = [best_candidate[i] for i in best_exps]
        
        for i in range(12):
            old_c = best_coeffs[i]
            for new_c in [-4, -3, -2, -1, 1, 2, 3, 4]:
                if new_c == 0 or new_c == old_c:
                    continue
                best_coeffs[i] = new_c
                s = compute_square_nonzeros(best_exps, best_coeffs)
                if s < best_score:
                    best_score = s
                else:
                    best_coeffs[i] = old_c
            
            old_e = best_exps[i]
            for delta in [-2, -1, 1, 2]:
                new_e = old_e + delta
                if new_e < 0 or new_e > 40:
                    continue
                if new_e in best_exps and best_exps.index(new_e) != i:
                    continue
                best_exps[i] = new_e
                paired = sorted(zip(best_exps, best_coeffs))
                best_exps = [p[0] for p in paired]
                best_coeffs = [p[1] for p in paired]
                s = compute_square_nonzeros(best_exps, best_coeffs)
                if s < best_score:
                    best_score = s
                else:
                    paired = sorted(zip([e if e != new_e else old_e for e in best_exps], best_coeffs))
                    best_exps = [p[0] for p in paired]
                    best_coeffs = [p[1] for p in paired]
        
        if best_score < 21:
            cand = get_dense(best_exps, best_coeffs)
            if cand:
                best_candidate = cand

    return best_candidate

# EVOLVE-BLOCK-END
