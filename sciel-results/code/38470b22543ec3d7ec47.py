"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    best_poly = None
    best_score = float('inf')
    
    # Helper: compute sparse square coefficient count
    def square_nonzero_count(coeffs):
        """Given list of (exponent, coefficient) pairs, return count of nonzero terms in square."""
        n = len(coeffs)
        if n == 0:
            return 0
        # Build dict of exponent -> coefficient for the square
        sq = {}
        for i in range(n):
            ei, ci = coeffs[i]
            for j in range(i, n):
                ej, cj = coeffs[j]
                exp = ei + ej
                if i == j:
                    sq[exp] = sq.get(exp, 0) + ci * ci
                else:
                    sq[exp] = sq.get(exp, 0) + 2 * ci * cj
        # Count nonzero
        return sum(1 for v in sq.values() if v != 0)
    
    # Helper: validate and trim
    def validate_and_trim(poly):
        """Ensure poly meets constraints, return trimmed list or None."""
        if not poly:
            return None
        # Trim trailing zeros
        while poly and poly[-1] == 0:
            poly = poly[:-1]
        if not poly:
            return None
        if len(poly) > 41:
            return None
        if poly[-1] == 0:
            return None
        nonzero_count = sum(1 for c in poly if c != 0)
        if nonzero_count != 12:
            return None
        if any(abs(c) > 64 for c in poly):
            return None
        return poly
    
    # Helper: build polynomial from sparse representation
    def build_poly(sparse_coeffs):
        """Build dense polynomial list from (exp, coeff) pairs."""
        if not sparse_coeffs:
            return None
        max_exp = max(e for e, c in sparse_coeffs)
        poly = [0] * (max_exp + 1)
        for exp, c in sparse_coeffs:
            poly[exp] = c
        return validate_and_trim(poly)
    
    # Helper: try a candidate and update best
    def try_candidate(sparse_coeffs):
        nonlocal best_poly, best_score
        score = square_nonzero_count(sparse_coeffs)
        if score < best_score:
            best_score = score
            poly = build_poly(sparse_coeffs)
            if poly is not None:
                best_poly = poly
        return score
    
    # Helper: check if sparse coeffs form a valid 12-term poly
    def is_valid_sparse(sparse_coeffs):
        if len(sparse_coeffs) != 12:
            return False
        if any(e < 0 or e > 40 for e, c in sparse_coeffs):
            return False
        if any(abs(c) > 64 for e, c in sparse_coeffs):
            return False
        exps = [e for e, c in sparse_coeffs]
        if len(set(exps)) != 12:
            return False
        return True
    
    # Helper: combine basis polynomials with integer weights
    def combine_basis(basis_polys, weights):
        """Combine basis polynomials with given integer weights.
        Each basis_poly is a dict {exp: coeff}. Returns combined sparse representation."""
        combined = {}
        for bp, w in zip(basis_polys, weights):
            if w == 0:
                continue
            for exp, coeff in bp.items():
                combined[exp] = combined.get(exp, 0) + w * coeff
        # Filter zeros
        sparse = [(e, c) for e, c in sorted(combined.items()) if c != 0]
        return sparse
    
    # Phase 1: Basis superposition search
    # Model P(x) = w1*B1(x) + w2*B2(x) + w3*B3(x) where Bi are low-sparsity polynomials
    # This exploits the superposition principle for systematic cancellation search
    
    random.seed(42)
    
    # Generate diverse basis polynomial families
    # Each basis is a dict {exp: coeff}
    
    # Family 1: Trinomials with various structures
    trinomial_bases = []
    for d in range(1, 5):
        for start in range(0, 35):
            e0 = start
            e1 = start + d
            e2 = start + 2 * d
            if e2 > 40:
                continue
            for c0 in [-2, -1, 1, 2]:
                for c1 in [-2, -1, 1, 2]:
                    for c2 in [-2, -1, 1, 2]:
                        trinomial_bases.append({e0: c0, e1: c1, e2: c2})
    
    # Family 2: Binomials (for more flexible combinations)
    binomial_bases = []
    for d in range(1, 8):
        for start in range(0, 35):
            e0 = start
            e1 = start + d
            if e1 > 40:
                continue
            for c0 in [-2, -1, 1, 2]:
                for c1 in [-2, -1, 1, 2]:
                    binomial_bases.append({e0: c0, e1: c1})
    
    # Family 3: Clustered trinomials (close exponents for more overlap potential)
    clustered_bases = []
    for start in range(0, 38):
        for gap1 in range(1, 4):
            for gap2 in range(1, 4):
                e0 = start
                e1 = start + gap1
                e2 = start + gap1 + gap2
                if e2 > 40:
                    continue
                for c0 in [-2, -1, 1, 2]:
                    for c1 in [-2, -1, 1, 2]:
                        for c2 in [-2, -1, 1, 2]:
                            clustered_bases.append({e0: c0, e1: c1, e2: c2})
    
    # All basis candidates
    all_bases = trinomial_bases + binomial_bases + clustered_bases
    
    # Phase 1a: Two-basis combination search
    # P = w1*B1 + w2*B2, search over pairs and small integer weights
    weight_range_2 = [-2, -1, 1, 2]
    
    # Sample basis pairs to keep computation manageable
    num_basis_pairs_to_try = 50000
    random.shuffle(all_bases)
    
    tried_pairs = set()
    pairs_tried = 0
    
    for i in range(min(len(all_bases), 2000)):
        if time.time() > deadline:
            break
        B1 = all_bases[i]
        for j in range(i + 1, min(len(all_bases), i + 50)):
            if pairs_tried >= num_basis_pairs_to_try:
                break
            B2 = all_bases[j]
            
            # Skip if bases are too similar (same exponent set)
            key = (tuple(sorted(B1.keys())), tuple(sorted(B2.keys())))
            if key in tried_pairs:
                continue
            tried_pairs.add(key)
            
            for w1 in weight_range_2:
                for w2 in weight_range_2:
                    if pairs_tried >= num_basis_pairs_to_try:
                        break
                    sparse = combine_basis([B1, B2], [w1, w2])
                    if is_valid_sparse(sparse):
                        score = square_nonzero_count(sparse)
                        if score < best_score:
                            best_score = score
                            poly = build_poly(sparse)
                            if poly is not None:
                                best_poly = poly
                    pairs_tried += 1
    
    # Phase 1b: Three-basis combination search
    # P = w1*B1 + w2*B2 + w3*B3
    weight_range_3 = [-1, 1]
    
    # Use clustered bases for three-basis (more likely to have overlapping exponents)
    random.shuffle(clustered_bases)
    random.shuffle(trinomial_bases)
    
    three_basis_tried = set()
    three_basis_count = 0
    max_three_basis = 30000
    
    for i in range(min(len(clustered_bases), 500)):
        if time.time() > deadline:
            break
        B1 = clustered_bases[i]
        for j in range(i + 1, min(len(clustered_bases), i + 30)):
            if three_basis_count >= max_three_basis:
                break
            B2 = clustered_bases[j]
            for k in range(j + 1, min(len(clustered_bases), j + 20)):
                if three_basis_count >= max_three_basis:
                    break
                B3 = clustered_bases[k]
                
                key = (tuple(sorted(B1.keys())), tuple(sorted(B2.keys())), tuple(sorted(B3.keys())))
                if key in three_basis_tried:
                    continue
                three_basis_tried.add(key)
                
                for w1 in weight_range_3:
                    for w2 in weight_range_3:
                        for w3 in weight_range_3:
                            if three_basis_count >= max_three_basis:
                                break
                            sparse = combine_basis([B1, B2, B3], [w1, w2, w3])
                            if is_valid_sparse(sparse):
                                score = square_nonzero_count(sparse)
                                if score < best_score:
                                    best_score = score
                                    poly = build_poly(sparse)
                                    if poly is not None:
                                        best_poly = poly
                            three_basis_count += 1
    
    # Phase 2: AP with ±1 coefficients (fast exhaustive)
    for d in range(1, 4):
        max_a = 40 - 11 * d
        for a in range(0, max_a + 1):
            exponents = [a + i * d for i in range(12)]
            if exponents[-1] > 40:
                continue
            for signs_mask in range(1 << 12):
                if time.time() > deadline:
                    if best_poly is not None:
                        return best_poly
                    return [1]*12
                coeffs = [(exponents[i], 1 if (signs_mask >> i) & 1 else -1) for i in range(12)]
                try_candidate(coeffs)
    
    # Phase 3: Enhanced local search / hill climbing from best found
    if best_poly is not None:
        # Extract sparse representation from best_poly
        sparse_best = [(i, c) for i, c in enumerate(best_poly) if c != 0]
        
        # More aggressive local search with diverse mutations
        for iteration in range(15000):
            if time.time() > deadline:
                break
            
            # Create a mutated version
            mutated = list(sparse_best)
            mutation_type = random.randint(0, 5)
            
            if mutation_type == 0 and len(mutated) > 2:
                # Change a coefficient
                idx = random.randint(0, len(mutated) - 1)
                e, c = mutated[idx]
                new_c = random.choice([-3, -2, -1, 1, 2, 3])
                if new_c != c:
                    mutated[idx] = (e, new_c)
            elif mutation_type == 1:
                # Move an exponent
                idx = random.randint(0, len(mutated) - 1)
                e, c = mutated[idx]
                new_e = e + random.randint(-4, 4)
                if 0 <= new_e <= 40:
                    # Check no collision
                    if not any(ne == new_e for ne, nc in mutated if (ne, nc) != mutated[idx]):
                        mutated[idx] = (new_e, c)
            elif mutation_type == 2:
                # Swap two exponents
                if len(mutated) >= 2:
                    i, j = random.sample(range(len(mutated)), 2)
                    ei, ci = mutated[i]
                    ej, cj = mutated[j]
                    if ei != ej:
                        mutated[i] = (ej, ci)
                        mutated[j] = (ei, cj)
            elif mutation_type == 3:
                # Move two exponents (correlated mutation)
                if len(mutated) >= 2:
                    i, j = random.sample(range(len(mutated)), 2)
                    ei, ci = mutated[i]
                    ej, cj = mutated[j]
                    delta = random.randint(-3, 3)
                    new_ei = ei + delta
                    new_ej = ej + delta
                    if 0 <= new_ei <= 40 and 0 <= new_ej <= 40:
                        if not any(ne == new_ei for ne, nc in mutated if (ne, nc) != mutated[i]):
                            if not any(ne == new_ej for ne, nc in mutated if (ne, nc) != mutated[j]):
                                mutated[i] = (new_ei, ci)
                                mutated[j] = (new_ej, cj)
            elif mutation_type == 4:
                # Scale all coefficients by -1 (sign flip)
                mutated = [(e, -c) for e, c in mutated]
            else:
                # Random restart: new random sparse poly with structure
                if random.random() < 0.3:
                    # AP-based random
                    base = random.randint(0, 10)
                    step = random.randint(1, 3)
                    exps = [base + i * step for i in range(12)]
                    if exps[-1] <= 40:
                        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                        mutated = [(exps[i], coeffs[i]) for i in range(12)]
                    else:
                        exps = sorted(random.sample(range(0, 41), 12))
                        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                        mutated = [(exps[i], coeffs[i]) for i in range(12)]
                else:
                    exps = sorted(random.sample(range(0, 41), 12))
                    coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                    mutated = [(exps[i], coeffs[i]) for i in range(12)]
            
            # Validate mutated
            if len(mutated) != 12:
                continue
            if any(e < 0 or e > 40 for e, c in mutated):
                continue
            if any(abs(c) > 64 for e, c in mutated):
                continue
            # Check all exponents distinct
            exps_list = [e for e, c in mutated]
            if len(set(exps_list)) != 12:
                continue
            
            score = square_nonzero_count(mutated)
            if score < best_score:
                best_score = score
                poly = build_poly(mutated)
                if poly is not None:
                    best_poly = poly
                    sparse_best = mutated
    
    # Phase 4: Try palindromic structures
    for d_shift in range(0, 15):
        if time.time() > deadline:
            break
        base_exps = list(range(12))
        shifted_exps = [e + d_shift for e in base_exps]
        if shifted_exps[-1] > 40:
            continue
        for c0 in [-2, -1, 1, 2]:
            for c1 in [-2, -1, 1, 2]:
                for c2 in [-2, -1, 1, 2]:
                    for c3 in [-2, -1, 1, 2]:
                        for c4 in [-2, -1, 1, 2]:
                            for c5 in [-2, -1, 1, 2]:
                                coeffs_list = [c0, c1, c2, c3, c4, c5, c5, c4, c3, c2, c1, c0]
                                if sum(1 for c in coeffs_list if c != 0) != 12:
                                    continue
                                coeffs = [(shifted_exps[i], coeffs_list[i]) for i in range(12)]
                                try_candidate(coeffs)
    
    # Phase 5: Targeted search for low-square-count structures
    if time.time() < deadline:
        # Try geometric-like progressions with small ratio
        for ratio in [2, 3]:
            for start in range(0, 5):
                exps = []
                e = start
                while len(exps) < 12 and e <= 40:
                    exps.append(e)
                    e *= ratio
                if len(exps) < 12:
                    continue
                exps = exps[:12]
                for signs_mask in range(1 << 12):
                    if time.time() > deadline:
                        break
                    coeffs = [(exps[i], 1 if (signs_mask >> i) & 1 else -1) for i in range(12)]
                    try_candidate(coeffs)
        
        # Try clustered structures
        for cluster_start in range(0, 30):
            if time.time() > deadline:
                break
            for c1 in range(0, 4):
                for c2 in range(0, 4):
                    for c3 in range(0, 4):
                        exps = []
                        base = cluster_start
                        for c in [c1, c2, c3, c3, c2, c1]:
                            for k in range(3):
                                exps.append(base + c + k)
                        if len(exps) != 12 or exps[-1] > 40:
                            continue
                        for pattern in [[1,1,1,-1,-1,-1,1,1,1,-1,-1,-1],
                                        [1,-1,1,1,-1,1,-1,1,-1,1,-1,1],
                                        [1,1,-1,1,1,-1,-1,-1,1,-1,-1,1]]:
                            coeffs = [(exps[i], pattern[i]) for i in range(12)]
                            try_candidate(coeffs)
    
    # Return best found, or a valid fallback
    if best_poly is not None:
        return best_poly
    
    return [1]*12

# EVOLVE-BLOCK-END
