"""Combined program: algebraic product structures + simulated annealing refinement."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

    random.seed(42)

    # ---------- helpers ----------
    def square_nonzero_count(sparse):
        """sparse = list of (exp, coeff). Return #nonzero terms in P^2."""
        n = len(sparse)
        if n == 0:
            return 0
        sq = {}
        for i in range(n):
            ei, ci = sparse[i]
            for j in range(i, n):
                ej, cj = sparse[j]
                exp = ei + ej
                if i == j:
                    sq[exp] = sq.get(exp, 0) + ci * ci
                else:
                    sq[exp] = sq.get(exp, 0) + 2 * ci * cj
        return sum(1 for v in sq.values() if v != 0)

    def validate_trim(poly):
        if not poly:
            return None
        while poly and poly[-1] == 0:
            poly = poly[:-1]
        if not poly or len(poly) > 41 or poly[-1] == 0:
            return None
        if sum(1 for c in poly if c != 0) != 12:
            return None
        if any(abs(c) > 64 for c in poly):
            return None
        return poly

    def build_poly(sparse):
        if not sparse:
            return None
        max_e = max(e for e, c in sparse)
        if max_e > 40:
            return None
        poly = [0] * (max_e + 1)
        for e, c in sparse:
            poly[e] = c
        return validate_trim(poly)

    best_poly = None
    best_score = float('inf')

    def try_candidate(sparse):
        nonlocal best_poly, best_score
        if len(sparse) != 12:
            return
        exps = [e for e, c in sparse]
        if len(set(exps)) != 12:
            return
        if any(e < 0 or e > 40 for e in exps):
            return
        if any(abs(c) > 64 for _, c in sparse):
            return
        score = square_nonzero_count(sparse)
        if score < best_score:
            best_score = score
            poly = build_poly(sparse)
            if poly is not None:
                best_poly = poly
        return score

    # ---------- Phase 1: algebraic product structures ----------
    # (1 - x^a)(1 - x^b)(1 - x^c) -> 8 terms with built-in cancellation in square
    # Multiply by (1 - x^d) -> up to 16 terms; overlaps or sign flips can give 12
    # Generalize: product of k binomials (1 + s_i x^{a_i}), s_i in {-1, +1}

    def product_binomials(a_list, s_list, shift=0):
        """Compute product of (1 + s_i x^{a_i}) shifted by `shift`.
        Returns list of (exp, coeff) for nonzero terms."""
        # Start with [(0, 1)]
        terms = [(0, 1)]
        for a, s in zip(a_list, s_list):
            new_terms = {}
            for e, c in terms:
                # keep
                new_terms[e] = new_terms.get(e, 0) + c
                # multiply by s*x^a
                ne = e + a
                new_terms[ne] = new_terms.get(ne, 0) + s * c
            terms = [(e, c) for e, c in sorted(new_terms.items()) if c != 0]
        # shift
        terms = [(e + shift, c) for e, c in terms]
        return terms

    # Try products of 3 and 4 binomials
    # 3 binomials: 8 terms base. Need to get to 12 by adding structure.
    # 4 binomials: up to 16 terms.

    # Phase 1a: 4-binomial products with overlaps to reduce to 12 terms
    for a in range(1, 12):
        for b in range(a + 1, 18):
            for c in range(b + 1, 24):
                if time.time() > deadline - 30:
                    break
                for d in range(c + 1, 30):
                    if a + b + c + d > 40:
                        continue
                    # All 16 sign combinations for 4 binomials
                    for mask in range(16):
                        s_list = [1 if (mask >> i) & 1 else -1 for i in range(4)]
                        a_list = [a, b, c, d]
                        terms = product_binomials(a_list, s_list)
                        if len(terms) == 12:
                            try_candidate(terms)
                        elif len(terms) > 12 and terms[0][0] == 0:
                            # Try removing the constant term (shift up)
                            # Actually just check if any subset of 12 works after shift
                            pass

    # Phase 1b: 3-binomial product * (1 + x^e) to add terms
    for a in range(1, 10):
        for b in range(a + 1, 16):
            for c in range(b + 1, 22):
                if time.time() > deadline - 25:
                    break
                # 3-binomial: 8 terms
                base = product_binomials([a, b, c], [-1, -1, -1])
                # Multiply by (1 + s*x^e) for various e
                for e in range(1, 30):
                    for s in [-1, 1]:
                        # Multiply base by (1 + s x^e)
                        new_terms = {}
                        for ex, cc in base:
                            new_terms[ex] = new_terms.get(ex, 0) + cc
                            ne = ex + e
                            new_terms[ne] = new_terms.get(ne, 0) + s * cc
                        terms = [(ex, cc) for ex, cc in sorted(new_terms.items()) if cc != 0]
                        if len(terms) == 12:
                            try_candidate(terms)

    # Phase 1c: AP with ±1 (fast exhaustive, kept from current)
    for d in range(1, 4):
        if time.time() > deadline - 20:
            break
        max_a = 40 - 11 * d
        for a in range(0, max_a + 1):
            exponents = [a + i * d for i in range(12)]
            if exponents[-1] > 40:
                continue
            for signs_mask in range(1 << 12):
                if time.time() > deadline - 15:
                    break
                coeffs = [(exponents[i], 1 if (signs_mask >> i) & 1 else -1) for i in range(12)]
                try_candidate(coeffs)

    # Phase 1d: Palindromic structures (kept from current)
    for d_shift in range(0, 15):
        if time.time() > deadline - 10:
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

    # Phase 1e: Random sparse polynomials
    coeff_choices = [-2, -1, 1, 2]
    for _ in range(300):
        if time.time() > deadline - 8:
            break
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice(coeff_choices) for _ in range(12)]
        sparse = [(exps[i], coeffs[i]) for i in range(12)]
        try_candidate(sparse)

    # ---------- Phase 2: Simulated annealing refinement ----------
    if best_poly is None:
        # Fallback: use a simple valid polynomial
        best_poly = [1] * 12
        best_score = square_nonzero_count([(i, 1) for i in range(12)])

    # Extract sparse representation
    sparse_best = [(i, c) for i, c in enumerate(best_poly) if c != 0]

    T = 2.0
    T_min = 0.01
    cooling = 0.99995
    no_improve = 0
    max_no_improve = 800

    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart from a random candidate
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice(coeff_choices) for _ in range(12)]
            sparse_best = [(exps[i], coeffs[i]) for i in range(12)]
            current_score = square_nonzero_count(sparse_best)
            if current_score < best_score:
                best_score = current_score
                poly = build_poly(sparse_best)
                if poly is not None:
                    best_poly = poly
            no_improve = 0
            T = 2.0
            continue

        # Generate a neighbor
        mutated = list(sparse_best)
        mutation_type = random.random()

        if mutation_type < 0.40:
            # Change a coefficient
            idx = random.randint(0, len(mutated) - 1)
            e, c = mutated[idx]
            new_c = random.choice([-2, -1, 1, 2])
            if new_c != c:
                mutated[idx] = (e, new_c)
        elif mutation_type < 0.70:
            # Move an exponent
            idx = random.randint(0, len(mutated) - 1)
            e, c = mutated[idx]
            new_e = e + random.randint(-4, 4)
            if 0 <= new_e <= 40:
                if not any(ne == new_e for k, (ne, nc) in enumerate(mutated) if k != idx):
                    mutated[idx] = (new_e, c)
        elif mutation_type < 0.85:
            # Swap two exponents
            if len(mutated) >= 2:
                i, j = random.sample(range(len(mutated)), 2)
                ei, ci = mutated[i]
                ej, cj = mutated[j]
                if ei != ej:
                    mutated[i] = (ej, ci)
                    mutated[j] = (ei, cj)
        else:
            # Perturb two exponents simultaneously
            if len(mutated) >= 2:
                i, j = random.sample(range(len(mutated)), 2)
                ei, ci = mutated[i]
                ej, cj = mutated[j]
                ne_i = ei + random.randint(-3, 3)
                ne_j = ej + random.randint(-3, 3)
                if 0 <= ne_i <= 40 and 0 <= ne_j <= 40:
                    if ne_i != ne_j:
                        # Check no collision with others
                        other_exps = set(mutated[k][0] for k in range(len(mutated)) if k not in (i, j))
                        if ne_i not in other_exps and ne_j not in other_exps:
                            mutated[i] = (ne_i, ci)
                            mutated[j] = (ne_j, cj)

        # Validate
        if len(mutated) != 12:
            no_improve += 1
            T *= cooling
            continue
        exps_list = [e for e, c in mutated]
        if len(set(exps_list)) != 12:
            no_improve += 1
            T *= cooling
            continue
        if any(e < 0 or e > 40 for e in exps_list):
            no_improve += 1
            T *= cooling
            continue
        if any(abs(c) > 64 for e, c in mutated):
            no_improve += 1
            T *= cooling
            continue

        new_score = square_nonzero_count(mutated)
        delta = new_score - best_score

        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0:
            sparse_best = mutated
            best_score = new_score
            poly = build_poly(mutated)
            if poly is not None:
                best_poly = poly
            no_improve = 0
        else:
            # Metropolis acceptance
            if T > T_min:
                import math
                prob = math.exp(-delta / T) if delta < 100 else 0.0
                if random.random() < prob:
                    sparse_best = mutated
                    no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1

        T *= cooling

    # Final validation
    if best_poly is None:
        return [1] * 12

    return best_poly
# EVOLVE-BLOCK-END
