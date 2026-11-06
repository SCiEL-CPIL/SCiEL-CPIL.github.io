import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    best_score = 21
    best_candidate = [1] * 12  # fallback

    def compute_square_nonzeros(exps, coeffs):
        square = defaultdict(int)
        n = len(exps)
        for i in range(n):
            ei = exps[i]
            ci = coeffs[i]
            square[2 * ei] += ci * ci
            for j in range(i + 1, n):
                square[ei + exps[j]] += 2 * ci * coeffs[j]
        return sum(1 for v in square.values() if v != 0)

    def try_candidate(exps, coeffs):
        nonlocal best_score, best_candidate
        if len(exps) != 12 or len(set(exps)) != 12:
            return
        if max(exps) > 40 or min(exps) < 0:
            return
        if any(abs(c) > 64 or c == 0 for c in coeffs):
            return
        score = compute_square_nonzeros(exps, coeffs)
        if score < best_score:
            best_score = score
            dense = [0] * (max(exps) + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if dense[-1] != 0:
                best_candidate = dense
        return score

    def local_search(exps, coeffs, max_iters=5000, deadline_t=None):
        """Local search over exponents and coefficients."""
        exps = list(exps)
        coeffs = list(coeffs)
        current_score = compute_square_nonzeros(exps, coeffs)
        coeff_choices = [-3, -2, -1, 1, 2, 3]
        
        for it in range(max_iters):
            if deadline_t and time.time() > deadline_t:
                break
            improved = False
            # Try coefficient changes
            for i in range(12):
                old = coeffs[i]
                for new_c in coeff_choices:
                    if new_c == old:
                        continue
                    coeffs[i] = new_c
                    s = compute_square_nonzeros(exps, coeffs)
                    if s < current_score:
                        current_score = s
                        improved = True
                    else:
                        coeffs[i] = old
                if improved:
                    break
            if improved:
                continue
            # Try exponent changes
            for i in range(12):
                old_e = exps[i]
                for new_e in range(0, 41):
                    if new_e == old_e:
                        continue
                    if new_e in [exps[j] for j in range(12) if j != i]:
                        continue
                    exps[i] = new_e
                    s = compute_square_nonzeros(exps, coeffs)
                    if s < current_score:
                        current_score = s
                        improved = True
                    else:
                        exps[i] = old_e
                    if improved:
                        break
            if not improved:
                break
        return exps, coeffs, current_score

    def factor_product(factors):
        """Multiply polynomials given as (exps, coeffs) lists."""
        result_exps = [0]
        result_coeffs = [1]
        for fe, fc in factors:
            new_exps = []
            new_coeffs = []
            for re, rc in zip(result_exps, result_coeffs):
                for fe_i, fc_i in zip(fe, fc):
                    new_exps.append(re + fe_i)
                    new_coeffs.append(rc * fc_i)
            # Merge same exponents
            merged = defaultdict(int)
            for e, c in zip(new_exps, new_coeffs):
                merged[e] += c
            result_exps = sorted(merged.keys())
            result_coeffs = [merged[e] for e in result_exps]
        return result_exps, result_coeffs

    def try_factorization_search():
        """Search over factorizations P = A*B*C where each has 4 terms."""
        nonlocal best_score, best_candidate
        random.seed(12345)
        
        # Generate candidate 4-term factors
        def gen_4term_factor():
            # Choose 4 distinct exponents
            base = random.randint(0, 15)
            exps = sorted(random.sample(range(base, base + 30), 4))
            if max(exps) > 40:
                exps = sorted(random.sample(range(0, 41), 4))
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(4)]
            return exps, coeffs

        # Try many random factorizations
        for trial in range(200):
            if time.time() > deadline - 30:
                break
            factors = [gen_4term_factor() for _ in range(3)]
            exps, coeffs = factor_product(factors)
            
            # Check if we have exactly 12 nonzero terms
            # Filter zeros
            nonzero = [(e, c) for e, c in zip(exps, coeffs) if c != 0]
            if len(nonzero) < 12:
                continue
            if len(nonzero) > 12:
                # Trim to 12 by removing smallest coefficients
                nonzero.sort(key=lambda x: abs(x[1]))
                nonzero = nonzero[-12:]
            if len(nonzero) != 12:
                continue
            
            exps12 = [e for e, c in nonzero]
            coeffs12 = [c for e, c in nonzero]
            
            if max(exps12) > 40 or min(exps12) < 0:
                continue
            if any(abs(c) > 64 for c in coeffs12):
                continue
            if len(set(exps12)) != 12:
                continue
            
            # Local search from here
            le, lc, ls = local_search(exps12, coeffs12, max_iters=2000, deadline_t=deadline - 5)
            try_candidate(le, lc)

    def try_arithmetic_structure():
        """Try structured exponent sets with arithmetic progressions."""
        nonlocal best_score, best_candidate
        random.seed(999)
        
        # P with exponents in arithmetic progression with signs
        for start in range(0, 10):
            for step in [1, 2, 3]:
                exps = [start + i * step for i in range(12)]
                if max(exps) > 40:
                    continue
                # Try various sign patterns
                for sign_pattern in range(min(8, 1 << 12)):
                    if time.time() > deadline - 20:
                        break
                    coeffs = []
                    for i in range(12):
                        if (sign_pattern >> i) & 1:
                            coeffs.append(-1)
                        else:
                            coeffs.append(1)
                    le, lc, ls = local_search(exps, coeffs, max_iters=1000, deadline_t=deadline - 5)
                    try_candidate(le, lc)

    def try_symmetric_sets():
        """Try symmetric exponent sets around a center."""
        nonlocal best_score, best_candidate
        random.seed(777)
        
        for center in range(8, 33):
            for _ in range(20):
                if time.time() > deadline - 15:
                    break
                # Pick 6 from left side
                left_range = list(range(max(0, center - 20), center))
                if len(left_range) < 6:
                    continue
                left = sorted(random.sample(left_range, 6))
                right = sorted(center * 2 - x for x in left)
                exps = sorted(left + right)
                if len(set(exps)) != 12 or max(exps) > 40:
                    continue
                coeffs = [random.choice([-1, 1]) for _ in range(12)]
                le, lc, ls = local_search(exps, coeffs, max_iters=1500, deadline_t=deadline - 5)
                try_candidate(le, lc)

    def try_dense_local_search():
        """Run local search from many random starting points."""
        nonlocal best_score, best_candidate
        random.seed(42)
        
        for trial in range(100):
            if time.time() > deadline - 10:
                break
            base = random.randint(0, 5)
            exps = sorted(random.sample(range(base, base + 36), 12))
            if max(exps) > 40:
                continue
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            le, lc, ls = local_search(exps, coeffs, max_iters=3000, deadline_t=deadline - 3)
            try_candidate(le, lc)

    # Run different strategies
    try_arithmetic_structure()
    try_symmetric_sets()
    try_factorization_search()
    try_dense_local_search()

    # Final: if we found something good, do more refinement
    if best_score < 21 and time.time() < deadline - 5:
        if best_candidate:
            best_exps = [i for i, c in enumerate(best_candidate) if c != 0]
            best_coeffs = [best_candidate[i] for i in best_exps]
            le, lc, ls = local_search(best_exps, best_coeffs, max_iters=5000, deadline_t=deadline - 1)
            try_candidate(le, lc)

    return best_candidate

# EVOLVE-BLOCK-END
