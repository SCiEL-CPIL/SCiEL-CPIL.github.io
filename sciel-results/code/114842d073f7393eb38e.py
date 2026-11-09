"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

    best_poly = None
    best_score = float('inf')

    # ---- fast square scorer with early pruning -----------------------------
    def square_nonzero_count(sparse):
        """sparse: list of (exp, coeff). Return count of nonzero terms in P^2."""
        n = len(sparse)
        if n == 0:
            return 0
        pre = [(sparse[i][0] + sparse[j][0],
                sparse[i][1] * sparse[j][1]) for i in range(n) for j in range(i, n)]
        sq = {}
        for e, p in pre:
            sq[e] = sq.get(e, 0) + p
        cnt = 0
        for v in sq.values():
            if v != 0:
                cnt += 1
                if cnt >= best_score:
                    break
        return cnt

    # ---- validation --------------------------------------------------------
    def validate_and_trim(poly):
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
        max_exp = max(e for e, c in sparse)
        if max_exp > 40:
            return None
        poly = [0] * (max_exp + 1)
        for exp, c in sparse:
            poly[exp] = c
        return validate_and_trim(poly)

    def try_candidate(sparse):
        nonlocal best_poly, best_score
        if len(sparse) != 12:
            return float('inf')
        exps = [e for e, c in sparse]
        if len(set(exps)) != 12:
            return float('inf')
        if any(e < 0 or e > 40 for e in exps):
            return float('inf')
        if any(abs(c) > 64 for e, c in sparse):
            return float('inf')
        score = square_nonzero_count(sparse)
        if score < best_score:
            best_score = score
            poly = build_poly(sparse)
            if poly is not None:
                best_poly = poly
        return score

    # ---- Phase 1: arithmetic progressions with ±1 (fast exhaustive) --------
    for d in range(1, 4):
        max_a = 40 - 11 * d
        for a in range(0, max_a + 1):
            exponents = [a + i * d for i in range(12)]
            for mask in range(1 << 12):
                if time.time() > deadline:
                    if best_poly is not None:
                        return best_poly
                    return [1] * 12
                coeffs = [(exponents[i], 1 if (mask >> i) & 1 else -1) for i in range(12)]
                try_candidate(coeffs)

    # ---- Phase 2: symmetry-constrained families ----------------------------
    random.seed(42)
    coeff_choices = [-2, -1, 1, 2]

    # Anti-palindromic: c[i] = -c[11-i]  (square has strong cancellation)
    for d_shift in range(0, 15):
        if time.time() > deadline:
            break
        shifted = [e + d_shift for e in range(12)]
        if shifted[-1] > 40:
            continue
        for c0 in coeff_choices:
            for c1 in coeff_choices:
                for c2 in coeff_choices:
                    for c3 in coeff_choices:
                        for c4 in coeff_choices:
                            for c5 in coeff_choices:
                                cl = [c0, c1, c2, c3, c4, c5, -c5, -c4, -c3, -c2, -c1, -c0]
                                if sum(1 for c in cl if c != 0) != 12:
                                    continue
                                sparse = [(shifted[i], cl[i]) for i in range(12)]
                                try_candidate(sparse)

    # Palindromic: c[i] = c[11-i]
    for d_shift in range(0, 15):
        if time.time() > deadline:
            break
        shifted = [e + d_shift for e in range(12)]
        if shifted[-1] > 40:
            continue
        for c0 in coeff_choices:
            for c1 in coeff_choices:
                for c2 in coeff_choices:
                    for c3 in coeff_choices:
                        for c4 in coeff_choices:
                            for c5 in coeff_choices:
                                cl = [c0, c1, c2, c3, c4, c5, c5, c4, c3, c2, c1, c0]
                                if sum(1 for c in cl if c != 0) != 12:
                                    continue
                                sparse = [(shifted[i], cl[i]) for i in range(12)]
                                try_candidate(sparse)

    # ---- Phase 3: product construction P = A*B (sparse factors) ------------
    for trial in range(400):
        if time.time() > deadline:
            break
        na = random.randint(2, 6)
        nb = random.randint(2, 6)
        if na * nb < 12:
            continue
        exps_a = sorted(random.sample(range(0, 21), na))
        exps_b = sorted(random.sample(range(0, 21), nb))
        ca = [random.choice(coeff_choices) for _ in range(na)]
        cb = [random.choice(coeff_choices) for _ in range(nb)]
        prod = {}
        for i, ea in enumerate(exps_a):
            for j, eb in enumerate(exps_b):
                e = ea + eb
                prod[e] = prod.get(e, 0) + ca[i] * cb[j]
        sparse = [(e, c) for e, c in sorted(prod.items()) if c != 0]
        if len(sparse) != 12 or sparse[-1][0] > 40:
            continue
        try_candidate(sparse)

    # ---- Phase 4: simulated annealing with smooth surrogate ----------------
    # smooth score: soft count of nonzero terms via log-sum-exp (nonsmooth->smooth)
    def smooth_score(sparse):
        pre = [(sparse[i][0] + sparse[j][0],
                sparse[i][1] * sparse[j][1]) for i in range(12) for j in range(i, 12)]
        sq = {}
        for e, p in pre:
            sq[e] = sq.get(e, 0) + p
        vals = [abs(v) for v in sq.values()]
        if not vals:
            return 0.0
        m = max(vals)
        if m == 0:
            return 0.0
        s = sum(math.exp((v - m) / 4.0) for v in vals)
        return m / 4.0 + math.log(s)

    def valid_sparse(sparse):
        exps = [e for e, c in sparse]
        return (len(sparse) == 12 and len(set(exps)) == 12
                and all(0 <= e <= 40 for e in exps)
                and all(abs(c) <= 64 for e, c in sparse))

    if best_poly is not None:
        cur = [(i, c) for i, c in enumerate(best_poly) if c != 0]
    else:
        exps = sorted(random.sample(range(0, 41), 12))
        cur = [(exps[i], random.choice(coeff_choices)) for i in range(12)]

    cur_score = square_nonzero_count(cur)
    if cur_score < best_score:
        best_score = cur_score
        best_poly = build_poly(cur)

    T = 3.0
    T_min = 0.05
    cooling = 0.995
    for it in range(20000):
        if time.time() > deadline:
            break
        mutated = list(cur)
        mt = random.random()
        if mt < 0.45:
            idx = random.randrange(12)
            e, c = mutated[idx]
            new_c = random.choice(coeff_choices)
            if new_c != c:
                mutated[idx] = (e, new_c)
        elif mt < 0.75:
            idx = random.randrange(12)
            e, c = mutated[idx]
            new_e = e + random.randint(-3, 3)
            if 0 <= new_e <= 40 and not any(ne == new_e for k, (ne, nc) in enumerate(mutated) if k != idx):
                mutated[idx] = (new_e, c)
        elif mt < 0.85:
            i, j = random.sample(range(12), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        else:
            exps = sorted(random.sample(range(0, 41), 12))
            mutated = [(exps[k], random.choice(coeff_choices)) for k in range(12)]

        if not valid_sparse(mutated):
            continue
        s_new = smooth_score(mutated)
        s_cur = smooth_score(cur)
        if s_new <= s_cur or random.random() < math.exp(-(s_new - s_cur) / max(T, 1e-6)):
            cur = mutated
            exact = square_nonzero_count(cur)
            if exact < best_score:
                best_score = exact
                best_poly = build_poly(cur)
        T = max(T * cooling, T_min)

    # ---- Phase 5: final exact local polish pass ----------------------------
    if best_poly is not None:
        cur = [(i, c) for i, c in enumerate(best_poly) if c != 0]
    polish_steps = 0
    while polish_steps < 4000 and time.time() < deadline:
        polish_steps += 1
        improved = False
        for idx in range(12):
            if time.time() > deadline:
                break
            e, c = cur[idx]
            for new_c in coeff_choices:
                if new_c == c:
                    continue
                cand = list(cur)
                cand[idx] = (e, new_c)
                if not valid_sparse(cand):
                    continue
                sc = square_nonzero_count(cand)
                if sc < best_score:
                    best_score = sc
                    cur = cand
                    best_poly = build_poly(cur)
                    improved = True
            for new_e in range(max(0, e - 3), min(40, e + 3) + 1):
                if new_e == e:
                    continue
                cand = list(cur)
                cand[idx] = (new_e, c)
                if not valid_sparse(cand):
                    continue
                sc = square_nonzero_count(cand)
                if sc < best_score:
                    best_score = sc
                    cur = cand
                    best_poly = build_poly(cur)
                    improved = True
        if not improved:
            break

    if best_poly is not None:
        return best_poly
    return [1] * 12

# EVOLVE-BLOCK-END
