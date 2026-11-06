import time
import random
import numpy as np

def _validate(poly):
    """Check if poly meets all constraints."""
    if len(poly) != 12:
        return False
    for e, c in poly:
        if c == 0:
            return False
        if c < -64 or c > 64:
            return False
        if e < 0 or not isinstance(e, int):
            return False
        if e > 40:
            return False
    exps = [e for e, _ in poly]
    if len(set(exps)) != 12:
        return False
    return True

def _square_nonzero_count_sparse(poly):
    """Fast sparse square evaluation: number of nonzero terms in poly^2."""
    if not poly:
        return 0
    sq = {}
    n = len(poly)
    for i in range(n):
        ei, ci = poly[i]
        for j in range(i, n):
            ej, cj = poly[j]
            key = ei + ej
            if i == j:
                sq[key] = sq.get(key, 0) + ci * ci
            else:
                sq[key] = sq.get(key, 0) + 2 * ci * cj
    return sum(1 for v in sq.values() if v != 0)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit

    best_poly = None
    best_score = float('inf')

    random.seed(42)
    np.random.seed(42)

    def _evaluate(poly):
        return _square_nonzero_count_sparse(poly)

    def _sa_optimize(initial_poly, max_iters=50000, T_start=10.0, T_end=0.05):
        """Simulated annealing to minimize nonzero terms in square."""
        current = list(initial_poly)
        current_score = _evaluate(current)
        best = list(current)
        best_score = current_score
        T = T_start

        coeff_choices = [-8, -4, -3, -2, -1, 1, 2, 3, 4, 8]

        for it in range(max_iters):
            if time.time() > deadline:
                break

            progress = it / max_iters
            T = T_start * (T_end / T_start) ** progress

            move_type = random.random()
            neighbor = list(current)

            if move_type < 0.45:
                # Change one exponent - bias toward local moves
                i = random.randint(0, 11)
                old_exp = neighbor[i][0]
                if random.random() < 0.6:
                    offset = random.choice([-3, -2, -1, 1, 2, 3, -5, 5])
                    new_exp = old_exp + offset
                    if new_exp < 0 or new_exp > 40:
                        continue
                else:
                    new_exp = random.randint(0, 40)
                if new_exp == old_exp:
                    continue
                if any(neighbor[j][0] == new_exp for j in range(12) if j != i):
                    continue
                neighbor[i] = (new_exp, neighbor[i][1])
            elif move_type < 0.75:
                # Change one coefficient
                i = random.randint(0, 11)
                new_coeff = random.choice(coeff_choices)
                if new_coeff == 0 or new_coeff == neighbor[i][1]:
                    continue
                neighbor[i] = (neighbor[i][0], new_coeff)
            elif move_type < 0.90:
                # Swap two exponents
                i, j = random.sample(range(12), 2)
                ei, ci = neighbor[i]
                ej, cj = neighbor[j]
                neighbor[i] = (ej, ci)
                neighbor[j] = (ei, cj)
            else:
                # Perturb two coefficients simultaneously
                i, j = random.sample(range(12), 2)
                new_ci = random.choice(coeff_choices)
                new_cj = random.choice(coeff_choices)
                if new_ci == 0 or new_cj == 0:
                    continue
                neighbor[i] = (neighbor[i][0], new_ci)
                neighbor[j] = (neighbor[j][0], new_cj)

            if not _validate(neighbor):
                continue

            neighbor_score = _evaluate(neighbor)
            delta = neighbor_score - current_score

            if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                current = neighbor
                current_score = neighbor_score
                if current_score < best_score:
                    best_score = current_score
                    best = list(current)

            if best_score <= 12:
                break

        return best, best_score

    # ---- Initial candidates: seed + structured families ----
    seed_poly = [(i, 1) for i in range(12)]
    if _validate(seed_poly):
        s = _evaluate(seed_poly)
        best_poly = list(seed_poly)
        best_score = s

    def _try_update(poly):
        nonlocal best_poly, best_score
        if _validate(poly):
            sc = _evaluate(poly)
            if sc < best_score:
                best_score = sc
                best_poly = list(poly)
            return sc
        return float('inf')

    # Arithmetic progressions with sign patterns
    for step in [1, 2, 3]:
        for start in range(0, 41 - 11 * step + 1):
            if time.time() > deadline:
                break
            for sign_pattern in [[1]*12, [1,-1]*6, [1,1,-1,-1]*3, [1,-1,1,-1]*3]:
                poly = [(start + i*step, sign_pattern[i]) for i in range(12)]
                if _validate(poly):
                    sc = _evaluate(poly)
                    if sc < best_score:
                        best_score = sc
                        best_poly = list(poly)
            if best_score <= 12:
                break
        if best_score <= 12:
            break

    # ---- Main search: diverse restarts + SA ----
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        if r == 0:
            initial = list(best_poly) if best_poly is not None else list(seed_poly)
        elif r == 1:
            step = random.randint(1, 3)
            start = random.randint(0, max(0, 40 - 11*step))
            signs = [random.choice([-1, 1]) for _ in range(12)]
            initial = [(start + i*step, signs[i]) for i in range(12)]
        elif r == 2:
            # Two clusters
            base1 = random.randint(0, 15)
            base2 = random.randint(25, 35)
            exps = []
            for b in [base1, base2]:
                for _ in range(6):
                    e = b + random.randint(0, 4)
                    if e not in exps and 0 <= e <= 40:
                        exps.append(e)
            while len(exps) < 12:
                e = random.randint(0, 40)
                if e not in exps:
                    exps.append(e)
            exps = sorted(exps[:12])
            initial = [(exps[i], random.choice([-1, 1])) for i in range(12)]
        elif r == 3:
            # Pairs with same coefficient magnitude
            base_exps = sorted(random.sample(range(0, 41), 6))
            exps = []
            for be in base_exps:
                exps.append(be)
                off = random.randint(1, 5)
                e2 = be + off
                if e2 > 40 or e2 in exps:
                    e2 = be - off
                if e2 < 0 or e2 in exps:
                    e2 = random.randint(0, 40)
                    while e2 in exps:
                        e2 = random.randint(0, 40)
                exps.append(e2)
            exps = sorted(exps[:12])
            coeffs = []
            for _ in range(6):
                c = random.choice([-2, -1, 1, 2])
                coeffs.extend([c, c])
            random.shuffle(coeffs)
            initial = list(zip(exps, coeffs))
        elif r == 4:
            # Palindromic coefficients
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1, -2, 2]) for _ in range(6)]
            coeffs = coeffs + coeffs[::-1]
            initial = list(zip(exps, coeffs))
        else:
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
            initial = list(zip(exps, coeffs))

        if not _validate(initial):
            continue

        sc = _evaluate(initial)
        if sc < best_score:
            best_score = sc
            best_poly = list(initial)

        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5:
            break
        max_iters = min(30000, int(remaining * 2000))
        local_poly, local_score = _sa_optimize(initial, max_iters=max(1000, max_iters))
        if local_score < best_score:
            best_score = local_score
            best_poly = local_poly

        if best_score <= 12:
            break

    # ---- Final polish: tighter SA on best found ----
    if best_poly is not None and time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 3:
            final_poly, final_score = _sa_optimize(
                best_poly, max_iters=int(remaining * 3000), T_start=5.0, T_end=0.01
            )
            if final_score < best_score:
                best_score = final_score
                best_poly = final_poly

    if best_poly is None:
        best_poly = seed_poly

    max_deg = max(e for e, _ in best_poly)
    coeff_list = [0] * (max_deg + 1)
    for exp, coeff in best_poly:
        coeff_list[exp] = coeff

    return coeff_list
