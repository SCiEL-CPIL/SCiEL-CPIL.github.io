import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n_mod = len(moduli)

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    gcd_mat = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute conflict masks: conflict[i][j][r_i] = bitmask of r_j that conflict
    conflict = [[None] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        mi = moduli[i]
        for j in range(n_mod):
            if i == j:
                continue
            mj = moduli[j]
            g = gcd_mat[i][j]
            masks = [0] * mi
            for r_i in range(mi):
                target = r_i % g
                mask = 0
                for r_j in range(mj):
                    if r_j % g == target:
                        mask |= (1 << r_j)
                masks[r_i] = mask
            conflict[i][j] = masks

    def popcount(x):
        return bin(x).count('1')

    def greedy_solve(order, temperature=0.0):
        n = len(order)
        avail = [(1 << moduli[i]) - 1 for i in range(n_mod)]
        chosen = {}

        # Precompute position map for fast future lookup
        pos_map = {idx: pos for pos, idx in enumerate(order)}

        for pos, idx in enumerate(order):
            if time.time() > deadline - 2.0:
                break
            mi = moduli[idx]
            if avail[idx] == 0:
                continue

            best_r = -1
            best_cost = float('inf')
            costs = []

            future = order[pos + 1:]

            for r in range(mi):
                if not (avail[idx] & (1 << r)):
                    continue
                cost = 0
                for j in future:
                    if avail[j] == 0:
                        continue
                    mask = conflict[idx][j][r]
                    cost += popcount(avail[j] & mask)
                costs.append((cost, r))
                if cost < best_cost:
                    best_cost = cost
                    best_r = r

            if best_r >= 0:
                if temperature > 0 and len(costs) > 1:
                    min_cost = min(c for c, r in costs)
                    weights = []
                    for c, r in costs:
                        weights.append(math.exp(-(c - min_cost) / temperature))
                    total_w = sum(weights)
                    rand_val = random.random() * total_w
                    cumulative = 0
                    for (c, r), w in zip(costs, weights):
                        cumulative += w
                        if rand_val <= cumulative:
                            best_r = r
                            break

                chosen[idx] = best_r
                for j in future:
                    mask = conflict[idx][j][best_r]
                    avail[j] &= ~mask

        return chosen

    def is_valid_solution(chosen):
        items = list(chosen.items())
        for i in range(len(items)):
            idx_i, r_i = items[i]
            for j in range(i + 1, len(items)):
                idx_j, r_j = items[j]
                g = gcd_mat[idx_i][idx_j]
                if r_i % g == r_j % g:
                    return False
        return True

    def local_search(chosen, max_iter=100):
        if time.time() > deadline - 5.0:
            return chosen
        chosen = dict(chosen)
        improved = True
        iterations = 0

        while improved and iterations < max_iter and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            missing = [i for i in range(n_mod) if i not in chosen]
            if not missing:
                break

            for idx in missing:
                if time.time() > deadline - 5.0:
                    break
                mi = moduli[idx]
                valid_residues = []
                for r in range(mi):
                    ok = True
                    for j, rj in chosen.items():
                        g = gcd_mat[idx][j]
                        if r % g == rj % g:
                            ok = False
                            break
                    if ok:
                        valid_residues.append(r)

                if not valid_residues:
                    continue

                best_r = None
                best_cost = float('inf')
                for r in valid_residues:
                    cost = 0
                    for j in missing:
                        if j == idx:
                            continue
                        mj = moduli[j]
                        g = gcd_mat[idx][j]
                        target = r % g
                        blocked_count = 0
                        for rj in range(mj):
                            if rj % g == target:
                                blocked_count += 1
                        cost += blocked_count
                    if cost < best_cost:
                        best_cost = cost
                        best_r = r

                if best_r is not None:
                    chosen[idx] = best_r
                    improved = True
                    break

        return chosen

    def swap_improve(chosen, max_iter=100):
        if time.time() > deadline - 5.0:
            return chosen
        chosen = dict(chosen)
        improved = True
        iterations = 0

        while improved and iterations < max_iter and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            missing = [i for i in range(n_mod) if i not in chosen]
            if not missing:
                break

            for idx in missing:
                if time.time() > deadline - 5.0:
                    break
                mi = moduli[idx]

                valid_residues = []
                for r in range(mi):
                    ok = True
                    for j, rj in chosen.items():
                        g = gcd_mat[idx][j]
                        if r % g == rj % g:
                            ok = False
                            break
                    if ok:
                        valid_residues.append(r)

                if not valid_residues:
                    for j, rj in list(chosen.items()):
                        if time.time() > deadline - 5.0:
                            break
                        mj = moduli[j]
                        g = gcd_mat[idx][j]
                        target = rj % g
                        for new_rj in range(mj):
                            if new_rj == rj or new_rj % g == target:
                                continue
                            ok = True
                            for k, rk in chosen.items():
                                if k == j:
                                    continue
                                g2 = gcd_mat[j][k]
                                if new_rj % g2 == rk % g2:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            for r in range(mi):
                                ok2 = True
                                for k, rk in chosen.items():
                                    if k == j:
                                        if r % g == new_rj % g:
                                            ok2 = False
                                            break
                                    else:
                                        g3 = gcd_mat[idx][k]
                                        if r % g3 == rk % g3:
                                            ok2 = False
                                            break
                                if ok2:
                                    chosen[j] = new_rj
                                    chosen[idx] = r
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break

                if improved:
                    break

        return chosen

    def two_swap_improve(chosen, max_iter=50):
        """Try 2-swap moves: swap residues of two chosen moduli to free space."""
        if time.time() > deadline - 5.0:
            return chosen
        chosen = dict(chosen)
        improved = True
        iterations = 0

        while improved and iterations < max_iter and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            missing = [i for i in range(n_mod) if i not in chosen]
            if not missing:
                break

            for idx in missing:
                if time.time() > deadline - 5.0:
                    break
                mi = moduli[idx]

                # Try to add idx by changing two existing residues
                keys = list(chosen.keys())
                for a in range(len(keys)):
                    if time.time() > deadline - 5.0:
                        break
                    for b in range(a + 1, len(keys)):
                        j1, j2 = keys[a], keys[b]
                        r1, r2 = chosen[j1], chosen[j2]
                        m1, m2 = moduli[j1], moduli[j2]

                        # Try all pairs of new residues for j1, j2
                        for nr1 in range(m1):
                            if nr1 == r1:
                                continue
                            # Check nr1 compatible with all chosen except j1
                            ok1 = True
                            for k, rk in chosen.items():
                                if k == j1:
                                    continue
                                g = gcd_mat[j1][k]
                                if nr1 % g == rk % g:
                                    ok1 = False
                                    break
                            if not ok1:
                                continue

                            for nr2 in range(m2):
                                if nr2 == r2:
                                    continue
                                # Check nr2 compatible with all chosen except j2
                                ok2 = True
                                for k, rk in chosen.items():
                                    if k == j2:
                                        continue
                                    g = gcd_mat[j2][k]
                                    if nr2 % g == rk % g:
                                        ok2 = False
                                        break
                                if not ok2:
                                    continue

                                # Check nr1 and nr2 compatible with each other
                                g12 = gcd_mat[j1][j2]
                                if nr1 % g12 == nr2 % g12:
                                    continue

                                # Now check if idx can be added
                                for r in range(mi):
                                    ok3 = True
                                    for k, rk in chosen.items():
                                        if k == j1:
                                            g3 = gcd_mat[idx][j1]
                                            if r % g3 == nr1 % g3:
                                                ok3 = False
                                                break
                                        elif k == j2:
                                            g3 = gcd_mat[idx][j2]
                                            if r % g3 == nr2 % g3:
                                                ok3 = False
                                                break
                                        else:
                                            g3 = gcd_mat[idx][k]
                                            if r % g3 == rk % g3:
                                                ok3 = False
                                                break
                                    if ok3:
                                        chosen[j1] = nr1
                                        chosen[j2] = nr2
                                        chosen[idx] = r
                                        improved = True
                                        break
                                if improved:
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        return chosen

    best_sol = {}
    best_len = 0

    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    def totient(n):
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result

    orderings = [
        list(range(n_mod)),
        list(range(n_mod - 1, -1, -1)),
        sorted(range(n_mod), key=lambda i: moduli[i]),
        sorted(range(n_mod), key=lambda i: -moduli[i]),
        sorted(range(n_mod), key=lambda i: num_prime_factors(moduli[i])),
        sorted(range(n_mod), key=lambda i: -num_prime_factors(moduli[i])),
        sorted(range(n_mod), key=lambda i: totient(moduli[i])),
        sorted(range(n_mod), key=lambda i: -totient(moduli[i])),
    ]

    for order in orderings:
        if time.time() > deadline - 10.0:
            break
        chosen = greedy_solve(order)
        chosen = local_search(chosen)
        chosen = swap_improve(chosen)
        chosen = two_swap_improve(chosen)
        if is_valid_solution(chosen):
            if len(chosen) > best_len:
                best_len = len(chosen)
                best_sol = dict(chosen)

    random.seed(42)
    trials = 0
    while time.time() < deadline - 10.0 and trials < 500:
        trials += 1
        order = list(range(n_mod))
        random.shuffle(order)
        temp = random.choice([0.0, 0.5, 1.0, 2.0, 5.0])
        chosen = greedy_solve(order, temperature=temp)
        chosen = local_search(chosen)
        chosen = swap_improve(chosen)
        chosen = two_swap_improve(chosen)
        if is_valid_solution(chosen):
            if len(chosen) > best_len:
                best_len = len(chosen)
                best_sol = dict(chosen)

    if time.time() < deadline - 5.0:
        best_sol = local_search(best_sol, max_iter=100)
        best_sol = swap_improve(best_sol, max_iter=200)
        best_sol = two_swap_improve(best_sol, max_iter=100)
        if is_valid_solution(best_sol):
            best_len = len(best_sol)
        else:
            valid_subset = {}
            for idx in sorted(best_sol.keys()):
                r = best_sol[idx]
                ok = True
                for j, rj in valid_subset.items():
                    g = gcd_mat[idx][j]
                    if r % g == rj % g:
                        ok = False
                        break
                if ok:
                    valid_subset[idx] = r
            best_sol = valid_subset
            best_len = len(best_sol)

    if not best_sol:
        best_sol = {0: 0}
        best_len = 1

    return [[r, moduli[idx]] for idx, r in best_sol.items()]
