"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(a, b):
    n = len(a)
    deadline = time.time() + 110.0  # safety margin under the 120 s limit

    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    bT = b.T

    # ---- initial solution: rank-matching (heavy flows -> short distances) ----
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    inv = np.empty(n, dtype=np.int64)
    inv[p] = np.arange(n)

    # ---- fast cost + per-row aggregate ----
    def full_cost():
        c = 0
        for i in range(n):
            c += int(np.dot(a[i], b[p[i]]))
        return c

    cur = full_cost()

    # ---- tabu search ----
    tabu_len = 18
    tabu = {}
    best = cur
    best_p = p.copy()
    it = 0
    a_rowsum = a.sum(axis=1)

    # We will do a few rounds of tabu search, interspersed with a perturbation
    # (kick) to escape local minima. The kick uses a random subset of swaps
    # guided by the worst-performing facilities (those with high row sums in a
    # but placed at locations with high distance sums in b).

    def kick(p, cur, num_swaps=5):
        """Perturb by swapping a few poorly-placed facilities."""
        # Identify facilities with high a_rowsum (important) and their current
        # locations. Swap them with random other locations.
        # Use a simple random selection biased toward high-impact facilities.
        indices = np.arange(n)
        # Weight by a_rowsum to target important facilities
        weights = a_rowsum + 1  # avoid zero weights
        weights = weights / weights.sum()
        chosen = np.random.choice(indices, size=min(num_swaps, n // 2), replace=False, p=weights)
        for i in chosen:
            # Pick a random j != i
            j = np.random.randint(0, n)
            if j == i:
                j = (j + 1) % n
            # Compute delta for swap (i, j)
            u, v = p[i], p[j]
            if u == v:
                continue
            ai, aj = a[i], a[j]
            diff_b = bT[v] - bT[u]
            diff_a = ai - aj
            cross = int(np.dot(diff_a, diff_b))
            delta = int((ai[v] + aj[v] - ai[u] - aj[u]) * b[u, v]) + cross
            # Accept with some probability even if delta > 0 (simulated-annealing-like)
            # For simplicity, accept if delta < 0 or randomly with small prob
            if delta < 0 or np.random.random() < 0.1:
                p[i], p[j] = v, u
                cur += delta
        return p, cur

    # Run tabu search in rounds with kicks between them
    num_rounds = 3
    for rnd in range(num_rounds):
        if time.time() > deadline - 10:
            break
        # Reset tabu for new round
        tabu = {}
        it = 0
        # Run tabu search until no improvement or time check
        no_improve_count = 0
        while time.time() < deadline - 10:
            cand = np.argsort(-a_rowsum, kind="stable")
            best_delta = 0
            best_i = -1
            best_j = -1

            for i in cand:
                if tabu.get(i, -1) >= it:
                    continue
                u = p[i]
                ai = a[i]
                for j in range(n):
                    if j == i:
                        continue
                    v = p[j]
                    if v == u:
                        continue
                    aj = a[j]
                    diff_b = bT[v] - bT[u]
                    diff_a = ai - aj
                    cross = int(np.dot(diff_a, diff_b))
                    delta = int((ai[v] + aj[v] - ai[u] - aj[u]) * b[u, v]) + cross
                    tabu_j = tabu.get(j, -1) >= it
                    if delta < best_delta or (delta < best and not tabu_j):
                        if delta < best:
                            best_delta = delta
                            best_i, best_j = i, j
                        elif not tabu_j:
                            best_delta = delta
                            best_i, best_j = i, j

            if best_i < 0:
                break

            i, j = best_i, best_j
            u, v = p[i], p[j]
            p[i], p[j] = v, u
            inv[u], inv[v] = j, i

            cur += best_delta
            if cur < best:
                best = cur
                best_p = p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            tabu[i] = it + tabu_len
            tabu[j] = it + tabu_len
            it += 1

            if no_improve_count > 50:
                break

        # Kick to escape local minimum (except after last round)
        if rnd < num_rounds - 1:
            p, cur = kick(p, cur, num_swaps=8)

    return best_p.tolist()
# EVOLVE-BLOCK-END
