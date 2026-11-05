"""Rewritten program: randomized greedy with restarts to maximize disjoint residue classes."""
import math
import random
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80.0  # safety margin under 90s wall-clock
    all_moduli = list(range(2, 129))
    n = len(all_moduli)
    idx = {m: i for i, m in enumerate(all_moduli)}

    # Precompute gcd matrix
    gmat = [[0]*n for _ in range(n)]
    for i in range(n):
        mi = all_moduli[i]
        for j in range(i+1, n):
            g = math.gcd(mi, all_moduli[j])
            gmat[i][j] = g
            gmat[j][i] = g

    # conflict count
    conflict_count = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if gmat[i][j] > 1:
                conflict_count[i] += 1
                conflict_count[j] += 1

    base_order = sorted(all_moduli, key=lambda m: conflict_count[idx[m]])

    def try_greedy(order, rng):
        chosen = []          # list of (r, m)
        chosen_idx = []      # list of index
        chosen_r = []
        chosen_m = []
        for m in order:
            i = idx[m]
            residues = list(range(m))
            rng.shuffle(residues)
            placed = False
            for r in residues:
                ok = True
                for k in range(len(chosen)):
                    g = gmat[i][chosen_idx[k]]
                    if g > 1 and (r - chosen_r[k]) % g == 0:
                        ok = False
                        break
                if ok:
                    chosen.append((r, m))
                    chosen_idx.append(i)
                    chosen_r.append(r)
                    chosen_m.append(m)
                    placed = True
                    break
        return chosen

    rng = random.Random(12345)
    best = try_greedy(base_order, rng)
    best_len = len(best)

    while time.time() < deadline:
        order = list(all_moduli)
        rng.shuffle(order)
        order.sort(key=lambda m: conflict_count[idx[m]] + rng.random()*5)
        chosen = try_greedy(order, rng)
        if len(chosen) > best_len:
            best = chosen
            best_len = len(chosen)
        if best_len == n:
            break

    return [[r, m] for (r, m) in best]
# EVOLVE-BLOCK-END
