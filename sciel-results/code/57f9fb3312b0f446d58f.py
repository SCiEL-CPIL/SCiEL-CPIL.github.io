"""Simulated Annealing search for a ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 140.0  # safety margin before the 150s kill
    rng = random.Random(12345)

    if q != 3 or n != 6 or radius != 1:
        # Generic fallback: direct sum of a q-ary perfect 1-covering on 4 coords.
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(q), repeat=n - 4)]

    # ---- Precompute the 729 words as integers in base-3, and each word's 19 neighbours ----
    words = np.array(list(itertools.product(range(3), repeat=6)), dtype=np.int8)
    wcol = np.arange(6)
    # 19 neighbours = self + 18 single-coordinate changes
    neigh = np.concatenate(
        [words[:, None, :],
         words[:, None, :] + np.array([[-1, 1] for _ in range(6)]).reshape(1, 6, 2)],
        axis=1)
    neigh = neigh % 3
    neigh = neigh.reshape(729, 19)

    # ---- Seed: direct sum of the ternary tetracode (9 words) with all 9 tails -> 81 words ----
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    seed_idx = np.array([int(np.ravel_multi_index(w, (3,) * 6)) for w in seed])

    # ---- SA machinery ----
    def recompute(sel):
        covered = np.zeros(729, dtype=bool)
        for i in sel:
            covered |= neigh[i]
        return covered

    best = seed_idx.copy()
    best_uncovered = 729 - recompute(best).sum()
    cur = seed_idx.copy()
    cur_cov = recompute(cur)
    cur_uncovered = 729 - cur_cov.sum()

    T = 1.0
    T_min = 1e-4
    cool = 0.9998

    while time.time() < deadline:
        if T < T_min:
            break
        move = rng.randrange(3)
        if move == 0 and len(cur) > 0:
            # REMOVE a random codeword
            k = rng.randrange(len(cur))
            cand = np.delete(cur, k)
            cand_cov = recompute(cand)
            cand_unc = 729 - cand_cov.sum()
            d = cand_unc - cur_uncovered  # >= 0
            if d == 0 or (rng.random() < np.exp(-d / T)):
                cur, cur_cov, cur_uncovered = cand, cand_cov, cand_unc
                if cur_uncovered < best_uncovered:
                    best, best_uncovered = cur.copy(), cur_uncovered
        elif move == 1:
            # ADD a random codeword (prefer one that covers the most new words)
            i = rng.randrange(729)
            if i in cur:
                continue
            cand = np.append(cur, i)
            cand_cov = recompute(cand)
            cand_unc = 729 - cand_cov.sum()
            d = cand_unc - cur_uncovered  # <= 0
            if d <= 0 or (rng.random() < np.exp(-d / T)):
                cur, cur_cov, cur_uncovered = cand, cand_cov, cand_unc
                if cur_uncovered < best_uncovered:
                    best, best_uncovered = cur.copy(), cur_uncovered
        else:
            # SWAP: remove one, add one
            if len(cur) == 0:
                continue
            k = rng.randrange(len(cur))
            i = rng.randrange(729)
            if i in cur and i != cur[k]:
                continue
            cand = np.delete(cur, k)
            if i not in cand:
                cand = np.append(cand, i)
            cand_cov = recompute(cand)
            cand_unc = 729 - cand_cov.sum()
            d = cand_unc - cur_uncovered
            if d <= 0 or (rng.random() < np.exp(-d / T)):
                cur, cur_cov, cur_uncovered = cand, cand_cov, cand_unc
                if cur_uncovered < best_uncovered:
                    best, best_uncovered = cur.copy(), cur_uncovered
        T *= cool

    # ---- Final polish: try to remove any redundant codeword from the best set ----
    sel = best.copy()
    for k in range(len(sel)):
        cand = np.delete(sel, k)
        if recompute(cand).all():
            sel = cand
    best = sel
    best_uncovered = 729 - recompute(best).sum()

    return [list(words[i]) for i in best]
# EVOLVE-BLOCK-END
