# EVOLVE-BLOCK-START
import time
import random
import itertools

import numpy as np


def _ball_indices(q, n, radius, word_idx):
    """Return a numpy int64 array of all indices within Hamming distance <= radius of word_idx."""
    if radius == 0:
        return np.array([word_idx], dtype=np.int64)
    word = np.array(list(np.unravel_index(word_idx, (q,) * n)), dtype=np.int64)
    balls = [word_idx]
    if radius >= 1:
        for i in range(n):
            orig = word[i]
            for v in range(q):
                if v == orig:
                    continue
                word[i] = v
                balls.append(int(np.ravel_multi_index(tuple(word.tolist()), (q,) * n)))
            word[i] = orig
    return np.array(balls, dtype=np.int64)


def construct(q, n, radius):
    start = time.time()
    deadline = start + 112.0  # safety margin before the 120s/150s kill

    total = q ** n
    shape = (q,) * n

    # ---------- Precompute every ball (index -> array of covered indices) ----------
    t0 = time.time()
    ball_list = []
    for idx in range(total):
        ball_list.append(_ball_indices(q, n, radius, idx))
    if time.time() > deadline:
        # cannot even precompute; return trivial full code
        return [list(w) for w in np.ndindex(*shape)]

    # ---------- Greedy cover (vectorized coverage counting) ----------
    def greedy(seed):
        rng = random.Random(seed)
        covered = np.zeros(total, dtype=np.int32)  # coverage count
        remaining = total
        code = []
        # order candidates; we will scan all each iteration (small total)
        # To add variety, shuffle the scan order once per restart
        order = list(range(total))
        rng.shuffle(order)
        while remaining > 0:
            if time.time() > deadline - 4.0:
                break
            best_idx = -1
            best_gain = -1
            for idx in order:
                if covered[idx] > 0:
                    continue
                gain = int(np.count_nonzero(covered[ball_list[idx]] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    if gain == remaining:
                        break
            if best_idx < 0:
                break
            code.append(best_idx)
            b = ball_list[best_idx]
            new = covered[b] == 0
            if new.any():
                remaining -= int(new.sum())
            covered[b] += 1
        return code, covered

    # ---------- Local search with coverage counts + add/remove/swap ----------
    def local_search(code, covered, deadline_local):
        # code: list of indices; covered: coverage count array
        m = len(code)
        code_set = set(code)
        improved = True
        # candidate pool for swaps: all indices
        all_idx = list(range(total))
        while improved and time.time() < deadline_local:
            improved = False
            # 1) Try removals
            for k in range(len(code)):
                if time.time() > deadline_local:
                    return code, covered
                b = ball_list[code[k]]
                if np.all(covered[b] >= 2):
                    # safe to remove
                    covered[b] -= 1
                    code.pop(k)
                    code_set.discard(code_set and 0)  # no-op keep
                    improved = True
                    break
            if improved:
                continue
            # 2) Try swaps: replace codeword k with a candidate c that improves coverage
            #    A swap is valid if after removing k and adding c, coverage stays >=1 everywhere.
            #    Removing k is safe where covered>=2. The "risky" words are ball(k) where covered==1.
            #    After removal, those risky words drop to 0; c must cover all of them.
            #    Also c should add value: cover some currently-uncovered? there are none (covered).
            #    Net benefit: c covers words that were covered==1 by k only AND c also covers
            #    words not covered by k (to allow future removals). We accept any valid swap that
            #    keeps size same but we only do swaps that enable a later removal; simpler:
            #    accept swap if it keeps coverage valid (we rely on removal pass to shrink).
            for k in range(len(code)):
                if time.time() > deadline_local:
                    return code, covered
                b_k = ball_list[code[k]]
                risky = b_k[covered[b_k] == 1]  # words uniquely covered by k
                if len(risky) == 0:
                    # then k is removable -> handled by removal pass; skip
                    continue
                risky_set = set(risky.tolist())
                cur = code[k]
                # candidate must cover all risky words
                # score candidate by how many words it covers that are NOT in b_k (new coverage)
                best_c = -1
                best_score = -1
                for c in all_idx:
                    if c == cur:
                        continue
                    b_c = ball_list[c]
                    # must cover all risky
                    if not all((covered[b_c] >= 1) or True for _ in [0]):
                        pass
                    # check risky covered by c
                    if not np.all(np.isin(risky, b_c)):
                        continue
                    # valid swap keeps coverage (since risky now covered by c, others fine)
                    # score: number of words c covers that k does not
                    newcov = int(np.count_nonzero(~np.isin(b_c, b_k)))
                    if newcov > best_score:
                        best_score = newcov
                        best_c = c
                if best_c >= 0 and best_score > 0:
                    # perform swap
                    covered[b_k] -= 1
                    covered[ball_list[best_c]] += 1
                    code[k] = best_c
                    improved = True
                    break
        return code, covered

    # ---------- Iterated greedy + local search ----------
    best_code = None
    best_size = float('inf')

    # First do a solid greedy to get a valid baseline
    base_code, base_cov = greedy(42)
    if base_cov.min() >= 1 and len(base_code) < best_size:
        lc, _ = local_search(list(base_code), base_cov.copy(), deadline - 6.0)
        if len(lc) < best_size:
            best_size = len(lc)
            best_code = list(lc)

    # More restarts with varied seeds while time permits
    seeds = [123, 7, 999, 2024, 5, 17, 31, 64, 128, 256]
    for seed in seeds:
        if time.time() > deadline - 12.0:
            break
        code, cov = greedy(seed)
        if cov.min() < 1:
            continue
        lc, _ = local_search(list(code), cov.copy(), deadline - 6.0)
        if len(lc) < best_size:
            best_size = len(lc)
            best_code = list(lc)
            if time.time() > deadline - 8.0:
                break

    if best_code is None:
        # fallback: full code
        return [list(w) for w in np.ndindex(*shape)]

    # Convert indices back to word lists
    result = [list(np.unravel_index(i, shape)) for i in best_code]
    # Ensure distinct (they are, by construction indices unique)
    return result
# EVOLVE-BLOCK-END
