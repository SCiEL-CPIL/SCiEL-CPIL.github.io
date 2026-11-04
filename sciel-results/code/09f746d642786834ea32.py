import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start = time.time()
    DEADLINE = 110.0  # safety margin under the 150s kill
    MAX_BLOCKS = int(v * (v - 1) * (v - 2) / (t * (t - 1) * (t - 2))) + 10  # C(v,t) cap

    def remaining():
        return DEADLINE - time.time()

    def time_ok():
        return time.time() < DEADLINE - 2.0

    # Precompute all t-combinations and their index map
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    comb_to_idx = {c: i for i, c in enumerate(all_t)}

    def greedy(seed_rng):
        rng = random.Random(seed_rng)
        uncovered = np.ones(n_t, dtype=bool)
        blocks = []

        while uncovered.any() and len(blocks) < MAX_BLOCKS:
            if not time_ok():
                return None
            # Pick the uncovered t-subset with the most potential (heuristic: random among uncovered)
            unc_idx = np.where(uncovered)[0]
            start_idx = unc_idx[rng.randrange(len(unc_idx))]
            block = list(all_t[start_idx])

            # Build the block greedily
            while len(block) < k:
                if not time_ok():
                    return None
                block_set = set(block)
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break

                # Compute gain for each candidate: number of uncovered t-subsets it would cover
                # A candidate x covers t-subsets {x} union (t-1)-subset of block
                # Precompute (t-1)-subsets of current block
                if t - 1 > 0:
                    t_minus_1_subs = list(itertools.combinations(block, t - 1))
                    # For each candidate, count how many (t-1)-subsets + x form an uncovered t-subset
                    gains = np.zeros(len(candidates), dtype=np.int32)
                    for ci, x in enumerate(candidates):
                        cnt = 0
                        for sub in t_minus_1_subs:
                            tsub = tuple(sorted(sub + (x,)))
                            idx = comb_to_idx.get(tsub)
                            if idx is not None and uncovered[idx]:
                                cnt += 1
                        gains[ci] = cnt
                    best_ci = int(np.argmax(gains))
                    if gains[best_ci] == 0:
                        # No gain; pick a random candidate to break tie
                        best_ci = rng.randrange(len(candidates))
                    block.append(candidates[best_ci])
                else:
                    # t == 1 edge case
                    block.append(candidates[rng.randrange(len(candidates))])

            block.sort()
            blocks.append(block)

            # Mark all t-subsets of this block as covered
            for tsub in itertools.combinations(block, t):
                idx = comb_to_idx[tsub]
                uncovered[idx] = False

        if uncovered.any():
            return None  # incomplete (shouldn't happen)
        return blocks

    best = None
    attempt = 0
    while time.time() < DEADLINE - 5.0:
        if best is not None and len(best) <= 76:
            break  # can't do better than known best of 76
        res = greedy(attempt)
        if res is not None:
            if best is None or len(res) < len(best):
                best = res
        attempt += 1
        if time.time() < DEADLINE - 5.0 and best is None:
            # keep trying
            continue

    if best is None:
        # Fallback: a single deterministic greedy (no random) to guarantee a valid answer
        # Rebuild with seed 0 and no time pressure on the final block
        uncovered = np.ones(n_t, dtype=bool)
        blocks = []
        while uncovered.any() and len(blocks) < MAX_BLOCKS:
            unc_idx = np.where(uncovered)[0]
            start_idx = unc_idx[0]
            block = list(all_t[start_idx])
            while len(block) < k:
                block_set = set(block)
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                if t - 1 > 0:
                    t_minus_1_subs = list(itertools.combinations(block, t - 1))
                    gains = np.zeros(len(candidates), dtype=np.int32)
                    for ci, x in enumerate(candidates):
                        cnt = 0
                        for sub in t_minus_1_subs:
                            tsub = tuple(sorted(sub + (x,)))
                            idx = comb_to_idx.get(tsub)
                            if idx is not None and uncovered[idx]:
                                cnt += 1
                        gains[ci] = cnt
                    best_ci = int(np.argmax(gains))
                    block.append(candidates[best_ci])
                else:
                    block.append(candidates[0])
            block.sort()
            blocks.append(block)
            for tsub in itertools.combinations(block, t):
                idx = comb_to_idx[tsub]
                uncovered[idx] = False
        best = blocks

    return best
