<<<<<<< SEARCH
import itertools
import time
import random
import numpy as np
from collections import defaultdict


def _greedy_construct(v, k, t, deadline):
    """
    Fast greedy covering algorithm using vectorized gain computation.
    Uses precomputed t-1 to additions mapping for O(1) lookup per (rest, x) pair.
    """
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    covered = np.zeros(n_subs, dtype=bool)
    blocks = []

    # Precompute: for each (t-1)-subset, the index of the t-subset formed by adding each point
    tminus1_to_additions = defaultdict(dict)
    for sub_idx, sub in enumerate(all_subs):
        for i in range(t):
            rest = tuple(sorted(sub[:i] + sub[i+1:]))
            x = sub[i]
            tminus1_to_additions[rest][x] = sub_idx

    while not covered.all():
        if time.time() > deadline:
            break

        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
        start_idx = uncovered_indices[0]
        start_sub = all_subs[start_idx]

        block = list(start_sub)
        block_set = set(block)

        while len(block) < k:
            if time.time() > deadline:
                break

            best_point = -1
            best_gain = -1

            # For each candidate point x, compute gain = number of uncovered t-subsets
            # that contain x and whose other t-1 elements are in block
            for x in range(v):
                if x in block_set:
                    continue

                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    rest_key = tuple(sorted(rest))
                    if x in tminus1_to_additions.get(rest_key, {}):
                        idx = tminus1_to_additions[rest_key][x]
                        if not covered[idx]:
                            gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_point = x

            if best_point == -1:
                break
            block.append(best_point)
            block_set.add(best_point)

        block.sort()
        blocks.append(block)

        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            covered[idx] = True

    return blocks, sub_to_idx, all_subs
=======
import itertools
import time
import random
import numpy as np
from collections import defaultdict


def _greedy_construct(v, k, t, deadline):
    """
    Fast greedy covering algorithm using vectorized gain computation.
    Precomputes for each t-subset its t-1 projections to enable fast gain lookup.
    """
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    covered = np.zeros(n_subs, dtype=bool)
    blocks = []

    # Precompute: for each (t-1)-subset, list of (point, t-subset-index) pairs
    tminus1_to_additions = defaultdict(list)
    for sub_idx, sub in enumerate(all_subs):
        for i in range(t):
            rest = tuple(sorted(sub[:i] + sub[i+1:]))
            x = sub[i]
            tminus1_to_additions[rest].append((x, sub_idx))

    while not covered.all():
        if time.time() > deadline:
            break

        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
        start_idx = uncovered_indices[0]
        start_sub = all_subs[start_idx]

        block = list(start_sub)
        block_set = set(block)

        while len(block) < k:
            if time.time() > deadline:
                break

            # Compute gain for each candidate point
            gains = np.zeros(v, dtype=np.int32)
            for rest in itertools.combinations(block, t - 1):
                rest_key = tuple(sorted(rest))
                if rest_key in tminus1_to_additions:
                    for x, idx in tminus1_to_additions[rest_key]:
                        if x not in block_set and not covered[idx]:
                            gains[x] += 1

            # Find best candidate
            candidates = np.where((gains > 0) | np.array([g in block_set for g in range(v)] == False))[0]
            if len(candidates) == 0:
                candidates = np.where(gains > 0)[0]
            if len(candidates) == 0:
                break

            best_point = candidates[np.argmax(gains[candidates])]
            block.append(best_point)
            block_set.add(best_point)

        block.sort()
        blocks.append(block)

        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            covered[idx] = True

    return blocks, sub_to_idx, all_subs
>>>>>>> REPLACE

