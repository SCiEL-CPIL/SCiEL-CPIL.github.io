"""Fresh design: greedy stochastic search over integer vectors in R^11.

Goal: maximize k such that min pairwise distance >= max norm.
We explore vectors with small integer coordinates (norm up to 4) and use
a greedy + local-search heuristic to pack as many as possible.
"""

import time
import random
import numpy as np

D = 11


def _norm_sq(v):
    return int(np.dot(v, v))


def _dist_sq(a, b):
    d = a - b
    return int(np.dot(d, d))


def _generate_candidate_pool():
    """Generate a diverse pool of candidate integer vectors."""
    pool = []
    seen = set()

    # Norm 1: unit vectors with signs
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            key = tuple(v)
            if key not in seen:
                seen.add(key)
                pool.append(v)

    # Norm 2: vectors with two non-zero entries of value +/-1
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    key = tuple(v)
                    if key not in seen:
                        seen.add(key)
                        pool.append(v)

    # Norm 4: vectors with four non-zero entries of value +/-1,
    # or two entries with one being +/-2, etc.
    # Include some norm-4 vectors: four positions with +/-1
    from itertools import combinations
    for combo in combinations(range(D), 4):
        for signs in [(1, 1, 1, 1), (1, 1, 1, -1), (1, 1, -1, -1),
                      (1, -1, -1, 1), (1, -1, -1, -1), (-1, 1, 1, -1),
                      (-1, 1, -1, 1), (-1, -1, 1, 1), (-1, -1, 1, -1),
                      (-1, -1, -1, 1)]:
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            key = tuple(v)
            if key not in seen:
                seen.add(key)
                pool.append(v)

    # Also include vectors with a 2 and a 1 in different positions (norm^2 = 4+1=5, skip for now)
    # Include vectors with two 2s (norm^2=8)
    for i in range(D):
        for j in range(i + 1, D):
            for si in (2, -2):
                for sj in (2, -2):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    key = tuple(v)
                    if key not in seen:
                        seen.add(key)
                        pool.append(v)

    return pool


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    random.seed(42)

    pool = _generate_candidate_pool()
    pool_arr = np.array(pool, dtype=np.int64)

    # We want: min pairwise distance >= max norm
    # Let M = max norm among selected vectors
    # For all pairs (i,j): ||v_i - v_j|| >= M

    # Strategy: greedy with restarts
    best_vectors = None
    best_count = 0

    def try_greedy(pool_indices, max_iters=50000):
        nonlocal best_vectors, best_count

        selected = []
        selected_arr = []
        used = set()

        # Shuffle
        order = list(range(len(pool_indices)))
        random.shuffle(order)

        for idx in order:
            if time.time() > deadline - 5:
                break
            v = pool[idx]
            vs = np.array(v, dtype=np.int64)
            vnorm_sq = int(np.dot(vs, vs))

            # Check if v can be added
            can_add = True
            for existing in selected:
                d = vs - existing
                dist_sq = int(np.dot(d, d))
                # We need dist >= max_norm_so_far including v
                # max_norm_sq = max(vnorm_sq, existing_norm_sq)
                # But we track max norm across all selected
                # Simpler: check dist_sq >= max(vnorm_sq, current_max_norm_sq)
                # Actually constraint: min pairwise dist >= max vector norm
                # So for each pair, dist >= max(norm_a, norm_b)
                # Let's track current_max_norm_sq
                existing_norm_sq = int(np.dot(existing, existing))
                required_sq = max(vnorm_sq, existing_norm_sq)
                if dist_sq < required_sq:
                    can_add = False
                    break

            if can_add:
                selected.append(vs)
                used.add(idx)

        if len(selected) > best_count:
            best_count = len(selected)
            best_vectors = np.array(selected, dtype=np.int64)

        return best_count

    # Run greedy with different random shuffles
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        try_greedy(range(len(pool)))

    # Local improvement: try swapping vectors
    if best_vectors is not None and len(best_vectors) > 0:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            # Try adding a random candidate that wasn't selected
            current_set = set(tuple(r) for r in best_vectors)
            improved = False
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                idx = random.randint(0, len(pool) - 1)
                v = pool[idx]
                if tuple(v) in current_set:
                    continue
                vs = np.array(v, dtype=np.int64)
                vnorm_sq = int(np.dot(vs, vs))
                can_add = True
                for existing in best_vectors:
                    d = vs - existing
                    dist_sq = int(np.dot(d, d))
                    existing_norm_sq = int(np.dot(existing, existing))
                    required_sq = max(vnorm_sq, existing_norm_sq)
                    if dist_sq < required_sq:
                        can_add = False
                        break
                if can_add:
                    best_vectors = np.vstack([best_vectors, vs])
                    current_set.add(tuple(v))
                    best_count = len(best_vectors)
                    improved = True
                    break
            if not improved:
                break

    if best_vectors is None or len(best_vectors) == 0:
        # Fallback: return at least something valid
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    return best_vectors
