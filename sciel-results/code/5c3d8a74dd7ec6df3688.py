import numpy as np
import time
import random
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = 580

    def fallback():
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        points.append(v)
        return np.array(points, dtype=np.int64)

    try:
        all_vectors = []
        for support in combinations(range(d), 4):
            for signs in range(16):
                v = np.zeros(d, dtype=np.int64)
                for k_idx, pos in enumerate(support):
                    v[pos] = 1 if (signs >> k_idx) & 1 else -1
                all_vectors.append(v)

        mat = np.array(all_vectors)
        n = len(mat)

        dots = mat @ mat.T
        conflict = dots > 2

        best_selected = None
        best_len = 0

        # Deterministic greedy first
        available = np.ones(n, dtype=bool)
        selected = []
        while available.any():
            if time.time() - start_time > deadline:
                break
            conflict_counts = conflict[:, available].sum(axis=1)
            conflict_counts[~available] = n + 1
            idx = int(np.argmin(conflict_counts))
            selected.append(idx)
            avail_conflicts = conflict[idx] & available
            available[avail_conflicts] = False
            available[idx] = False
        best_selected = selected[:]
        best_len = len(selected)

        # Random restarts with different orderings
        rng = random.Random(42)
        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() - start_time > deadline - 30:
                break
            # Randomize tie-breaking by adding small random noise to conflict counts
            available = np.ones(n, dtype=bool)
            selected = []
            while available.any():
                if time.time() - start_time > deadline:
                    break
                conflict_counts = conflict[:, available].sum(axis=1).astype(np.float64)
                # Add small random perturbation to break ties randomly
                noise = rng.random() * 0.5
                conflict_counts[~available] = n + 1
                # Pick among the fewest conflict counts with random tie-breaking
                min_val = np.min(conflict_counts[available])
                candidates = np.where((conflict_counts <= min_val + 0.1) & available)[0]
                if len(candidates) == 0:
                    candidates = np.where(available)[0]
                idx = int(candidates[rng.randint(0, len(candidates) - 1)])
                selected.append(idx)
                avail_conflicts = conflict[idx] & available
                available[avail_conflicts] = False
                available[idx] = False
            if len(selected) > best_len:
                best_len = len(selected)
                best_selected = selected[:]

        if best_len > 220:
            return mat[best_selected].astype(np.int64)

    except Exception:
        pass

    return fallback()
