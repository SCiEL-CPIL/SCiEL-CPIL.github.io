"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    # Generate Weight 3 and Weight 4 vectors
    candidates = []
    norms_sq = []

    # Weight 3
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(3)

    # Weight 4
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(4)

    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.array(norms_sq, dtype=np.int64)
    n_candidates = len(candidates)

    # Compute dot products
    dots = np.dot(candidates, candidates.T)

    # Determine conflicts
    # dist_sq[i, j] = norms_sq[i] + norms_sq[j] - 2*dots[i, j]
    # Conflict if dist_sq[i, j] < 4 (max norm squared is 4)
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflicts = dist_sq < 4
    np.fill_diagonal(conflicts, False)

    # Precompute neighbors for each node
    neighbors = [set() for _ in range(n_candidates)]
    for i in range(n_candidates):
        neigh_indices = np.where(conflicts[i])[0]
        neighbors[i] = set(neigh_indices.tolist())

    def greedy_min_degree(seed):
        """
        Greedy selection picking the vertex with the minimum number of
        conflicts with the remaining available vertices.
        Uses random sampling to speed up the min-degree search.
        """
        rng = random.Random(seed)
        available = set(range(n_candidates))
        selected = []

        while available:
            if time.time() > deadline - 2.0:
                break

            # Find vertex with min degree in available subgraph
            # Sample if large
            if len(available) > 200:
                sample = rng.sample(list(available), 200)
            else:
                sample = list(available)

            best_v = None
            min_deg = float('inf')
            for v in sample:
                deg = len(neighbors[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v

            if best_v is None:
                break

            selected.append(best_v)
            available.discard(best_v)
            # Remove neighbors of best_v from available
            for n in neighbors[best_v]:
                if n in available:
                    available.discard(n)

        return selected

    best_selected = []

    # Run many trials with different seeds
    max_trials = 500
    for trial in range(max_trials):
        if time.time() > deadline - 5.0:
            break

        seed = trial * 7919 + 1  # deterministic seed sequence
        sel = greedy_min_degree(seed)
        if len(sel) > len(best_selected):
            best_selected = sel

    if not best_selected:
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)

    return candidates[best_selected]
# EVOLVE-BLOCK-END
