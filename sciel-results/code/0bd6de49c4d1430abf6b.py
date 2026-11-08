"""Initial program for the suite task; SCiEL evolves the EVOLVE-BBLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    # Baseline: 220 vectors of form (±1, ±1, 0, ..., 0), all norm sqrt(2)
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)

    arr = np.array(vectors, dtype=np.int64)

    # Current state
    norms_sq = np.sum(arr ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)  # 2 for baseline
    max_norm = np.sqrt(max_norm_sq)  # sqrt(2)

    def can_add_batch(candidates, arr, max_norm_sq):
        """Vectorized check: which candidates can be added?
        Returns mask of acceptable candidates."""
        n_cand = len(candidates)
        if n_cand == 0:
            return np.array([], dtype=bool)
        
        cand_arr = np.array(candidates, dtype=np.int64)
        cand_norms_sq = np.sum(cand_arr ** 2, axis=1)
        new_max_norm_sq = np.maximum(max_norm_sq, np.max(cand_norms_sq))
        
        # For each candidate, compute min squared distance to existing vectors
        # diffs shape: (n_cand, n_existing, D)
        # dists_sq shape: (n_cand, n_existing)
        diffs = cand_arr[:, np.newaxis, :] - arr[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        min_dists_sq = np.min(dists_sq, axis=1)
        
        # Constraint: min_dist >= new_max_norm, i.e., min_dists_sq >= new_max_norm_sq
        mask = min_dists_sq >= (new_max_norm_sq - 1e-9)
        return mask

    # Generate candidate vectors of various types (small norms preferred)
    candidates = []
    seen = set(map(tuple, arr.tolist()))

    # Type A: (±2, 0, ..., 0)  norm 2
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s * 2
            t = tuple(v)
            if t not in seen:
                candidates.append(v)
                seen.add(t)

    # Type B: (±1,±1,±1,±1,0^7)  norm 2
    for pos in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for p, s in zip(pos, signs):
                v[p] = s
            t = tuple(v)
            if t not in seen:
                candidates.append(v)
                seen.add(t)

    # Type C: (±1,±1,±1,±1,±1,0^6)  norm sqrt(5)
    for pos in itertools.combinations(range(D), 5):
        for signs in itertools.product([1, -1], repeat=5):
            v = [0] * D
            for p, s in zip(pos, signs):
                v[p] = s
            t = tuple(v)
            if t not in seen:
                candidates.append(v)
                seen.add(t)

    # Sort by norm ascending (keep max_norm as small as possible)
    candidates.sort(key=lambda v: sum(x * x for x in v))

    # Greedy add with multiple passes using deterministic ordering
    # Use batch checking for efficiency
    remaining = list(candidates)
    
    # Multiple passes to handle order-dependence
    for pass_num in range(10):
        if time.time() > deadline:
            break
        if not remaining:
            break
        
        # Check all remaining candidates in one batch
        mask = can_add_batch(remaining, arr, max_norm_sq)
        if not np.any(mask):
            break
        
        # Add all acceptable candidates
        new_vectors = [remaining[i] for i in range(len(remaining)) if mask[i]]
        if new_vectors:
            new_arr = np.array(new_vectors, dtype=np.int64)
            arr = np.vstack([arr, new_arr])
            new_norms_sq = np.sum(new_arr ** 2, axis=1)
            max_norm_sq = max(max_norm_sq, np.max(new_norms_sq))
            max_norm = np.sqrt(max_norm_sq)
        
        # Remove added candidates from remaining
        remaining = [remaining[i] for i in range(len(remaining)) if not mask[i]]
        
        # Check time after each pass
        if time.time() > deadline:
            break

    return arr
# EVOLVE-BLOCK-END
