"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _start_time

def construct():
    global _start_time, _DEADLINE
    _start_time = time.time()
    _DEADLINE = _start_time + 140.0
    
    # Start with the base 220 vectors
    base_vectors = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                base_vectors.append(v)
    
    vectors = [np.array(v, dtype=np.int64) for v in base_vectors]
    
    def max_norm():
        return np.max(np.linalg.norm(vectors, axis=1)) if vectors else 0
    
    def min_pairwise_dist():
        if len(vectors) < 2:
            return float('inf')
        n = len(vectors)
        arr = np.array(vectors)
        # Compute all pairwise distances
        # Use broadcasting for efficiency
        # For large n, this might be memory intensive
        # n=220 -> 220*220 = 48400, fine
        # n=500 -> 250000, still fine
        # n=1000 -> 1000000, fine
        diff = arr[:, np.newaxis, :] - arr[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, float('inf'))
        return np.min(dists)
    
    def is_valid(new_v):
        """Check if adding new_v maintains the constraint."""
        if not vectors:
            return True
        arr = np.array(vectors)
        new_v = np.array(new_v, dtype=np.float64)
        # Distance from new_v to all existing
        diffs = arr - new_v[np.newaxis, :]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        min_d = np.min(dists)
        # New max norm
        new_norm = np.linalg.norm(new_v)
        cur_max = max_norm()
        new_max = max(cur_max, new_norm)
        # Need min_pairwise_dist >= new_max
        # Current min pairwise dist among existing is min_d_to_existing (but we need full min)
        # Actually we need: min(existing_min_pairwise, min_d_to_new) >= new_max
        # We don't have existing_min_pairwise cached, so let's compute it
        # For efficiency, let's just check the new distances and assume existing is valid
        # Since we only add valid vectors, existing should be valid
        # So we need min_d_to_new >= new_max
        return min_d >= new_max - 1e-9
    
    # Try to add more vectors
    # Candidate pool: vectors with entries in {-2,-1,0,1,2} with small L2 norm
    # Focus on vectors with norm <= current max norm (which is 2)
    # So we want vectors with norm <= 2, i.e., sum of squares <= 4
    
    candidates = []
    # Generate candidates with entries in {-1, 0, 1, 2}
    # But limit to those with norm^2 <= 4 (so norm <= 2)
    # This is a large space, so let's be smart
    
    # First, try adding unit vectors and their negatives
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = sign
            candidates.append(v)
    
    # Try vectors with two non-zero entries of value 1 (already in base)
    # Try vectors with one entry of 2
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = 2 * sign
            candidates.append(v)
    
    # Try vectors with three non-zero entries, each ±1, with norm^2 = 3 <= 4
    for idxs in combinations(range(D), 3):
        for signs in product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
    
    # Try vectors with four non-zero entries, each ±1, norm^2 = 4
    for idxs in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
    
    # Remove duplicates
    seen = set()
    unique_candidates = []
    for v in candidates:
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(v)
    
    # Try to add candidates greedily
    added = 0
    for v in unique_candidates:
        if _elapsed() > _DEADLINE - 5:
            break
        if is_valid(v):
            vectors.append(np.array(v, dtype=np.int64))
            added += 1
    
    # Local search: try to improve by replacing or adding
    # For now, return what we have
    if not vectors:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
