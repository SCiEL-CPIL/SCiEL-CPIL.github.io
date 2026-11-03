"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from itertools import combinations

# Deadline guard: evaluator kills at 150s; we target finishing by ~120s
_DEADLINE = 120.0


def _norm_sq(v):
    return int(np.dot(v, v))


def _min_pairwise_dist_sq(vecs):
    """Minimum squared pairwise distance among rows of vecs (int array)."""
    if len(vecs) < 2:
        return np.inf
    # Use NumPy for speed: compute all pairwise squared distances
    # To avoid O(n^2) memory blowup for large n, do chunked computation
    n = len(vecs)
    min_d2 = np.inf
    chunk = 512
    for i in range(0, n, chunk):
        block = vecs[i:i+chunk]
        # dist from block to all vectors
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        norms_block = np.sum(block * block, axis=1)  # (chunk,)
        norms_all = np.sum(vecs * vecs, axis=1)      # (n,)
        # cross terms
        cross = block @ vecs.T  # (chunk, n)
        d2 = norms_block[:, None] + norms_all[None, :] - 2 * cross
        # set diagonal to inf to ignore self-distance
        # but block rows correspond to indices i..i+chunk-1 in full matrix
        for k in range(block.shape[0]):
            idx = i + k
            d2[k, idx] = np.inf
        # only consider pairs where j > i to avoid double counting, but min is symmetric
        # actually we just need global min off-diagonal
        # mask out the self entries properly
        # The diagonal entries in d2 are at (k, i+k) for k in range(chunk)
        # We already set them. Now take min
        block_min = np.min(d2)
        if block_min < min_d2:
            min_d2 = block_min
    return min_d2


def _max_norm_sq(vecs):
    if len(vecs) == 0:
        return 0
    return int(np.max(np.sum(vecs * vecs, axis=1)))


def _is_valid(vecs):
    """Check if minimum pairwise distance >= max norm."""
    if len(vecs) < 2:
        return True
    min_d2 = _min_pairwise_dist_sq(vecs)
    max_n2 = _max_norm_sq(vecs)
    return min_d2 >= max_n2


def _candidate_pool(max_coord=3):
    """Generate all nonzero integer vectors in R^11 with coordinates in [-max_coord, max_coord]."""
    coords = list(range(-max_coord, max_coord + 1))
    # Exclude zero vector
    # For D=11, 7^11 is huge. We need a smarter approach.
    # Instead, let's generate vectors with small weight (number of nonzero entries)
    # Weight 1: 11 * 2 * max_coord = 11*2*3 = 66
    # Weight 2: C(11,2) * (2*max_coord)^2 = 55 * 36 = 1980
    # Weight 3: C(11,3) * (2*max_coord)^3 = 165 * 216 = 35640
    # Weight 4: C(11,4) * (2*max_coord)^4 = 330 * 1296 = 427680
    # Total for weight <= 3: ~37686 vectors. Manageable.
    # For weight 4: 427k vectors. Total ~465k. Still manageable for conflict graph if we're careful.
    # Let's start with weight <= 3 and max_coord=2 for speed, then expand.
    
    vectors = []
    # Weight 1
    for i in range(D):
        for c in range(1, max_coord + 1):
            for s in (1, -1):
                v = [0] * D
                v[i] = s * c
                vectors.append(tuple(v))
    # Weight 2
    for i, j in combinations(range(D), 2):
        for ci in range(1, max_coord + 1):
            for cj in range(1, max_coord + 1):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i] = si * ci
                        v[j] = sj * cj
                        vectors.append(tuple(v))
    # Weight 3
    for i, j, k in combinations(range(D), 3):
        for ci in range(1, max_coord + 1):
            for cj in range(1, max_coord + 1):
                for ck in range(1, max_coord + 1):
                    for si in (1, -1):
                        for sj in (1, -1):
                            for sk in (1, -1):
                                v = [0] * D
                                v[i] = si * ci
                                v[j] = sj * cj
                                v[k] = sk * ck
                                vectors.append(tuple(v))
    
    return np.array(vectors, dtype=np.int64)


def construct():
    start_time = time.time()
    
    # Start with the seed solution: all weight-2 vectors with coords ±1
    # That's C(11,2) * 4 = 55 * 4 = 220 vectors
    # These have norm^2 = 2, and pairwise distance^2 >= 2 (actually >= 2 for most, need to check)
    # Actually two weight-2 vectors sharing one coordinate: distance^2 = (1-1)^2 + (1-0)^2 + (0-1)^2 + ... 
    # Let's just verify the seed is valid.
    
    seed_vecs = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                seed_vecs.append(v)
    seed_arr = np.array(seed_vecs, dtype=np.int64)
    
    # Verify seed
    if not _is_valid(seed_arr):
        # Should not happen, but fallback
        return seed_arr
    
    best_vecs = seed_arr.copy()
    best_k = len(best_vecs)
    
    # Try to improve using GRASP-like approach
    # Build candidate pool: weight-1, weight-2, weight-3 vectors with small coords
    # We'll try to add vectors one by one, maintaining validity
    
    # Generate candidates sorted by norm (smaller norm first, easier to add)
    candidates = _candidate_pool(max_coord=2)
    
    # Remove candidates that are already in best_vecs (up to sign and permutation? No, exact match)
    # Actually, let's just try to add any candidate that maintains validity
    
    # For efficiency, precompute norms of candidates
    cand_norms = np.sum(candidates * candidates, axis=1)
    
    # Sort candidates by norm ascending
    order = np.argsort(cand_norms)
    candidates_sorted = candidates[order]
    
    # Try to greedily add candidates
    current = best_vecs.copy()
    current_set = set(map(tuple, current))
    
    for idx in range(len(candidates_sorted)):
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        cand = candidates_sorted[idx]
        cand_t = tuple(cand)
        if cand_t in current_set:
            continue
        
        # Check if adding cand maintains validity
        # New min distance must be >= new max norm
        # New max norm = max(old_max_norm, cand_norm)
        old_max_n2 = _max_norm_sq(current)
        new_max_n2 = max(old_max_n2, int(cand_norms[order[idx]]))
        
        # Need: min distance between cand and all current >= new_max_n2
        # AND: old min distance >= new_max_n2 (but old min distance >= old_max_n2, so if new_max_n2 <= old_max_n2, ok)
        # If new_max_n2 > old_max_n2, we need old min distance >= new_max_n2 too
        
        # Compute distance from cand to all current vectors
        # ||cand - v||^2 for all v in current
        diff = current - cand
        d2s = np.sum(diff * diff, axis=1)
        min_d_to_cand = np.min(d2s)
        
        # Also need old pairwise distances to be >= new_max_n2
        # If new_max_n2 <= old_max_n2, this is automatically satisfied
        # If new_max_n2 > old_max_n2, we need to check
        if new_max_n2 > old_max_n2:
            # Check if old min distance >= new_max_n2
            # This is expensive, but rare
            old_min_d2 = _min_pairwise_dist_sq(current)
            if old_min_d2 < new_max_n2:
                continue
        
        if min_d_to_cand >= new_max_n2:
            current = np.vstack([current, cand])
            current_set.add(cand_t)
            if len(current) > best_k:
                best_k = len(current)
                best_vecs = current.copy()
    
    # Try some randomized restarts / swaps if time permits
    if time.time() - start_time < _DEADLINE - 20:
        # Try to replace some vectors with others to allow more additions
        # Simple local search: try removing one vector and adding two others
        for trial in range(50):
            if time.time() - start_time > _DEADLINE - 10:
                break
            
            # Pick a random vector to remove
            if len(best_vecs) < 2:
                break
            rem_idx = random.randint(0, len(best_vecs) - 1)
            candidate_pool = [v for i, v in enumerate(best_vecs) if i != rem_idx]
            candidate_pool = np.array(candidate_pool, dtype=np.int64)
            
            # Try to add 2 candidates
            added = 0
            new_vecs = candidate_pool.copy()
            new_set = set(map(tuple, new_vecs))
            
            for cand in candidates_sorted:
                if time.time() - start_time > _DEADLINE - 10:
                    break
                cand_t = tuple(cand)
                if cand_t in new_set:
                    continue
                # Check validity
                old_max_n2 = _max_norm_sq(new_vecs)
                cand_n2 = int(np.sum(cand * cand))
                new_max_n2 = max(old_max_n2, cand_n2)
                
                diff = new_vecs - cand
                d2s = np.sum(diff * diff, axis=1)
                min_d = np.min(d2s) if len(new_vecs) > 0 else np.inf
                
                if new_max_n2 > old_max_n2:
                    old_min_d2 = _min_pairwise_dist_sq(new_vecs)
                    if old_min_d2 < new_max_n2:
                        continue
                
                if min_d >= new_max_n2:
                    new_vecs = np.vstack([new_vecs, cand])
                    new_set.add(cand_t)
                    added += 1
                    if added >= 2:
                        break
            
            if len(new_vecs) > len(best_vecs):
                best_vecs = new_vecs
                best_k = len(best_vecs)
    
    return best_vecs
# EVOLVE-BLOCK-END
