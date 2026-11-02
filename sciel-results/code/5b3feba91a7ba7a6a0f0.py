"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial.distance import cdist

D = 11

# EVOLVE-BLOCK-START

def _min_pairwise_dist_sq(vectors):
    """Compute minimum pairwise squared distance for a list/array of vectors."""
    if len(vectors) < 2:
        return float('inf')
    v = np.array(vectors, dtype=np.float64)
    dists = cdist(v, v)
    # Zero out diagonal
    np.fill_diagonal(dists, float('inf'))
    return float(np.min(dists))

def _max_norm_sq(vectors):
    """Compute maximum squared norm."""
    if not vectors:
        return 0.0
    v = np.array(vectors, dtype=np.float64)
    return float(np.max(np.sum(v * v, axis=1)))

def _is_valid(vectors):
    """Check if vectors satisfy min pairwise dist >= max norm."""
    if not vectors:
        return False
    min_d = _min_pairwise_dist_sq(vectors)
    max_n = _max_norm_sq(vectors)
    return min_d >= max_n - 1e-9

def _greedy_add(vectors, candidates, max_time=10.0):
    """
    Greedily add candidates to vectors, maintaining the invariant that
    min pairwise distance >= max norm.
    """
    start = time.time()
    result = list(vectors)
    if not result:
        # No constraint on first vector
        for c in candidates:
            result.append(c)
            break
        return result
    
    v_arr = np.array(result, dtype=np.float64)
    current_max_norm = float(np.max(np.sum(v_arr * v_arr, axis=1)))
    
    # Precompute pairwise distances from existing vectors to candidates
    # For efficiency, we'll check candidates one by one
    
    # Sort candidates by norm (prefer smaller norms first to keep max norm low)
    norms = np.array([float(np.sum(c*c)) for c in candidates])
    order = np.argsort(norms)
    
    for idx in order:
        if time.time() - start > max_time:
            break
        c = candidates[idx]
        c_norm_sq = float(np.sum(c*c))
        
        # New max norm would be max(current_max_norm, c_norm_sq)
        new_max_norm = max(current_max_norm, c_norm_sq)
        
        # Check if c is far enough from all existing vectors
        # We need dist(c, v_i) >= new_max_norm for all i
        if not result:
            result.append(c)
            current_max_norm = new_max_norm
            continue
        
        v_arr = np.array(result, dtype=np.float64)
        diffs = v_arr - c
        dists_sq = np.sum(diffs * diffs, axis=1)
        min_dist_sq = float(np.min(dists_sq))
        
        if min_dist_sq >= new_max_norm - 1e-9:
            result.append(c)
            current_max_norm = new_max_norm
    
    return result

def _generate_candidates():
    """Generate a diverse set of candidate integer vectors in R^11."""
    candidates = []
    
    # Unit vectors
    for i in range(D):
        for s in (1, -1):
            v = [0]*D
            v[i] = s
            candidates.append(v)
    
    # Pairs of unit vectors (e_i +/- e_j)
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0]*D
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Triples of unit vectors (e_i +/- e_j +/- e_k)
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0]*D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
    
    # Some vectors with a 2 in one coordinate
    for i in range(D):
        for s in (1, -1):
            v = [0]*D
            v[i] = 2*s
            candidates.append(v)
    
    # Vectors with 2 and 1
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0]*D
                    v[i] = 2*si
                    v[j] = sj
                    candidates.append(v)
    
    # Simplex-like: all +1
    candidates.append([1]*D)
    candidates.append([-1]*D)
    
    # Vectors with one -1 and rest +1, and variants
    for i in range(D):
        v = [1]*D
        v[i] = -1
        candidates.append(v)
        v2 = [-1]*D
        v2[i] = 1
        candidates.append(v2)
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    return unique_candidates

def _local_search(vectors, max_time=10.0):
    """
    Try to improve the configuration by removing and re-adding vectors,
    or by trying to add more vectors from a pool.
    """
    start = time.time()
    result = list(vectors)
    
    # Try to add more vectors from a broader candidate pool
    candidates = _generate_candidates()
    
    # Also try some random vectors
    rng = np.random.RandomState(42)
    for _ in range(200):
        v = rng.randint(-2, 3, size=D)
        if np.any(v != 0):
            candidates.append(v.tolist())
    
    result = _greedy_add(result, candidates, max_time=max_time)
    
    return result

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    """
    total_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Generate candidates
    candidates = _generate_candidates()
    
    # Greedy construction
    vectors = _greedy_add([], candidates, max_time=30.0)
    
    # Local search / refinement
    if time.time() - total_start < deadline - 20:
        vectors = _local_search(vectors, max_time=20.0)
    
    # Additional greedy pass with more candidates
    if time.time() - total_start < deadline - 10:
        # Generate more diverse candidates
        rng = np.random.RandomState(123)
        extra_candidates = []
        for _ in range(1000):
            v = rng.randint(-3, 4, size=D)
            if np.any(v != 0):
                extra_candidates.append(v.tolist())
        
        vectors = _greedy_add(vectors, extra_candidates, max_time=10.0)
    
    # Final validation
    if not _is_valid(vectors):
        # Fallback: just return unit vectors which are always valid
        vectors = []
        for i in range(D):
            v = [0]*D
            v[i] = 1
            vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
