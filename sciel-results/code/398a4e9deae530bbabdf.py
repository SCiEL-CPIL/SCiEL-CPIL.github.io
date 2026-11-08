"""Improved program for kissing number in dimension 11."""
import time
import numpy as np

D = 11
DEADLINE_MARGIN = 5.0  # seconds of safety margin before the 150s kill


def _build_candidates(weight, limit=None):
    """Generate all nonzero integer vectors with given L1 weight (sum of abs values).
    
    For weight w, we consider vectors where entries are in {-1, 0, 1} and
    exactly w entries are nonzero. This gives C(D, w) * 2^w candidates.
    """
    from itertools import combinations, product
    
    candidates = []
    for support in combinations(range(D), weight):
        for signs in product([1, -1], repeat=weight):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            if limit is not None and len(candidates) >= limit:
                return candidates
    return candidates


def _max_norm_sq(candidates):
    """Compute the maximum squared norm among candidates."""
    if len(candidates) == 0:
        return 0
    norms_sq = np.array([np.dot(c, c) for c in candidates])
    return int(np.max(norms_sq))


def _pairwise_dist_sq(candidates):
    """Compute pairwise squared distances as a matrix."""
    n = len(candidates)
    if n == 0:
        return np.array([])
    arr = np.array(candidates, dtype=np.float64)
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    norms_sq = np.sum(arr * arr, axis=1)
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2.0 * (arr @ arr.T)
    return dist_sq


def _greedy_independent_set(candidates, min_dist_sq, time_budget=None):
    """Greedy selection: pick vectors that maintain min pairwise distance >= min_dist_sq.
    
    Uses a deterministic greedy approach with random permutations for diversity.
    """
    n = len(candidates)
    if n == 0:
        return []
    
    best_selected = []
    best_count = 0
    
    # Try multiple random orderings
    num_tries = 10
    rng = np.random.default_rng(42)
    
    for trial in range(num_tries):
        if time_budget is not None and time.time() - time_budget > 0:
            break
        
        order = rng.permutation(n)
        selected = []
        selected_indices = []
        
        for idx in order:
            if time_budget is not None and time.time() - time_budget > 0:
                break
            
            v = candidates[idx]
            # Check distance to all already-selected vectors
            ok = True
            for sidx in selected_indices:
                sv = candidates[sidx]
                diff = v - sv
                d_sq = int(np.dot(diff, diff))
                if d_sq < min_dist_sq:
                    ok = False
                    break
            
            if ok:
                selected.append(idx)
                selected_indices.append(idx)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected
    
    return best_selected


def construct():
    start_time = time.time()
    
    # Strategy: try different weights and use greedy independent set
    # The constraint is: min pairwise dist >= max norm
    # For vectors with entries in {-1, 0, 1} and weight w:
    #   norm^2 = w
    #   pairwise dist^2 = w + w - 2*(dot product) = 2w - 2*dot
    #   We need 2w - 2*dot >= w  =>  w >= 2*dot
    #   dot product is sum of products of corresponding entries
    #   For two vectors with supports of size w, dot can range from -w to w
    #   We need dot <= w/2
    
    # For weight 2: norm^2 = 2, need dist^2 >= 2 => 4 - 2*dot >= 2 => dot <= 1
    #   Two weight-2 vectors: dot can be -2,-1,0,1,2
    #   dot=2 means same support and same signs (identical vectors, excluded)
    #   dot=1 means they share one position with same sign
    #   So we need to avoid pairs sharing a position with same sign
    #   This is equivalent to: no two selected vectors have the same (position, sign)
    #   Actually more nuanced. Let's just use the greedy approach.
    
    # Weight 2 gives C(11,2)*4 = 55*4 = 220 candidates
    # Weight 3 gives C(11,3)*8 = 165*8 = 1320 candidates
    # Weight 4 gives C(11,4)*16 = 330*16 = 5280 candidates
    
    # Let's try weight 2 first (fast), then weight 3, then weight 4
    
    best_vectors = []
    best_k = 0
    
    for weight in [2, 3, 4]:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_MARGIN:
            break
        
        candidates = _build_candidates(weight)
        max_norm_sq = _max_norm_sq(candidates)
        
        # For weight w with entries in {-1,0,1}, norm^2 = w
        # min pairwise dist^2 must be >= w
        # So min_dist_sq = w
        
        min_dist_sq = max_norm_sq  # which equals weight for these candidates
        
        # Use greedy independent set
        selected = _greedy_independent_set(candidates, min_dist_sq, 
                                           time_budget=start_time + 150.0 - DEADLINE_MARGIN)
        
        k = len(selected)
        if k > best_k:
            best_k = k
            best_vectors = [candidates[i] for i in selected]
    
    # If we have time, try a local improvement: try to add more vectors
    # by checking if any non-selected candidate can be added
    elapsed = time.time() - start_time
    if elapsed < 150.0 - DEADLINE_MARGIN and best_k > 0:
        # Build a set of selected vectors for quick lookup
        selected_set = set(tuple(v) for v in best_vectors)
        
        # Try each weight separately and attempt to add vectors
        for weight in [2, 3, 4]:
            if time.time() - start_time > 150.0 - DEADLINE_MARGIN:
                break
            
            candidates = _build_candidates(weight)
            max_norm_sq = _max_norm_sq(candidates)
            min_dist_sq = max_norm_sq
            
            # Try to add any candidate that's not already selected and satisfies distance
            new_vectors = []
            for c in candidates:
                if tuple(c) in selected_set:
                    continue
                # Check against all current best_vectors
                ok = True
                for bv in best_vectors:
                    diff = c - bv
                    d_sq = int(np.dot(diff, diff))
                    if d_sq < min_dist_sq:
                        ok = False
                        break
                if ok:
                    new_vectors.append(c)
            
            if new_vectors:
                best_vectors.extend(new_vectors)
                best_k += len(new_vectors)
    
    if best_k == 0:
        # Fallback: return at least the seed program's result
        import itertools
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(best_vectors, dtype=np.int64)
