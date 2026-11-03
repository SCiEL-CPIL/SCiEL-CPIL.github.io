"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time

D = 11


def _generate_vectors():
    """Generate all nonzero integer vectors with exactly 3 nonzero entries in {-1, 1}.
    Returns a 2D numpy array of shape (n, D).
    """
    vecs_list = []
    for idx in combinations(range(D), 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[idx[0]] = s1
                    v[idx[1]] = s2
                    v[idx[2]] = s3
                    vecs_list.append(v)
    return np.array(vecs_list, dtype=np.int64)


def _build_conflict_matrix(vecs, max_norm_sq):
    """
    Build a boolean conflict matrix.
    conflict[i, j] is True if ||v_i - v_j||^2 < max_norm_sq.
    We want to find the Maximum Independent Set (MIS) of this conflict graph,
    which corresponds to a set of vectors where all pairwise distances are >= sqrt(max_norm_sq).
    """
    n = len(vecs)
    # Compute dot products
    dots = vecs @ vecs.T
    # Compute squared norms
    norms_sq = np.sum(vecs**2, axis=1)
    # Compute squared distances: ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Conflict if distance is strictly less than the required minimum squared distance
    # The requirement is min pairwise distance >= max norm.
    # So we need ||v_i - v_j|| >= max_norm.
    # Squared: dist_sq >= max_norm_sq.
    # Conflict if dist_sq < max_norm_sq.
    conflict = dist_sq < max_norm_sq
    # No self-conflict
    np.fill_diagonal(conflict, False)
    return conflict


def _greedy_mis(conflict, n, order=None):
    """
    Greedy Maximum Independent Set.
    conflict: (n, n) boolean matrix where True means edge.
    Returns list of indices in the MIS.
    """
    if order is None:
        # Order by degree ascending (pick vertices with fewest conflicts first)
        degrees = conflict.sum(axis=1)
        order = np.argsort(degrees)
        
    selected = []
    removed = np.zeros(n, dtype=bool)
    
    for v in order:
        if removed[v]:
            continue
        # Select v
        selected.append(v)
        # Remove v and all its neighbors from consideration
        # neighbors are those where conflict[v, u] is True
        # Also remove v itself
        removed[v] = True
        # Vectorized update: mark all u such that conflict[v, u] is True as removed
        removed[conflict[v]] = True
        
    return selected


def _local_search_improve(selected, conflict, n, iterations=50):
    """
    Try to improve the greedy solution by looking for vertices not in selected
    that can be added.
    """
    selected_set = set(selected)
    # Create a boolean mask for selected
    sel_mask = np.zeros(n, dtype=bool)
    for v in selected:
        sel_mask[v] = True
    
    # Vertices not selected
    candidates = np.where(~sel_mask)[0]
    
    improved = True
    while improved and iterations > 0:
        improved = False
        iterations -= 1
        
        # Try to add any candidate that doesn't conflict with any selected vertex
        # For each candidate c, check if conflict[c, selected] has any True.
        # We can check this efficiently:
        # If conflict[c, sel_mask].any() is False, we can add c.
        
        # To speed up, we can check candidates in random order or degree order.
        # Let's just check all candidates.
        for c in candidates:
            # Check if c conflicts with any selected vertex
            if not conflict[c, sel_mask].any():
                # Add c
                selected.append(c)
                sel_mask[c] = True
                improved = True
                # We don't need to remove c from candidates immediately as the mask check
                # will prevent re-adding it or adding its neighbors in subsequent iterations.
                # However, iterating over the full candidates list every time is O(N^2).
                # For N=1320, N^2 is ~1.7M, which is fast enough for a few iterations.
                pass
    
    return selected


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Generate candidate vectors with 3 nonzero entries
    vecs = _generate_vectors()
    n = len(vecs)
    
    # All vectors have norm^2 = 3.
    # Max norm is sqrt(3).
    # We need pairwise distance >= sqrt(3).
    # So squared distance >= 3.
    max_norm_sq = 3
    
    # Build conflict graph
    conflict = _build_conflict_matrix(vecs, max_norm_sq)
    
    # Greedy MIS
    # Order by degree ascending
    degrees = conflict.sum(axis=1)
    order = np.argsort(degrees)
    
    selected = _greedy_mis(conflict, n, order)
    
    # Check time
    if time.time() < deadline - 5:
        # Try local search improvement
        try:
            selected = _local_search_improve(selected, conflict, n, iterations=20)
        except Exception:
            pass
    
    # Convert selected indices to the actual vectors
    if len(selected) == 0:
        # Fallback: return empty or a single vector if something went wrong
        # But greedy should always find at least 1.
        return vecs[:1]
        
    result = vecs[selected]
    return result
# EVOLVE-BLOCK-END
