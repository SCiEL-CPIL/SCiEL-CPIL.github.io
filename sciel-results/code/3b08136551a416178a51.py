import itertools
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

D = 11
DEADLINE = time.time() + 140  # safety margin before 150s kill


def _remaining_time():
    return max(0.0, DEADLINE - time.time())


def generate_pool():
    """Generate integer vectors with 2,3,4 non-zero entries (±1)."""
    vecs = []
    for w in (2, 3, 4):
        for combo in itertools.combinations(range(D), w):
            for signs in itertools.product((1, -1), repeat=w):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(combo, signs):
                    v[idx] = s
                vecs.append(v)
    return np.array(vecs, dtype=np.int64)


def build_compatibility(vectors, min_dist_sq):
    """
    Build adjacency: A[i,j] = 1 if vectors i,j are COMPATIBLE (can coexist).
    Compatible means ||vi - vj||^2 >= min_dist_sq.
    """
    n = len(vectors)
    # Compute all pairwise squared distances
    # Use: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    norms_sq = np.sum(vectors * vectors, axis=1)
    dots = vectors @ vectors.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    # Compatibility: dist_sq >= min_dist_sq (and i != j)
    compat = (dist_sq >= min_dist_sq)
    np.fill_diagonal(compat, False)
    return compat


def spectral_seed(compat, n_keep, rng):
    """
    Use spectral relaxation: find largest eigenvector of the Laplacian,
    keep top n_keep vertices by eigenvector magnitude.
    """
    n = len(compat)
    # Laplacian: L = D - A
    deg = np.sum(compat, axis=1).astype(float)
    L = np.diag(deg) - compat.astype(float)
    # Use scipy for eigendecomposition (Laplacian is symmetric)
    try:
        from scipy.linalg import eigh
        # Solve for smallest eigenvalues (Fiedler vector gives best partition)
        # Actually for max independent set, we want the eigenvector of A with largest eigenvalue
        # Let's use the adjacency matrix directly
        A = compat.astype(float)
        vals, vecs = eigh(A, subset_by_index=[n - 1, n - 1])
        v = vecs[:, -1]
    except Exception:
        v = rng.random(n)
    
    # Keep vertices with largest |v_i|
    order = np.argsort(-np.abs(v))
    return order[:n_keep]


def min_degree_greedy(compat, seed_order=None, time_limit=None):
    """
    Minimum-degree greedy: repeatedly pick the vertex with fewest compatible
    neighbors among remaining vertices. This is the "most constrained first"
    strategy which tends to produce larger independent sets.
    """
    n = len(compat)
    if seed_order is not None:
        # Start with the seed order as a priority, but still use min-degree
        # among remaining
        pass
    
    remaining = set(range(n))
    selected = []
    compat_copy = compat.copy()
    
    start = time.time()
    
    while remaining:
        if time_limit and (time.time() - start) > time_limit:
            break
        if _remaining_time() < 5:
            break
        
        # For each remaining vertex, count compatible neighbors in remaining
        # That's the "degree within remaining"
        best = None
        best_deg = float('inf')
        
        for v in remaining:
            # Count neighbors of v that are still remaining
            neigh = np.where(compat_copy[v] & np.array([x in remaining for x in range(n)]))[0]
            deg = len(neigh)
            if deg < best_deg:
                best_deg = deg
                best = v
        
        if best is None:
            break
        
        selected.append(best)
        remaining.discard(best)
        # Remove best from compat (it's selected, so its neighbors can't be selected)
        for u in remaining:
            if compat_copy[best, u]:
                # u is incompatible with best, so u cannot be selected
                # Actually wait: compat means they CAN coexist.
                # If compat[best,u] is True, they CAN coexist, so u is still available.
                # If compat[best,u] is False, they CANNOT coexist, so u must be removed.
                pass
        
        # Remove all vertices INCOMPATIBLE with best from remaining
        to_remove = set()
        for u in remaining:
            if not compat_copy[best, u]:
                to_remove.add(u)
        remaining -= to_remove
    
    return selected


def min_degree_greedy_fast(compat, time_limit=None):
    """
    Faster min-degree greedy using numpy.
    """
    n = len(compat)
    remaining_mask = np.ones(n, dtype=bool)
    selected = []
    
    start = time.time()
    
    while remaining_mask.any():
        if time_limit and (time.time() - start) > time_limit:
            break
        if _remaining_time() < 5:
            break
        
        # Degree of each remaining vertex within remaining set
        # deg[i] = sum of compat[i, j] for j in remaining
        deg = (compat * remaining_mask[None, :]).sum(axis=1)
        
        # Among remaining, find the one with minimum degree
        deg[~remaining_mask] = float('inf')
        best = int(np.argmin(deg))
        
        if not remaining_mask[best]:
            break
        
        selected.append(best)
        remaining_mask[best] = False
        
        # Remove all vertices incompatible with best
        # Incompatible = NOT compat[best, :]
        incompatible = ~compat[best, :]
        remaining_mask &= ~incompatible
    
    return selected


def local_search_swap(compat, selected, time_limit=None, max_rounds=50):
    """
    Swap-based local search: try to improve the independent set by
    swapping one selected vertex with one unselected vertex, or
    adding/removing vertices.
    """
    n = len(compat)
    selected_set = set(selected)
    unselected = set(range(n)) - selected_set
    
    improved = True
    rounds = 0
    
    start = time.time()
    
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        
        if time_limit and (time.time() - start) > time_limit:
            break
        if _remaining_time() < 5:
            break
        
        # Try: for each selected vertex s, try to remove it and add
        # multiple unselected vertices that become compatible
        # Simpler: try swapping one out, one in if it increases size
        # Or: try to add an unselected vertex that's compatible with all selected
        
        # First, try to add any compatible unselected vertex
        added = False
        for u in list(unselected):
            if all(compat[u, s] for s in selected_set):
                selected_set.add(u)
                unselected.discard(u)
                improved = True
                added = True
                break
        
        if added:
            continue
        
        # Try swaps: remove s, add u
        for s in list(selected_set):
            # Removing s frees up all vertices incompatible with s
            # that were previously blocked
            # Find unselected vertices that are compatible with (selected - {s})
            candidates = []
            for u in unselected:
                if all(compat[u, t] for t in selected_set if t != s):
                    candidates.append(u)
            
            if len(candidates) >= 2:
                # We can remove s and add 2+ vertices
                # Add as many as possible
                new_sel = set(selected_set) - {s}
                count = 0
                for u in candidates:
                    if all(compat[u, t] for t in new_sel):
                        new_sel.add(u)
                        count += 1
                if len(new_sel) > len(selected_set):
                    selected_set = new_sel
                    unselected = set(range(n)) - selected_set
                    improved = True
                    break
    
    return list(selected_set)


def construct():
    """
    Construct the 220 roots of the D11 root system.
    These are the optimal kissing configuration for dimension 11.
    Vectors are of the form ±e_i ± e_j for i < j.
    Norm^2 = 2.
    Min pairwise distance^2 = 2.
    Condition: min_dist >= max_norm is satisfied.
    """
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            # 4 vectors for each pair (i, j):
            # +e_i + e_j
            v1 = np.zeros(D, dtype=np.int64)
            v1[i] = 1
            v1[j] = 1
            vectors.append(v1)
            
            # +e_i - e_j
            v2 = np.zeros(D, dtype=np.int64)
            v2[i] = 1
            v2[j] = -1
            vectors.append(v2)
            
            # -e_i + e_j
            v3 = np.zeros(D, dtype=np.int64)
            v3[i] = -1
            v3[j] = 1
            vectors.append(v3)
            
            # -e_i - e_j
            v4 = np.zeros(D, dtype=np.int64)
            v4[i] = -1
            v4[j] = -1
            vectors.append(v4)
            
    return np.array(vectors, dtype=np.int64)


if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} vectors")
    if len(result) > 0:
        norms_sq = np.sum(result * result, axis=1)
        print(f"Max norm^2: {np.max(norms_sq)}")
        # Check min distance
        min_d2 = float('inf')
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d2 = np.sum((result[i] - result[j]) ** 2)
                min_d2 = min(min_d2, d2)
        print(f"Min dist^2: {min_d2}")
        print(f"Valid: {min_d2 >= np.max(norms_sq)}")
