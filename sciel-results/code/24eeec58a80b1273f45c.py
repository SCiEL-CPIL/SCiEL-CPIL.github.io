"""Kissing number in dimension 11 - Spectral Clustering and Local Search."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.csgraph as csgraph

def _build_candidates():
    """Generate candidate vectors: weight-2 (264) and weight-11 all-sign (2048)."""
    cands = []
    # Weight-2 vectors: two nonzero ±1 entries
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                cands.append(v)
    # Weight-11 vectors: all entries ±1
    for mask in range(1 << D):
        v = []
        for i in range(D):
            v.append(1 if (mask >> i) & 1 else -1)
        cands.append(v)
    return cands

def _build_compat_matrix(cands):
    """Build sparse adjacency matrix of the compatibility graph."""
    n = len(cands)
    norms = np.array([sum(x*x for x in v) for v in cands])
    
    # Use numpy for vectorized distance calculation to speed up construction
    # cands is a list of lists, convert to array
    cands_arr = np.array(cands, dtype=np.int64)
    
    # Calculate all pairwise squared distances
    # d2[i, j] = ||cands[i] - cands[j]||^2
    # (a-b)^2 = a^2 - 2ab + b^2
    # d2 = norms[:, None] + norms[None, :] - 2 * (cands_arr @ cands_arr.T)
    
    # This creates an NxN matrix. N ~ 2300. 2300^2 ~ 5M. 
    # 5M * 8 bytes ~ 40MB. Acceptable.
    dot_prod = cands_arr @ cands_arr.T
    d2 = norms[:, None] + norms[None, :] - 2 * dot_prod
    
    # Compatibility condition: d2[i, j] >= max(norms[i], norms[j])
    max_norms = np.maximum(norms[:, None], norms[None, :])
    compat_mask = d2 >= max_norms
    
    # We only need upper triangle for the graph, but let's keep it symmetric for Laplacian
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(compat_mask, False)
    
    # Convert to sparse matrix
    # indices of True
    rows, cols = np.where(compat_mask)
    data = np.ones(len(rows))
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    return A, norms

def _kmeans(X, K, n_iters=10, seed=42):
    """Simple K-Means implementation."""
    rng = np.random.RandomState(seed)
    n, d = X.shape
    # Initialize centers randomly
    centers = X[rng.choice(n, K, replace=False)]
    
    for _ in range(n_iters):
        # Assign clusters
        # dists: (n, K)
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        
        # Update centers
        new_centers = np.zeros_like(centers)
        for k in range(K):
            mask = labels == k
            if np.sum(mask) > 0:
                new_centers[k] = np.mean(X[mask], axis=0)
            else:
                # If empty, keep old center or reassign randomly
                new_centers[k] = centers[k]
        
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
        
    return labels

def _greedy_clique(sub_indices, A, norms):
    """
    Find a large clique in the subgraph induced by sub_indices.
    Uses greedy degree-based selection.
    """
    if not sub_indices:
        return []
    
    # Map sub_indices to local indices 0..m-1
    m = len(sub_indices)
    # Get subgraph adjacency
    # A is global sparse. We need A[sub_indices, :][:, sub_indices]
    sub_A = A[sub_indices, :][:, sub_indices]
    
    # Degrees in subgraph
    degrees = sub_A.sum(axis=1).A1
    
    # Greedy: pick vertex with highest degree, remove it and its neighbors
    selected = []
    remaining = list(range(m))
    remaining_set = set(remaining)
    
    # Precompute neighbors for fast removal? 
    # Or just iterate.
    # Since m is small (cluster size), this is fast.
    
    while remaining:
        # Find vertex with max degree in remaining subgraph
        # Note: degrees change as we remove vertices.
        # For simplicity, use original degrees or recompute?
        # Recomputing is O(m^2). m is ~100-200. OK.
        
        best_idx = -1
        best_deg = -1
        for i in remaining:
            # Count neighbors in remaining
            # sub_A[i, remaining]
            # This is slow if done in python loop.
            # Let's just use the current degree approximation or recompute sparse row sum.
            # Actually, let's just sort by initial degree and pick the first one that is still valid?
            # Greedy with static degree ordering is a common heuristic.
            pass
        
        # Better: Sort sub_indices by degree once, then iterate.
        # This is "Greedy Coloring" style, but for Clique.
        # Order: High degree first.
        order = np.argsort(-degrees)
        
        # Now iterate through order
        # We need to check if vertex is compatible with all previously selected.
        # This is O(K^2) where K is clique size.
        
        selected = []
        for i in order:
            if i not in remaining_set:
                continue
            # Check compatibility with all in selected
            ok = True
            for s in selected:
                if not sub_A[i, s]:
                    ok = False
                    break
            if ok:
                selected.append(i)
                # Remove i and its neighbors from remaining?
                # For Clique, we don't remove neighbors, we just can't select them if they conflict.
                # Actually, if we select i, we can still select j if j is compatible with i.
                # So we don't remove neighbors. We just add i.
                # The "remaining" logic above was for Independent Set in Conflict Graph.
                # Here we are finding Clique in Compatibility Graph.
                # So we just pick vertices that are compatible with all picked so far.
                # This is the same as the original greedy!
                # But the *order* is determined by spectral clustering + degree.
        
        # Map back to global indices
        global_selected = [sub_indices[i] for i in selected]
        return global_selected

def _local_search(clique, A, norms, n_iters=50):
    """
    Refine clique by swapping vertices.
    Try to add a vertex v not in clique. If it conflicts with some vertices in clique,
    try to remove one of the conflicting vertices and add v if it leads to a larger clique.
    """
    current_clique = set(clique)
    current_size = len(current_clique)
    
    # Get all vertices not in clique
    all_vertices = set(range(A.shape[0]))
    candidates = list(all_vertices - current_clique)
    
    for _ in range(n_iters):
        improved = False
        # Shuffle candidates for diversity
        random.shuffle(candidates)
        
        for v in candidates:
            # Check if v is compatible with all in current_clique
            # If yes, add it (size increases)
            # If no, find conflicts.
            
            conflicts = []
            for u in current_clique:
                if not A[v, u]:
                    conflicts.append(u)
            
            if not conflicts:
                # Can add v directly
                current_clique.add(v)
                current_size += 1
                candidates.remove(v)
                improved = True
                # Continue searching for more additions
                continue
            
            # If conflicts, try swapping
            # For each conflict u, try removing u and adding v
            # This is valid only if v is compatible with all others in (current_clique - {u})
            # And ideally, we don't want to lose size, so we just swap.
            # But maybe removing u allows adding multiple new vertices?
            # Simple swap: remove u, add v. Size stays same.
            # We want to increase size.
            # So we only do this if we can then add another vertex?
            # For simplicity, let's just try to find a vertex w that is compatible with (current_clique - {u}) U {v}.
            
            # This is getting complex. Let's stick to simple addition if possible.
            # If no simple addition, try a random swap to escape local minima?
            # Actually, the greedy step already found a local maximum.
            # Let's try to replace a low-degree vertex in the clique with a high-degree vertex outside.
            
            if not improved and random.random() < 0.1:
                # Random swap
                u = random.choice(list(current_clique))
                # Remove u
                current_clique.remove(u)
                # Try to add v
                ok = True
                for w in current_clique:
                    if not A[v, w]:
                        ok = False
                        break
                if ok:
                    current_clique.add(v)
                    candidates.remove(v)
                    improved = True
                    break
                else:
                    # Revert
                    current_clique.add(u)
        
        if not improved:
            break
            
    return list(current_clique)

def construct():
    deadline = time.time() + 110
    random.seed(12345)

    cands = _build_candidates()
    n = len(cands)
    
    # Build compatibility matrix
    A, norms = _build_compat_matrix(cands)
    
    # Spectral Clustering
    # Compute Laplacian
    L = csgraph.laplacian(A, normed=False)
    
    # Get smallest eigenvalues/vectors
    # k=5
    try:
        eigenvalues, eigenvectors = spla.eigsh(L, k=5, which='SM')
        # eigenvectors are columns
        X = eigenvectors.T # (5, n) -> (n, 5)
    except Exception:
        # Fallback to random embedding if eigsh fails
        X = np.random.rand(n, 5)
        
    # K-Means
    K = 20
    labels = _kmeans(X, K)
    
    best_clique = []
    
    # Process each cluster
    for k in range(K):
        if time.time() > deadline:
            break
            
        cluster_indices = np.where(labels == k)[0]
        if len(cluster_indices) < 2:
            continue
            
        # Find clique in this cluster
        clique = _greedy_clique(cluster_indices, A, norms)
        
        if len(clique) > len(best_clique):
            best_clique = clique[:]
            
    # Local Search on best clique
    if best_clique:
        best_clique = _local_search(best_clique, A, norms, n_iters=100)
        
    # Fallback if no clique found
    if not best_clique:
        # Pick one random vector
        best_clique = [0]
        
    result = np.array([cands[i] for i in best_clique], dtype=np.int64)
    return result
# EVOLVE-BLOCK-END
