import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def priority(adjacency):
    """
    Assigns a priority score to each vertex for a greedy decoder that picks
    vertices in descending score order, skipping neighbors of already picked vertices.
    
    Uses BFS-based "uncontested territory" scoring:
    - For each vertex, simulate greedy selection with that vertex first
    - Count how many vertices become "contested" (blocked) due to early selection
    - Vertices with less contested territory get higher priority
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    A = np.asarray(adjacency)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Precompute features
    degree = A.sum(axis=1).astype(float)
    
    # Helper: Fast Greedy Decode with early termination for territory counting
    def greedy_decode_with_territory(scores, target_vertex=None):
        """
        Simulates greedy decoder. If target_vertex is given, counts how many
        vertices are blocked by its early selection (territory lost).
        Returns (independent_set_size, territory_lost)
        """
        order = np.argsort(-scores)
        
        # If target_vertex specified, force it to be first
        if target_vertex is not None:
            order = np.concatenate([[target_vertex], order[order != target_vertex]])
        
        selected = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)  # Vertices blocked due to selection
        count = 0
        territory_lost = 0
        
        for v in order:
            if selected[v] or blocked[v]:
                continue
            
            # Check if v has any selected neighbor
            if np.dot(selected, A[v]):
                blocked[v] = True
                continue
            
            selected[v] = True
            count += 1
            
            # Mark neighbors as blocked (they can't be selected)
            neighbors = A[v]
            for u in np.where(neighbors)[0]:
                if not selected[u] and not blocked[u]:
                    blocked[u] = True
                    if target_vertex is not None and v == target_vertex:
                        territory_lost += 1
        
        return count, territory_lost
    
    # Candidate Strategies
    candidates = []
    
    # 1. Pure Degree (baseline)
    candidates.append(-degree)
    
    # 2. BFS-based "uncontested territory" score
    # For each vertex, estimate how much "territory" it blocks if selected first
    # Lower territory_lost = better candidate (less damage to potential IS)
    territory_scores = np.zeros(n)
    for v in range(n):
        if time.time() > deadline - 20.0:
            break
        # Create a score vector where v has highest priority
        test_scores = -degree.copy()
        test_scores[v] = 1e10  # Force v to be selected first
        
        _, territory_lost = greedy_decode_with_territory(test_scores, target_vertex=v)
        territory_scores[v] = -territory_lost  # Lower territory lost = higher score
    
    candidates.append(territory_scores)
    
    # 3. Hybrid: combine degree and territory
    candidates.append(-degree + 0.5 * territory_scores)
    candidates.append(-degree + 1.0 * territory_scores)
    candidates.append(-degree + 0.2 * territory_scores)
    
    # 4. Spectral features (keep from original)
    try:
        L = np.diag(degree) - A
        if n > 10:
            vals, vecs = eigsh(L, k=2, which='SM')
            idx = np.argsort(vals)
            fiedler = vecs[:, idx[1]]
        else:
            eigvals, eigvecs = np.linalg.eigh(L)
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
        
        f_min = fiedler.min()
        f_max = fiedler.max()
        if f_max - f_min > 1e-9:
            fiedler_norm = (fiedler - f_min) / (f_max - f_min)
        else:
            fiedler_norm = np.zeros(n)
        
        # Add spectral candidates
        candidates.append(-degree + 0.5 * fiedler_norm)
        candidates.append(-degree - 0.5 * fiedler_norm)
        candidates.append(territory_scores + 0.5 * fiedler_norm)
        
    except Exception:
        pass
    
    # 5. Mean neighbor degree
    neighbor_deg_sum = A @ degree
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbor_deg = np.where(degree > 0, neighbor_deg_sum / degree, 0.0)
    
    candidates.append(-(degree + 0.5 * mean_neighbor_deg))
    candidates.append(territory_scores - 0.5 * mean_neighbor_deg)
    
    # Helper: Fast Greedy Decode for evaluation
    def greedy_decode_size(scores):
        """
        Simulates the greedy decoder.
        Returns the size of the independent set.
        """
        order = np.argsort(-scores)
        
        selected = np.zeros(n, dtype=bool)
        count = 0
        
        for v in order:
            if not np.dot(selected, A[v]):
                selected[v] = True
                count += 1
        
        return count
    
    # Evaluate candidates
    best_size = -1
    best_scores = candidates[0]
    
    for s in candidates:
        if time.time() > deadline - 10.0:
            break
        # Ensure finite
        if np.any(~np.isfinite(s)):
            s = np.where(np.isfinite(s), s, 0.0)
        
        size = greedy_decode_size(s)
        if size > best_size:
            best_size = size
            best_scores = s
    
    return best_scores

# EVOLVE-BLOCK-END
