import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def priority(adjacency):
    """
    Assigns a priority score to each vertex for a greedy decoder that picks
    vertices in descending score order, skipping neighbors of already picked vertices.
    
    Strategy:
    1. Compute local structural features (degree, neighbor degree stats).
    2. Compute spectral features (Fiedler vector) for global structure.
    3. Generate a diverse pool of candidate scoring vectors.
    4. Simulate the greedy decoder for each candidate to find the one that yields
       the largest independent set on the *current* graph.
    5. Return the winning score vector.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin for safety
    
    A = np.asarray(adjacency)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Ensure symmetric
    A = (A + A.T) / 2.0
    A = (A > 0.5).astype(float)
    
    # Precompute features
    degree = A.sum(axis=1).astype(float)
    
    # Helper: Fast Greedy Decode
    def greedy_decode_size(scores):
        """
        Simulates the greedy decoder.
        Returns the size of the independent set.
        """
        order = np.argsort(-scores)
        selected = np.zeros(n, dtype=bool)
        count = 0
        
        for v in order:
            # Check if v is adjacent to any selected vertex
            # np.dot(selected, A[v]) is fast enough for n=200
            if not np.any(selected & A[v]):
                selected[v] = True
                count += 1
                
        return count

    candidates = []
    
    # 1. Basic Degree-based scores
    candidates.append(-degree)
    
    # 2. Neighbor Degree Statistics
    neighbor_deg_sum = A @ degree
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbor_deg = np.where(degree > 0, neighbor_deg_sum / degree, 0.0)
    
    # Score: -(degree + alpha * mean_neighbor_deg)
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        candidates.append(-(degree + alpha * mean_neighbor_deg))
        
    # Score: -(degree * (1 + mean_neighbor_deg))
    candidates.append(-(degree * (1 + mean_neighbor_deg)))
    
    # Score: -degree / (1 + mean_neighbor_deg)
    with np.errstate(divide='ignore', invalid='ignore'):
        strat_div = -(degree / (1 + mean_neighbor_deg))
        strat_div[~np.isfinite(strat_div)] = 0
    candidates.append(strat_div)
    
    # 3. Spectral Features (Fiedler Vector)
    try:
        L = np.diag(degree) - A
        if n > 10:
            # eigsh is faster for large n, but for n=200 eigh is fine and robust
            # Let's use eigh for reliability, it's O(n^3) ~ 8M ops, very fast
            eigvals, eigvecs = np.linalg.eigh(L)
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
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
        
        # Combine spectral with degree
        # Fiedler vector components indicate "side" of the cut.
        # High degree + high fiedler value might be "bad" or "good" depending on sign.
        # We try both signs.
        for beta in [0.5, 1.0, 2.0]:
            candidates.append(-degree + beta * fiedler_norm)
            candidates.append(-degree - beta * fiedler_norm)
            
    except Exception:
        pass
    
    # 4. "Territory" or "Impact" Heuristic
    # Idea: If we pick vertex v, we remove its neighbors. 
    # How "valuable" are those neighbors?
    # If neighbors have high degree, picking v is "expensive" because we lose high-degree nodes
    # that might have been hard to place anyway? Or does it free up low-degree nodes?
    
    # Let's try: Score = - (degree + sum_of_neighbor_degrees)
    candidates.append(-(degree + neighbor_deg_sum))
    
    # 5. Randomized Restarts / Perturbations
    # If we have time, add some random perturbations to break ties or escape local minima
    # in the meta-search space of scorings?
    # Actually, the meta-search is over the *candidates* list.
    # Let's add a few random linear combinations of the features we have.
    
    if time.time() < deadline - 10.0:
        # Normalize features for random combination
        deg_norm = (degree - degree.min()) / (degree.max() - degree.min() + 1e-9)
        mnd_norm = (mean_neighbor_deg - mean_neighbor_deg.min()) / (mean_neighbor_deg.max() - mean_neighbor_deg.min() + 1e-9)
        
        # Random weights
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            w1 = np.random.uniform(0, 1)
            w2 = np.random.uniform(0, 1)
            # We want to penalize high degree and high mean neighbor degree
            random_score = -(w1 * degree + w2 * mean_neighbor_deg)
            candidates.append(random_score)

    # Evaluate candidates
    best_size = -1
    best_scores = candidates[0]
    
    for s in candidates:
        if time.time() > deadline - 5.0:
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
