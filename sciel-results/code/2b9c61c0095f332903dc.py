import numpy as np
import time
from scipy.linalg import eigh

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 140.0
    
    A = np.asarray(adjacency, dtype=bool)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
        
    # 1. Compute Features
    D = A.sum(axis=1).astype(float)
    
    # Avoid division by zero
    D_safe = D + 1e-9
    
    SND = A @ D
    MeanND = SND / D_safe
    
    # Triangles
    A_sq = A @ A
    Tri = np.diag(A_sq).astype(float) / 2.0
    
    # Clustering Coefficient
    Clus = np.zeros(N)
    mask = D > 1
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))
    
    # Perron Vector (Power Iteration)
    v = np.ones(N) / np.sqrt(N)
    for _ in range(100):
        v_new = A @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-9:
            break
        v = v_new / norm
    PC = v
    
    # Fiedler Vector (Second smallest eigenvector of Laplacian)
    L = np.diag(D) - A.astype(float)
    try:
        eigvals, eigvecs = eigh(L)
        # eigvecs columns are eigenvectors. Index 1 is Fiedler.
        fiedler_vec = eigvecs[:, 1]
        Fiedler = np.abs(fiedler_vec)
    except Exception:
        Fiedler = np.zeros(N)

    # Normalize all features to [0, 1]
    def norm(x):
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)
        
    D_n = norm(D)
    MeanND_n = norm(MeanND)
    Tri_n = norm(Tri)
    Clus_n = norm(Clus)
    PC_n = norm(PC)
    Fiedler_n = norm(Fiedler)
    
    # 2. Define Candidate Scores
    def evaluate(scores):
        indices = np.argsort(-scores)
        is_picked = np.zeros(N, dtype=bool)
        count = 0
        for v in indices:
            if not is_picked[v]:
                count += 1
                is_picked[v] = True
                is_picked[A[v]] = True
        return count

    def local_swap_refine(scores, max_swaps=50):
        """Try local swaps to improve the greedy selection."""
        best_size = evaluate(scores)
        indices = np.argsort(-scores)
        is_picked = np.zeros(N, dtype=bool)
        picked_order = []
        for v in indices:
            if not is_picked[v]:
                picked_order.append(v)
                is_picked[v] = True
                is_picked[A[v]] = True
        
        improved = True
        swap_count = 0
        while improved and swap_count < max_swaps:
            improved = False
            for i in range(len(picked_order) - 1):
                if time.time() > DEADLINE:
                    return scores, best_size
                vi = picked_order[i]
                for j in range(i + 1, len(picked_order)):
                    vj = picked_order[j]
                    # Try swapping: remove vj, add vi's alternative neighbor
                    # Simple swap: try replacing vj with a non-neighbor of other picked vertices
                    # Find a candidate to replace vj
                    replaced = False
                    for candidate in indices:
                        if candidate == vj or candidate == vi:
                            continue
                        if is_picked[candidate]:
                            continue
                        # Check if candidate is not neighbor of any other picked vertex (except vj)
                        ok = True
                        for pk in picked_order:
                            if pk == vj:
                                continue
                            if A[candidate, pk]:
                                ok = False
                                break
                        if ok:
                            # Try the swap
                            new_scores = scores.copy()
                            new_scores[vj] = -1e10  # Remove vj
                            new_scores[candidate] = scores[vi] + 1e-6  # Add candidate with slightly higher priority than vi
                            new_size = evaluate(new_scores)
                            if new_size > best_size:
                                best_size = new_size
                                scores = new_scores
                                picked_order[j] = candidate
                                improved = True
                                swap_count += 1
                                replaced = True
                                break
                    if not replaced:
                        continue
            if not improved:
                break
        
        return scores, best_size

    best_size = -1
    best_scores = None
    
    # Baseline: Low Degree
    base_score = -D_n
    base_size = evaluate(base_score)
    if base_size > best_size:
        best_size = base_size
        best_scores = base_score.copy()
        
    # "Shelter" Heuristic: Low Degree, High MeanND
    shelter_score = -2.0 * D_n + 1.5 * MeanND_n
    shelter_size = evaluate(shelter_score)
    if shelter_size > best_size:
        best_size = shelter_size
        best_scores = shelter_score.copy()
        
    # Random Search with local refinement
    rng = np.random.default_rng(12345)
    
    num_trials = 5000
    
    for i in range(num_trials):
        if time.time() > DEADLINE:
            break
            
        # Weights for features - use smaller ranges for more focused search
        w1 = rng.uniform(0.3, 2.5) # -D_n
        w2 = rng.uniform(0.0, 1.8) # MeanND_n
        w3 = rng.uniform(0.0, 0.8) # -Tri_n
        w4 = rng.uniform(0.0, 0.8) # -Clus_n
        w5 = rng.uniform(0.0, 1.5) # -PC_n
        w6 = rng.uniform(0.0, 1.2) # +/- Fiedler_n
        sign6 = 1.0 if rng.random() > 0.5 else -1.0
        
        scores = -w1 * D_n + w2 * MeanND_n - w3 * Tri_n - w4 * Clus_n - w5 * PC_n + sign6 * w6 * Fiedler_n
        
        size = evaluate(scores)
        
        if size > best_size:
            best_size = size
            best_scores = scores.copy()
    
    if best_scores is None:
        return base_score
    
    # Apply local swap refinement to the best solution found
    if time.time() < DEADLINE - 5.0:
        best_scores, best_size = local_swap_refine(best_scores, max_swaps=100)
    
    return best_scores
