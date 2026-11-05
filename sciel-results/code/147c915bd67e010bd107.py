import numpy as np
import time
from scipy.linalg import eigh

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 115.0
    
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
        
    # Random Search
    rng = np.random.default_rng(12345)
    
    num_trials = 2000
    
    for i in range(num_trials):
        if time.time() > DEADLINE:
            break
            
        # Weights for features
        w1 = rng.uniform(0.5, 3.0) # -D_n
        w2 = rng.uniform(0.0, 2.0) # MeanND_n
        w3 = rng.uniform(0.0, 1.0) # -Tri_n
        w4 = rng.uniform(0.0, 1.0) # -Clus_n
        w5 = rng.uniform(0.0, 2.0) # -PC_n
        w6 = rng.uniform(0.0, 1.5) # +/- Fiedler_n
        sign6 = 1.0 if rng.random() > 0.5 else -1.0
        
        scores = -w1 * D_n + w2 * MeanND_n - w3 * Tri_n - w4 * Clus_n - w5 * PC_n + sign6 * w6 * Fiedler_n
        
        size = evaluate(scores)
        
        if size > best_size:
            best_size = size
            best_scores = scores.copy()
            
    if best_scores is None:
        return base_score
        
    return best_scores
