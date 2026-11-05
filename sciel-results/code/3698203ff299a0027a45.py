import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 115.0

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric (just in case)
    A = (A + A.T) / 2.0
    # Zero diagonal
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5

    D = A.sum(axis=1)
    D_safe = D + 1e-9

    # Features
    SND = A @ D
    MeanND = SND / D_safe

    # Triangles
    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0

    # Clustering
    Clus = np.zeros(N)
    mask = D > 1
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))

    # Degree-2 subgraph: neighbors of neighbors that are not neighbors and not self
    # A2 = A @ A. For i, A2[i] counts length-2 paths. 
    # Common neighbors of i and j is A2[i,j].
    # We want for each vertex i, the number of "open" pairs among its neighbors.
    # Number of closed triangles: Tri[i]
    # Number of neighbor pairs: C(D[i], 2)
    # Open pairs = C(D[i], 2) - Tri[i]
    # This measures how much of i's neighborhood is "spread out" (low clustering).
    OpenPairs = np.zeros(N)
    C_D_2 = 0.5 * D * (D - 1.0)
    OpenPairs = np.maximum(0, C_D_2 - Tri)

    # Normalize
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
    OpenPairs_n = norm(OpenPairs)

    # Evaluate function
    def evaluate(scores):
        indices = np.argsort(-scores)
        is_picked = np.zeros(N, dtype=bool)
        count = 0
        for v in indices:
            if not is_picked[v]:
                count += 1
                is_picked[v] = True
                is_picked[A_bool[v]] = True
        return count

    best_size = -1
    best_scores = None

    # Baseline: Low degree
    base_score = -D_n
    base_size = evaluate(base_score)
    best_size = base_size
    best_scores = base_score.copy()

    # Search for better weights
    # Weights for: -D_n, MeanND_n, -Clus_n, OpenPairs_n, -Tri_n
    # We want to penalize high degree, reward high mean neighbor degree (hub blocking),
    # reward open neighborhoods (low clustering), penalize triangles.
    
    # Grid search / Random search over weights
    rng = np.random.default_rng(42)
    
    # Try a few deterministic combinations first
    # 1. Low Degree
    # 2. Low Degree + Open Pairs
    # 3. Low Degree + Mean ND
    # 4. Low Degree + Low Clustering
    
    candidates = [
        (1.0, 0.0, 0.0, 0.0, 0.0), # -D
        (1.0, 0.0, 0.0, 1.0, 0.0), # -D + Open
        (1.0, 1.0, 0.0, 0.0, 0.0), # -D + MeanND
        (1.0, 0.0, 1.0, 0.0, 0.0), # -D - Clus
        (1.0, 0.5, 0.5, 0.5, 0.0), # Mix
        (0.5, 1.0, 0.0, 0.0, 0.0), # MeanND + -D
    ]
    
    for w in candidates:
        scores = -w[0] * D_n + w[1] * MeanND_n - w[2] * Clus_n + w[3] * OpenPairs_n - w[4] * Tri_n
        size = evaluate(scores)
        if size > best_size:
            best_size = size
            best_scores = scores.copy()

    # Random search
    # We have ~100s left. Each eval takes ~5ms for N=200? 
    # 2000 evals = 10s. 
    # Let's do more if fast.
    
    max_trials = 5000
    trial_count = 0
    
    while trial_count < max_trials:
        if time.time() > DEADLINE:
            break
            
        # Sample weights
        # w1 (deg) in [0.5, 2.0]
        # w2 (meannd) in [0.0, 1.5]
        # w3 (clus) in [0.0, 1.0]
        # w4 (open) in [0.0, 1.5]
        # w5 (tri) in [0.0, 0.5]
        
        w1 = rng.uniform(0.5, 2.0)
        w2 = rng.uniform(0.0, 1.5)
        w3 = rng.uniform(0.0, 1.0)
        w4 = rng.uniform(0.0, 1.5)
        w5 = rng.uniform(0.0, 0.5)
        
        scores = -w1 * D_n + w2 * MeanND_n - w3 * Clus_n + w4 * OpenPairs_n - w5 * Tri_n
        
        size = evaluate(scores)
        
        if size > best_size:
            best_size = size
            best_scores = scores.copy()
            
        trial_count += 1

    return best_scores
