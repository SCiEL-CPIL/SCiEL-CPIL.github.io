import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5

    # Precompute adjacency rows for fast evaluation
    adj_rows = A_bool

    # Features
    D = A.sum(axis=1)
    D_safe = D + 1e-9

    # Sum of Neighbor Degrees
    SND = A @ D
    MeanND = SND / D_safe

    # Triangles
    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0

    # Clustering coefficient
    Clus = np.zeros(N)
    mask = D > 1
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))

    # Open pairs: C(D,2) - Tri
    OpenPairs = np.maximum(0, 0.5 * D * (D - 1.0) - Tri)

    # Core Numbers (k-core decomposition)
    core = np.zeros(N, dtype=int)
    deg = D.copy().astype(int)
    nbrs = [np.where(A_bool[i])[0] for i in range(N)]
    remaining = set(range(N))
    while remaining:
        min_deg = min(deg[i] for i in remaining)
        to_remove = [i for i in remaining if deg[i] == min_deg]
        if not to_remove:
            break
        for i in to_remove:
            core[i] = min_deg
            remaining.remove(i)
            for j in nbrs[i]:
                if j in remaining:
                    deg[j] -= 1

    # Distance 2 Non-Neighbors count
    mask_non_nbr = ~A_bool
    np.fill_diagonal(mask_non_nbr, False)
    Dist2NonNbr = (A_sq * mask_non_nbr).sum(axis=1)

    # PageRank-like centrality (simple power iteration)
    pr = np.ones(N) / N
    for _ in range(20):
        new_pr = 0.85 * (A.T @ pr) / N + 0.15 / N
        pr = new_pr

    # Normalize features
    def norm(x):
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    D_n = norm(D)
    SND_n = norm(SND)
    MeanND_n = norm(MeanND)
    Tri_n = norm(Tri)
    Clus_n = norm(Clus)
    OpenPairs_n = norm(OpenPairs)
    Core_n = norm(core.astype(float))
    Dist2_n = norm(Dist2NonNbr)
    PR_n = norm(pr)

    # Fast evaluation function using precomputed neighbor lists
    def evaluate_fast(scores):
        indices = np.argsort(-scores, kind='stable')
        blocked = np.zeros(N, dtype=bool)
        count = 0
        for v in indices:
            if not blocked[v]:
                count += 1
                blocked[v] = True
                blocked[nbrs[v]] = True
        return count

    # Baseline: Low Degree
    scores_base = -D_n
    best_size = evaluate_fast(scores_base)
    best_scores = scores_base.copy()

    # Random Search
    rng = np.random.default_rng(42)

    # Initial candidates with various weight combinations
    candidates = [
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),    # -D
        (-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),   # -D - 0.5 SND
        (-1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),   # -D - 0.5 MeanND
        (-1.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0),   # -D - 0.5 Tri
        (-1.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0),   # -D - 0.5 Clus
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0),    # -D + 0.5 OpenPairs
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0),   # -D - 0.5 Core
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0),    # -D + 0.5 Dist2
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),    # -D + 0.5 PR
        (-0.8, -0.3, -0.2, -0.2, -0.2, 0.3, -0.2, 0.3, 0.2),  # Mixed
        (-1.0, -0.5, -0.3, -0.3, -0.3, 0.5, -0.3, 0.5, 0.3),  # Mixed 2
    ]

    for w in candidates:
        s = (w[0]*D_n + w[1]*SND_n + w[2]*MeanND_n + w[3]*Tri_n +
             w[4]*Clus_n + w[5]*OpenPairs_n + w[6]*Core_n + w[7]*Dist2_n +
             w[8]*PR_n)
        sz = evaluate_fast(s)
        if sz > best_size:
            best_size = sz
            best_scores = s.copy()

    # Simulated annealing style search
    current_scores = best_scores.copy()
    current_size = best_size
    T = 0.5  # initial temperature
    cooling_rate = 0.9995
    
    max_iterations = 50000
    for it in range(max_iterations):
        if time.time() > DEADLINE - 15.0:
            break
        
        # Perturb scores
        perturbation_scale = T * 0.1
        perturbed = current_scores + rng.normal(0, perturbation_scale, size=N)
        
        sz = evaluate_fast(perturbed)
        
        if sz > current_size:
            current_scores = perturbed.copy()
            current_size = sz
            if sz > best_size:
                best_size = sz
                best_scores = perturbed.copy()
        else:
            # Accept worse solutions with probability exp((sz - current_size)/T)
            delta = sz - current_size
            if delta < 0:
                prob = np.exp(delta / max(T, 1e-6))
                if rng.random() < prob:
                    current_scores = perturbed.copy()
                    current_size = sz
        
        T *= cooling_rate
    
    # Local Refinement: perturb best scores and evaluate
    if time.time() < DEADLINE - 5.0:
        for _ in range(2000):
            if time.time() > DEADLINE - 2.0:
                break
            perturbed = best_scores + rng.normal(0, 0.03, size=N)
            sz = evaluate_fast(perturbed)
            if sz > best_size:
                best_size = sz
                best_scores = perturbed.copy()

    # Final polish: try small adjustments to individual vertex scores
    if time.time() < DEADLINE - 1.0:
        for _ in range(500):
            if time.time() > DEADLINE - 0.5:
                break
            # Pick a random vertex and adjust its score slightly
            v = rng.integers(0, N)
            delta = rng.uniform(-0.05, 0.05)
            perturbed = best_scores.copy()
            perturbed[v] += delta
            sz = evaluate_fast(perturbed)
            if sz > best_size:
                best_size = sz
                best_scores = perturbed.copy()

    return best_scores
