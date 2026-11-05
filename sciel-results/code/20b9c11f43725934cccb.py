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

    # Precompute for evaluation
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

    # Core Numbers - use a faster approximation via iterative peeling
    core = np.zeros(N, dtype=int)
    deg = D.copy().astype(int)
    nbrs = [np.where(A_bool[i])[0] for i in range(N)]
    remaining = set(range(N))
    # Sort by degree initially
    order = np.argsort(deg)
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

    # Distance 2 Non-Neighbors
    mask_non_nbr = ~A_bool
    np.fill_diagonal(mask_non_nbr, False)
    Dist2NonNbr = (A_sq * mask_non_nbr).sum(axis=1)

    # OutwardExp: how much the neighborhood expands
    # For each vertex, count neighbors that have more neighbors than it
    OutwardExp = np.zeros(N)
    for i in range(N):
        nbrs_i = nbrs[i]
        if len(nbrs_i) == 0:
            OutwardExp[i] = 0
        else:
            OutwardExp[i] = np.mean(D[nbrs_i] > D[i])

    # MinNbrDeg: minimum degree among neighbors
    MinNbrDeg = np.zeros(N)
    for i in range(N):
        nbrs_i = nbrs[i]
        if len(nbrs_i) == 0:
            MinNbrDeg[i] = 0
        else:
            MinNbrDeg[i] = np.min(D[nbrs_i])

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
    Core_n = norm(core)
    Dist2_n = norm(Dist2NonNbr)
    OutwardExp_n = norm(OutwardExp)
    MinNbrDeg_n = norm(MinNbrDeg)

    # Fast evaluation function
    def evaluate_fast(scores):
        indices = np.argsort(-scores, kind='stable')
        is_picked = np.zeros(N, dtype=bool)
        count = 0
        for v in indices:
            if not is_picked[v]:
                count += 1
                is_picked[v] = True
                is_picked[adj_rows[v]] = True
        return count

    # Baseline: Low Degree
    scores_base = -D_n
    best_size = evaluate_fast(scores_base)
    best_scores = scores_base.copy()

    # Random Search
    rng = np.random.default_rng(42)

    # Weights for: -D, -SND, -MeanND, -Tri, -Clus, OpenPairs, -Core, Dist2, OutwardExp, -MinNbrDeg
    # Let's search over a broader range

    # Initial candidates
    candidates = [
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5),
    ]

    for w in candidates:
        s = (w[0]*D_n + w[1]*SND_n + w[2]*MeanND_n + w[3]*Tri_n + w[4]*Clus_n +
             w[5]*OpenPairs_n + w[6]*Core_n + w[7]*Dist2_n + w[8]*OutwardExp_n + w[9]*MinNbrDeg_n)
        sz = evaluate_fast(s)
        if sz > best_size:
            best_size = sz
            best_scores = s.copy()

    # Random Search with broader ranges
    max_trials = 20000
    trial = 0
    while trial < max_trials:
        if time.time() > DEADLINE - 5.0:
            break

        # Sample weights with broader ranges
        w0 = rng.uniform(0.5, 2.0)
        w1 = rng.uniform(0.0, 2.5)
        w2 = rng.uniform(0.0, 2.0)
        w3 = rng.uniform(0.0, 1.5)
        w4 = rng.uniform(0.0, 1.5)
        w5 = rng.uniform(0.0, 2.0)
        w6 = rng.uniform(0.0, 1.5)
        w7 = rng.uniform(0.0, 1.5)
        w8 = rng.uniform(0.0, 1.0)
        w9 = rng.uniform(0.0, 1.0)

        # Score = -w0*D - w1*SND - w2*MeanND - w3*Tri - w4*Clus + w5*Open - w6*Core + w7*Dist2 + w8*OutwardExp - w9*MinNbrDeg
        s = (-w0*D_n - w1*SND_n - w2*MeanND_n - w3*Tri_n - w4*Clus_n +
             w5*OpenPairs_n - w6*Core_n + w7*Dist2_n + w8*OutwardExp_n - w9*MinNbrDeg_n)

        sz = evaluate_fast(s)
        if sz > best_size:
            best_size = sz
            best_scores = s.copy()

        trial += 1

    # Local Refinement
    if time.time() < DEADLINE - 2.0:
        for _ in range(500):
            if time.time() > DEADLINE:
                break
            perturbed = best_scores + rng.normal(0, 0.1, size=N)
            sz = evaluate_fast(perturbed)
            if sz > best_size:
                best_size = sz
                best_scores = perturbed.copy()

    return best_scores
