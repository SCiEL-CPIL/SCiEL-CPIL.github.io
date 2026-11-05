import numpy as np
import time
from collections import deque

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
    nbrs = [np.where(A_bool[i])[0] for i in range(N)]

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

    # Core Numbers (k-core decomposition) using bucket approach
    core = np.zeros(N, dtype=int)
    deg = D.copy().astype(int)
    max_deg = int(deg.max()) if N > 0 else 0
    buckets = [deque() for _ in range(max_deg + 1)]
    for i in range(N):
        if deg[i] >= 0:
            buckets[min(deg[i], max_deg)].append(i)
    remaining = np.ones(N, dtype=bool)
    current_max = max_deg
    while current_max >= 0:
        while buckets[current_max]:
            i = buckets[current_max].popleft()
            if not remaining[i]:
                continue
            core[i] = current_max
            remaining[i] = False
            for j in nbrs[i]:
                if remaining[j]:
                    old_deg = deg[j]
                    deg[j] -= 1
                    try:
                        buckets[old_deg].remove(j)
                    except ValueError:
                        pass
                    new_deg = max(0, min(deg[j], max_deg))
                    buckets[new_deg].append(j)
        current_max -= 1

    # Distance 2 Non-Neighbors count
    mask_non_nbr = ~A_bool
    np.fill_diagonal(mask_non_nbr, False)
    Dist2NonNbr = (A_sq * mask_non_nbr).sum(axis=1)

    # Fiedler Vector (second smallest eigenvector of Laplacian)
    L = np.diag(D) - A
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        fiedler = eigvecs[:, 1]
        f_min, f_max = fiedler.min(), fiedler.max()
        if f_max > f_min:
            fiedler_n = (fiedler - f_min) / (f_max - f_min)
        else:
            fiedler_n = np.zeros(N)
        abs_fiedler = np.abs(fiedler)
        af_min, af_max = abs_fiedler.min(), abs_fiedler.max()
        if af_max > af_min:
            abs_fiedler_n = (abs_fiedler - af_min) / (af_max - af_min)
        else:
            abs_fiedler_n = np.zeros(N)
    except Exception:
        fiedler_n = np.zeros(N)
        abs_fiedler_n = np.zeros(N)

    # OutwardExp: how much the neighborhood expands
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

    # MaxNbrDeg: maximum degree among neighbors
    MaxNbrDeg = np.zeros(N)
    for i in range(N):
        nbrs_i = nbrs[i]
        if len(nbrs_i) == 0:
            MaxNbrDeg[i] = 0
        else:
            MaxNbrDeg[i] = np.max(D[nbrs_i])

    # Normalize features
    def norm(x):
        x = np.asarray(x, dtype=float)
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
    OutwardExp_n = norm(OutwardExp)
    MinNbrDeg_n = norm(MinNbrDeg)
    MaxNbrDeg_n = norm(MaxNbrDeg)
    Fiedler_n = fiedler_n
    AbsFiedler_n = abs_fiedler_n

    # Fast evaluation function using precomputed neighbor lists
    # Optimized: use numpy vectorized operations where possible
    def evaluate_fast(scores):
        indices = np.argsort(-scores, kind='stable')
        blocked = np.zeros(N, dtype=bool)
        count = 0
        for v in indices:
            if not blocked[v]:
                count += 1
                blocked[v] = True
                if len(nbrs[v]) > 0:
                    blocked[nbrs[v]] = True
        return count

    # Even faster evaluation using bitset-like approach for small N
    # Precompute neighbor bitsets as integers for fast blocking
    # For N=200, we can use Python ints as bitsets
    nbr_bitsets = []
    for i in range(N):
        mask = 0
        for j in nbrs[i]:
            mask |= (1 << j)
        nbr_bitsets.append(mask)

    def evaluate_fast_bitset(scores):
        indices = np.argsort(-scores, kind='stable')
        blocked_bits = 0
        count = 0
        for v in indices:
            if not (blocked_bits & (1 << v)):
                count += 1
                blocked_bits |= (1 << v) | nbr_bitsets[v]
        return count

    # Baseline: Low Degree
    scores_base = -D_n
    best_size = evaluate_fast_bitset(scores_base)
    best_scores = scores_base.copy()

    # Random Search
    rng = np.random.default_rng(42)

    # Initial candidates with various weight combinations
    candidates = [
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5),
        # Now add candidates that PREFER high |Fiedler| (positive weight on AbsFiedler)
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0),
        (-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0),
        # Mixed: low degree + high |Fiedler|
        (-0.8, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.3, 1.0),
        (-0.7, -0.3, -0.2, 0.0, 0.0, 0.2, -0.3, 0.2, 0.1, -0.1, -0.1, 0.2, 0.8),
    ]

    for w in candidates:
        s = (w[0]*D_n + w[1]*SND_n + w[2]*MeanND_n + w[3]*Tri_n +
             w[4]*Clus_n + w[5]*OpenPairs_n + w[6]*Core_n + w[7]*Dist2_n +
             w[8]*OutwardExp_n + w[9]*MinNbrDeg_n + w[10]*MaxNbrDeg_n +
             w[11]*Fiedler_n + w[12]*AbsFiedler_n)
        sz = evaluate_fast_bitset(s)
        if sz > best_size:
            best_size = sz
            best_scores = s.copy()

    # Random Search with more diverse weight sampling
    max_trials = 50000
    trial = 0
    while trial < max_trials:
        if time.time() > DEADLINE - 10.0:
            break

        # Sample weights with varied ranges
        w0 = rng.uniform(0.5, 2.0)
        w1 = rng.uniform(-0.5, 1.0)
        w2 = rng.uniform(-0.5, 1.0)
        w3 = rng.uniform(-0.5, 1.0)
        w4 = rng.uniform(-0.5, 1.0)
        w5 = rng.uniform(-0.5, 1.5)
        w6 = rng.uniform(-1.0, 0.5)
        w7 = rng.uniform(-0.5, 1.0)
        w8 = rng.uniform(-0.5, 1.0)
        w9 = rng.uniform(-0.5, 1.0)
        w10 = rng.uniform(-0.5, 1.0)
        w11 = rng.uniform(-1.0, 1.0)
        w12 = rng.uniform(-0.5, 2.0)

        s = (-w0*D_n + w1*SND_n + w2*MeanND_n + w3*Tri_n +
             w4*Clus_n + w5*OpenPairs_n + w6*Core_n + w7*Dist2_n +
             w8*OutwardExp_n + w9*MinNbrDeg_n + w10*MaxNbrDeg_n +
             w11*Fiedler_n + w12*AbsFiedler_n)

        sz = evaluate_fast_bitset(s)
        if sz > best_size:
            best_size = sz
            best_scores = s.copy()

        trial += 1

    # Local Refinement: perturb best scores and evaluate
    if time.time() < DEADLINE - 5.0:
        # Start with larger steps, then refine
        for iteration in range(3000):
            if time.time() > DEADLINE - 2.0:
                break
            # Use adaptive step size: larger early, smaller later
            progress = (time.time() - start_time) / (DEADLINE - 5.0 - start_time)
            sigma = 0.1 * (1.0 - progress) + 0.01
            perturbed = best_scores + rng.normal(0, sigma, size=N)
            sz = evaluate_fast_bitset(perturbed)
            if sz > best_size:
                best_size = sz
                best_scores = perturbed.copy()

    # Swap-based local search: try swapping the order of two vertices
    if time.time() < DEADLINE - 1.0:
        for _ in range(2000):
            if time.time() > DEADLINE - 0.5:
                break
            # Pick two random vertices and try to swap their relative order
            # by adjusting their scores
            v1, v2 = rng.choice(N, size=2, replace=False)
            # Try small adjustments to push v1 above v2 or vice versa
            delta = rng.uniform(0.01, 0.1)
            for sign in [1, -1]:
                perturbed = best_scores.copy()
                perturbed[v1] += sign * delta
                perturbed[v2] -= sign * delta
                sz = evaluate_fast_bitset(perturbed)
                if sz > best_size:
                    best_size = sz
                    best_scores = perturbed.copy()

    # Final polish: try small adjustments to individual vertex scores
    if time.time() < DEADLINE - 0.5:
        for _ in range(500):
            if time.time() > DEADLINE - 0.1:
                break
            v = rng.integers(0, N)
            delta = rng.uniform(-0.03, 0.03)
            perturbed = best_scores.copy()
            perturbed[v] += delta
            sz = evaluate_fast_bitset(perturbed)
            if sz > best_size:
                best_size = sz
                best_scores = perturbed.copy()

    return best_scores
