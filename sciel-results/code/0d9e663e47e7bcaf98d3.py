import numpy as np
import time

def priority(adjacency):
    start = time.time()
    DEADLINE = 118.0

    A = np.asarray(adjacency, dtype=np.float64)
    N = A.shape[0]

    if N == 0:
        return np.array([], dtype=np.float64)

    # Symmetrize and zero diagonal
    A = (A + A.T) * 0.5
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5

    D = A_bool.sum(axis=1).astype(np.float64)
    D_safe = D + 1e-12

    # --- Feature extraction (vectorized where possible) ---
    # Degree
    D_f = D.copy()

    # Mean neighbor degree
    SND = A_bool @ D
    MeanND = SND / D_safe

    # Triangles
    A_sq = A_bool @ A_bool
    Tri = np.diag(A_sq).astype(np.float64) * 0.5

    # Clustering coefficient
    Clus = np.zeros(N, dtype=np.float64)
    mask = D > 1
    Clus[mask] = (2.0 * Tri[mask]) / (D[mask] * (D[mask] - 1.0))

    # Open pairs: number of non-triangular edges among neighbors
    OpenPairs = np.maximum(0.0, 0.5 * D * (D - 1.0) - Tri)

    # Outward expansion: sum of (deg(nbr) - 1) over neighbors, normalized
    OutwardExp = (A_bool @ (D - 1.0)) / D_safe

    # Min neighbor degree
    MinNbrDeg = np.full(N, 0.0, dtype=np.float64)
    for i in range(N):
        nbrs = np.where(A_bool[i])[0]
        if len(nbrs) > 0:
            MinNbrDeg[i] = D[nbrs].min()

    # Leaf neighbors count
    LeafNbr = (A_bool @ (D == 1).astype(np.float64))

    # Variance of neighbor degrees (proxy for 2-hop spread)
    VarNbrDeg = np.zeros(N, dtype=np.float64)
    for i in range(N):
        nbrs = np.where(A_bool[i])[0]
        if len(nbrs) > 1:
            VarNbrDeg[i] = np.var(D[nbrs])

    # --- Additional structural features ---
    # Common neighbors count (from A_sq)
    CommonNbrs = (A_sq - A_bool).clip(min=0)  # remove direct edges
    CommonNbrs[np.diag_indices(N)] = 0.0
    CN_sum = CommonNbrs.sum(axis=1)

    # Second-neighbor degree sum: for each i, sum deg(j) over j at distance 2
    # = sum over nbrs k of i: sum over nbrs j of k (j != i, j not nbr of i) deg(j)
    # Approximate: A_bool @ A_bool @ D - contributions
    TwoHopDegSum = A_bool @ (A_bool @ D) - A_bool @ D  # rough proxy

    # Eigenvector centrality (power iteration, few steps)
    eig = np.ones(N, dtype=np.float64) / N
    for _ in range(8):
        eig_new = A_bool @ eig
        eig_new /= (eig_new.sum() + 1e-12)
        eig = eig_new

    # PageRank-like (damped)
    pr = np.ones(N, dtype=np.float64) / N
    d = 0.85
    for _ in range(8):
        pr_new = (1 - d) / N + d * (A_bool.T @ pr / (D_safe + 1e-12))
        pr = pr_new

    # --- Normalize all features to [0,1] ---
    def norm(x):
        x = x.astype(np.float64)
        mn = x.min()
        mx = x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    D_n = norm(D_f)
    MeanND_n = norm(MeanND)
    Tri_n = norm(Tri)
    Clus_n = norm(Clus)
    OpenPairs_n = norm(OpenPairs)
    OutwardExp_n = norm(OutwardExp)
    MinNbrDeg_n = norm(MinNbrDeg)
    LeafNbr_n = norm(LeafNbr)
    VarNbrDeg_n = norm(VarNbrDeg)
    CN_n = norm(CN_sum)
    TwoHopDeg_n = norm(TwoHopDegSum)
    Eig_n = norm(eig)
    PR_n = norm(pr)

    # --- Fast greedy evaluator (vectorized marking) ---
    adj_rows = A_bool  # boolean matrix

    def eval_greedy(scores):
        order = np.argsort(-scores, kind='stable')
        picked = np.zeros(N, dtype=np.bool_)
        cnt = 0
        for v in order:
            if not picked[v]:
                cnt += 1
                picked[v] = True
                picked[adj_rows[v]] = True
        return cnt

    # --- Feature matrix for weight search ---
    # Each column is a feature; sign chosen so that higher score = prefer pick
    # We want to prefer low degree, low clustering, high open pairs, etc.
    # Build signed feature matrix F where score = w @ F
    F = np.column_stack([
        -D_n,          # prefer low degree
        MeanND_n,      # prefer high mean neighbor degree (contrarian: pick hub-adjacent?)
        -Clus_n,       # prefer low clustering
        OpenPairs_n,   # prefer high open pairs
        -Tri_n,        # prefer low triangles
        OutwardExp_n,  # prefer high outward expansion
        MinNbrDeg_n,   # prefer high min nbr degree
        LeafNbr_n,     # prefer high leaf nbrs
        -VarNbrDeg_n,  # prefer low variance (homogeneous neighbors)
        -CN_n,         # prefer low common neighbors
        TwoHopDeg_n,   # prefer high 2-hop degree sum
        Eig_n,         # prefer high eigenvector centrality
        PR_n,          # prefer high pagerank
    ])

    n_feats = F.shape[1]
    best_size = -1
    best_w = np.zeros(n_feats)

    # Evaluate a weight vector
    def eval_w(w):
        scores = w @ F
        return eval_greedy(scores)

    # --- Phase 1: coordinate descent / greedy weight search ---
    # Start from a reasonable baseline
    w = np.zeros(n_feats)
    w[0] = 1.0  # -D
    best_size = eval_w(w)
    best_w = w.copy()

    # Try single-feature additions
    for j in range(n_feats):
        if j == 0:
            continue
        w_try = best_w.copy()
        for alpha in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            w_try[j] = alpha
            sz = eval_w(w_try)
            if sz > best_size:
                best_size = sz
                best_w = w_try.copy()

    # --- Phase 2: random weight search with adaptive sampling ---
    rng = np.random.default_rng(123)
    max_trials = 6000
    for t in range(max_trials):
        if time.time() > DEADLINE - 10.0:
            break

        # Sample weights: mix of sparse and dense
        if rng.random() < 0.3:
            # Sparse: pick 2-4 features
            k = rng.integers(2, 5)
            idx = rng.choice(n_feats, size=k, replace=False)
            w_try = np.zeros(n_feats)
            for i in idx:
                w_try[i] = rng.uniform(0.3, 2.0) * (1 if rng.random() < 0.7 else -1)
        else:
            # Dense: all features with small weights
            w_try = rng.uniform(0.0, 1.0, size=n_feats)
            # Ensure degree term is present and negative
            w_try[0] = -rng.uniform(0.5, 2.0)

        sz = eval_w(w_try)
        if sz > best_size:
            best_size = sz
            best_w = w_try.copy()

    # --- Phase 3: local refinement around best weights ---
    if time.time() < DEADLINE - 15.0:
        for _ in range(300):
            if time.time() > DEADLINE - 10.0:
                break
            # Perturb each weight slightly
            w_try = best_w + rng.normal(0, 0.15, size=n_feats)
            sz = eval_w(w_try)
            if sz > best_size:
                best_size = sz
                best_w = w_try.copy()

    # --- Phase 4: direct score perturbation (fine-tune the final scores) ---
    if time.time() < DEADLINE - 5.0:
        best_scores = best_w @ F
        for _ in range(150):
            if time.time() > DEADLINE - 3.0:
                break
            perturbed = best_scores + rng.normal(0, 0.05, size=N)
            sz = eval_greedy(perturbed)
            if sz > best_size:
                best_size = sz
                best_scores = perturbed

    # Final scores
    final_scores = best_w @ F
    if time.time() < DEADLINE - 2.0:
        # One more check
        sz = eval_greedy(final_scores)
        if sz > best_size:
            best_size = sz
            best_scores = final_scores

    return best_scores
