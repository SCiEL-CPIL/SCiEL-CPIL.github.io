"""Combined program: Rich feature engineering with weight-space search + spectral centrality + Fiedler vector + local swap refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin below 150s limit

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric and zero diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5

    # --- Feature Extraction ---
    
    # 1. Degree
    D = A.sum(axis=1)
    D_safe = D + 1e-9

    # 2. Mean Neighbor Degree
    SND = A @ D
    MeanND = SND / D_safe

    # 3. Triangles (number of triangles each vertex is in)
    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0

    # 4. Clustering Coefficient
    Clus = np.zeros(N)
    mask = D > 1
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))

    # 5. Open Pairs (pairs of neighbors that are NOT connected)
    OpenPairs = np.maximum(0, 0.5 * D * (D - 1.0) - Tri)

    # 6. Outward Expansion (how "spread out" the neighborhood is)
    OutwardExp_raw = A @ (D - 1.0)
    OutwardExp = OutwardExp_raw / D_safe

    # 7. Min Neighbor Degree
    MinNbrDeg = np.zeros(N)
    for i in range(N):
        nbrs = np.where(A_bool[i])[0]
        if len(nbrs) > 0:
            MinNbrDeg[i] = D[nbrs].min()
        else:
            MinNbrDeg[i] = 0.0

    # 8. Spectral Centrality via Power Iteration
    # Compute principal eigenvector of A
    # Higher eigenvector centrality = more central, we want lower for independent set
    x = np.ones(N) / N
    for _ in range(80):
        x_new = A @ x
        norm_x = np.linalg.norm(x_new)
        if norm_x < 1e-12:
            break
        x_new /= norm_x
        if np.linalg.norm(x_new - x) < 1e-10:
            break
        x = x_new
    SpectralCentrality = x

    # 9. Fiedler Vector (2nd smallest eigenvector of Laplacian)
    # Identifies vertices at natural graph cuts/periphery
    L = D[:, np.newaxis] - A  # Laplacian
    # Use power iteration on (I - L/lambda_max) to find smallest eigenvector,
    # then orthogonalize against ones vector to get Fiedler vector
    # Simpler: use scipy if available, else approximate via inverse power iteration
    try:
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        L_sp = csr_matrix(L)
        # Find 2 smallest eigenvalues/eigenvectors
        vals, vecs = eigsh(L_sp, k=2, which='SM')
        # Sort by eigenvalue
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]]
        # Ensure consistent sign
        if np.sum(fiedler) < 0:
            fiedler = -fiedler
    except Exception:
        # Fallback: use power iteration to find smallest eigenvector of L
        # Shift: work with (I - L/lambda_max)
        lambda_max_est = np.max(D) + 1e-6
        M = np.eye(N) - L / lambda_max_est
        y = np.ones(N) / N
        for _ in range(100):
            y_new = M @ y
            ny = np.linalg.norm(y_new)
            if ny < 1e-12:
                break
            y_new /= ny
            if np.linalg.norm(y_new - y) < 1e-10:
                break
            y = y_new
        # Orthogonalize against ones
        ones = np.ones(N) / N
        fiedler = y - np.dot(y, ones) * ones
        nf = np.linalg.norm(fiedler)
        if nf < 1e-12:
            fiedler = np.zeros(N)
        else:
            fiedler /= nf
            if np.sum(fiedler) < 0:
                fiedler = -fiedler
    FiedlerVec = fiedler

    # Normalize features to [0, 1]
    def norm(x):
        min_val = np.min(x)
        max_val = np.max(x)
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    D_n = norm(D)
    MeanND_n = norm(MeanND)
    Tri_n = norm(Tri)
    Clus_n = norm(Clus)
    OpenPairs_n = norm(OpenPairs)
    OutwardExp_n = norm(OutwardExp)
    MinNbrDeg_n = norm(MinNbrDeg)
    Spectral_n = norm(SpectralCentrality)
    Fiedler_n = norm(FiedlerVec)

    # --- Evaluation Function ---
    adj_rows = A_bool

    def evaluate_fast(scores):
        """Vectorized greedy independent set evaluation."""
        indices = np.argsort(-scores, kind='stable')
        is_picked = np.zeros(N, dtype=bool)
        count = 0
        for v in indices:
            if not is_picked[v]:
                count += 1
                is_picked[v] = True
                is_picked[adj_rows[v]] = True
        return count

    best_size = -1
    best_scores = None

    # --- Search Phase 1: Deterministic Candidates ---
    # Weights: (w_deg, w_meanND, w_clus, w_open, w_tri, w_outward, w_minND, w_spectral, w_fiedler)
    # Score = -w_deg*D + w_meanND*MeanND - w_clus*Clus + w_open*Open - w_tri*Tri + w_outward*Outward + w_minND*MinND - w_spectral*Spectral - w_fiedler*Fiedler
    
    candidates = [
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # Baseline: -Degree
        (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # -Deg + Open
        (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # -Deg + MeanND
        (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # -Deg - Clus
        (1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),  # Mix
        (0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # MeanND heavy
        (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),  # -Deg + Outward
        (1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0), # -Deg + Open + Outward
        (1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0), # Mix + Outward
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0), # -Deg + MinND
        (1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0), # All mixed
        (1.5, 0.5, 0.5, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0), # Stronger Degree weight
        # Candidates with spectral centrality
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0), # -Deg - Spectral
        (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0), # Lighter Deg - Spectral
        (1.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0), # Mix + Spectral
        (1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0), # -Deg + Open + Outward + Spectral
        (0.8, 0.5, 0.3, 0.5, 0.0, 0.3, 0.3, 1.0, 0.0), # Balanced + Spectral
        # Candidates with Fiedler vector
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), # -Deg - Fiedler
        (1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 1.0), # All + Fiedler
        (0.8, 0.5, 0.3, 0.5, 0.0, 0.3, 0.3, 0.5, 1.0), # Balanced + Spectral + Fiedler
        (1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 1.0), # -Deg + Open + Outward + Spectral + Fiedler
    ]
    
    for w in candidates:
        scores = (-w[0] * D_n + w[1] * MeanND_n - w[2] * Clus_n 
                  + w[3] * OpenPairs_n - w[4] * Tri_n 
                  + w[5] * OutwardExp_n + w[6] * MinNbrDeg_n
                  - w[7] * Spectral_n - w[8] * Fiedler_n)
        size = evaluate_fast(scores)
        if size > best_size:
            best_size = size
            best_scores = scores.copy()

    # --- Search Phase 2: Random Search ---
    rng = np.random.default_rng(42)
    max_trials = 20000  # Upper bound, will be cut by time
    
    # Ranges based on Inspiration 1's successful ranges + spectral + fiedler
    ranges = {
        'w1': (0.3, 2.5),   # Degree
        'w2': (0.0, 2.0),   # MeanND
        'w3': (0.0, 1.5),   # Clustering
        'w4': (0.0, 2.0),   # Open Pairs
        'w5': (0.0, 0.8),   # Triangles
        'w6': (0.0, 1.5),   # Outward Exp
        'w7': (0.0, 1.0),   # Min Nbr Deg
        'w8': (0.0, 2.0),   # Spectral Centrality
        'w9': (0.0, 2.0),   # Fiedler Vector
    }

    trial_count = 0
    while trial_count < max_trials:
        if time.time() > DEADLINE - 15.0: # Reserve 15s for local refinement
            break
            
        w1 = rng.uniform(*ranges['w1'])
        w2 = rng.uniform(*ranges['w2'])
        w3 = rng.uniform(*ranges['w3'])
        w4 = rng.uniform(*ranges['w4'])
        w5 = rng.uniform(*ranges['w5'])
        w6 = rng.uniform(*ranges['w6'])
        w7 = rng.uniform(*ranges['w7'])
        w8 = rng.uniform(*ranges['w8'])
        w9 = rng.uniform(*ranges['w9'])
        
        scores = (-w1 * D_n + w2 * MeanND_n - w3 * Clus_n 
                  + w4 * OpenPairs_n - w5 * Tri_n 
                  + w6 * OutwardExp_n + w7 * MinNbrDeg_n
                  - w8 * Spectral_n - w9 * Fiedler_n)
        
        size = evaluate_fast(scores)
        
        if size > best_size:
            best_size = size
            best_scores = scores.copy()
            
        trial_count += 1

    # --- Search Phase 3: Local Swap Refinement ---
    # Directly optimize the greedy outcome by swapping vertex priorities
    if best_scores is not None and time.time() < DEADLINE - 1.0:
        # First: small perturbations
        for _ in range(200):
            if time.time() > DEADLINE - 2.0:
                break
            perturbed = best_scores + rng.normal(0, 0.03, size=N)
            size = evaluate_fast(perturbed)
            if size > best_size:
                best_size = size
                best_scores = perturbed.copy()
        
        # Then: targeted swap-based local search
        # Get current ordering
        order = np.argsort(-best_scores, kind='stable')
        # Try swapping adjacent pairs in the ordering that might improve the result
        n_swaps_attempted = 0
        max_swap_iterations = 5000
        while n_swaps_attempted < max_swap_iterations:
            if time.time() > DEADLINE - 0.5:
                break
            
            # Pick a random position in the ordering
            pos = rng.integers(0, N - 1)
            # Swap with a nearby position (within ±10)
            offset = rng.integers(-10, 11)
            pos2 = pos + offset
            if pos2 < 0 or pos2 >= N:
                n_swaps_attempted += 1
                continue
            
            # Create a modified score vector by swapping the scores at these positions
            v1 = order[pos]
            v2 = order[pos2]
            if v1 == v2:
                n_swaps_attempted += 1
                continue
            
            modified = best_scores.copy()
            modified[v1], modified[v2] = best_scores[v2], best_scores[v1]
            
            size = evaluate_fast(modified)
            if size > best_size:
                best_size = size
                best_scores = modified.copy()
                order = np.argsort(-best_scores, kind='stable')
            n_swaps_attempted += 1

    if best_scores is None:
        return -D.astype(float)

    return best_scores
# EVOLVE-BLOCK-END
