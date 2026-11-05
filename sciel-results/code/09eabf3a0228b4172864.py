import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5

    D = A.sum(axis=1)
    D_safe = D + 1e-9

    # Spectral feature: principal eigenvector of adjacency matrix
    # This gives a global 2-coloring-like signal
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        # Principal eigenvector (largest eigenvalue)
        principal_idx = np.argmax(eigenvalues)
        principal_vec = eigenvectors[:, principal_idx]
        # Normalize to [0, 1]
        p_min = principal_vec.min()
        p_max = principal_vec.max()
        if p_max > p_min:
            principal_norm = (principal_vec - p_min) / (p_max - p_min)
        else:
            principal_norm = np.zeros(N)
    except Exception:
        principal_norm = np.zeros(N)

    # Eigenvector centrality (power iteration as backup / additional feature)
    # Actually eigh already gives us this. Let's also compute a "signed" version
    # where we use the absolute value as a centrality measure
    eig_centrality = np.abs(principal_vec)
    ec_min = eig_centrality.min()
    ec_max = eig_centrality.max()
    if ec_max > ec_min:
        eig_cent_norm = (eig_centrality - ec_min) / (ec_max - ec_min)
    else:
        eig_cent_norm = np.zeros(N)

    # Second eigenvector for additional structure
    if len(eigenvalues) > 1:
        # Sort by eigenvalue magnitude
        sorted_idx = np.argsort(-np.abs(eigenvalues))
        if sorted_idx[0] == principal_idx and len(sorted_idx) > 1:
            second_idx = sorted_idx[1]
        else:
            second_idx = sorted_idx[0]
        second_vec = eigenvectors[:, second_idx]
        s_min = second_vec.min()
        s_max = second_vec.max()
        if s_max > s_min:
            second_norm = (second_vec - s_min) / (s_max - s_min)
        else:
            second_norm = np.zeros(N)
    else:
        second_norm = np.zeros(N)

    # Local features
    SND = A @ D
    MeanND = SND / D_safe

    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0

    Clus = np.zeros(N)
    mask = D > 1
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))

    OpenPairs = np.maximum(0, 0.5 * D * (D - 1.0) - Tri)

    # Degree of neighbors (average)
    # Outward expansion: sum of (D[j]-1) for neighbors j
    OutwardExp = np.zeros(N)
    for i in range(N):
        nbrs = np.where(A_bool[i])[0]
        if len(nbrs) > 0:
            OutwardExp[i] = np.sum(D[nbrs] - 1.0)
    OutwardExp = OutwardExp / D_safe

    # Normalize all features
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
    OutwardExp_n = norm(OutwardExp)
    principal_n = norm(principal_norm)
    eig_cent_n = norm(eig_cent_norm)
    second_n = norm(second_norm)

    # Fast greedy evaluation
    adj_rows = A_bool

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

    best_size = -1
    best_scores = None

    # Baseline: low degree
    base_score = -D_n
    base_size = evaluate_fast(base_score)
    best_size = base_size
    best_scores = base_score.copy()

    # Spectral baseline: prefer vertices with high principal eigenvector value
    # and low degree (to avoid picking adjacent high-centrality vertices)
    spectral_score = principal_n - 0.5 * D_n
    spectral_size = evaluate_fast(spectral_score)
    if spectral_size > best_size:
        best_size = spectral_size
        best_scores = spectral_score.copy()

    # Another spectral variant: use the sign/magnitude to prefer one side
    # Vertices with positive principal eigenvector component form one "color class"
    # We want to pick from the larger color class, but also consider degree
    # Score: high if in dominant color class AND low degree
    color_class = principal_vec > 0
    spectral_color = np.where(color_class, 1.0, 0.0) - 0.3 * D_n
    sc_size = evaluate_fast(spectral_color)
    if sc_size > best_size:
        best_size = sc_size
        best_scores = spectral_color.copy()

    # Combine spectral with local features via weight search
    rng = np.random.default_rng(42)

    # Feature vector: [D_n, MeanND_n, Clus_n, OpenPairs_n, Tri_n, OutwardExp_n,
    #                  principal_n, eig_cent_n, second_n]
    # Score = -w1*D + w2*MeanND - w3*Clus + w4*Open - w5*Tri + w6*Outward
    #         + w7*principal + w8*eig_cent + w9*second

    candidates = [
        # (w1..w9)
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # -D
        (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),  # -0.5D + principal
        (0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),  # -0.3D + principal
        (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),  # -0.5D + eig_cent
        (0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),  # -0.5D + 0.5MeanND + principal
        (0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),  # -0.5D - 0.5Clus + principal
        (0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),  # -0.5D + Open + principal
        (0.3, 0.3, 0.3, 0.3, 0.0, 0.3, 1.0, 0.5, 0.3),  # Mix + spectral
        (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0),  # -0.5D + 0.5principal + 0.5eig_cent
        (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5),  # -0.5D + principal + 0.5second
        (0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0),  # -0.7D + 0.7principal
        (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),  # -0.2D + principal
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0),  # -D + 0.5principal + 0.5eig_cent
        (0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 1.0, 0.5, 0.5),  # All mixed
    ]

    for w in candidates:
        scores = (-w[0] * D_n + w[1] * MeanND_n - w[2] * Clus_n
                  + w[3] * OpenPairs_n - w[4] * Tri_n + w[5] * OutwardExp_n
                  + w[6] * principal_n + w[7] * eig_cent_n + w[8] * second_n)
        size = evaluate_fast(scores)
        if size > best_size:
            best_size = size
            best_scores = scores.copy()

    # Random search
    max_trials = 6000
    trial_count = 0

    while trial_count < max_trials:
        if time.time() > DEADLINE:
            break

        w1 = rng.uniform(0.1, 2.0)
        w2 = rng.uniform(0.0, 1.5)
        w3 = rng.uniform(0.0, 1.0)
        w4 = rng.uniform(0.0, 1.5)
        w5 = rng.uniform(0.0, 0.5)
        w6 = rng.uniform(0.0, 1.0)
        w7 = rng.uniform(0.0, 2.0)
        w8 = rng.uniform(0.0, 1.5)
        w9 = rng.uniform(0.0, 1.0)

        scores = (-w1 * D_n + w2 * MeanND_n - w3 * Clus_n
                  + w4 * OpenPairs_n - w5 * Tri_n + w6 * OutwardExp_n
                  + w7 * principal_n + w8 * eig_cent_n + w9 * second_n)

        size = evaluate_fast(scores)

        if size > best_size:
            best_size = size
            best_scores = scores.copy()

        trial_count += 1

    # Local refinement: perturb the best score vector
    if best_scores is not None and time.time() < DEADLINE - 5.0:
        for _ in range(300):
            if time.time() > DEADLINE:
                break
            perturbed = best_scores + rng.normal(0, 0.05, size=N)
            size = evaluate_fast(perturbed)
            if size > best_size:
                best_size = size
                best_scores = perturbed.copy()

    return best_scores
