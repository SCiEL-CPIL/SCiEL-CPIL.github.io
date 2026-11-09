"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_decode(adjacency, scores):
    """Greedy decoder: pick vertices in descending score order, skipping neighbours."""
    n = adjacency.shape[0]
    order = np.argsort(-scores, kind='stable')
    chosen = np.zeros(n, dtype=bool)
    forbidden = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if chosen[v] or forbidden[v]:
            continue
        chosen[v] = True
        count += 1
        forbidden[adjacency[v]] = True
    return count


def _compute_degree(adjacency):
    return adjacency.sum(axis=1).astype(float)


def _compute_neighbor_degree_sum(adjacency):
    deg = adjacency.sum(axis=1)
    return (adjacency @ deg).astype(float)


def _compute_coreness(adjacency):
    """Compute k-core number for each vertex using iterative peeling."""
    n = adjacency.shape[0]
    deg = adjacency.sum(axis=1).astype(int).copy()
    core = np.zeros(n, dtype=int)
    removed = np.zeros(n, dtype=bool)
    current_deg = deg.copy()

    for _ in range(n):
        active_mask = ~removed
        if not active_mask.any():
            break
        active_indices = np.where(active_mask)[0]
        min_deg = current_deg[active_indices].min()
        candidates = active_indices[current_deg[active_indices] == min_deg]
        v = candidates[0]
        removed[v] = True
        core[v] = min_deg
        for u in np.where(adjacency[v])[0]:
            if not removed[u]:
                current_deg[u] -= 1

    return core.astype(float)


def _compute_effective_degree(adjacency):
    A_int = adjacency.astype(np.int32)
    degree = A_int.sum(axis=1).astype(float)
    M = A_int @ A_int.T
    S = (A_int * M).sum(axis=1).astype(float)
    edges_in_N = 0.5 * S
    return degree - edges_in_N


def _compute_neighborhood_sparsity(adjacency):
    A_int = adjacency.astype(np.int32)
    degree = A_int.sum(axis=1).astype(float)
    M = A_int @ A_int.T
    S = (A_int * M).sum(axis=1).astype(float)
    edges_in_N = 0.5 * S
    d_v = degree
    total_pairs = d_v * (d_v - 1.0) / 2.0
    sparsity_ratio = np.where(total_pairs > 0,
                              1.0 - edges_in_N / np.maximum(total_pairs, 1.0),
                              1.0)
    return sparsity_ratio


def _compute_mean_neighbor_degree(adjacency):
    degree = adjacency.sum(axis=1).astype(float)
    neighbor_deg_sum = adjacency @ degree
    return neighbor_deg_sum / np.maximum(degree, 1.0)


def _compute_spectral_features(adjacency):
    """
    Compute spectral features using the graph Laplacian.
    Returns (fiedler_vector, eigenvalue_gap, partition_indicator)
    - fiedler_vector: eigenvector corresponding to 2nd smallest eigenvalue
    - eigenvalue_gap: lambda_2 - lambda_1 (spectral gap)
    - partition_indicator: +1/-1 based on sign of fiedler vector
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.zeros(0), 0.0, np.zeros(0)

    A = adjacency.astype(float)
    degree = A.sum(axis=1)
    D = np.diag(degree)
    L = D - A  # Graph Laplacian

    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # eigvals sorted in ascending order
        # lambda_0 = 0 (for connected graph), lambda_1 = Fiedler value
        if len(eigvals) >= 2:
            fiedler_val = eigvals[1]
            fiedler_vec = eigvecs[:, 1]
        else:
            fiedler_val = 0.0
            fiedler_vec = np.zeros(n)

        # Spectral gap: difference between lambda_1 and lambda_0
        gap = eigvals[1] - eigvals[0] if len(eigvals) >= 2 else 0.0

        # Partition indicator based on sign of Fiedler vector
        partition = np.sign(fiedler_vec)
        # Where sign is 0, assign based on which side has more vertices
        if np.sum(partition > 0) < n / 2:
            partition = -partition

        return fiedler_vec, gap, partition

    except Exception:
        return np.zeros(n), 0.0, np.zeros(n)


def _compute_spectral_partition_scores(adjacency):
    """
    Compute scores based on spectral partition.
    Vertices in the smaller partition get higher scores (prioritized),
    as they are in the "sparse" side of the cut.
    """
    fiedler_vec, gap, partition = _compute_spectral_features(adjacency)

    # Score: vertices in smaller partition get bonus
    # Also, vertices with smaller |fiedler_vec| are more "central" to the cut
    # and may be better to exclude, while vertices far from 0 are more peripheral

    n = len(fiedler_vec)
    if n == 0:
        return np.zeros(0)

    # Normalize fiedler vector
    max_abs = np.max(np.abs(fiedler_vec))
    if max_abs > 1e-12:
        fiedler_norm = fiedler_vec / max_abs
    else:
        fiedler_norm = np.zeros(n)

    # Score components:
    # 1. Partition bonus: +1 for smaller partition
    part_scores = np.zeros(n)
    pos_count = np.sum(partition > 0)
    neg_count = np.sum(partition < 0)
    if pos_count <= neg_count:
        part_scores[partition > 0] = 1.0
        part_scores[partition < 0] = -0.5
    else:
        part_scores[partition < 0] = 1.0
        part_scores[partition > 0] = -0.5

    # 2. Distance from cut: vertices far from 0 in fiedler vec are more peripheral
    #    Give bonus to vertices with large |fiedler_vec|
    dist_scores = np.abs(fiedler_norm)

    # Combine
    scores = 0.6 * part_scores + 0.4 * dist_scores

    return scores


def _compute_territory_size(adjacency, max_depth=3):
    n = adjacency.shape[0]
    degree = adjacency.sum(axis=1).astype(np.int32)
    territory = np.zeros(n, dtype=np.int32)

    contested = np.zeros(n, dtype=bool)
    for v in range(n):
        if degree[v] > 0:
            neighbors = np.where(adjacency[v])[0]
            if np.any(degree[neighbors] > degree[v]):
                contested[v] = True

    for v in range(n):
        if contested[v]:
            territory[v] = 0
            continue

        visited = np.zeros(n, dtype=bool)
        visited[v] = True
        current_frontier = [v]
        count = 1

        for depth in range(max_depth):
            next_frontier = []
            for node in current_frontier:
                neighbors = np.where(adjacency[node])[0]
                for nb in neighbors:
                    if not visited[nb] and not contested[nb]:
                        visited[nb] = True
                        count += 1
                        next_frontier.append(nb)
            current_frontier = next_frontier
            if not current_frontier:
                break

        territory[v] = count

    return territory.astype(float)


def _search_best_scores(adjacency, time_limit=100.0):
    n = adjacency.shape[0]
    start_time = time.time()
    deadline = start_time + time_limit

    # Precompute base features
    degree = _compute_degree(adjacency)
    nds = _compute_neighbor_degree_sum(adjacency)
    coreness = _compute_coreness(adjacency)
    effective_deg = _compute_effective_degree(adjacency)
    sparsity_ratio = _compute_neighborhood_sparsity(adjacency)
    mean_neighbor_deg = _compute_mean_neighbor_degree(adjacency)

    # Spectral features
    spectral_scores = _compute_spectral_partition_scores(adjacency)
    fiedler_vec, spectral_gap, partition = _compute_spectral_features(adjacency)

    # Compute territory size
    territory = _compute_territory_size(adjacency, max_depth=3)

    # Normalize features to [0, 1] range
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    deg_norm = normalize(degree)
    nds_norm = normalize(nds)
    core_norm = normalize(coreness)
    eff_deg_norm = normalize(effective_deg)
    sparsity_norm = normalize(sparsity_ratio)
    mnd_norm = normalize(mean_neighbor_deg)
    territory_norm = normalize(territory)
    spectral_norm = normalize(spectral_scores)

    # Fiedler vector absolute value as a feature
    fiedler_abs_norm = normalize(np.abs(fiedler_vec))

    best_scores = -degree
    best_count = _greedy_decode(adjacency, best_scores)

    # Features with positive weight = prefer high value
    features = {
        '-deg': -deg_norm,
        '-eff': -eff_deg_norm,
        'sparsity': sparsity_norm,
        'core': core_norm,
        'mnd': mnd_norm,
        'territory': territory_norm,
        'spectral': spectral_norm,
        'fiedler_abs': fiedler_abs_norm,
    }

    feature_names = list(features.keys())
    feature_arrays = [features[name] for name in feature_names]

    # Structured combinations
    structured_combos = [
        # (w_deg, w_eff, w_sparsity, w_core, w_mnd, w_territory, w_spectral, w_fiedler)
        (-1, 0, 0, 0, 0, 0, 0, 0),
        (0, -1, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 0, 0, 1),
        (-0.5, -0.5, 0, 0, 0, 0, 0, 0),
        (-0.3, -0.3, 0.2, 0, 0, 0, 0, 0),
        (-0.2, -0.4, 0.2, 0, 0, 0, 0, 0),
        (-0.4, -0.2, 0.2, 0, 0, 0, 0, 0),
        (-0.5, -0.5, 0, 0.2, 0, 0, 0, 0),
        (-0.5, -0.5, 0, 0, 0.2, 0, 0, 0),
        (-0.5, -0.5, 0, 0, 0, 0.2, 0, 0),
        (-0.5, -0.5, 0, 0, 0, 0, 0.2, 0),
        (-0.5, -0.5, 0, 0, 0, 0, 0, 0.2),
        # Spectral-heavy combos
        (-0.3, -0.3, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1),
        (-0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1),
        (-0.3, -0.3, 0.2, 0.1, 0.1, 0.1, 0.3, 0.1),
        (-0.2, -0.4, 0.2, 0.1, 0.1, 0.1, 0.3, 0.1),
        (-0.4, -0.2, 0.2, 0.1, 0.1, 0.1, 0.3, 0.1),
        (-0.5, -0.5, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1),
        # Fiedler-heavy combos
        (-0.3, -0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4),
        (-0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5),
        (-0.3, -0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.3),
        # Mixed spectral + fiedler
        (-0.3, -0.3, 0.1, 0.1, 0.1, 0.1, 0.25, 0.25),
        (-0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.3, 0.2),
        (-0.3, -0.3, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2),
        # Territory-heavy with spectral
        (-0.2, -0.2, 0.1, 0.1, 0.1, 0.3, 0.2, 0.1),
        (-0.3, -0.3, 0.1, 0.1, 0.1, 0.25, 0.2, 0.1),
        # Balanced combos
        (-0.3, -0.3, 0.2, 0.1, 0.1, 0.1, 0.15, 0.1),
        (-0.2, -0.4, 0.2, 0.1, 0.1, 0.1, 0.15, 0.1),
        (-0.4, -0.2, 0.2, 0.1, 0.1, 0.1, 0.15, 0.1),
        (-0.5, -0.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1),
    ]

    for weights in structured_combos:
        if time.time() > deadline - 5:
            break
        scores = sum(w * f for w, f in zip(weights, feature_arrays))
        count = _greedy_decode(adjacency, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()

    # Random search
    rng = np.random.default_rng(42)
    num_random_tries = 3000
    for i in range(num_random_tries):
        if time.time() > deadline - 5:
            break
        w1 = rng.uniform(-1, 0)
        w2 = rng.uniform(-1, 0)
        w3 = rng.uniform(0, 0.5)
        w4 = rng.uniform(0, 0.5)
        w5 = rng.uniform(0, 0.3)
        w6 = rng.uniform(0, 0.5)
        w7 = rng.uniform(0, 0.5)
        w8 = rng.uniform(0, 0.5)
        weights = [w1, w2, w3, w4, w5, w6, w7, w8]
        scores = sum(w * f for w, f in zip(weights, feature_arrays))
        count = _greedy_decode(adjacency, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()

    # Local search
    local_search_steps = 1500
    current_scores = best_scores.copy()
    current_count = best_count

    for i in range(local_search_steps):
        if time.time() > deadline - 5:
            break
        noise = rng.normal(0, 0.05, size=n)
        perturbed = current_scores + noise
        count = _greedy_decode(adjacency, perturbed)
        if count > current_count:
            current_count = count
            current_scores = perturbed.copy()
            if count > best_count:
                best_count = count
                best_scores = perturbed.copy()
        elif i % 10 == 0:
            noise = rng.normal(0, 0.2, size=n)
            perturbed = current_scores + noise
            count = _greedy_decode(adjacency, perturbed)
            if count > current_count:
                current_count = count
                current_scores = perturbed.copy()
                if count > best_count:
                    best_count = count
                    best_scores = perturbed.copy()

    # Polish
    polish_steps = 500
    for i in range(polish_steps):
        if time.time() > deadline - 2:
            break
        noise = rng.normal(0, 0.01, size=n)
        perturbed = best_scores + noise
        count = _greedy_decode(adjacency, perturbed)
        if count > best_count:
            best_count = count
            best_scores = perturbed.copy()
        elif i % 5 == 0:
            v = rng.integers(0, n)
            for delta in [0.1, -0.1, 0.05, -0.05]:
                perturbed = best_scores.copy()
                perturbed[v] += delta
                count = _greedy_decode(adjacency, perturbed)
                if count > best_count:
                    best_count = count
                    best_scores = perturbed.copy()
                    break

    return best_scores


def priority(adjacency):
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([], dtype=float)

    start_time = time.time()
    time_limit = 100.0

    try:
        scores = _search_best_scores(adjacency, time_limit=time_limit)
    except Exception:
        scores = -adjacency.sum(axis=1).astype(float)

    return scores
# EVOLVE-BLOCK-END
