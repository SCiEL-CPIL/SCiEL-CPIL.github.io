"""Improved local-search optimization of greedy decoder for MIS."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetry and remove self-loops
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)

    # Convert to boolean for faster neighbor lookups
    A_bool = A > 0.5

    # Precompute neighbor lists and degree
    neighbors_list = [np.where(A_bool[i])[0] for i in range(N)]
    deg = A.sum(axis=1)

    # --- 1. Rich initial heuristic ---
    # Multiple structural features
    neighbor_deg_sum = A.dot(deg)
    safe_deg = deg + 1e-6
    avg_neighbor_deg = neighbor_deg_sum / safe_deg

    # Second-order: sum of neighbor degrees of neighbors
    neighbor2_deg_sum = A.dot(neighbor_deg_sum)
    avg_neighbor2_deg = neighbor2_deg_sum / safe_deg

    # Clustering coefficient proxy: for each vertex, fraction of neighbor pairs that are connected
    A2_diag = np.diag(A @ A)
    # Clustering proxy: A2_diag / (deg*(deg-1)) where defined
    denom = deg * (deg - 1)
    denom = np.where(denom > 0, denom, 1e-6)
    clustering = A2_diag / denom

    # Combine features: we want low degree, low neighbor-degree, low clustering
    w1, w2, w3, w4 = 1.0, 0.5, 0.3, 0.2
    raw_scores = -(w1 * deg + w2 * avg_neighbor_deg + w3 * clustering + w4 * avg_neighbor2_deg)

    # Normalize to [0, 1]
    if raw_scores.max() > raw_scores.min():
        scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    else:
        scores = np.ones(N) / N

    def simulate_fast(s):
        """
        Fast greedy simulation using vectorized operations.
        Returns the size of the independent set.
        """
        order = np.argsort(-s, kind='stable')
        active = np.ones(N, dtype=bool)
        count = 0

        for v in order:
            if active[v]:
                count += 1
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return count

    def simulate_with_min_degree(s):
        """
        Greedy simulation that at each step picks the vertex with minimum degree
        among active vertices, breaking ties by score. This is a well-known
        approximation for MIS.
        """
        active = np.ones(N, dtype=bool)
        count = 0
        
        while active.any():
            # Find active vertices
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            
            # Compute degree of each active vertex within the active subgraph
            # degree within active = number of active neighbors
            degs = np.zeros(N, dtype=float)
            for v in active_indices:
                nbrs = neighbors_list[v]
                degs[v] = np.sum(active[nbrs]) if len(nbrs) > 0 else 0
            
            # We want to pick the vertex with minimum degree, breaking ties by higher score
            # So we want to minimize degree, and among those maximize score
            # Create a combined key: primary = degree, secondary = -score
            # But we need to pick min of (degree, -score)
            # Actually, let's just find the min degree among active, then pick max score among those
            min_deg = degs[active].min()
            # Vertices with min degree
            min_deg_mask = (degs == min_deg) & active
            min_deg_vertices = np.where(min_deg_mask)[0]
            
            if len(min_deg_vertices) == 1:
                v = min_deg_vertices[0]
            else:
                # Pick the one with highest score
                v = min_deg_vertices[np.argmax(s[min_deg_vertices])]
            
            active[v] = False
            nbrs = neighbors_list[v]
            if len(nbrs) > 0:
                active[nbrs] = False
            count += 1
            
        return count

    # --- 2. Baseline ---
    best_val = simulate_fast(scores)
    best_scores = scores.copy()

    # Also try the min-degree greedy as an alternative baseline
    min_deg_val = simulate_with_min_degree(scores)
    if min_deg_val > best_val:
        best_val = min_deg_val
        # We don't have scores for min-degree, but we can use it as a reference
        # Actually, the min-degree strategy is a different decoder. Since the task
        # specifies a fixed greedy decoder that picks by descending score, we
        # can only optimize the scores. So the min-degree result is just a
        # reference upper bound for what's possible.
        # Let's not use it as best_val since it's not from our score-based decoder.
        # Revert: best_val should be from simulate_fast only.
        best_val = simulate_fast(scores)

    # --- 3. Local Search with Simulated Annealing ---
    rng = np.random.default_rng(42)

    num_restarts = 20
    current_temp = 1.0
    cooling = 0.95

    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 2:
            break

        if restart == 0:
            current_scores = best_scores.copy()
            current_val = best_val
        else:
            # Perturb the best score found so far
            noise_scale = 0.08 + 0.03 * restart
            noise = rng.normal(0, noise_scale, N)
            current_scores = best_scores + noise
            current_val = simulate_fast(current_scores)

        # Local search with simulated annealing
        no_improve = 0
        max_no_improve = 12000
        temp = current_temp

        while time.time() - start_time < DEADLINE - 3 and no_improve < max_no_improve:
            # Choose move type based on temperature
            r = rng.random()
            if r < 0.4:
                # Random swap
                i, j = rng.integers(0, N, size=2)
                if i == j:
                    continue
            elif r < 0.7:
                # Rank-adjacent swap
                order = np.argsort(-current_scores, kind='stable')
                i_idx = rng.integers(0, N)
                offset = rng.integers(1, min(15, N))
                j_idx = (i_idx + offset) % N
                i, j = order[i_idx], order[j_idx]
                if i == j:
                    continue
            else:
                # Score perturbation: add small noise to a random vertex
                i = rng.integers(0, N)
                j = rng.integers(0, N)
                if i == j:
                    continue
                # Perturb scores rather than swap
                delta = rng.normal(0, 0.1)
                current_scores[i] += delta
                new_val = simulate_fast(current_scores)
                if new_val > current_val:
                    current_val = new_val
                    no_improve = 0
                    if current_val > best_val:
                        best_val = current_val
                        best_scores = current_scores.copy()
                else:
                    current_scores[i] -= delta
                    no_improve += 1
                continue

            # Swap-based move
            si, sj = current_scores[i], current_scores[j]
            current_scores[i], current_scores[j] = sj, si
            new_val = simulate_fast(current_scores)

            delta_val = new_val - current_val
            if delta_val > 0 or (temp > 0.01 and rng.random() < np.exp(delta_val / temp)):
                current_val = new_val
                no_improve = 0
                if current_val > best_val:
                    best_val = current_val
                    best_scores = current_scores.copy()
            else:
                current_scores[i], current_scores[j] = si, sj
                no_improve += 1

            # Cool down
            temp *= cooling

        # Reset temperature for next restart
        current_temp = max(0.1, current_temp * 0.8)

    return best_scores
# EVOLVE-BLOCK-END
