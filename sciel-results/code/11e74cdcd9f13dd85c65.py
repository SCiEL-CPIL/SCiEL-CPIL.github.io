"""Combined program: local-search optimization of greedy decoder output."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetry and remove self-loops
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)

    # Convert to boolean for faster neighbor lookups
    A_bool = A > 0.5

    # Precompute neighbor lists for fast simulation
    neighbors_list = [np.where(A_bool[i])[0] for i in range(N)]

    # --- 1. Initial Heuristic: Combine degree and neighbor-degree signals ---
    deg = A.sum(axis=1)
    neighbor_deg_sum = A.dot(deg)
    safe_deg = deg + 1e-6
    avg_neighbor_deg = neighbor_deg_sum / safe_deg

    # Heuristic: Low degree is good. Being near hubs is bad (penalty).
    # Score = -(deg + 0.5 * avg_neighbor_deg)
    # Then normalize to [0, 1] to make noise scaling easier.
    raw_scores = -(deg + 0.5 * avg_neighbor_deg)

    # Normalize to [0, 1]
    if raw_scores.max() > raw_scores.min():
        scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    else:
        scores = np.ones(N) / N

    def simulate_fast(s):
        """
        Fast greedy simulation.
        Returns the size of the independent set.
        """
        order = np.argsort(-s, kind='stable')
        active = np.ones(N, dtype=bool)
        count = 0

        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return count

    # --- 2. Baseline ---
    best_val = simulate_fast(scores)
    best_scores = scores.copy()

    # --- 3. Local Search with Multiple Restarts ---
    rng = np.random.default_rng(42)

    # We will perform multiple restarts.
    # For each restart, we do a local search.
    # To improve, let's try a mix of:
    # 1. Random swaps (global exploration)
    # 2. Rank-adjacent swaps (local exploitation)

    num_restarts = 15  # More restarts, shorter time each?

    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break

        if restart == 0:
            current_scores = best_scores.copy()
            current_val = best_val
        else:
            # Perturb the best score found so far
            # Add Gaussian noise with increasing scale
            noise_scale = 0.1 + 0.05 * restart
            noise = rng.normal(0, noise_scale, N)
            current_scores = best_scores + noise
            current_val = simulate_fast(current_scores)

        # Local search loop
        no_improve = 0
        max_no_improve = 8000  # Reset on improvement

        while time.time() - start_time < DEADLINE and no_improve < max_no_improve:
            # Alternate between random and rank-adjacent swaps
            if no_improve % 5 == 0 and N > 10:
                # Rank-adjacent swap: pick a random vertex, try swapping with nearby in rank
                order = np.argsort(-current_scores, kind='stable')
                i_idx = rng.integers(0, N)
                i = order[i_idx]
                # Pick a neighbor in rank space
                offset = rng.integers(1, min(10, N))
                j_idx = (i_idx + offset) % N
                j = order[j_idx]
            else:
                # Random swap
                i, j = rng.integers(0, N, size=2)
                if i == j:
                    continue

            # Swap
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            new_val = simulate_fast(current_scores)

            if new_val > current_val:
                current_val = new_val
                no_improve = 0
                if current_val > best_val:
                    best_val = current_val
                    best_scores = current_scores.copy()
            else:
                # Revert
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                no_improve += 1

        # If time is running out, break
        if time.time() - start_time > DEADLINE - 5:
            break

    return best_scores
# EVOLVE-BLOCK-END
