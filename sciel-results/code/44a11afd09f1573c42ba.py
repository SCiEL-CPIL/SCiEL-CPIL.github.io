"""Heuristic matrix for ant colony optimization of capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _aco_eval_fast(dm, demands, capacity, H, n_ants, n_iterations, seed, time_limit):
    """
    Fast ACO simulation to estimate heuristic quality.
    """
    start_time = time.time()
    n = dm.shape[0]
    tau = np.ones((n, n), dtype=np.float64)
    rng = np.random.RandomState(seed)
    best_cost = np.inf
    demands = np.asarray(demands, dtype=np.float64)

    for it in range(n_iterations):
        if time.time() - start_time > time_limit:
            break

        iteration_costs = []

        for ant in range(n_ants):
            if time.time() - start_time > time_limit:
                break

            current_node = 0
            remaining_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            total_dist = 0.0

            while True:
                valid_mask = ~visited
                valid_mask[0] = False
                valid_mask &= (demands <= remaining_cap)

                if not np.any(valid_mask):
                    total_dist += dm[current_node, 0]
                    remaining_cap = capacity
                    current_node = 0
                    if np.all(visited):
                        break
                    continue

                row_tau = tau[current_node]
                row_eta = H[current_node]

                weights = np.zeros(n)
                weights[valid_mask] = row_tau[valid_mask] * row_eta[valid_mask]

                total_w = np.sum(weights)
                if total_w <= 0:
                    valid_indices = np.where(valid_mask)[0]
                    next_node = rng.choice(valid_indices)
                else:
                    probs = weights / total_w
                    next_node = rng.choice(n, p=probs)

                total_dist += dm[current_node, next_node]
                visited[next_node] = True
                remaining_cap -= demands[next_node]
                current_node = next_node

                if np.all(visited):
                    total_dist += dm[current_node, 0]
                    break

            iteration_costs.append(total_dist)

        if iteration_costs:
            current_best = min(iteration_costs)
            if current_best < best_cost:
                best_cost = current_best
            tau *= 0.9
            tau += 1.0 / current_best

    return best_cost


def _evaluate_heuristic(dm, demands, capacity, H, time_limit, n_seeds=3):
    """
    Evaluate a heuristic by running ACO simulations with multiple seeds.
    Returns the average best cost.
    """
    costs = []
    for s in range(n_seeds):
        cost = _aco_eval_fast(dm, demands, capacity, H,
                              n_ants=6, n_iterations=6, seed=s,
                              time_limit=time_limit / n_seeds)
        if cost < np.inf:
            costs.append(cost)
    if costs:
        return np.mean(costs)
    return np.inf


def _gaussian_spatial_heuristic(dm, demands, capacity, coords, sigma=0.1, gamma=1.5, demand_weight=1.0):
    """
    Smooth Gaussian spatial kernel heuristic with demand-aware depot weighting.
    Uses a smooth spatial proximity kernel (no discontinuities) combined with
    inverse distance and demand-based depot preference.
    """
    n = dm.shape[0]
    dm_safe = np.where(dm > 1e-10, dm, 1e10)
    
    # Euclidean distance matrix
    eucl_dm = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))
    eucl_safe = np.where(eucl_dm > 1e-10, eucl_dm, 1e10)
    
    # Smooth Gaussian spatial kernel: high for nearby nodes, decays smoothly
    # This provides gradient information for route construction without
    # discontinuities at cluster boundaries
    spatial_kernel = np.exp(-eucl_safe ** 2 / (2.0 * sigma ** 2))
    
    # Base inverse distance component
    inv_dist = 1.0 / (dm_safe ** gamma)
    
    # Demand-aware depot weighting: prefer high-demand nodes when leaving depot
    # This encourages early service of large orders to reduce total routes
    max_demand = np.max(demands) if np.max(demands) > 0 else 1.0
    norm_demands = demands / max_demand
    
    # Depot row: weight by demand to prioritize large orders early
    depot_weight = 1.0 + demand_weight * norm_demands
    depot_weight = depot_weight / np.mean(depot_weight)  # normalize to mean 1
    
    # Combine: spatial proximity * inverse distance
    H = spatial_kernel * inv_dist
    
    # Apply depot weighting only to row 0 (edges from depot)
    H[0, :] *= depot_weight
    # Also apply mild demand preference to other rows (slightly prefer
    # moving to high-demand nodes to keep routes dense)
    for i in range(1, n):
        H[i, :] *= (1.0 + 0.3 * norm_demands)
    
    # Ensure positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO using a combined flow-based potential
    and demand-aware multi-criteria approach.
    """
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin before 330s kill

    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)

    dm_safe = np.where(dm > 1e-10, dm, 1e10)

    # Precompute Euclidean distances from coordinates
    eucl_dm = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))
    eucl_safe = np.where(eucl_dm > 1e-10, eucl_dm, 1e10)

    # Demand-related quantities
    max_demand = np.max(demands) if np.max(demands) > 0 else 1.0
    norm_demands = demands / max_demand

    # Demand penalty factor: higher demand nodes get lower heuristic weight
    demand_penalty = 1.0 / (1.0 + 2.0 * norm_demands)

    best_H = None
    best_score = np.inf

    def try_candidate(H_candidate, description=""):
        nonlocal best_H, best_score, deadline, start_time
        if time.time() > deadline - 20.0:
            return

        H_candidate = np.where(np.isfinite(H_candidate) & (H_candidate > 0),
                              H_candidate, 1e-9)
        H_candidate = H_candidate.astype(np.float64)

        # Quick evaluation with reduced simulation
        remaining = deadline - 20.0 - time.time()
        if remaining < 1.0:
            return

        sim_time = min(1.5, max(0.5, remaining / 35))
        score = _evaluate_heuristic(dm, demands, capacity, H_candidate,
                                    time_limit=sim_time, n_seeds=3)
        if score < best_score:
            best_score = score
            best_H = H_candidate.copy()

    # ---- Phase 1: Coarse search over Gaussian spatial kernel parameters ----
    # Coarse grid: broader sigma and gamma ranges
    coarse_sigmas = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    coarse_gammas = [0.5, 1.0, 1.5, 2.0, 2.5]
    coarse_dw = [0.5, 1.0, 1.5, 2.0]

    for sigma in coarse_sigmas:
        if time.time() > deadline - 60.0:
            break
        for gamma in coarse_gammas:
            if time.time() > deadline - 60.0:
                break
            for dw in coarse_dw:
                if time.time() > deadline - 60.0:
                    break
                H_cand = _gaussian_spatial_heuristic(
                    dm, demands, capacity, coords,
                    sigma=sigma, gamma=gamma, demand_weight=dw)
                try_candidate(H_cand, f"gauss_s={sigma} g={gamma} dw={dw}")

    # ---- Phase 2: Fine search around best found parameters ----
    if best_H is not None and time.time() < deadline - 40.0:
        # Fine-tune sigma
        for sigma in np.linspace(0.04, 0.25, 12):
            if time.time() > deadline - 40.0:
                break
            for gamma in np.linspace(0.5, 2.5, 8):
                if time.time() > deadline - 40.0:
                    break
                for dw in np.linspace(0.5, 2.5, 5):
                    if time.time() > deadline - 40.0:
                        break
                    H_cand = _gaussian_spatial_heuristic(
                        dm, demands, capacity, coords,
                        sigma=sigma, gamma=gamma, demand_weight=dw)
                    try_candidate(H_cand, f"gauss_fine s={sigma:.3f} g={gamma:.3f} dw={dw:.2f}")

    # ---- Phase 3: Very fine local search if time permits ----
    if best_H is not None and time.time() < deadline - 30.0:
        # Track best parameters implicitly by re-evaluating a tight neighborhood
        for sigma in np.linspace(0.06, 0.18, 10):
            if time.time() > deadline - 30.0:
                break
            for gamma in np.linspace(0.8, 2.2, 8):
                if time.time() > deadline - 30.0:
                    break
                for dw in np.linspace(0.8, 2.0, 5):
                    if time.time() > deadline - 30.0:
                        break
                    H_cand = _gaussian_spatial_heuristic(
                        dm, demands, capacity, coords,
                        sigma=sigma, gamma=gamma, demand_weight=dw)
                    try_candidate(H_cand, f"gauss_vfine s={sigma:.3f} g={gamma:.3f} dw={dw:.2f}")

    # Final fallback
    if best_H is None:
        best_H = _gaussian_spatial_heuristic(dm, demands, capacity, coords,
                                             sigma=0.1, gamma=1.5, demand_weight=1.0)
        best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)

    # Ensure output shape matches input and all values are valid
    assert best_H.shape == dm.shape, "Shape mismatch"
    best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)

    return best_H

# EVOLVE-BLOCK-END
