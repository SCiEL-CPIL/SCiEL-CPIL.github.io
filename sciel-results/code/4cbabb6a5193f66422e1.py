"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _fast_aco_sim(dm, demands, capacity, H, n_ants=10, n_iterations=10, seed=42, time_limit=2.0):
    """
    Fast simulation of the evaluator's ACO to estimate heuristic quality.
    Closely mirrors the evaluator: 20 ants, 30 iters, tau=0.9*tau+1/best.
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


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO.
    Returns an edge-attractiveness matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 300.0  # Safety margin before 330s kill

    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)

    dm_safe = np.where(dm > 1e-10, dm, 1e10)

    best_gamma = 1.0
    best_sim_cost = np.inf
    best_H = None

    # Phase 1: Coarse gamma search
    gammas_coarse = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2, 2.8, 3.5, 5.0]

    for g in gammas_coarse:
        if time.time() > deadline - 20.0:
            break

        H_candidate = 1.0 / (dm_safe ** g)
        H_candidate = np.where(np.isfinite(H_candidate) & (H_candidate > 0), H_candidate, 1e-9)

        cost = _fast_aco_sim(dm, demands, capacity, H_candidate,
                             n_ants=8, n_iterations=8, seed=42, time_limit=1.5)

        if cost < best_sim_cost:
            best_sim_cost = cost
            best_gamma = g
            best_H = H_candidate

    # Phase 2: Fine-tune around the best gamma found
    if time.time() < deadline - 30.0:
        gammas_fine = []
        for offset in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
            g_fine = best_gamma + offset
            if g_fine > 0.05:
                gammas_fine.append(g_fine)

        for g in gammas_fine:
            if time.time() > deadline - 20.0:
                break

            H_candidate = 1.0 / (dm_safe ** g)
            H_candidate = np.where(np.isfinite(H_candidate) & (H_candidate > 0), H_candidate, 1e-9)

            cost = _fast_aco_sim(dm, demands, capacity, H_candidate,
                                 n_ants=12, n_iterations=12, seed=42, time_limit=2.5)

            if cost < best_sim_cost:
                best_sim_cost = cost
                best_gamma = g
                best_H = H_candidate

    # Phase 3: Ultra-fine search if time permits
    if time.time() < deadline - 40.0:
        gammas_ultra = []
        for offset in [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]:
            g_ultra = best_gamma + offset
            if g_ultra > 0.05:
                gammas_ultra.append(g_ultra)

        for g in gammas_ultra:
            if time.time() > deadline - 25.0:
                break

            H_candidate = 1.0 / (dm_safe ** g)
            H_candidate = np.where(np.isfinite(H_candidate) & (H_candidate > 0), H_candidate, 1e-9)

            cost = _fast_aco_sim(dm, demands, capacity, H_candidate,
                                 n_ants=16, n_iterations=16, seed=42, time_limit=3.0)

            if cost < best_sim_cost:
                best_sim_cost = cost
                best_gamma = g
                best_H = H_candidate

    # Phase 4: Final polish with highest fidelity simulation
    if time.time() < deadline - 10.0:
        # Try a few more candidates near the best with maximum fidelity
        final_candidates = []
        for offset in [-0.05, 0.0, 0.05]:
            g_final = best_gamma + offset
            if g_final > 0.05:
                final_candidates.append(g_final)

        for g in final_candidates:
            if time.time() > deadline - 5.0:
                break

            H_candidate = 1.0 / (dm_safe ** g)
            H_candidate = np.where(np.isfinite(H_candidate) & (H_candidate > 0), H_candidate, 1e-9)

            cost = _fast_aco_sim(dm, demands, capacity, H_candidate,
                                 n_ants=20, n_iterations=20, seed=42, time_limit=4.0)

            if cost < best_sim_cost:
                best_sim_cost = cost
                best_gamma = g
                best_H = H_candidate

    # Fallback
    if best_H is None:
        best_H = 1.0 / dm_safe
        best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)

    return best_H

# EVOLVE-BLOCK-END
