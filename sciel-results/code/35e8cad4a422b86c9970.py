"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _greedy_cost(distance_matrix, demands, capacity, heuristic):
    """
    Fast greedy evaluation: construct a route by always picking the most attractive
    feasible unvisited node. Returns total distance.
    """
    n = len(demands)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    remaining_cap = capacity
    total_dist = 0.0

    for _ in range(n - 1):
        scores = heuristic[current].copy()
        feasible_mask = (~visited) & (demands <= remaining_cap)
        scores[~feasible_mask] = -np.inf
        scores[0] = -np.inf

        if not np.any(np.isfinite(scores)):
            total_dist += distance_matrix[current, 0]
            current = 0
            remaining_cap = capacity
            if np.all(visited):
                break
            continue

        best_j = int(np.argmax(scores))
        total_dist += distance_matrix[current, best_j]
        remaining_cap -= demands[best_j]
        visited[best_j] = True
        current = best_j

    total_dist += distance_matrix[current, 0]
    return total_dist

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix (larger = more attractive).
    
    Strategy:
    1. Define a parametric heuristic family combining:
       - Power-law distance decay (1/d^alpha)
       - Clarke-Wright savings boost
       - Demand-efficiency node potential (depot-pressure field)
       - Centroid proximity / local density field
    2. Use Nelder-Mead to tune parameters on a fast greedy objective.
    3. Return the optimized heuristic matrix.
    """
    n = len(demands)
    d_safe = distance_matrix + 1e-9
    start_time = time.time()
    deadline = start_time + 270.0  # Safety margin before 300s limit

    # --- Precompute structural features ---
    
    # Clarke-Wright savings: savings[i,j] = d(0,i) + d(0,j) - d(i,j)
    d0 = distance_matrix[0, :]
    savings = d0[:, None] + d0[None, :] - distance_matrix
    savings = np.maximum(savings, 0.0)
    max_savings = np.max(savings)
    if max_savings > 0:
        savings_norm = savings / max_savings
    else:
        savings_norm = np.zeros_like(savings)

    # Demand efficiency: lower demand -> higher efficiency
    demand_eff = 1.0 / (1.0 + demands)
    
    # Depot-pressure: high-demand nodes are harder to serve, should be visited early
    # Inverse demand as a node potential
    depot_pressure = 1.0 / (1.0 + demands * 0.5)

    # Centroid proximity: distance from each node to the centroid of all customers
    coords = np.array(coordinates, dtype=float)
    # Exclude depot (index 0) from centroid calculation for customer centroid
    cust_coords = coords[1:]
    centroid = np.mean(cust_coords, axis=0)
    dist_to_centroid = np.sqrt(np.sum((coords - centroid) ** 2, axis=1))
    # Normalize: closer to centroid -> higher potential
    max_dtc = np.max(dist_to_centroid)
    if max_dtc > 0:
        centroid_potential = 1.0 - dist_to_centroid / max_dtc
    else:
        centroid_potential = np.ones(n)
    centroid_potential = np.maximum(centroid_potential, 0.01)

    # Local density: for each node, count of nearby nodes within a radius
    # Use a simple grid-based approximation for speed
    # Normalize coordinates to [0,1] and use a small grid
    grid_size = 5
    grid = np.zeros((grid_size, grid_size))
    for i in range(1, n):  # exclude depot
        gx = int(np.clip(coords[i, 0] * grid_size, 0, grid_size - 1))
        gy = int(np.clip(coords[i, 1] * grid_size, 0, grid_size - 1))
        grid[gy, gx] += 1
    max_density = np.max(grid)
    if max_density > 0:
        grid_norm = grid / max_density
    else:
        grid_norm = np.ones_like(grid)
    
    node_density = np.zeros(n)
    node_density[0] = 0.0
    for i in range(1, n):
        gx = int(np.clip(coords[i, 0] * grid_size, 0, grid_size - 1))
        gy = int(np.clip(coords[i, 1] * grid_size, 0, grid_size - 1))
        node_density[i] = grid_norm[gy, gx]

    # Combine node potentials: depot-pressure * centroid_potential * (1 + density_boost * node_density)
    node_potential = depot_pressure * centroid_potential

    def make_heuristic(params):
        alpha, beta, gamma, delta, epsilon = params
        # Clamp parameters
        alpha = np.clip(alpha, 0.5, 3.0)
        beta = np.clip(beta, 0.0, 3.0)
        gamma = np.clip(gamma, 0.0, 3.0)
        delta = np.clip(delta, 0.0, 3.0)
        epsilon = np.clip(epsilon, 0.0, 3.0)

        # Base distance component: 1 / d^alpha
        base = 1.0 / (d_safe ** alpha)

        # Savings boost: 1 + beta * savings_norm
        savings_term = 1.0 + beta * savings_norm

        # Demand efficiency at destination: 1 + gamma * demand_eff[j]
        # This boosts edges going to low-demand nodes
        demand_term = 1.0 + gamma * demand_eff[None, :]

        # Node potential (depot-pressure * centroid) at destination
        potential_term = 1.0 + delta * node_potential[None, :]

        # Density boost at destination
        density_term = 1.0 + epsilon * node_density[None, :]

        h = base * savings_term * demand_term * potential_term * density_term
        h = np.maximum(h, 1e-9)
        np.fill_diagonal(h, 1e-9)
        return h

    def objective(params):
        h = make_heuristic(params)
        return _greedy_cost(distance_matrix, demands, capacity, h)

    # --- Optimization ---
    # Initial guess: alpha=1.5 (stronger distance decay), beta=1.0, gamma=0.5, delta=0.5, epsilon=0.3
    x0 = np.array([1.5, 1.0, 0.5, 0.5, 0.3])
    
    best_params = x0.copy()
    best_cost = objective(x0)

    try:
        from scipy.optimize import minimize
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 200,
                'maxfev': 200,
                'xatol': 1e-4,
                'fatol': 1e-5
            }
        )
        if result.fun < best_cost:
            best_cost = result.fun
            best_params = result.x.copy()
    except Exception:
        pass

    # Check time before final construction
    if time.time() > deadline - 5.0:
        # Use whatever we have
        pass

    # Construct final heuristic
    h = make_heuristic(best_params)

    # Validate
    if not np.all(np.isfinite(h)):
        h = 1.0 / d_safe
        np.fill_diagonal(h, 1e-9)
    if h.shape != distance_matrix.shape:
        h = 1.0 / d_safe
        np.fill_diagonal(h, 1e-9)
    h = np.maximum(h, 1e-9)
    np.fill_diagonal(h, 1e-9)

    return h
# EVOLVE-BLOCK-END
