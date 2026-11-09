"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Spatial density potential heuristic with budget-aware decay.
    
    Combines:
    - Gaussian kernel density estimate of high-prize nodes (cluster awareness)
    - Exponential decay based on remaining path budget
    - Prize/distance ratio for individual edge attractiveness
    - Feasibility masking for moves that exceed the length budget
    """
    t0 = time.time()
    _DEADLINE = 322.0  # safety margin before the 330s wall-clock kill

    prizes = np.asarray(prizes, dtype=np.float64)
    d = np.asarray(distance_matrix, dtype=np.float64)
    max_len = float(max_len)
    n = len(prizes)

    eps = 1e-9

    # Ensure prizes[0] is 0 (depot has no prize)
    prizes = prizes.copy()
    prizes[0] = 0.0

    # Compute the "cost" of visiting node j from any node i and returning to depot:
    # cost[i, j] = d[i, j] + d[j, 0]
    d_to_depot = d[:, 0]  # shape (n,)
    cost = d + d_to_depot[None, :]  # shape (n, n)

    # Safety: avoid negative or zero costs
    cost_safe = np.maximum(cost, eps)

    # ---- Spatial density potential via Gaussian KDE of high-prize nodes ----
    # Estimate a density field where high-prize nodes contribute more mass
    # Use a bandwidth based on the problem scale
    # sigma_kde controls how "local" the density estimate is
    sigma_kde = 0.2 * max_len
    if sigma_kde <= 0:
        sigma_kde = 1e-3

    # For each node j, compute the density of prize mass around it
    # density[j] = sum_k prizes[k] * exp(-d[j,k]^2 / (2*sigma_kde^2))
    # This is a matrix operation: d is (n,n), prizes is (n,)
    # density[j] = sum over k of prizes[k] * exp(-d[j,k]^2 / (2*sigma_kde^2))
    d_sq = d ** 2  # (n, n)
    kde_weights = np.exp(-d_sq / (2.0 * sigma_kde ** 2))  # (n, n)
    density = kde_weights @ prizes  # (n,) - density at each node

    # Normalize density to be in a reasonable range
    max_density = np.max(density)
    if max_density > 0:
        density_norm = density / max_density
    else:
        density_norm = np.ones(n) * eps

    # ---- Budget-aware exponential decay ----
    # Penalize moves that consume a disproportionate fraction of the budget
    lambda_decay = 1.5
    decay = np.exp(-lambda_decay * cost_safe / max_len)  # (n, n)

    # ---- Base prize/distance ratio (proven signal from seed) ----
    prize_vec = prizes[None, :] + eps  # shape (1, n)
    base_ratio = prize_vec / cost_safe  # (n, n)

    # ---- Combine all components ----
    # The heuristic for edge (i, j) is:
    # eta[i, j] = density[j] * decay[i, j] * base_ratio[i, j]
    # This rewards:
    # 1. Moving to nodes in high-prize-density clusters (density[j])
    # 2. Moves that don't consume too much budget (decay[i, j])
    # 3. High prize relative to cost (base_ratio[i, j])

    eta = density_norm[None, :] * decay * base_ratio

    # ---- Feasibility mask: moves that would make the tour infeasible get very low weight ----
    # An ant at node i moving to node j needs d[i,j] + d[j,0] <= max_len to still return
    feas = np.where(cost_safe < max_len, 1.0, 1e-9)
    eta = eta * feas

    # For the diagonal (i == j), the ant can't move to itself
    np.fill_diagonal(eta, eps)

    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, eps)

    # Clamp to avoid extreme values
    eta = np.clip(eta, eps, 1e6)

    # Deadline guard: if we're running out of time, return what we have
    if time.time() - t0 > _DEADLINE:
        return eta

    # Final safety: enforce positivity, finiteness, and shape
    eta = np.nan_to_num(eta, nan=eps, posinf=1e6, neginf=eps)
    eta = np.where(eta > 0.0, eta, eps)
    if eta.shape != (n, n):
        eta = np.ones((n, n), dtype=np.float64) * eps

    return eta
# EVOLVE-BLOCK-END
