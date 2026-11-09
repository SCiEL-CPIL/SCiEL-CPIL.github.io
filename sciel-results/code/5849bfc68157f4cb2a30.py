"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Budget-aware expected utility heuristic with Gaussian smoothing.
    
    Combines:
    - Prize/distance ratio (structural idea from seed)
    - Gaussian kernel smoothing to avoid premature convergence
    - Budget-aware exponential decay to prevent getting stuck far from depot
    - Feasibility masking for moves that exceed the length budget
    
    eta[i, j] = (pi_j + eps) / (d_ij + d_j0 + eps) * exp(-lambda * (d_ij + d_j0) / L_max)
    
    where the exponential term penalizes moves that consume a disproportionate
    fraction of the remaining length budget.
    """
    t0 = time.time()
    _DEADLINE = 322.0  # safety margin before the 330 s wall-clock kill
    
    prizes = np.asarray(prizes, dtype=np.float64)
    d = np.asarray(distance_matrix, dtype=np.float64)
    max_len = float(max_len)
    n = len(prizes)
    
    # Ensure prizes[0] is 0 (depot has no prize)
    prizes = prizes.copy()
    prizes[0] = 0.0
    
    eps = 1e-9
    
    # Compute the "cost" of visiting node j from any node i and returning to depot:
    # cost[i, j] = d[i, j] + d[j, 0]
    d_to_depot = d[:, 0]  # shape (n,)
    cost = d + d_to_depot[None, :]  # shape (n, n)
    
    # Safety: avoid negative or zero costs
    cost_safe = np.maximum(cost, eps)
    
    # Prize vector (broadcast over rows)
    prize_vec = prizes[None, :] + eps  # shape (1, n)
    
    # Base ratio: prize / cost
    base_ratio = prize_vec / cost_safe
    
    # Gaussian smoothing: use a kernel based on cost relative to max_len
    # This smooths the heuristic landscape to prevent premature convergence
    # sigma is chosen as a fraction of max_len to provide meaningful smoothing
    sigma = 0.3 * max_len
    if sigma <= 0:
        sigma = 1e-3
    
    # Gaussian kernel: exp(-(cost^2) / (2 * sigma^2))
    # This down-weights very expensive moves smoothly rather than abruptly
    gauss_kernel = np.exp(-(cost_safe ** 2) / (2.0 * sigma ** 2))
    
    # Budget-aware exponential decay: penalizes moves that use large fraction of budget
    lambda_decay = 1.0
    decay = np.exp(-lambda_decay * cost_safe / max_len)
    
    # Combine: base ratio * gaussian smoothing * budget decay
    eta = base_ratio * gauss_kernel * decay
    
    # Feasibility mask: moves that would make the tour infeasible get very low weight
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
