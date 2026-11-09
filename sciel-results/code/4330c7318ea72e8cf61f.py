"""Coarse-to-fine annealing for Sutton-Chen transition-metal clusters."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time

def _sc_energy_gradient(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy and gradient with given parameters."""
    n = positions.shape[0]
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff**2, axis=-1))
    # Avoid division by zero
    mask = r > 1e-10
    r_safe = np.where(mask, r, 1.0)
    
    # r^-n and r^-m
    rn = r_safe ** (-n_exp)
    rm = r_safe ** (-m_exp)
    
    # Sum over j != i for each i
    # Set diagonal to 0
    rn[~mask] = 0
    rm[~mask] = 0
    
    phi_i = np.sum(rn, axis=1)  # sum_j r_ij^-n
    psi_i = np.sum(rm, axis=1)  # sum_j r_ij^-m
    
    # Energy: E = 1/2 sum_i [phi_i - c * sqrt(psi_i)]
    # Note: the formula is sum_i [1/2 phi_i - c sqrt(psi_i)] which equals 1/2 sum_i phi_i - c sum_i sqrt(psi_i)
    # But wait, the standard Sutton-Chen is E = 1/2 sum_i phi_i - sum_i epsilon_i sqrt(1 + c sum_j r_ij^-m)
    # The task says: E = sum_i [1/2 sum_{j!=i} r_ij^-n_exp - c sqrt(sum_{j!=i} r_ij^-m_exp)]
    # So E = 1/2 * sum_i sum_j r_ij^-n - c * sum_i sqrt(sum_j r_ij^-m)
    
    E_rep = 0.5 * np.sum(phi_i)
    E_att = -c * np.sum(np.sqrt(psi_i))
    energy = E_rep + E_att
    
    # Gradient
    # dE_rep/dx_i = 1/2 * sum_j d(r_ij^-n)/dx_i
    # d(r_ij^-n)/dx_i = -n * r_ij^(-n-1) * (x_i - x_j) / r_ij = -n * r_ij^(-n-2) * (x_i - x_j)
    # Actually d/dx_i (r_ij^-n) = -n * r_ij^(-n-1) * d(r_ij)/dx_i
    # d(r_ij)/dx_i = (x_i - x_j) / r_ij
    # So d(r_ij^-n)/dx_i = -n * r_ij^(-n-1) * (x_i - x_j) / r_ij = -n * r_ij^(-n-2) * (x_i - x_j)
    
    # For E_rep = 1/2 sum_i sum_{j!=i} r_ij^-n
    # dE_rep/dx_i = 1/2 sum_{j!=i} (-n) r_ij^(-n-2) (x_i - x_j)
    # = -n/2 sum_{j!=i} r_ij^(-n-2) (x_i - x_j)
    
    # For E_att = -c sum_i sqrt(psi_i) where psi_i = sum_{j!=i} r_ij^-m
    # dE_att/dx_i = -c * 1/(2 sqrt(psi_i)) * d(psi_i)/dx_i
    # d(psi_i)/dx_i = sum_{j!=i} d(r_ij^-m)/dx_i = sum_{j!=i} (-m) r_ij^(-m-2) (x_i - x_j)
    # So dE_att/dx_i = -c/(2 sqrt(psi_i)) * sum_{j!=i} (-m) r_ij^(-m-2) (x_i - x_j)
    # = c*m/(2 sqrt(psi_i)) * sum_{j!=i} r_ij^(-m-2) (x_i - x_j)
    
    # Compute r^(-n-2) and r^(-m-2)
    rn2 = r_safe ** (-(n_exp + 2))
    rm2 = r_safe ** (-(m_exp + 2))
    rn2[~mask] = 0
    rm2[~mask] = 0
    
    # grad_rep[i] = -n/2 * sum_j rn2[i,j] * (x_i - x_j)
    # = -n/2 * (sum_j rn2[i,j] * x_i - sum_j rn2[i,j] * x_j)
    # = -n/2 * (x_i * sum_j rn2[i,j] - sum_j rn2[i,j] * x_j)
    
    sum_rn2 = np.sum(rn2, axis=1)
    dot_rn2_x = rn2 @ positions  # shape (n, 3)
    
    grad_rep = -0.5 * n_exp * (positions * sum_rn2[:, np.newaxis] - dot_rn2_x)
    
    # grad_att[i] = c*m/(2 sqrt(psi_i)) * sum_j rm2[i,j] * (x_i - x_j)
    sum_rm2 = np.sum(rm2, axis=1)
    dot_rm2_x = rm2 @ positions
    
    # Avoid division by zero when psi_i is small
    sqrt_psi = np.sqrt(np.maximum(psi_i, 1e-10))
    grad_att = c * m_exp / (2.0 * sqrt_psi[:, np.newaxis]) * (positions * sum_rm2[:, np.newaxis] - dot_rm2_x)
    
    gradient = grad_rep + grad_att
    
    return energy, gradient


def _sc_energy(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy only (faster, no gradient)."""
    n = positions.shape[0]
    if n < 2:
        return 0.0
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff**2, axis=-1))
    mask = r > 1e-10
    r_safe = np.where(mask, r, 1.0)
    
    rn = r_safe ** (-n_exp)
    rm = r_safe ** (-m_exp)
    rn[~mask] = 0
    rm[~mask] = 0
    
    phi_i = np.sum(rn, axis=1)
    psi_i = np.sum(rm, axis=1)
    
    E_rep = 0.5 * np.sum(phi_i)
    E_att = -c * np.sum(np.sqrt(psi_i))
    return E_rep + E_att


def _soften_params(n_exp, m_exp, c, t):
    """
    Soften parameters for coarse-to-fine annealing.
    t in [0, 1]: t=0 is fully softened, t=1 is target parameters.
    """
    # Reduce exponents for softer potential at high t (early annealing)
    # n_exp_soft = n_exp * (0.5 + 0.5 * t)  -> from 0.5*n_exp to n_exp
    # m_exp_soft = m_exp * (0.5 + 0.5 * t)
    # c_soft = c * (0.3 + 0.7 * t)  -> weaker attraction initially
    
    n_exp_s = n_exp * (0.4 + 0.6 * t)
    m_exp_s = m_exp * (0.4 + 0.6 * t)
    c_s = c * (0.2 + 0.8 * t)
    
    return n_exp_s, m_exp_s, c_s


def _local_relax(positions, n_exp, m_exp, c, maxiter=200):
    """Local relaxation using L-BFGS-B with the given parameters."""
    def fun(x):
        pos = x.reshape(-1, 3)
        e, g = _sc_energy_gradient(pos, n_exp, m_exp, c)
        return e, g.ravel()
    
    try:
        result = minimize(fun, positions.ravel(), method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
        return result.fun, result.x.reshape(-1, 3)
    except:
        return _sc_energy(positions, n_exp, m_exp, c), positions


def _basin_hop(positions, n_exp, m_exp, c, temperature, n_hops=3, maxiter_per_relax=100):
    """
    Basin hopping with soft potential.
    Returns best energy and positions found.
    """
    best_e, best_pos = _local_relax(positions, n_exp, m_exp, c, maxiter=maxiter_per_relax)
    
    current_pos = best_pos
    current_e = best_e
    
    for hop in range(n_hops):
        # Perturb with Gaussian noise scaled by temperature
        noise_scale = temperature * 0.3
        perturbed = current_pos + np.random.normal(0, noise_scale, size=current_pos.shape)
        
        # Relax the perturbed structure
        e_new, pos_new = _local_relax(perturbed, n_exp, m_exp, c, maxiter=maxiter_per_relax)
        
        # Metropolis accept/reject
        delta_e = e_new - current_e
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / max(temperature, 1e-10)):
            current_pos = pos_new
            current_e = e_new
        
        # Track best
        if e_new < best_e:
            best_e = e_new
            best_pos = pos_new
    
    return best_e, best_pos


def construct(instance):
    """
    Coarse-to-fine annealing for Sutton-Chen clusters.
    """
    n = instance["n"]
    kind = instance["kind"]
    n_exp_target = instance["n_exp"]
    m_exp_target = instance["m_exp"]
    c_target = instance["c"]
    
    # Determine starting geometry: compact random cluster
    rng = np.random.default_rng(42 + n)
    spacing = 0.72
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    
    # Generate initial compact cluster
    def generate_initial():
        points = []
        attempts = 0
        while len(points) < n and attempts < n * 100:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                    points.append(p)
            attempts += 1
        # If not enough points, fill with random positions
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        return np.array(points)
    
    # Coarse-to-fine schedule
    # Phase 1: High temperature, soft potential, many basin hops
    # Phase 2: Medium temperature, medium softness
    # Phase 3: Low temperature, target potential, final relaxation
    
    n_exp_0, m_exp_0, c_0 = _soften_params(n_exp_target, m_exp_target, c_target, 0.0)
    
    # Phase 1: Soft potential, high T
    pos = generate_initial()
    e1, pos = _basin_hop(pos, n_exp_0, m_exp_0, c_0, temperature=0.5, n_hops=5, maxiter_per_relax=80)
    
    # Phase 2: Medium softness
    n_exp_1, m_exp_1, c_1 = _soften_params(n_exp_target, m_exp_target, c_target, 0.5)
    e2, pos = _basin_hop(pos, n_exp_1, m_exp_1, c_1, temperature=0.2, n_hops=5, maxiter_per_relax=100)
    
    # Phase 3: Target potential, low T
    e3, pos = _basin_hop(pos, n_exp_target, m_exp_target, c_target, temperature=0.05, n_hops=5, maxiter_per_relax=150)
    
    # Phase 4: Final precise relaxation with target parameters
    e_final, pos_final = _local_relax(pos, n_exp_target, m_exp_target, c_target, maxiter=500)
    
    # Validate: check no atoms too close
    if n > 1:
        min_dist = np.min(pdist(pos_final))
        if min_dist < 0.35:
            # Try to fix by pushing apart
            for _ in range(50):
                diff = pos_final[:, np.newaxis, :] - pos_final[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=-1))
                mask = (dists < 0.35) & (dists > 1e-10)
                if not np.any(mask):
                    break
                # Push apart
                for i in range(n):
                    for j in range(i + 1, n):
                        if dists[i, j] < 0.35 and dists[i, j] > 1e-10:
                            direction = diff[i, j] / dists[i, j]
                            push = (0.35 - dists[i, j]) * 0.5
                            pos_final[i] += direction * push
                            pos_final[j] -= direction * push
                e_final = _sc_energy(pos_final, n_exp_target, m_exp_target, c_target)
    
    # Final check
    if n > 1:
        min_dist = np.min(pdist(pos_final))
        if min_dist < 0.35:
            # Fallback: return a valid configuration
            pos_final = generate_initial()
            e_final, pos_final = _local_relax(pos_final, n_exp_target, m_exp_target, c_target, maxiter=200)
    
    # Ensure finite
    if not np.all(np.isfinite(pos_final)):
        pos_final = generate_initial()
        e_final, pos_final = _local_relax(pos_final, n_exp_target, m_exp_target, c_target, maxiter=200)
    
    return pos_final
# EVOLVE-BLOCK-END
