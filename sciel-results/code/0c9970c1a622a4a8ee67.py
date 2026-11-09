"""Sutton-Chen transition-metal cluster global minima: Multi-start with Soft-Potential Basin Hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Per-kind parameters: (n_exp, m_exp, c)
_KIND_PARAMS = {
    '12-6': (12, 6, 144.41),
    '9-6': (9, 6, 39.432),
    '10-8': (10, 8, 34.408),
}

# Spacing per kind for seed generation
_SPACING = {
    '12-6': 0.72,
    '9-6': 0.72,
    '10-8': 0.72,
}

# Soft potential parameters for basin exploration
# We reduce exponents to flatten the landscape
_SOFT_PARAMS = {
    '12-6': {'n': 6, 'm': 3, 'c_scale': 0.5}, 
    '9-6': {'n': 4.5, 'm': 3, 'c_scale': 0.5},
    '10-8': {'n': 5, 'm': 4, 'c_scale': 0.5},
}


def _validate_positions(pos):
    """Vectorized validity check: finite coords and min pair distance >= 0.35."""
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _fcc_cluster(n, spacing=0.72):
    """Generate n-atom FCC-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k], dtype=float) + b
                    sites.append(p)
    sites = np.array(sites)
    sites = sites * spacing
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _hcp_cluster(n, spacing=0.72):
    """Generate n-atom HCP-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    v1 = np.array([a, 0, 0])
    v2 = np.array([-a / 2, a * np.sqrt(3) / 2, 0])
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                base = i * v1 + j * v2
                z = k * c
                p1 = base + np.array([0, 0, z])
                p2 = base + np.array([a / 2, a * np.sqrt(3) / 6, z + c / 2])
                sites.append(p1)
                sites.append(p2)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _icosahedral_cluster(n, spacing=0.72):
    """Generate n-atom icosahedral-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)

    points = [np.zeros(3)]
    if n <= 1:
        return np.array(points[:n])

    r1 = spacing
    for v in ico_verts:
        points.append(v * r1)
    if len(points) >= n:
        return np.array(points[:n])

    r2 = spacing * np.sqrt(3.0)
    ico_norms = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico_norms[i] - ico_norms[j])
            if 1.0 < d < 2.0:
                mid = (ico_norms[i] + ico_norms[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                points.append(mid * r2)
    if len(points) >= n:
        return np.array(points[:n])

    r3 = spacing * np.sqrt(5.0)
    remaining = n - len(points)
    if remaining > 0:
        for k in range(remaining):
            t = (k + 0.5) / remaining
            inc = np.pi * (3 - np.sqrt(5))
            z = 1 - 2 * t
            r = np.sqrt(1 - z * z)
            theta = inc * k
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * r3)

    return np.array(points[:n])


def _random_compact(n, spacing=0.72, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = n * 500
    attempt = 0
    while len(points) < n and attempt < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
                points.append(p)
        attempt += 1
    while len(points) < n:
        base = points[len(points) % max(1, len(points))] if points else np.zeros(3)
        p = base + rng.normal(0, spacing * 0.3, 3)
        points.append(p)
    return np.array(points[:n])


def _soft_energy_gradient(x, n, n_s, m_s, c_s):
    """
    Compute energy and gradient for the softened Sutton-Chen potential.
    x: (n, 3) array
    """
    x = x.reshape(n, 3)
    # Pair indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # Distances
    diff = x[i_idx] - x[j_idx]
    r2 = np.sum(diff * diff, axis=1)
    r2 = np.maximum(r2, 1e-10)
    r = np.sqrt(r2)
    
    # Repulsive part: sum_{i<j} r^-n_s
    r_inv_n = r ** (-n_s)
    E_rep = np.sum(r_inv_n)
    
    # Attractive part: -c_s * sum_i sqrt(sum_{j!=i} r^-m_s)
    # We need S_i = sum_{j!=i} r_ij^-m_s
    # Create a matrix of r^-m_s for all pairs, then sum rows excluding diagonal
    # Efficient way: use the pair list to build the sum for each i
    # Initialize S array
    S = np.zeros(n)
    r_inv_m = r ** (-m_s)
    
    # Add to S[i] and S[j]
    np.add.at(S, i_idx, r_inv_m)
    np.add.at(S, i_idx, r_inv_m) # Wait, i_idx and j_idx are from triu.
    # Let's do it properly:
    S = np.zeros(n)
    np.add.at(S, i_idx, r_inv_m)
    np.add.at(S, j_idx, r_inv_m)
    
    # S[i] is sum_{j!=i} r_ij^-m_s
    sqrt_S = np.sqrt(S)
    E_att = -c_s * np.sum(sqrt_S)
    
    E = E_rep + E_att
    
    # Gradient
    # dE_rep/dx_k = sum_{j!=k} -n_s * r_kj^(-n_s-1) * (x_k - x_j)/r_kj
    # = sum_{j!=k} -n_s * r_kj^(-n_s-2) * (x_k - x_j)
    
    grad = np.zeros((n, 3))
    
    # Repulsive gradient contribution
    # For each pair (i, j), force on i is -n_s * r^(-n_s-2) * (x_i - x_j)
    # Force on j is -n_s * r^(-n_s-2) * (x_j - x_i) = -Force on i
    coeff_rep = -n_s * r ** (-n_s - 2)
    # diff is x_i - x_j
    forces_rep = coeff_rep[:, np.newaxis] * diff
    # Add to grad[i] and subtract from grad[j]
    np.add.at(grad, i_idx, forces_rep)
    np.add.at(grad, j_idx, -forces_rep)
    
    # Attractive gradient contribution
    # E_att = -c_s * sum_i sqrt(S_i)
    # dE_att/dx_k = -c_s * sum_i (1/(2 sqrt(S_i))) * dS_i/dx_k
    # dS_i/dx_k:
    # If k == i: sum_{j!=i} -m_s * r_ij^(-m_s-2) * (x_i - x_j)
    # If k != i: -m_s * r_ik^(-m_s-2) * (x_k - x_i)  (Note: S_i depends on x_k via r_ik)
    # Actually, S_i = sum_{j!=i} r_ij^-m.
    # dS_i/dx_k = sum_{j!=i} d(r_ij^-m)/dx_k.
    # If k=i: sum_{j!=i} -m r_ij^(-m-2) (x_i - x_j)
    # If k!=i: term j=k is present: -m r_ik^(-m-2) (x_i - x_k) = m r_ik^(-m-2) (x_k - x_i)
    
    # We can compute dS_i/dx_k for all i,k using the pair list.
    # Let's build a matrix of dS_i/dx_k? No, n is small.
    
    # dE_att/dx_k = -c_s * sum_i (1/(2 sqrt(S_i))) * dS_i/dx_k
    
    # We can iterate over pairs again to accumulate the gradient of S.
    # For each pair (i, j):
    # Contribution to dS_i/dx_i: -m_s * r^(-m_s-2) * (x_i - x_j)
    # Contribution to dS_i/dx_j: -m_s * r^(-m_s-2) * (x_j - x_i)  (Wait, r_ij depends on x_j)
    # d(r_ij^-m)/dx_j = -m r^(-m-2) (x_j - x_i)
    
    # So, for pair (i, j):
    # dS_i/dx_i += -m_s * r^(-m_s-2) * (x_i - x_j)
    # dS_i/dx_j += -m_s * r^(-m_s-2) * (x_j - x_i)
    # dS_j/dx_j += -m_s * r^(-m_s-2) * (x_j - x_i)
    # dS_j/dx_i += -m_s * r^(-m_s-2) * (x_i - x_j)
    
    # Let's define coeff_att = -m_s * r^(-m_s-2)
    coeff_att = -m_s * r ** (-m_s - 2)
    # diff = x_i - x_j
    # dS_i/dx_i contribution: coeff_att * diff
    # dS_i/dx_j contribution: coeff_att * (-diff)
    # dS_j/dx_j contribution: coeff_att * (-diff)
    # dS_j/dx_i contribution: coeff_att * diff
    
    # We need to sum these weighted by 1/(2 sqrt(S_i)) for each i.
    # This is complex to vectorize directly in one shot without a matrix.
    # Since n is <= 80, we can loop over pairs for the attractive gradient part.
    
    # Precompute inv_2sqrt_S
    inv_2sqrt_S = np.zeros(n)
    mask = S > 1e-10
    inv_2sqrt_S[mask] = 1.0 / (2.0 * np.sqrt(S[mask]))
    
    # Initialize gradient accumulator for attractive part
    grad_att = np.zeros((n, 3))
    
    # Loop over pairs (n^2/2 ~ 3000, very fast)
    for idx in range(len(i_idx)):
        i = i_idx[idx]
        j = j_idx[idx]
        r_val = r[idx]
        # coeff_att[idx] is -m_s * r^(-m_s-2)
        c_att = coeff_att[idx]
        d_vec = diff[idx] # x_i - x_j
        
        # dS_i/dx_i part
        dSi_dxi = c_att * d_vec
        # dS_i/dx_j part
        dSi_dxj = c_att * (-d_vec)
        
        # dS_j/dx_j part
        dSj_dxj = c_att * (-d_vec)
        # dS_j/dx_i part
        dSj_dxi = c_att * d_vec
        
        # dE_att/dx_k = -c_s * sum_i inv_2sqrt_S[i] * dS_i/dx_k
        
        # Contribution to x_i:
        # from i's sum: inv_2sqrt_S[i] * dSi_dxi
        # from j's sum: inv_2sqrt_S[j] * dSj_dxi
        grad_att[i] += -c_s * (inv_2sqrt_S[i] * dSi_dxi + inv_2sqrt_S[j] * dSj_dxi)
        
        # Contribution to x_j:
        # from i's sum: inv_2sqrt_S[i] * dSi_dxj
        # from j's sum: inv_2sqrt_S[j] * dSj_dxj
        grad_att[j] += -c_s * (inv_2sqrt_S[i] * dSi_dxj + inv_2sqrt_S[j] * dSj_dxj)
        
    grad += grad_att
    
    return E, grad


def construct(instance):
    """
    Multi-start with Soft-Potential Basin Hopping.
    1. Generate diverse seeds.
    2. For each seed, perform Soft Relaxation (scipy minimize) then Hard Relaxation (provided relax).
    3. Iterate with perturbations if time permits.
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    spacing = _SPACING.get(kind, 0.72)
    
    n_exp, m_exp, c_exp = _KIND_PARAMS[kind]
    soft_params = _SOFT_PARAMS[kind]
    n_s = soft_params['n']
    m_s = soft_params['m']
    c_s = c_exp * soft_params['c_scale']
    
    # Generate seeds
    seeds = []
    for scale in [0.85, 1.0, 1.15]:
        s = spacing * scale
        seeds.append(_fcc_cluster(size, s))
        seeds.append(_hcp_cluster(size, s))
        seeds.append(_icosahedral_cluster(size, s))
        
    for rs in [size, size + 1, size + 7, size + 13, size + 42]:
        seeds.append(_random_compact(size, spacing, seed=rs))
        
    # Remove duplicates and validate
    unique_seeds = []
    seen = set()
    for s in seeds:
        if s.shape[0] != size:
            continue
        key = tuple(np.round(s, 3).flatten())
        if key not in seen:
            seen.add(key)
            if _validate_positions(s):
                unique_seeds.append(s)
                
    best_energy = np.inf
    best_positions = None
    
    # Helper to run soft then hard relaxation
    def process_seed(seed_pos):
        nonlocal best_energy, best_positions
        if time.time() > deadline:
            return
        
        # 1. Soft Relaxation
        try:
            # Flatten for scipy
            x0 = seed_pos.flatten()
            
            def func(x):
                E, G = _soft_energy_gradient(x, size, n_s, m_s, c_s)
                if not np.isfinite(E):
                    return 1e10, np.zeros_like(x)
                return E, G
                
            res = minimize(func, x0, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'ftol': 1e-8})
            if res.success or np.isfinite(res.fun):
                soft_pos = res.x.reshape(size, 3)
                if _validate_positions(soft_pos):
                    # 2. Hard Relaxation
                    hard_energy, hard_pos = relax(soft_pos, potential, kind)
                    if np.isfinite(hard_energy) and _validate_positions(hard_pos):
                        if hard_energy < best_energy:
                            best_energy = hard_energy
                            best_positions = hard_pos
        except Exception:
            # Fallback: direct hard relaxation if soft fails
            try:
                hard_energy, hard_pos = relax(seed_pos, potential, kind)
                if np.isfinite(hard_energy) and _validate_positions(hard_pos):
                    if hard_energy < best_energy:
                        best_energy = hard_energy
                        best_positions = hard_pos
            except Exception:
                pass

    # Process all seeds
    for seed in unique_seeds:
        if time.time() > deadline - 10.0: # Leave time for polish
            break
        process_seed(seed)
        
    # If no valid result, fallback
    if best_positions is None:
        positions = np.zeros((size, 3))
        positions[:, 0] = np.arange(size) * spacing
        best_positions = positions
        best_energy = np.inf
        
    # Polish phase: Perturb best result and re-process
    if best_positions is not None and time.time() < deadline - 5.0:
        rng = np.random.default_rng(0)
        for perturbation_scale in [0.1, 0.2, 0.3, 0.5]:
            if time.time() > deadline - 5.0:
                break
            perturbed = best_positions + rng.normal(0, perturbation_scale * spacing, best_positions.shape)
            if _validate_positions(perturbed):
                process_seed(perturbed)
                
    # Final validation
    if not _validate_positions(best_positions):
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * spacing
        
    return best_positions
# EVOLVE-BLOCK-END
