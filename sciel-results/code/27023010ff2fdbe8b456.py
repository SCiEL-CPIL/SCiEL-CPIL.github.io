# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _sc_energy_grad(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy and gradient for given parameters."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Compute pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r = np.sqrt(r2)
    
    # Pairwise terms
    r_neg_n = r ** (-n_exp)
    r_neg_m = r ** (-m_exp)
    
    # Sum over j != i for each i
    sum_r_neg_n = np.sum(r_neg_n, axis=1)  # sum_j r_ij^(-n_exp)
    sum_r_neg_m = np.sum(r_neg_m, axis=1)  # sum_j r_ij^(-m_exp)
    
    # Energy
    E_rep = 0.5 * np.sum(sum_r_neg_n)
    E_att = -c * np.sum(np.sqrt(np.maximum(sum_r_neg_m, 0)))
    energy = E_rep + E_att
    
    # Gradient
    # dE/dx_i = sum_{j!=i} [ -n_exp * r_ij^(-n_exp-2) * (x_i - x_j) 
    #                         + c * (1/(2*sqrt(sum_m_i))) * (-m_exp) * r_ij^(-m_exp-2) * (x_i - x_j) ]
    # Note: the 1/2 factor in repulsive part and the sqrt in attractive part need careful handling
    
    # Repulsive gradient contribution
    # d/dx_i [0.5 * sum_j r_ij^(-n_exp)] = 0.5 * sum_j (-n_exp) r_ij^(-n_exp-2) * (x_i - x_j)
    # = -0.5 * n_exp * sum_j r_ij^(-n_exp-2) * (x_i - x_j)
    
    r_neg_n2 = r ** (-(n_exp + 2))
    rep_grad = -0.5 * n_exp * np.sum(r_neg_n2[:, :, np.newaxis] * diff, axis=1)
    
    # Attractive gradient contribution
    # d/dx_i [-c * sqrt(sum_j r_ij^(-m_exp))] = -c * (1/(2*sqrt(sum_m_i))) * sum_j (-m_exp) r_ij^(-m_exp-2) * (x_i - x_j)
    # = c * m_exp / (2*sqrt(sum_m_i)) * sum_j r_ij^(-m_exp-2) * (x_i - x_j)
    
    r_neg_m2 = r ** (-(m_exp + 2))
    sum_m2 = np.sum(r_neg_m2, axis=1)  # sum_j r_ij^(-m_exp-2)
    
    # For each i, the coefficient is c * m_exp / (2 * sqrt(sum_m_i))
    # But we need to be careful: sum_m_i = sum_j r_ij^(-m_exp), not sum_j r_ij^(-m_exp-2)
    # The gradient of sqrt(S) where S = sum_j r_ij^(-m_exp) is (1/(2*sqrt(S))) * dS/dx_i
    # dS/dx_i = sum_j (-m_exp) r_ij^(-m_exp-2) * (x_i - x_j)
    
    sqrt_sum_m = np.sqrt(np.maximum(sum_r_neg_m, 1e-15))
    att_coeff = c * m_exp / (2 * sqrt_sum_m)  # shape (n,)
    
    # att_grad_i = att_coeff[i] * sum_j r_ij^(-m_exp-2) * (x_i - x_j)
    # But note: the term inside is sum_j [r_ij^(-m_exp-2) * (x_i - x_j)]
    att_grad = np.zeros_like(positions)
    for i in range(n):
        att_grad[i] = att_coeff[i] * np.sum(r_neg_m2[i, :, np.newaxis] * diff[i, :, :], axis=0)
    
    gradient = rep_grad + att_grad
    
    return energy, gradient


def _coarse_fine_core(n_exp, m_exp, c, core_size, rng):
    """
    Use a coarse-to-fine approach to find a good core configuration.
    Start with a small number of atoms on a grid, then refine.
    """
    # Start with a small core (e.g., 4-8 atoms) and find good configurations
    # Then use gradient-based optimization to refine
    
    best_core = None
    best_energy = np.inf
    
    # Try several initial configurations for the core
    n_attempts = min(20, max(5, core_size * 2))
    
    for attempt in range(n_attempts):
        if attempt == 0:
            # Start with a simple cubic-like arrangement
            positions = []
            s = 1.0
            for i in range(core_size):
                x = (i % 2) * s
                y = ((i // 2) % 2) * s
                z = (i // 4) * s
                positions.append([x, y, z])
            positions = np.array(positions, dtype=float)
        elif attempt == 1:
            # FCC-like arrangement
            positions = []
            s = 1.0
            for i in range(core_size):
                x = (i % 2) * s
                y = ((i // 2) % 2) * s
                z = (i // 4) * s
                if i % 2 == 0 and (i // 2) % 2 == 0:
                    x += 0.5 * s
                    y += 0.5 * s
                positions.append([x, y, z])
            positions = np.array(positions, dtype=float)
        else:
            # Random positions in a small sphere
            radius = 1.5 * core_size ** (1.0 / 3.0)
            positions = rng.uniform(-radius, radius, (core_size, 3))
            # Ensure no atoms are too close
            valid = True
            for i in range(core_size):
                for j in range(i + 1, core_size):
                    if np.linalg.norm(positions[i] - positions[j]) < 0.4:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
        
        # Optimize the core
        try:
            result = minimize(
                fun=lambda x: _sc_energy_grad(x.reshape(-1, 3), n_exp, m_exp, c)[0],
                x0=positions.flatten(),
                jac=lambda x: _sc_energy_grad(x.reshape(-1, 3), n_exp, m_exp, c)[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-8}
            )
            opt_positions = result.x.reshape(-1, 3)
            opt_energy = result.fun
            
            # Check validity
            if not np.all(np.isfinite(opt_positions)):
                continue
            for i in range(len(opt_positions)):
                for j in range(i + 1, len(opt_positions)):
                    if np.linalg.norm(opt_positions[i] - opt_positions[j]) < 0.35:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_core = opt_positions.copy()
        except Exception:
            continue
    
    if best_core is None:
        # Fallback: simple arrangement
        s = 1.0
        positions = []
        for i in range(core_size):
            x = (i % 2) * s
            y = ((i // 2) % 2) * s
            z = (i // 4) * s
            positions.append([x, y, z])
        best_core = np.array(positions, dtype=float)
    
    return best_core


def _insert_atom_optimal(positions, n_exp, m_exp, c, rng):
    """
    Find the optimal position to insert a new atom given existing positions.
    Use gradient-based search starting from several candidate positions.
    """
    n = len(positions)
    
    # Generate candidate insertion positions
    candidates = []
    
    # 1. Positions near existing atoms (but not too close)
    for i in range(n):
        for angle in range(0, 360, 45):
            for dist in [0.8, 1.0, 1.2, 1.5]:
                theta = np.radians(angle)
                phi = np.radians(rng.uniform(0, 360))
                offset = dist * np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ])
                candidates.append(positions[i] + offset)
    
    # 2. Random positions in a reasonable range
    radius = 2.0 * n ** (1.0 / 3.0)
    for _ in range(50):
        candidates.append(rng.uniform(-radius, radius, 3))
    
    # 3. Positions that are symmetric with respect to the cluster
    centroid = np.mean(positions, axis=0)
    for _ in range(20):
        candidates.append(centroid + rng.uniform(-1.5, 1.5, 3))
    
    candidates = np.array(candidates)
    
    # Filter out candidates that are too close to existing atoms
    valid_candidates = []
    for cand in candidates:
        if not np.all(np.isfinite(cand)):
            continue
        min_dist = min(np.linalg.norm(cand - p) for p in positions)
        if min_dist > 0.35:
            valid_candidates.append(cand)
    
    if len(valid_candidates) == 0:
        # Fallback: place atom far away
        return positions + np.array([[3.0, 0.0, 0.0]])
    
    # Optimize each candidate and keep the best
    best_energy = np.inf
    best_position = None
    
    # Limit the number of candidates to optimize for speed
    max_candidates = min(len(valid_candidates), 30)
    if len(valid_candidates) > max_candidates:
        # Sample randomly
        indices = rng.choice(len(valid_candidates), max_candidates, replace=False)
        valid_candidates = valid_candidates[indices]
    
    for cand in valid_candidates:
        new_positions = np.vstack([positions, cand.reshape(1, 3)])
        
        try:
            result = minimize(
                fun=lambda x: _sc_energy_grad(x.reshape(-1, 3), n_exp, m_exp, c)[0],
                x0=new_positions.flatten(),
                jac=lambda x: _sc_energy_grad(x.reshape(-1, 3), n_exp, m_exp, c)[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8}
            )
            opt_positions = result.x.reshape(-1, 3)
            opt_energy = result.fun
            
            # Check validity
            if not np.all(np.isfinite(opt_positions)):
                continue
            valid = True
            for i in range(len(opt_positions)):
                for j in range(i + 1, len(opt_positions)):
                    if np.linalg.norm(opt_positions[i] - opt_positions[j]) < 0.35:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_position = opt_positions.copy()
        except Exception:
            continue
    
    if best_position is None:
        # Fallback: use the best unoptimized candidate
        best_cand = valid_candidates[0]
        return np.vstack([positions, best_cand.reshape(1, 3)])
    
    return best_position


def construct(instance):
    """
    Build a Sutton-Chen cluster using coarse-to-fine grid search for the core
    and gradient-based optimal insertion for subsequent atoms.
    """
    size = instance["n"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    # Map kind to parameters
    param_map = {
        "12-6": (12, 6, 144.41),
        "9-6": (9, 6, 39.432),
        "10-8": (10, 8, 34.408),
    }
    n_exp, m_exp, c = param_map.get(kind, (12, 6, 144.41))
    
    rng = np.random.default_rng(size)
    
    # Step 1: Build a good core using coarse-to-fine search
    core_size = min(size, max(8, size // 4))
    core = _coarse_fine_core(n_exp, m_exp, c, core_size, rng)
    
    # Step 2: Insert remaining atoms one by one using gradient-based optimization
    positions = core.copy()
    
    for i in range(core_size, size):
        positions = _insert_atom_optimal(positions, n_exp, m_exp, c, rng)
    
    # Step 3: Final full relaxation using the provided relax function
    try:
        energy, final_positions = relax(positions, "sutton_chen", kind)
        if np.all(np.isfinite(final_positions)):
            # Check minimum distance
            valid = True
            for i in range(len(final_positions)):
                for j in range(i + 1, len(final_positions)):
                    if np.linalg.norm(final_positions[i] - final_positions[j]) < 0.35:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return final_positions
    except Exception:
        pass
    
    # Fallback: return the positions we have
    return positions
# EVOLVE-BLOCK-END
