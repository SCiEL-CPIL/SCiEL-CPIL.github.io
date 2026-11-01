"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the maximum common radius for a given set of centers."""
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists, axis=1)
    
    radii = np.minimum(wall_dists, min_pair_dist / 2.0)
    
    return np.min(radii)

def _repulsion_energy(centers, power=6, wall_power=6, epsilon=1e-6):
    """
    Compute a repulsion-based energy.
    E = sum_{i<j} (1 / (d_ij + eps)^power) + sum_i (1 / (wall_i + eps)^power)
    Minimizing this spreads points out.
    """
    energy = 0.0
    
    # Pairwise repulsion
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Use upper triangle to avoid double counting
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx]
    energy += np.sum(1.0 / (pair_dists + epsilon) ** power)
    
    # Wall repulsion
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    # Ensure wall_dists are positive (clamp if needed, though they should be)
    wall_dists_clamped = np.maximum(wall_dists, epsilon)
    energy += np.sum(1.0 / (wall_dists_clamped ** wall_power))
    
    return energy

def _repulsion_grad(centers, power=6, wall_power=6, epsilon=1e-6):
    """
    Gradient of the repulsion energy.
    """
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx]
    diff_vecs = diff[i_idx, j_idx] # shape (M, 3)
    
    # d/dx_i (1/(d+eps)^p) = -p * (x_i - x_j) / (d+eps)^(p+1) * (1/d) ? No.
    # Let f(d) = (d+eps)^-p. df/dd = -p (d+eps)^(-p-1).
    # d/dx_i d_ij = (x_i - x_j) / d_ij.
    # So grad_i = -p (d+eps)^(-p-1) * (x_i - x_j) / d
    # grad_j = -p (d+eps)^(-p-1) * (x_j - x_i) / d
    
    factor = -power * (pair_dists + epsilon) ** (-power - 1) / np.maximum(pair_dists, epsilon)
    factor = factor[:, np.newaxis] # shape (M, 1)
    
    grad_terms = factor * diff_vecs # shape (M, 3)
    
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        grad[i] += grad_terms[k]
        grad[j] -= grad_terms[k]
    
    # Wall gradient
    # wall_i = 0.5 - max(|c_ik|)
    # Let m_i = argmax_k |c_ik|
    # d wall_i / d c_im = -sign(c_im)
    # E_wall = sum (1/wall_i^p). dE/dwall = -p wall^(-p-1).
    # dE/dc_im = -p wall^(-p-1) * (-sign(c_im)) = p wall^(-p-1) sign(c_im)
    
    abs_centers = np.abs(centers)
    max_idx = np.argmax(abs_centers, axis=1)
    wall_dists = 0.5 - np.max(abs_centers, axis=1)
    wall_dists_clamped = np.maximum(wall_dists, epsilon)
    
    wall_factor = wall_power * wall_dists_clamped ** (-wall_power - 1)
    
    for i in range(N):
        dim = max_idx[i]
        sign_val = np.sign(centers[i, dim])
        if sign_val == 0:
            sign_val = 1.0
        grad[i, dim] += wall_factor[i] * sign_val
        
    return grad

def _anneal(centers, start_temp=0.1, end_temp=0.001, n_iter=500, power=6):
    """
    Simulated annealing using repulsion energy.
    Moves are random perturbations. Accept if energy decreases or with prob exp(-dE/T).
    """
    current_energy = _repulsion_energy(centers, power=power)
    best_centers = centers.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(12345)
    
    for it in range(n_iter):
        # Temperature schedule (geometric cooling)
        T = start_temp * (end_temp / start_temp) ** (it / n_iter)
        
        # Generate a random move: pick a random sphere and perturb it
        i = rng.integers(0, N)
        step_size = 0.01 * (1.0 + it / n_iter) # Slightly increase step size? Or decrease?
        # Usually step size decreases. Let's keep it constant or slightly decreasing.
        step_size = 0.02 * (1.0 - 0.5 * it / n_iter)
        
        new_centers = centers.copy()
        new_centers[i] += rng.normal(0, step_size, size=3)
        
        # Clip to bounds
        new_centers[i] = np.clip(new_centers[i], -0.5, 0.5)
        
        new_energy = _repulsion_energy(new_centers, power=power)
        
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            centers = new_centers
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_centers = centers.copy()
                
    return best_centers

def _polish(centers, beta=500, max_iter=50):
    """
    Final polish using L-BFGS-B on a smooth approximation of the radius.
    We maximize min distance. Equivalent to minimizing -min distance.
    We use a soft-min approximation.
    """
    def _softmin_radius(centers, beta):
        wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_half_dists = dists / 2.0
        np.fill_diagonal(pair_half_dists, np.inf)
        
        M = np.zeros((N, N + 1))
        M[:, 0] = wall_dists
        M[:, 1:] = pair_half_dists
        
        # Smooth min over distances for each sphere
        x = -beta * M
        x_max = np.max(x, axis=1, keepdims=True)
        x_max = np.where(x_max == -np.inf, 0, x_max)
        lse_i = np.log(np.sum(np.exp(x - x_max), axis=1)) + np.squeeze(x_max, axis=1)
        r_i_smooth = -1.0 / beta * lse_i
        
        # Smooth min over spheres
        y = -beta * r_i_smooth
        y_max = np.max(y, keepdims=True)
        y_max = np.where(y_max == -np.inf, 0, y_max)
        lse_r = np.log(np.sum(np.exp(y - y_max))) + np.squeeze(y_max)
        r_smooth = -1.0 / beta * lse_r
        
        return r_smooth

    def _softmin_grad(centers, beta):
        wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_half_dists = dists / 2.0
        np.fill_diagonal(pair_half_dists, np.inf)
        
        M = np.zeros((N, N + 1))
        M[:, 0] = wall_dists
        M[:, 1:] = pair_half_dists
        
        # Weights v_kj
        x = -beta * M
        x_max = np.max(x, axis=1, keepdims=True)
        x_max = np.where(x_max == -np.inf, 0, x_max)
        exp_x = np.exp(x - x_max)
        V_k = np.sum(exp_x, axis=1, keepdims=True)
        v = exp_x / V_k
        
        # r_k_smooth
        r_k = -1.0 / beta * (np.log(np.squeeze(V_k, axis=1)) + np.squeeze(x_max, axis=1))
        
        # Weights w_k
        y = -beta * r_k
        y_max = np.max(y, keepdims=True)
        y_max = np.where(y_max == -np.inf, 0, y_max)
        exp_y = np.exp(y - y_max)
        U = np.sum(exp_y, keepdims=True)
        w = exp_y / U
        
        grad_d = -w * v
        
        grad_flat = np.zeros_like(centers.flatten())
        
        # Wall gradients
        abs_centers = np.abs(centers)
        max_idx = np.argmax(abs_centers, axis=1)
        
        for k in range(N):
            g_wall = grad_d[k, 0]
            if g_wall != 0:
                dim = max_idx[k]
                c_val = centers[k, dim]
                grad_flat[k * 3 + dim] += g_wall * (-np.sign(c_val))
                
        # Pair gradients
        for k in range(N):
            for j in range(1, N + 1):
                l = j - 1
                if k == l:
                    continue
                g_pair = grad_d[k, j]
                if g_pair != 0:
                    diff_vec = centers[k] - centers[l]
                    dist_val = np.linalg.norm(diff_vec)
                    if dist_val > 1e-9:
                        grad_vec = 0.5 * diff_vec / dist_val
                        grad_flat[k * 3 : (k + 1) * 3] += g_pair * grad_vec
                        grad_flat[l * 3 : (l + 1) * 3] += g_pair * (-grad_vec)
                        
        return grad_flat

    flat_centers = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def obj(x):
        return -_softmin_radius(x.reshape(N, 3), beta)
        
    def grad(x):
        return _softmin_grad(x.reshape(N, 3), beta)
        
    try:
        res = minimize(
            obj,
            flat_centers,
            jac=grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        new_centers = res.x.reshape((N, 3))
        if np.all(np.isfinite(new_centers)):
            return new_centers
    except Exception:
        pass
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_centers = None
    best_radius = -np.inf
    
    # Multi-start strategy
    # 1. Random starts with annealing
    # 2. Grid starts with annealing
    # 3. FCC-like starts with annealing
    
    rng = np.random.default_rng(42)
    
    # Start 1: Random
    if time.time() < deadline - 30:
        centers = rng.uniform(-0.4, 0.4, size=(N, 3))
        annealed = _anneal(centers, start_temp=0.2, end_temp=0.005, n_iter=1000, power=6)
        polished = _polish(annealed, beta=100, max_iter=30)
        r = _compute_radius(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()
            
    # Start 2: Grid
    if time.time() < deadline - 30:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        indices = rng.choice(len(grid), size=N, replace=False)
        centers = grid[indices].copy()
        # Add small noise
        centers += rng.normal(0, 0.01, size=(N, 3))
        annealed = _anneal(centers, start_temp=0.1, end_temp=0.005, n_iter=1000, power=6)
        polished = _polish(annealed, beta=100, max_iter=30)
        r = _compute_radius(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()
            
    # Start 3: FCC-like
    if time.time() < deadline - 30:
        # Generate FCC lattice points in a cube
        # FCC unit cell has 4 points.
        # We need 50 points.
        # Let's generate a dense FCC lattice and pick 50.
        # Scale factor s.
        # Points in FCC: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) + integer shifts.
        pts = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    pts.append((i, j, k))
                    pts.append((i + 0.5, j + 0.5, k))
                    pts.append((i + 0.5, j, k + 0.5))
                    pts.append((i, j + 0.5, k + 0.5))
        pts = np.array(pts)
        # Normalize to fit in [-0.5, 0.5]
        # Current range is 0 to 3.5. Center is 1.75.
        pts = (pts - 1.75) / 1.75 # Scale to roughly [-1, 1]
        pts = pts * 0.45 # Scale to fit inside with margin
        # Shuffle and take 50
        indices = rng.choice(len(pts), size=N, replace=False)
        centers = pts[indices].copy()
        centers += rng.normal(0, 0.01, size=(N, 3))
        annealed = _anneal(centers, start_temp=0.1, end_temp=0.005, n_iter=1000, power=6)
        polished = _polish(annealed, beta=100, max_iter=30)
        r = _compute_radius(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()
            
    # Final polish on best
    if best_centers is not None and time.time() < deadline - 5:
        polished = _polish(best_centers, beta=1000, max_iter=50)
        r = _compute_radius(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()
            
    return best_centers

# EVOLVE-BLOCK-END
