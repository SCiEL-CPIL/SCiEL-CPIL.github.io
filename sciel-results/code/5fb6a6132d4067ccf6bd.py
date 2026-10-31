import time
import numpy as np
from scipy.optimize import minimize
import random

N = 38

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    # Vectorized calculation
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0) # Avoid division by zero
    
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient: 
    # dU/dr = 4 * (-12/r^13 + 6/r^7)
    # dU/drij = 4 * (-12/r^13 + 6/r^7) * (1/r) * rij  ... wait
    # U = 4 * (r^-12 - r^-6)
    # dU/dr = 4 * (-12 r^-13 - (-6) r^-7) = 4 * (-12 r^-13 + 6 r^-7)
    # dU/dx_i = sum_j dU/dr_ij * dr_ij/dx_i
    # r_ij^2 = (x_i - x_j)^2 + ...
    # dr_ij/dx_i = (x_i - x_j) / r_ij
    # dU/dx_i = sum_j 4 * (-12 r_ij^-13 + 6 r_ij^-7) * (x_i - x_j) / r_ij
    #          = sum_j 4 * (-12 r_ij^-14 + 6 r_ij^-8) * (x_i - x_j)
    
    r8 = r2 ** 4
    r14 = r2 ** 7
    # coeff for (x_i - x_j)
    # 4 * (-12/r^14 + 6/r^8)
    coeff = 4.0 * (-12.0 / r14 + 6.0 / r8)
    np.fill_diagonal(coeff, 0.0)
    
    # grad_i = sum_j coeff_ij * (x_i - x_j)
    # = x_i * sum_j coeff_ij - sum_j coeff_ij * x_j
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _generate_fcc_38():
    """
    Generate the 38 closest points in the FCC lattice to the origin.
    FCC lattice points: (x,y,z) where x,y,z are integers or half-integers, 
    and x+y+z is an integer.
    """
    pts = []
    # Generate candidates in a cube [-4, 4]
    # Step 0.5
    for i in range(-8, 9):
        for j in range(-8, 9):
            for k in range(-8, 9):
                x = i * 0.5
                y = j * 0.5
                z = k * 0.5
                # Check FCC condition: x+y+z is integer
                s = x + y + z
                if abs(s - round(s)) < 1e-9:
                    d2 = x*x + y*y + z*z
                    pts.append((d2, x, y, z))
    
    pts = np.array(pts)
    # Sort by distance squared
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    
    # Take first 38
    selected = pts[:N, 1:4]
    return selected

def _relax(pos, max_iter=1000, gtol=1e-10):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-14},
        )
        p_out = res.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        return p_out, e_out
    except Exception:
        return pos, _lj_energy_and_grad(pos)[0]

def _optimize_scale(pos, e_current):
    """
    Optimize the global scale factor to minimize energy.
    Uses 1D minimization on the scale factor.
    """
    def energy_scaled(scale):
        p_scaled = pos * scale
        e, _ = _lj_energy_and_grad(p_scaled)
        return e
    
    # Initial scale is 1.0
    # Use scalar minimization
    from scipy.optimize import minimize_scalar
    try:
        res = minimize_scalar(energy_scaled, bounds=(0.8, 1.2), method='bounded')
        if res.success:
            best_scale = res.x
            p_scaled = pos * best_scale
            e_new, _ = _lj_energy_and_grad(p_scaled)
            if e_new < e_current:
                return p_scaled, e_new
    except Exception:
        pass
    return pos, e_current

def _basin_hop(pos, n_steps=20, step_size=0.5, seed=42):
    """
    Simple Basin Hopping algorithm.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_e, _ = _lj_energy_and_grad(current_pos)
    best_pos = current_pos.copy()
    best_e = current_e
    
    for i in range(n_steps):
        # Perturb
        noise = rng.normal(0, step_size, size=current_pos.shape)
        new_pos = current_pos + noise
        
        # Relax
        relaxed_pos, relaxed_e = _relax(new_pos, max_iter=200, gtol=1e-8)
        
        # Accept if better
        if relaxed_e < current_e:
            current_pos = relaxed_pos
            current_e = relaxed_e
        
        # Update best
        if current_e < best_e:
            best_e = current_e
            best_pos = current_pos.copy()
            
        # Optionally anneal step size
        step_size *= 0.95
        
    return best_pos, best_e

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure minimum inter-atomic separation."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin

    # 1. Generate FCC truncated octahedron
    pos = _generate_fcc_38()
    
    # Center
    pos -= np.mean(pos, axis=0)
    
    # Initial scaling: 
    # The FCC lattice generated has NN distance sqrt(0.5) in the grid units?
    # Let's check distance between (0,0,0) and (0.5, 0.5, 0).
    # d = sqrt(0.25+0.25) = sqrt(0.5) ~ 0.707.
    # We want NN distance to be 2^(1/6) ~ 1.122.
    # So scale factor should be approx 1.122 / 0.707 ~ 1.58.
    
    # Compute current NN distance
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_nn = np.sqrt(np.min(r2))
    
    target_nn = 2.0 ** (1.0 / 6.0)
    scale = target_nn / min_nn
    pos = pos * scale
    
    # Initial Relaxation
    pos, e = _relax(pos, max_iter=500, gtol=1e-10)
    
    # Optimize Scale
    pos, e = _optimize_scale(pos, e)
    
    # Final Relaxation
    pos, e = _relax(pos, max_iter=1000, gtol=1e-12)
    
    # Basin Hopping to ensure global minimum
    if time.time() < deadline - 30.0:
        # Run a few basin hopping steps
        # Use a smaller step size since we are close
        pos_bh, e_bh = _basin_hop(pos, n_steps=10, step_size=0.3, seed=123)
        
        if e_bh < e:
            pos = pos_bh
            e = e_bh
            
            # Relax and scale again if improved
            pos, e = _relax(pos, max_iter=500, gtol=1e-10)
            pos, e = _optimize_scale(pos, e)
            pos, e = _relax(pos, max_iter=1000, gtol=1e-12)

    # Final check and separation enforcement
    pos = _ensure_min_sep(pos)
    
    return pos

if __name__ == "__main__":
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
