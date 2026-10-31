"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 38
TIME_LIMIT = 115.0  # safety margin before 120s kill


def _lj_energy_and_grad(x):
    """Compute LJ energy and gradient for N atoms.
    
    x: (N, 3) array of coordinates
    returns: energy, gradient
    """
    x = x.reshape(N, 3)
    # Pairwise differences
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.einsum('ijk,ijk->ij', diff, diff)  # (N, N)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r = np.sqrt(r2)
    
    # LJ energy: 4 * sum(r^-12 - r^-6) over all pairs (i < j)
    r6 = r2 ** 3  # r^6
    r12 = r6 ** 2  # r^12
    
    # Energy per pair (upper triangle)
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)
    
    # Gradient: dE/dx_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j) / r
    # = sum_j 4*(-12/r^13 + 6/r^7) * diff[i,j]
    # Actually: dE/dr = 4*(-12*r^-13 + 6*r^-7)
    # dE/dx_i = sum_{j != i} dE/dr * (x_i - x_j)/r = sum_j 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)
    
    r3 = r2 ** 1.5  # r^3
    r7 = r3 * r2  # r^5 * r^2 = r^7... let me redo
    # r^7 = r * r^6 = sqrt(r2) * r2^3
    # Let's use: grad factor = 4 * (-12 * r^-14 + 6 * r^-8) = 4 * (-12/r^14 + 6/r^8)
    # = 4 * r^-14 * (-12 + 6*r^6) = 24 * r^-14 * (r^6 - 2)
    
    # r^14 = (r^2)^7
    r14 = r2 ** 7
    factor = 24.0 * (r6 - 2.0) / r14  # (N, N)
    np.fill_diagonal(factor, 0.0)
    
    # grad[i] = sum_j factor[i,j] * (x[i] - x[j])
    grad = np.einsum('ij,ijk->ik', factor, diff)
    
    return energy, grad.flatten()


def _lj_energy(x):
    """Just the energy."""
    e, _ = _lj_energy_and_grad(x)
    return e


def _check_validity(x):
    """Check that all pairwise distances are at least 0.5."""
    x = x.reshape(N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    min_r = np.sqrt(np.min(r2))
    return min_r >= 0.5


def _fcct_octahedron():
    """Generate a truncated octahedron (FCC) fragment with 38 atoms.
    
    FCC lattice: basis at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) * a
    where a = 2^(1/6) * sqrt(2) for LJ sigma=1.
    Nearest neighbor distance = a/sqrt(2) = 2^(1/6).
    """
    a = 2.0 ** (1.0 / 6.0) * math.sqrt(2.0)  # lattice constant
    
    # Generate FCC points in a cubic region
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ])
    
    # Generate lattice points in a box large enough
    max_coord = 3
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b * a
                    points.append(p)
    
    points = np.array(points)
    
    # Center on the origin
    center = points.mean(axis=0)
    points -= center
    
    # Sort by distance from center, take 38 closest
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:N]]
    
    return points


def _optimize_scale(x, time_budget=5.0):
    """Optimize the global scale factor to minimize energy."""
    start = time.time()
    best_scale = 1.0
    best_e = _lj_energy(x)
    
    # Try a range of scales
    for scale in np.linspace(0.9, 1.1, 21):
        if time.time() - start > time_budget:
            break
        x_scaled = x * scale
        if _check_validity(x_scaled):
            e = _lj_energy(x_scaled)
            if e < best_e:
                best_e = e
                best_scale = scale
    
    return x * best_scale


def _icosahedron_seed():
    """Generate an icosahedral seed (known local minimum, but we start here and basin hop)."""
    # icosahedron: 12 vertices + inner shells
    phi = (1 + math.sqrt(5)) / 2.0
    # 12 vertices of icosahedron (edge length 2)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Scale so nearest neighbor = 2^(1/6)
    # Edge length of this icosahedron is 2
    scale = 2.0 ** (1.0 / 6.0) / 2.0
    verts *= scale
    
    # Add more atoms in shells
    # For 38 atoms: 12 outer + 12 middle + 12 inner + 1 center + 1 more?
    # Let's just use the FCC seed as primary and this as secondary
    
    # Actually, let's generate a more realistic icosahedral cluster
    # 12 outer, then a second shell of 20, etc.
    # For simplicity, use FCC truncated octahedron as the main seed
    return _fcct_octahedron()


def construct():
    """Construct 38-atom LJ cluster configuration minimizing total energy."""
    start_time = time.time()
    
    # Try multiple seeds and basin hopping
    best_x = None
    best_e = np.inf
    
    # Seed 1: FCC truncated octahedron (putative global minimum)
    x0 = _fcct_octahedron()
    if _check_validity(x0):
        e0, _ = _lj_energy_and_grad(x0)
        if e0 < best_e:
            best_e = e0
            best_x = x0.copy()
    
    # Local optimization from FCC seed
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 15:
        try:
            res = minimize(
                _lj_energy_and_grad, x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success or res.fun < best_e:
                if _check_validity(res.x):
                    best_e = res.fun
                    best_x = res.x.reshape(N, 3).copy()
        except Exception:
            pass
    
    # Optimize scale
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 15 and best_x is not None:
        try:
            x_scaled = _optimize_scale(best_x, time_budget=3.0)
            e_scaled, _ = _lj_energy_and_grad(x_scaled)
            if e_scaled < best_e:
                best_e = e_scaled
                best_x = x_scaled.copy()
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed >= TIME_LIMIT - 10:
        return best_x if best_x is not None else x0
    
    # Basin hopping from the best configuration
    rng = np.random.RandomState(42)
    step_size = 0.12
    max_iter = 300
    
    for iteration in range(max_iter):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 10:
            break
        
        # Perturb
        x_pert = best_x + rng.normal(0, step_size, size=(N, 3))
        
        # Local minimize
        try:
            res = minimize(
                _lj_energy_and_grad, x_pert,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if _check_validity(res.x):
                e_new = res.fun
                if e_new < best_e:
                    best_e = e_new
                    best_x = res.x.reshape(N, 3).copy()
                    # Reduce step size when improving
                    step_size = max(step_size * 0.9, 0.05)
                else:
                    # Metropolis-like acceptance
                    dE = e_new - best_e
                    T = 0.08 * (1 - iteration / max_iter) + 0.02
                    if rng.random() < math.exp(-dE / max(T, 1e-10)):
                        best_x = res.x.reshape(N, 3).copy()
                    # Gradually increase step size
                    step_size = min(step_size * 1.01, 0.25)
        except Exception:
            continue
    
    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 5 and best_x is not None:
        try:
            res = minimize(
                _lj_energy_and_grad, best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if _check_validity(res.x) and res.fun < best_e:
                best_e = res.fun
                best_x = res.x.reshape(N, 3).copy()
        except Exception:
            pass
    
    # One more polish pass with even tighter tolerance
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 2 and best_x is not None:
        try:
            res = minimize(
                _lj_energy_and_grad, best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-15}
            )
            if _check_validity(res.x) and res.fun < best_e:
                best_e = res.fun
                best_x = res.x.reshape(N, 3).copy()
        except Exception:
            pass
    
    if best_x is None:
        best_x = x0
    
    return best_x


# EVOLVE-BLOCK-START
# (The above code is the evolved block)
# EVOLVE-BLOCK-END
