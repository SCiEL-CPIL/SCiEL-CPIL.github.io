"""Rewritten program: FCC-truncated-octahedron parameterization + local polish."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy(pos):
    """Total LJ energy 4*sum(r^-12 - r^-6) over all pairs."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)  # avoid div by zero on diagonal
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6) - 4.0 * n  # subtract diagonal contribution (4*(1-1)=0, but let's be safe)
    # Actually diagonal is r=0 which we set to 1, so contribution is 4*(1-1)=0. No subtraction needed.
    return 4.0 * (np.sum(r12 - r6) - 0.0)  # diagonal is 0 contribution

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    # gradient: dE/dr2 = 4*(-6*r2^-7 + 12*r2^-13) = 4*(-6/r2^7 + 12/r2^13)
    # dE/dpos_i = sum_j 2*diff_ij * dE/dr2_ij
    dr2 = 4.0 * (-6.0 * r2 ** (-7) + 12.0 * r2 ** (-13))
    np.fill_diagonal(dr2, 0.0)
    grad = 2.0 * np.einsum('ij,ijk->ik', dr2, diff)
    return energy, grad

def _fcc_truncated_octahedron(a):
    """
    Generate FCC-truncated octahedron cluster of 38 atoms with lattice parameter a.
    Structure: FCC layers.
    Layer 1 (z=0): 10 atoms - hexagonal arrangement
    Layer 2 (z=a/2): 12 atoms - 
    Layer 3 (z=a): 10 atoms - hexagonal arrangement
    Layer 4 (z=3a/2): 6 atoms - 
    Wait, standard LJ38 global min is FCC truncated octahedron.
    Let me use the known coordinates or generate from FCC.
    
    FCC unit cell atoms at:
    (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)
    
    For a truncated octahedron of 38 atoms, it's a specific cluster.
    Let me use a parameterized approach:
    - Core: 1 atom at center
    - Shell 1: 12 atoms at distance r1 (icosahedral-like but FCC)
    - Shell 2: 25 atoms at distance r2
    
    Actually, the FCC truncated octahedron has:
    - 1 atom at center
    - 12 atoms at first shell (FCC neighbors)
    - 25 atoms at second shell
    
    For FCC, the first shell neighbors are at (±a/2, ±a/2, 0) and permutations: 12 atoms
    Second shell: various positions
    
    Let me just generate an FCC cluster and take the 38 closest to center.
    """
    # Generate FCC lattice points
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) in units of a
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    # Generate lattice points in a cube of side L
    L = 4  # enough to get 38+ atoms
    points = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b * a
                    points.append(p)
    points = np.array(points)
    
    # Find the 38 points closest to the origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:38]
    return points[idx]

def _breathing_fcc(params):
    """
    params[0] = a (lattice parameter)
    params[1] = breathing factor for shell 1 (multiplier on first-shell distances)
    params[2] = breathing factor for shell 2 (multiplier on second-shell distances)
    """
    a = params[0]
    b1 = params[1]
    b2 = params[2]
    
    # Generate FCC points with lattice parameter a
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    L = 4
    points = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b * a
                    points.append(p)
    points = np.array(points)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:38]
    pts = points[idx]
    d_pts = np.linalg.norm(pts, axis=1)
    
    # Apply breathing: scale points based on their shell
    # Shell 1: d < a*0.75 (approximately first shell)
    # Shell 2: d >= a*0.75
    shell1_mask = d_pts < a * 0.75
    shell2_mask = ~shell1_mask
    
    pts[shell1_mask] *= b1
    pts[shell2_mask] *= b2
    
    return pts

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin
    
    # Start with FCC truncated octahedron at optimal spacing
    # LJ minimum at r = 2^(1/6) ≈ 1.12246
    # For FCC, nearest neighbor distance = a/sqrt(2)
    # So a = sqrt(2) * 2^(1/6) = 2^(2/3) ≈ 1.5874
    a0 = 2.0 ** (1.0/3.0)
    
    best_pos = _fcc_truncated_octahedron(a0)
    best_energy = _lj_energy(best_pos)
    
    # Local relaxation with L-BFGS
    try:
        result = minimize(
            _lj_energy_and_grad,
            best_pos,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success or result.fun < best_energy:
            best_pos = result.x
            best_energy = result.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_pos
    
    # Try breathing optimization
    # Parameterize: a, b1, b2
    def _breathing_energy(params):
        pos = _breathing_fcc(params)
        # Check minimum distance constraint
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        min_r2 = np.min(r2[np.triu_indices_from(r2, k=1)])
        if min_r2 < 0.25:  # 0.5^2
            return 1e10
        return _lj_energy(pos)
    
    def _breathing_energy_grad(params):
        pos = _breathing_fcc(params)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        min_r2 = np.min(r2[np.triu_indices_from(r2, k=1)])
        if min_r2 < 0.25:
            return 1e10, np.zeros(3)
        e, g = _lj_energy_and_grad(pos)
        # We need dE/dparams, which requires chain rule. For simplicity, use finite differences or just return energy.
        return e, np.zeros(3)
    
    # Optimize breathing parameters
    try:
        x0 = np.array([a0, 1.0, 1.0])
        result_b = minimize(
            _breathing_energy,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if result_b.fun < best_energy:
            pos_b = _breathing_fcc(result_b.x)
            # Final L-BFGS polish
            try:
                result_c = minimize(
                    _lj_energy_and_grad,
                    pos_b,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if result_c.fun < best_energy:
                    best_pos = result_c.x
                    best_energy = result_c.fun
            except Exception:
                pass
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_pos
    
    # Try multiple random perturbations + relaxation
    rng = np.random.RandomState(42)
    for trial in range(20):
        if time.time() - start_time > deadline - 5:
            break
        
        # Small random perturbation of the best structure
        perturbed = best_pos + rng.normal(0, 0.05, size=(38, 3))
        
        try:
            result = minimize(
                _lj_energy_and_grad,
                perturbed,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_energy:
                best_pos = result.x
                best_energy = result.fun
        except Exception:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
