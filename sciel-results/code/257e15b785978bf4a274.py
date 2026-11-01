"""Lennard-Jones cluster global minimum, 55 atoms."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE = 115.0  # Safety margin before 120s kill

def _lj_energy(pos):
    """Compute total LJ energy for N atoms (vectorized)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def _lj_energy_grad(pos):
    """Compute LJ energy and gradient (vectorized)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    # dE/dpos_i = sum_{j!=i} 4 * (-12*r_ij^-14 + 6*r_ij^-8) * (pos_i - pos_j)
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def _min_distance(positions):
    """Compute minimum pairwise distance."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def _get_mackay_directions():
    """
    Get the 55 unit direction vectors for a Mackay icosahedron M(2,2,2).
    Uses exact icosahedral symmetry operations.
    Returns:
        directions: (55, 3) array of unit vectors (center is zero)
        shell_ids: (55,) array indicating which shell each atom belongs to
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 icosahedron vertices (exact)
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # 30 edge midpoints: for each pair of adjacent vertices
    # Adjacency: each vertex connects to 5 others. We find edges by distance.
    # The edge length for unit icosahedron is 2*sin(pi/5) = 2*sqrt((5-sqrt(5))/8)
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - edge_len) < 1e-12:
                edges.append((i, j))
    
    # Verify we have exactly 30 edges
    assert len(edges) == 30, f"Expected 30 edges, got {len(edges)}"
    
    # 30 edge midpoints (normalized)
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    # Total: 1 (center) + 12 (shell 1) + 30 (shell 2) + 12 (shell 3) = 55
    directions = np.zeros((55, 3))
    shell_ids = np.zeros(55, dtype=int)
    
    # Atom 0: center
    directions[0] = [0.0, 0.0, 0.0]
    shell_ids[0] = 0
    
    # Atoms 1-12: Shell 1 (icosahedron vertices)
    for i in range(12):
        directions[1+i] = verts[i]
        shell_ids[1+i] = 1
    
    # Atoms 13-42: Shell 2 (edge midpoints)
    for i in range(30):
        directions[13+i] = midpoints[i]
        shell_ids[13+i] = 2
    
    # Atoms 43-54: Shell 3 (icosahedron vertices, outer)
    for i in range(12):
        directions[43+i] = verts[i]
        shell_ids[43+i] = 3
    
    return directions, shell_ids

def _make_mackay_from_radii(radii):
    """Generate Mackay icosahedron positions given shell radii [r1, r2, r3]."""
    directions, shell_ids = _get_mackay_directions()
    
    pos = np.zeros((55, 3))
    for i in range(1, 55):
        shell = shell_ids[i]
        if shell == 1:
            pos[i] = directions[i] * radii[0]
        elif shell == 2:
            pos[i] = directions[i] * radii[1]
        elif shell == 3:
            pos[i] = directions[i] * radii[2]
    
    return pos

def _relax(positions, tol=1e-6, maxiter=300):
    """Relax positions using L-BFGS-B."""
    def objective(x):
        pos = x.reshape(N, 3)
        e, g = _lj_energy_grad(pos)
        return e, g.flatten()
    
    x0 = positions.flatten()
    try:
        result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                          options={'gtol': tol, 'maxiter': maxiter, 'maxls': 50, 'ftol': 1e-10})
        if result.success or result.fun < 1e6:
            return result.x.reshape(N, 3), result.fun
    except Exception:
        pass
    return positions, _lj_energy(positions)

def _liquid_quench(seed=42, max_time=100.0):
    """
    Liquid quench: start from disordered high-density config, cool via
    progressively tighter L-BFGS relaxations.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Initialize: random positions in a small sphere (high density liquid)
    # Typical LJ nearest-neighbor ~1.122, so sphere radius ~3-4 for 55 atoms
    radius = 3.0
    pos = rng.randn(N, 3)
    norms = np.linalg.norm(pos, axis=1, keepdims=True)
    pos = pos / norms * (radius * (rng.rand(N, 1) ** (1/3)))
    
    # Ensure minimum distance constraint during initialization
    for _ in range(1000):
        min_d = _min_distance(pos)
        if min_d >= 0.5:
            break
        # Push apart any atoms too close
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, np.inf)
        too_close = r2 < 0.25  # distance < 0.5
        if not np.any(too_close):
            break
        # Separate pairs that are too close
        for i in range(N):
            for j in range(i+1, N):
                if too_close[i, j]:
                    d = pos[i] - pos[j]
                    dist = np.linalg.norm(d)
                    if dist < 1e-10:
                        d = rng.randn(3)
                        dist = np.linalg.norm(d)
                    d = d / dist
                    # Push apart to exactly 0.5
                    push = (0.5 - dist) / 2.0
                    pos[i] += push * d
                    pos[j] -= push * d
    
    current_energy = _lj_energy(pos)
    
    # Cooling schedule: progressively tighter tolerances
    # Phase 1: Coarse relaxation (high "temperature" - allow larger moves)
    # Phase 2: Medium relaxation
    # Phase 3: Fine relaxation
    # Phase 4: Very fine polish
    
    cooling_steps = [
        (1e-2, 50),   # coarse, few iterations
        (1e-3, 100),  # medium
        (1e-4, 200),  # fine
        (1e-5, 300),  # very fine
        (1e-6, 500),  # polish
        (1e-8, 500),  # high precision
    ]
    
    best_energy = current_energy
    best_pos = pos.copy()
    
    for tol, maxiter in cooling_steps:
        if time.time() - start_time > max_time:
            break
        
        pos, e = _relax(pos, tol=tol, maxiter=maxiter)
        
        if e < best_energy:
            best_energy = e
            best_pos = pos.copy()
        
        # If energy increased significantly, try to recover
        if e > best_energy * 1.5 and e > 1e3:
            # Reset to best and try with slightly different tolerance
            pos = best_pos.copy()
            pos, e = _relax(pos, tol=max(tol*10, 1e-3), maxiter=maxiter//2)
            if e < best_energy:
                best_energy = e
                best_pos = pos.copy()
    
    return best_pos, best_energy

def _optimize_radii(radii_init, maxiter_lbfgs=150, deadline=None):
    """
    Optimize shell radii using L-BFGS with numerical gradient,
    with L-BFGS relaxation at each step.
    """
    def objective(radii):
        radii = np.maximum(radii, 0.5)
        pos = _make_mackay_from_radii(radii)
        pos_relaxed, e = _relax(pos, tol=1e-4, maxiter=maxiter_lbfgs)
        return e
    
    def objective_grad(radii):
        e = objective(radii)
        # Numerical gradient
        g = np.zeros(3)
        h = 1e-4
        for i in range(3):
            r_plus = radii.copy()
            r_plus[i] += h
            r_minus = radii.copy()
            r_minus[i] = max(r_minus[i] - h, 0.5)
            g[i] = (objective(r_plus) - objective(r_minus)) / (r_plus[i] - r_minus[i])
        return e, g
    
    try:
        # Use L-BFGS on the 3D radial space
        res = minimize(objective_grad, radii_init, method='L-BFGS-B', jac=True,
                      options={'maxiter': 30, 'gtol': 1e-4, 'maxls': 20, 'ftol': 1e-6})
        best_radii = np.maximum(res.x, 0.5)
        best_pos = _make_mackay_from_radii(best_radii)
        best_pos, best_e = _relax(best_pos, tol=1e-6, maxiter=200)
        return best_pos, best_e, best_radii
    except Exception:
        return None, np.inf, radii_init

def _perturb(positions, scale=0.1, seed=None):
    """Add small random perturbation."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    return positions + rng.randn(*positions.shape) * scale

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE - elapsed()
    
    # Get Mackay directions
    directions, shell_ids = _get_mackay_directions()
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Try multiple Mackay radius configurations with tight relaxation
    # The known global minimum has specific shell radii
    radius_candidates = [
        np.array([1.0, 1.75, 2.5]),
        np.array([1.0, 1.8, 2.6]),
        np.array([1.05, 1.85, 2.7]),
        np.array([0.95, 1.7, 2.45]),
        np.array([1.1, 1.9, 2.75]),
        np.array([1.0, 1.78, 2.55]),
    ]
    
    for radii_init in radius_candidates:
        if remaining() < 10:
            break
        try:
            pos = _make_mackay_from_radii(radii_init)
            # Tight relaxation to converge to local minimum
            relaxed, e = _relax(pos, tol=1e-8, maxiter=500)
            if e < best_energy:
                best_energy = e
                best_positions = relaxed.copy()
        except Exception:
            continue
    
    # Fallback if nothing worked
    if best_positions is None:
        best_positions = _make_mackay_from_radii(np.array([1.0, 1.8, 2.6]))
        best_energy = _lj_energy(best_positions)
    
    # Phase 2: Multi-start perturbation-relax around best configuration
    # Small perturbations to escape shallow local minima
    num_starts = 12
    for i in range(num_starts):
        if remaining() < 8:
            break
        
        # Varying perturbation scales
        if i < 6:
            scale = 0.01 * (i + 1)
        else:
            scale = 0.05 * (i - 4)
        
        try:
            perturbed = _perturb(best_positions, scale=scale, seed=200 + i)
            relaxed, e = _relax(perturbed, tol=1e-7, maxiter=300)
            if e < best_energy:
                best_energy = e
                best_positions = relaxed.copy()
        except Exception:
            continue
    
    # Phase 3: Radial optimization - optimize per-atom radial distances
    if remaining() > 15:
        try:
            def radial_objective(radial_params):
                pos = np.zeros((55, 3))
                pos[0] = [0, 0, 0]
                for i in range(1, 55):
                    pos[i] = directions[i] * max(radial_params[i-1], 0.5)
                return _lj_energy(pos)
            
            def radial_grad(radial_params):
                # Numerical gradient for radial parameters
                e = radial_objective(radial_params)
                g = np.zeros(54)
                h = 1e-6
                for i in range(54):
                    rp = radial_params.copy()
                    rm = radial_params.copy()
                    rp[i] += h
                    rm[i] = max(rm[i] - h, 0.5)
                    g[i] = (radial_objective(rp) - radial_objective(rm)) / (rp[i] - rm[i])
                return e, g
            
            radial_init = np.zeros(54)
            for i in range(1, 55):
                radial_init[i-1] = np.linalg.norm(best_positions[i])
            
            res = minimize(radial_objective, radial_init, method='L-BFGS-B',
                          options={'maxiter': 300, 'gtol': 1e-8, 'maxls': 50, 'ftol': 1e-10})
            
            if res.fun < best_energy:
                pos_radial = np.zeros((55, 3))
                pos_radial[0] = [0, 0, 0]
                for i in range(1, 55):
                    pos_radial[i] = directions[i] * max(res.x[i-1], 0.5)
                # Full relaxation after radial optimization
                relaxed, e = _relax(pos_radial, tol=1e-8, maxiter=300)
                if e < best_energy:
                    best_energy = e
                    best_positions = relaxed.copy()
        
        except Exception:
            pass
    
    # Phase 4: Full 3D optimization with tight tolerance
    if remaining() > 10:
        try:
            def full_objective(x):
                pos = x.reshape(N, 3)
                e, g = _lj_energy_grad(pos)
                return e, g.flatten()
            
            x0 = best_positions.flatten()
            result = minimize(full_objective, x0, method='L-BFGS-B', jac=True,
                            options={'gtol': 1e-10, 'maxiter': 500, 'maxls': 50, 'ftol': 1e-12})
            if result.fun < best_energy:
                best_energy = result.fun
                best_positions = result.x.reshape(N, 3)
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish
    if remaining() > 5:
        try:
            relaxed_final, e_final = _relax(best_positions, tol=1e-12, maxiter=1000)
            if e_final < best_energy:
                best_energy = e_final
                best_positions = relaxed_final
        except Exception:
            pass
    
    # Ensure minimum distance constraint
    min_d = _min_distance(best_positions)
    if min_d < 0.5:
        scale = 0.5 / min_d
        best_positions = best_positions * scale
    
    # Final energy check and return
    final_energy = _lj_energy(best_positions)
    return best_positions
