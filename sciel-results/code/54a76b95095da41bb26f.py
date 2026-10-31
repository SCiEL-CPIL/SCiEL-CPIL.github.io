"""Direct FCC truncated octahedron construction with multi-stage high-precision relaxation."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    mask = ~np.eye(len(pos), dtype=bool)
    r2 = r2[mask]
    r6 = r2 ** 3
    r12 = r6 ** 2
    energy = 4.0 * np.sum(1.0 / r12 - 1.0 / r6)
    return energy

def _lj_energy_and_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    r2_safe = np.where(r2 < 1e-12, 1e-12, r2)
    r6 = r2_safe ** 3
    r12 = r6 ** 2
    
    mask = ~np.eye(len(pos), dtype=bool)
    energy = 4.0 * np.sum(1.0 / r12[mask] - 1.0 / r6[mask])
    
    coeff = 4.0 * (-12.0 / (r2_safe ** 7) + 6.0 / (r2_safe ** 4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _make_fcc_truncated_octahedron():
    """Generate the exact 38-atom FCC truncated octahedron positions.
    
    The 38-atom FCC cluster is a truncated octahedron with:
    - 1 center atom
    - 12 atoms in the first shell (vertices of cuboctahedron)
    - 14 atoms in the second shell (face centers)
    - 11 atoms in the third shell (remaining vertices)
    
    Using FCC lattice with nearest-neighbor distance = 1 (in lattice units),
    the actual LJ optimal spacing is 2^(1/6).
    """
    # FCC lattice: points where i+j+k is even, with spacing a
    # Nearest neighbor distance in FCC = a * sqrt(2)
    # We want nearest neighbor = 1 in "lattice units" so a = 1/sqrt(2)
    # But let's just generate integer FCC points and scale at the end
    
    # Generate FCC points within a radius
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 0:
                    points.append([i, j, k])
    
    pts = np.array(points, dtype=float)
    dists = np.sum(pts ** 2, axis=1)
    sorted_idx = np.argsort(dists)
    
    # Take 38 closest points
    cluster = pts[sorted_idx[:N]].copy()
    
    # The nearest neighbor distance in this lattice is sqrt(2)
    # For LJ minimum, we want r_min = 2^(1/6)
    # So scale = 2^(1/6) / sqrt(2)
    r_min = 2.0 ** (1.0 / 6.0)
    scale = r_min / np.sqrt(2.0)
    cluster *= scale
    
    return cluster

def _relax(pos, max_iter=2000, tol=1e-12):
    """L-BFGS relaxation with tight tolerance."""
    res = minimize(
        _lj_energy,
        pos,
        jac=lambda p: _lj_energy_and_grad(p)[1],
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 100}
    )
    return res.x

def _polish(pos, max_iter=5000, tol=1e-14):
    """High-precision polish pass."""
    res = minimize(
        _lj_energy,
        pos,
        jac=lambda p: _lj_energy_and_grad(p)[1],
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 200}
    )
    return res.x

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure minimum separation between atoms."""
    for _ in range(100):
        min_dist = np.inf
        min_pair = None
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
        if min_dist >= min_sep:
            break
        # Push apart
        diff = pos[min_pair[0]] - pos[min_pair[1]]
        norm = np.linalg.norm(diff)
        if norm < 1e-12:
            diff = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        push = (min_sep - min_dist) / 2.0
        pos[min_pair[0]] += push * diff / norm
        pos[min_pair[1]] -= push * diff / norm
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Stage 1: Construct exact FCC truncated octahedron
    pos = _make_fcc_truncated_octahedron()
    
    # Stage 2: Initial relaxation with moderate tolerance
    try:
        pos = _relax(pos, max_iter=1000, tol=1e-10)
    except Exception:
        pass
    
    if time.time() > deadline - 25.0:
        return _ensure_min_sep(pos)
    
    # Stage 3: High-precision polish
    try:
        pos_polished = _polish(pos, max_iter=5000, tol=1e-14)
        e_orig = _lj_energy(pos)
        e_polished = _lj_energy(pos_polished)
        if e_polished < e_orig:
            pos = pos_polished
    except Exception:
        pass
    
    if time.time() > deadline - 15.0:
        return _ensure_min_sep(pos)
    
    # Stage 4: Try perturbations to escape any shallow local minima
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    rng = np.random.RandomState(12345)
    
    # Multiple perturbation scales for better exploration
    perturbation_scales = [0.005, 0.01, 0.02, 0.03, 0.05]
    
    for attempt in range(40):
        if time.time() > deadline - 10.0:
            break
        
        # Vary perturbation scale
        scale = perturbation_scales[attempt % len(perturbation_scales)]
        
        # Small random perturbation
        perturbed = best_pos + rng.normal(0, scale, size=(N, 3))
        
        try:
            relaxed = _relax(perturbed, max_iter=500, tol=1e-12)
            energy = _lj_energy(relaxed)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
        except Exception:
            continue
    
    # Final ultra-tight polish
    if time.time() < deadline - 5.0:
        try:
            final = _polish(best_pos, max_iter=5000, tol=1e-15)
            e_final = _lj_energy(final)
            if e_final < best_energy:
                best_pos = final
                best_energy = e_final
        except Exception:
            pass
    
    return _ensure_min_sep(best_pos)
# EVOLVE-BLOCK-END
