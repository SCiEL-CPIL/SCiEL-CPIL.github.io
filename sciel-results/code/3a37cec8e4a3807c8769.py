"""LJ38 global minimum search via FCC truncated octahedron seed + L-BFGS relaxation."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for positions."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return 4.0 * np.sum(r12[mask] - r6[mask])

def _lj_energy_grad(pos):
    """Compute energy and gradient."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    energy = 4.0 * np.sum(r12[mask] - r6[mask])
    
    coeff = 4.0 * (6.0 * r6 - 12.0 * r12)
    coeff = coeff * mask
    row_sum = coeff.sum(axis=1, keepdims=True)
    grad = row_sum * pos - (coeff @ pos)
    return energy, grad

def _fcc_truncated_octahedron(scale=1.0):
    """Generate FCC truncated octahedron cluster of 38 atoms with optional scale factor."""
    a_base = np.sqrt(2.0) * 2.0 ** (1.0/6.0)
    a = a_base * scale
    
    points = []
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                for b in basis:
                    pt = a * (np.array([i, j, k]) + b)
                    points.append(pt)
    
    points = np.array(points)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    selected = points[idx[:N]]
    
    return selected

def _make_truncated_octahedron_start():
    """
    Create a starting geometry using proper FCC primitive vectors.
    This ensures the lattice is correctly oriented and scaled.
    """
    a1 = np.array([0.0, 1.0, 1.0])
    a2 = np.array([1.0, 0.0, 1.0])
    a3 = np.array([1.0, 1.0, 0.0])
    
    pts = []
    max_c = 4
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                p = i * a1 + j * a2 + k * a3
                d2 = np.dot(p, p)
                if d2 <= 20.0:
                    pts.append([p[0], p[1], p[2], d2])
    
    pts = np.array(pts)
    pts_sorted = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]
    unique_pts = [pts_sorted[0]]
    for p in pts_sorted[1:]:
        if np.linalg.norm(p[:3] - unique_pts[-1][:3]) > 1e-10:
            unique_pts.append(p)
    pts = np.array(unique_pts)
    
    pts = pts[np.argsort(pts[:, 3])]
    selected = pts[:38, :3].copy()
    selected -= np.mean(selected, axis=0)
    
    # Scale so that nearest neighbor distance is ~2^(1/6)
    diff = selected[:, np.newaxis, :] - selected[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        selected *= scale
    
    return selected

def _relax(pos, maxiter=2000, ftol=1e-10, gtol=1e-8):
    """Relax positions using L-BFGS with energy and gradient."""
    def obj(x):
        e, g = _lj_energy_grad(x.reshape(N, 3))
        return e, g
    
    res = minimize(obj, pos.flatten(), method='L-BFGS-B', 
                   jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
    return res.x.reshape(N, 3)

def _optimize_scale_simple(pos, maxiter=50):
    """Optimize the scale factor using a simple 1D search on the energy of the relaxed structure."""
    # We assume pos is already a good local minimum.
    # We want to find the optimal scale s such that E(pos * s) is minimized.
    # However, scaling a non-uniform structure doesn't just scale distances uniformly if it's not centered/scaled properly.
    # But for a cluster, scaling from the center of mass is a good approximation for the "lattice constant" optimization.
    
    # Center the structure
    pos_centered = pos - np.mean(pos, axis=0)
    
    def energy_at_scale(s):
        scaled_pos = pos_centered * s
        # Quick relaxation to account for anisotropic distortions induced by scaling?
        # No, that's expensive. Just evaluate energy.
        # Actually, the optimal scale for a fixed topology is found by minimizing E(s).
        # But E(s) is not simply a function of s if the structure relaxes.
        # The inspiration suggests: relax first, then optimize scale on the relaxed geometry?
        # Or optimize scale on the template?
        
        # Let's try: Relax the scaled template.
        # This is expensive. 
        # Alternative: Use the relaxed position, scale it, and evaluate energy.
        # This assumes the topology doesn't change.
        return _lj_energy(scaled_pos)
    
    res = minimize_scalar(energy_at_scale, bounds=(0.9, 1.1), method='bounded', options={'xatol': 1e-6})
    if res.success:
        return pos_centered * res.x + np.mean(pos, axis=0), res.fun
    else:
        return pos, _lj_energy(pos)

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    # Strategy:
    # 1. Generate two high-quality FCC seeds:
    #    a) The simple cubic-based FCC (current program's method)
    #    b) The primitive-vector-based FCC (inspiration's method)
    # 2. Relax both.
    # 3. Perform scale optimization on the best one.
    # 4. Perform basin hopping / perturbation search from the best one.
    # 5. Final polish.
    
    best_pos = None
    best_energy = np.inf
    
    # Seed 1: Current program's FCC
    pos1 = _fcc_truncated_octahedron()
    try:
        relaxed1 = _relax(pos1, maxiter=2000, ftol=1e-10, gtol=1e-8)
        e1 = _lj_energy(relaxed1)
        if e1 < best_energy:
            best_energy = e1
            best_pos = relaxed1
    except Exception:
        pass
    
    # Seed 2: Inspiration's FCC
    pos2 = _make_truncated_octahedron_start()
    try:
        relaxed2 = _relax(pos2, maxiter=2000, ftol=1e-10, gtol=1e-8)
        e2 = _lj_energy(relaxed2)
        if e2 < best_energy:
            best_energy = e2
            best_pos = relaxed2
    except Exception:
        pass
    
    # If we didn't get a valid result, fallback to raw seed
    if best_pos is None:
        best_pos = pos1
        best_energy = _lj_energy(pos1)
    
    # Scale optimization on the best result
    if time.time() < deadline - 60.0:
        try:
            # The inspiration's _optimize_scale is simpler and might be more robust.
            # Let's try optimizing the scale of the current best_pos.
            # But we need to be careful: scaling a relaxed structure might move it out of the basin.
            # Better approach: Re-relax from the scaled structure.
            
            # Let's try a few scales and re-relax.
            for s in [0.98, 0.99, 1.01, 1.02]:
                if time.time() > deadline - 30.0:
                    break
                scaled_pos = best_pos * s
                try:
                    relaxed_scaled = _relax(scaled_pos, maxiter=1000, ftol=1e-10, gtol=1e-8)
                    e_scaled = _lj_energy(relaxed_scaled)
                    if e_scaled < best_energy:
                        best_energy = e_scaled
                        best_pos = relaxed_scaled
                except Exception:
                    pass
        except Exception:
            pass
            
    # Basin hopping / Perturbation search
    if time.time() < deadline - 30.0:
        rng = np.random.RandomState(42)
        # Try different noise levels
        noise_levels = [0.02, 0.05, 0.1]
        for noise in noise_levels:
            if time.time() > deadline - 15.0:
                break
            for trial in range(3):
                if time.time() > deadline - 15.0:
                    break
                perturbed = best_pos + rng.normal(0, noise, size=(N, 3))
                try:
                    relaxed = _relax(perturbed, maxiter=1000, ftol=1e-10, gtol=1e-8)
                    e = _lj_energy(relaxed)
                    if e < best_energy:
                        best_pos = relaxed
                        best_energy = e
                except Exception:
                    pass
    
    # Final high-precision polish
    if time.time() < deadline - 10.0:
        try:
            polished = _relax(best_pos, maxiter=5000, ftol=1e-12, gtol=1e-10)
            e = _lj_energy(polished)
            if e < best_energy:
                best_pos = polished
                best_energy = e
        except Exception:
            pass
    
    # Ensure minimum separation of 0.5
    for _ in range(10):
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, np.inf)
        r = np.sqrt(r2)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        if np.min(r[mask]) >= 0.5:
            break
        close_pairs = np.argwhere(mask & (r < 0.5))
        for i, j in close_pairs:
            direction = best_pos[i] - best_pos[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            else:
                direction = np.array([1.0, 0.0, 0.0])
            best_pos[i] += 0.01 * direction
            best_pos[j] -= 0.01 * direction
    
    return best_pos
# EVOLVE-BLOCK-END
