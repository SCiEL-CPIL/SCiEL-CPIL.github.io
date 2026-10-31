"""
LJ38 cluster global minimum search.

Approach:
1. Construct the 19-atom FCC core (global min for LJ19).
2. Generate 24 candidate outer atoms from the next FCC shell.
3. Use a combination of deterministic symmetric subsets and random subsets to initialize the 19 outer atoms.
4. Relax each configuration using L-BFGS.
5. Perform a final 1D scale optimization on the best result to ensure the geometry is at its energy minimum for that topology.
6. A final high-precision polish ensures the result is at the local minimum.
"""
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 38

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _generate_fcc_core_and_outer():
    """
    Generate the 19 core FCC atoms and 24 outer candidate FCC atoms.
    Core: 1 (0,0,0) + 12 (110 type) + 6 (200 type).
    Outer: 24 (211 type).
    Using proper FCC primitive vectors: a1=(0,1,1), a2=(1,0,1), a3=(1,1,0)
    """
    # FCC primitive vectors
    a1 = np.array([0.0, 1.0, 1.0])
    a2 = np.array([1.0, 0.0, 1.0])
    a3 = np.array([1.0, 1.0, 0.0])
    
    pts = []
    max_c = 3
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                p = i * a1 + j * a2 + k * a3
                d2 = np.dot(p, p)
                if d2 <= 6.5:
                    pts.append([p[0], p[1], p[2], d2])
    
    pts = np.array(pts)
    # Remove duplicates (FCC lattice can have coincident points from different representations)
    # Sort and deduplicate
    pts_sorted = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]
    unique_pts = [pts_sorted[0]]
    for p in pts_sorted[1:]:
        if np.linalg.norm(p[:3] - unique_pts[-1][:3]) > 1e-10:
            unique_pts.append(p)
    pts = np.array(unique_pts)
    
    core_pts = pts[pts[:, 3] <= 4.1]
    outer_pts = pts[(pts[:, 3] > 4.1) & (pts[:, 3] <= 6.1)]
    
    return core_pts[:, :3], outer_pts[:, :3]

def _make_truncated_octahedron_start():
    """
    Create a starting geometry that closely resembles the FCC truncated octahedron.
    Uses proper FCC lattice generation.
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
    
    # Sort by distance from origin
    pts = pts[np.argsort(pts[:, 3])]
    
    # Take the 38 closest points
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

def _relax(pos, max_iter=500, gtol=1e-10):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

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

def _optimize_scale(pos):
    """
    Optimize the global scale of the cluster to minimize energy.
    This is a 1D optimization problem.
    """
    def energy_at_scale(s):
        scaled_pos = pos * s
        e, _ = _lj_energy_and_grad(scaled_pos)
        return e

    # Initial scale is 1.0. We expect the optimal scale to be close to 1.0.
    # The LJ minimum is at r = 2^(1/6) ~ 1.122.
    # If the initial structure is too compressed, scale > 1.
    # If too expanded, scale < 1.
    res = minimize_scalar(energy_at_scale, bounds=(0.8, 1.5), method='bounded', options={'xatol': 1e-8})
    if res.success:
        optimal_scale = res.x
        optimal_pos = pos * optimal_scale
        optimal_energy = res.fun
        return optimal_pos, optimal_energy
    else:
        return pos, energy_at_scale(1.0)

def _ensure_min_sep(pos, min_sep=0.5):
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
    deadline = t0 + 140.0  # Safety margin

    rng = np.random.default_rng(42)
    
    core, outer = _generate_fcc_core_and_outer()
    
    # Ensure we have exactly 19 core and 24 outer
    # Core should be 1 + 12 + 6 = 19
    # Outer should be 24
    if len(core) < 19:
        # Fallback to truncated octahedron start
        pos = _make_truncated_octahedron_start()
        pos, e = _relax(pos, max_iter=200, gtol=1e-10)
        pos, e = _optimize_scale(pos)
        pos, e = _relax(pos, max_iter=500, gtol=1e-12)
        return _ensure_min_sep(pos)
        
    if len(outer) < 24:
        # Fallback
        pos = _make_truncated_octahedron_start()
        pos, e = _relax(pos, max_iter=200, gtol=1e-10)
        pos, e = _optimize_scale(pos)
        pos, e = _relax(pos, max_iter=500, gtol=1e-12)
        return _ensure_min_sep(pos)
        
    core = core[:19]
    outer = outer[:24]
    
    best_pos = None
    best_energy = np.inf
    
    # Strategy 1: Use the 19 outer atoms that are closest to the core (symmetric subset)
    # We need to select 19 out of 24 outer atoms.
    # The 24 outer atoms form a cuboctahedron shell.
    # We can try different subsets.
    
    # Try the full truncated octahedron start first as a baseline
    pos_to = _make_truncated_octahedron_start()
    pos_relaxed, e_relaxed = _relax(pos_to, max_iter=200, gtol=1e-10)
    pos_scaled, e_scaled = _optimize_scale(pos_relaxed)
    pos_polished, e_polished = _relax(pos_scaled, max_iter=500, gtol=1e-12)
    
    if e_polished < best_energy:
        best_energy = e_polished
        best_pos = pos_polished.copy()
    
    # Try combinations of core + subsets of outer
    # Since we have 24 outer atoms, and need 19, we are omitting 5.
    # We can try omitting atoms from different symmetry classes or random subsets.
    
    # For efficiency, let's just try a few random subsets of the outer shell
    # combined with the core, then relax.
    
    n_tries = 10
    for i in range(n_tries):
        if time.time() > deadline - 10.0:
            break
            
        # Select 19 random outer atoms
        indices = rng.choice(len(outer), size=19, replace=False)
        selected_outer = outer[indices]
        
        pos = np.vstack([core, selected_outer])
        pos -= np.mean(pos, axis=0)
        
        # Scale to reasonable size
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, np.inf)
        nn = np.sqrt(np.min(r2, axis=1))
        avg_nn = np.mean(nn)
        if avg_nn > 1e-12:
            scale = (2.0 ** (1.0 / 6.0)) / avg_nn
            pos *= scale
            
        try:
            pos_relaxed, e_relaxed = _relax(pos, max_iter=200, gtol=1e-10)
            if e_relaxed < best_energy:
                best_energy = e_relaxed
                best_pos = pos_relaxed.copy()
        except Exception:
            continue
            
    # Final polish on the best found
    if best_pos is not None:
        if time.time() < deadline - 2.0:
            pos_scaled, e_scaled = _optimize_scale(best_pos)
            pos_polished, e_polished = _relax(pos_scaled, max_iter=1000, gtol=1e-12)
            if e_polished < best_energy:
                best_energy = e_polished
                best_pos = pos_polished
        
        best_pos = _ensure_min_sep(best_pos)
        
    return best_pos

if __name__ == "__main__":
    # Test
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
