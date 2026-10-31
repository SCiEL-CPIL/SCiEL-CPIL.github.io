"""
LJ38 cluster global minimum search.

Approach:
1. Construct the 19-atom FCC core (global min for LJ19).
2. Generate 24 candidate outer atoms from the next FCC shell.
3. Use a combination of deterministic symmetric subsets and random subsets to initialize the 19 outer atoms.
4. Relax each configuration using L-BFGS.
5. Perform a final 1D scale optimization on the best result to ensure the geometry is at its energy minimum for that topology.
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
    """
    pts = []
    max_c = 3
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    d2 = i*i + j*j + k*k
                    if d2 <= 6:
                        pts.append([float(i), float(j), float(k), d2])
    
    pts = np.array(pts)
    core_pts = pts[pts[:, 3] <= 4]
    outer_pts = pts[pts[:, 3] == 6]
    
    return core_pts[:, :3], outer_pts[:, :3]

def _make_icosahedral_structure():
    """
    Icosahedral starting geometry.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=np.float64)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico /= norms

    layers = [1.0, 1.7, 2.4]
    pts = [np.zeros(3)]
    for r in layers:
        pts.extend(ico * r)
    pts.append(np.array([0.25, 0.25, 0.25]))
    pts = np.array(pts[:N], dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pts *= scale
    return pts

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

def _make_truncated_octahedron_start():
    """
    Create a starting geometry that closely resembles the FCC truncated octahedron.
    The 38-atom LJ global minimum has a specific symmetry: it's an FCC cluster
    with layers. We build it layer by layer.
    
    FCC truncated octahedron for N=38:
    - Layer 0 (z=0): 1 atom at center
    - Layer 1 (z=1): 6 atoms (hexagonal ring)
    - Layer 2 (z=2): 12 atoms
    - Layer 3 (z=3): 12 atoms  
    - Layer 4 (z=4): 6 atoms
    - Layer 5 (z=5): 1 atom
    
    But we need exactly 38. The standard truncated octahedron has 38 atoms.
    Let's use the FCC lattice points within a certain radius and select 38.
    """
    # Generate FCC lattice points
    pts = []
    max_c = 4
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    d2 = i*i + j*j + k*k
                    pts.append([float(i), float(j), float(k), d2])
    
    pts = np.array(pts)
    # Sort by distance from origin
    pts = pts[np.argsort(pts[:, 3])]
    
    # Take the 38 closest points
    selected = pts[:38, :3]
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

def _basin_hop(pos, rng, n_iter=50, max_amp=0.5):
    """
    Basin hopping: perturb, relax, accept/reject.
    Returns best position and energy found.
    """
    best_pos = pos.copy()
    best_energy, _ = _lj_energy_and_grad(pos)
    current_pos = pos.copy()
    current_energy = best_energy
    
    for i in range(n_iter):
        # Perturb with random displacement
        amp = max_amp * (1.0 - i / max(n_iter, 1))
        noise = rng.normal(0.0, amp, size=pos.shape)
        perturbed = current_pos + noise
        
        # Relax the perturbed structure
        try:
            relaxed, e_relaxed = _relax(perturbed, max_iter=200, gtol=1e-10)
            relaxed, e_relaxed = _optimize_scale(relaxed)
        except Exception:
            continue
        
        # Accept if lower energy, or with probability exp(-dE/T)
        if e_relaxed < current_energy:
            current_pos = relaxed
            current_energy = e_relaxed
            if e_relaxed < best_energy:
                best_energy = e_relaxed
                best_pos = relaxed.copy()
        else:
            # Metropolis acceptance with simple temperature
            T = 0.1 * (1.0 + i / max(n_iter, 1))
            dE = e_relaxed - current_energy
            if rng.random() < np.exp(-dE / T):
                current_pos = relaxed
                current_energy = e_relaxed
    
    return best_pos, best_energy

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s kill

    rng = np.random.default_rng(42)
    
    core, outer = _generate_fcc_core_and_outer()
    
    # 1. Best guess: truncated octahedron from FCC lattice
    to_start = _make_truncated_octahedron_start()
    to_start = _ensure_min_sep(to_start)
    
    # First, quickly relax the TO start to get a good baseline
    best_pos, best_energy = _relax(to_start, max_iter=300, gtol=1e-11)
    best_pos, best_energy = _optimize_scale(best_pos)
    
    # Check if we're already at the global minimum (energy ~ -173.928)
    # If so, we can spend remaining time on polish
    if best_energy < -173.9:
        # We found the global minimum! Polish it.
        if time.time() < deadline - 5.0:
            try:
                p, e = _relax(best_pos, max_iter=1000, gtol=1e-13)
                p, e = _optimize_scale(p)
                if e < best_energy:
                    best_energy = e
                    best_pos = p.copy()
            except Exception:
                pass
        best_pos = _ensure_min_sep(best_pos)
        return best_pos
    
    # If not at global min, use basin hopping to search
    # Start basin hopping from the TO start (best initial guess)
    remaining_time = deadline - time.time() - 10.0  # Leave 10s buffer
    max_bh_iters = max(1, int(remaining_time / 2.0))  # ~2s per BH iteration
    
    bh_pos, bh_energy = _basin_hop(to_start, rng, n_iter=max_bh_iters, max_amp=0.4)
    if bh_energy < best_energy:
        best_energy = bh_energy
        best_pos = bh_pos.copy()
    
    # Also try basin hopping from icosahedral start as fallback
    if time.time() < deadline - 15.0:
        try:
            ico = _make_icosahedral_structure()
            ico = _ensure_min_sep(ico)
            remaining2 = deadline - time.time() - 10.0
            max_bh2 = max(1, int(remaining2 / 3.0))
            bh_pos2, bh_energy2 = _basin_hop(ico, rng, n_iter=max_bh2, max_amp=0.5)
            if bh_energy2 < best_energy:
                best_energy = bh_energy2
                best_pos = bh_pos2.copy()
        except Exception:
            pass
    
    # Final polish on the best result
    if time.time() < deadline - 3.0:
        try:
            p, e = _relax(best_pos, max_iter=500, gtol=1e-12)
            p, e = _optimize_scale(p)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass
    
    best_pos = _ensure_min_sep(best_pos)
    return best_pos

if __name__ == "__main__":
    # Test
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
