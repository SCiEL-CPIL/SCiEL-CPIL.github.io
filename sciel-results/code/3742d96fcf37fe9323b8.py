import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 38
DEADLINE = 145.0  # 150s limit, keep 5s margin

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

def _generate_fcc_points(max_c=5):
    """Generate FCC lattice points using primitive vectors."""
    a1 = np.array([0.0, 1.0, 1.0])
    a2 = np.array([1.0, 0.0, 1.0])
    a3 = np.array([1.0, 1.0, 0.0])
    
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                p = i * a1 + j * a2 + k * a3
                d2 = np.dot(p, p)
                if d2 <= 30.0:
                    pts.append([p[0], p[1], p[2], d2])
    
    pts = np.array(pts)
    pts_sorted = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]
    unique_pts = [pts_sorted[0]]
    for p in pts_sorted[1:]:
        if np.linalg.norm(p[:3] - unique_pts[-1][:3]) > 1e-10:
            unique_pts.append(p)
    return np.array(unique_pts)

# Pre-compute the base FCC structure for LJ38
# We take the 38 closest points to the origin in the FCC lattice.
_FCC_PTS = _generate_fcc_points(max_c=5)
_FCC_PTS = _FCC_PTS[np.argsort(_FCC_PTS[:, 3])]
_BASE_COORDS = _FCC_PTS[:N, :3].copy()
_BASE_COORDS -= np.mean(_BASE_COORDS, axis=0)

# Identify shells based on distance from center
_radii = np.linalg.norm(_BASE_COORDS, axis=1)
# Sort indices by radius
_sorted_idx = np.argsort(_radii)
_sorted_radii = _radii[_sorted_idx]

# Determine unique shells
# Tolerance for grouping
shell_ids = np.zeros(N, dtype=int)
current_shell = 0
prev_r = -1
for i in _sorted_idx:
    r = _radii[i]
    if prev_r < 0 or abs(r - prev_r) > 0.1:
        current_shell += 1
        prev_r = r
    shell_ids[i] = current_shell

N_SHELLS = current_shell
# Store shell indices for fast access
_SHELL_INDICES = [np.where(shell_ids == s)[0] for s in range(N_SHELLS)]

def _sa_to_coords(params):
    """Convert symmetry-adapted parameters to 38x3 coordinates.
    params[0]: global scale
    params[1:]: shell breathing factors (relative to 1.0)
    """
    scale = params[0]
    breaths = params[1:]
    coords = np.zeros((N, 3))
    for s in range(N_SHELLS):
        b = 1.0 + breaths[s]
        idx = _SHELL_INDICES[s]
        coords[idx] = _BASE_COORDS[idx] * b * scale
    return coords

def _sa_energy_and_grad(params):
    """Energy and gradient for SA parameters."""
    coords = _sa_to_coords(params)
    e, grad_coords = _lj_energy_and_grad(coords)
    
    # Backpropagate gradient to params
    # dE/dscale = sum_i grad_i . dcoords_i/dscale
    # dE/dbreath_s = sum_{i in shell s} grad_i . dcoords_i/dbreath_s
    
    grad_params = np.zeros_like(params)
    
    # Global scale: coords_i = base_i * breath_i * scale
    # dcoords_i/dscale = base_i * breath_i
    breaths = 1.0 + params[1:]
    
    # Compute sum for global scale
    # grad_params[0] = sum_i grad_i . (base_i * breath_i)
    # We can do this efficiently
    # Let's just loop over shells for clarity and speed (N is small)
    scale_grad = 0.0
    for s in range(N_SHELLS):
        idx = _SHELL_INDICES[s]
        b = breaths[s]
        # grad_i . (base_i * b)
        # grad_i is row i of grad_coords
        # base_i is row i of _BASE_COORDS
        dot_prods = np.sum(grad_coords[idx] * (_BASE_COORDS[idx] * b), axis=1)
        scale_grad += np.sum(dot_prods)
    
    grad_params[0] = scale_grad
    
    for s in range(N_SHELLS):
        idx = _SHELL_INDICES[s]
        # dcoords_i/dbreath_s = base_i * scale
        dot_prods = np.sum(grad_coords[idx] * (_BASE_COORDS[idx] * params[0]), axis=1)
        grad_params[s+1] = np.sum(dot_prods)
        
    return e, grad_params

def _relax_sa(params, max_iter=500, gtol=1e-10):
    """Relax in SA space."""
    res = minimize(
        _sa_energy_and_grad,
        params,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-12, 'maxls': 50}
    )
    return res.x, res.fun

def _relax_full(coords, max_iter=1000, gtol=1e-12, ftol=1e-14):
    """Relax in full 3D space."""
    res = minimize(
        _lj_energy_and_grad,
        coords.ravel(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': gtol, 'ftol': ftol, 'maxls': 100}
    )
    return res.x.reshape(-1, 3), res.fun

def _optimize_scale(pos):
    """1D scale optimization."""
    def energy_at_scale(s):
        scaled_pos = pos * s
        e, _ = _lj_energy_and_grad(scaled_pos)
        return e

    res = minimize_scalar(energy_at_scale, bounds=(0.9, 1.1), method='bounded', options={'xatol': 1e-8})
    if res.success:
        optimal_pos = pos * res.x
        return optimal_pos, res.fun
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
    rng = np.random.default_rng(42)
    
    best_pos = None
    best_energy = np.inf
    
    # Phase 1: Generate diverse SA starting points
    # The optimal scale is around 1.0 for the base coords scaled by 2^(1/6)/sqrt(2)?
    # Let's just start with a range of scales and breaths.
    # Base coords are raw FCC. We need to scale them to LJ equilibrium.
    # NN distance in raw FCC is sqrt(2). LJ eq is 2^(1/6).
    # Scale factor = 2^(1/6) / sqrt(2) ~ 0.91.
    
    sa_starts = []
    
    # 1. Deterministic: Ideal FCC scale
    ideal_scale = (2.0 ** (1.0/6.0)) / np.sqrt(2.0)
    sa_starts.append(np.array([ideal_scale] + [0.0] * N_SHELLS))
    
    # 2. Perturbed scales
    for i in range(10):
        scale = ideal_scale * (0.95 + 0.1 * rng.random())
        breaths = rng.normal(0, 0.05, size=N_SHELLS)
        sa_starts.append(np.array([scale] + list(breaths)))
        
    # 3. Specific shell variations (outer shell expansion/contraction)
    for i in range(10):
        scale = ideal_scale
        breaths = np.zeros(N_SHELLS)
        # Perturb the last shell (outer)
        breaths[-1] = rng.normal(0, 0.1)
        # Perturb the second to last
        if N_SHELLS > 1:
            breaths[-2] = rng.normal(0, 0.05)
        sa_starts.append(np.array([scale] + list(breaths)))

    # Phase 2: Relax in SA space (Fast)
    relaxed_sa_results = []
    for start in sa_starts:
        if time.time() > t0 + DEADLINE - 20:
            break
        try:
            p_opt, e_opt = _relax_sa(start, max_iter=200, gtol=1e-8)
            relaxed_sa_results.append((e_opt, p_opt))
        except Exception:
            continue
            
    # Sort by energy
    relaxed_sa_results.sort(key=lambda x: x[0])
    
    # Phase 3: Convert top SA results to 3D and Relax in 3D (Slow but accurate)
    # Take top 10 SA results
    candidates_3d = []
    for e, p in relaxed_sa_results[:10]:
        if time.time() > t0 + DEADLINE - 15:
            break
        coords = _sa_to_coords(p)
        candidates_3d.append((e, coords))
        
    # Also add some random 3D starts to break symmetry if needed
    # But SA is very good for FCC. Let's stick to SA-derived candidates first.
    
    # Relax each candidate in 3D
    for e_init, coords in candidates_3d:
        if time.time() > t0 + DEADLINE - 10:
            break
        try:
            # Initial energy check
            e, _ = _lj_energy_and_grad(coords)
            p_relaxed, e_relaxed = _relax_full(coords, max_iter=500, gtol=1e-10)
            
            if e_relaxed < best_energy:
                best_energy = e_relaxed
                best_pos = p_relaxed.copy()
        except Exception:
            continue
            
    if best_pos is None:
        # Fallback: Use the best SA result directly
        if relaxed_sa_results:
            best_p_sa = relaxed_sa_results[0][1]
            best_pos = _sa_to_coords(best_p_sa)
            best_energy, _ = _lj_energy_and_grad(best_pos)
        else:
            best_pos = _sa_to_coords(np.array([ideal_scale] + [0.0] * N_SHELLS))
            best_energy, _ = _lj_energy_and_grad(best_pos)

    # Phase 4: Basin Hopping in SA space from the best 3D result
    # We need to map the best 3D result back to SA parameters?
    # Or just perturb the SA params that generated the best 3D result?
    # The best 3D result came from an SA relaxation. Let's find which one.
    # Actually, it's easier to just run BH on the SA params that gave the best 3D energy.
    
    # Identify the SA params that led to the best 3D result
    # We didn't track which SA param set led to the best 3D result directly in the loop above.
    # Let's re-run relaxation for the top 5 SA results to find the best 3D energy, 
    # and keep track of the SA params.
    
    best_sa_params = None
    best_sa_energy_3d = np.inf
    
    for e_sa, p_sa in relaxed_sa_results[:5]:
        if time.time() > t0 + DEADLINE - 5:
            break
        coords = _sa_to_coords(p_sa)
        try:
            p_relaxed, e_relaxed = _relax_full(coords, max_iter=300, gtol=1e-10)
            if e_relaxed < best_sa_energy_3d:
                best_sa_energy_3d = e_relaxed
                best_sa_params = p_sa
        except Exception:
            continue
            
    if best_sa_params is not None:
        # Run Basin Hopping in SA space starting from best_sa_params
        current_sa = best_sa_params.copy()
        current_e_sa, _ = _sa_energy_and_grad(current_sa)
        
        T0 = 0.1
        T1 = 0.01
        max_bh_iters = 20
        
        for i in range(max_bh_iters):
            if time.time() > t0 + DEADLINE - 2:
                break
            T_curr = T0 * (T1 / T0) ** (i / max_bh_iters)
            perturbed_sa = current_sa + rng.normal(0, T_curr, size=current_sa.shape)
            
            # Relax perturbed SA
            try:
                p_relaxed_sa, e_relaxed_sa = _relax_sa(perturbed_sa, max_iter=100, gtol=1e-8)
                
                # Convert to 3D and check energy (approximate, but good for comparison)
                # Actually, to be fair, we should relax in 3D? No, that's too slow for BH.
                # Use SA energy as a proxy for acceptance, then check 3D for the best.
                
                if e_relaxed_sa < current_e_sa or rng.random() < np.exp(-(e_relaxed_sa - current_e_sa) / max(T_curr, 1e-10)):
                    current_sa = p_relaxed_sa
                    current_e_sa = e_relaxed_sa
                    
                    # Check if this SA state yields a better 3D minimum
                    coords_test = _sa_to_coords(current_sa)
                    # Quick 3D relax (low iter) to see potential
                    if time.time() < t0 + DEADLINE - 3:
                        try:
                            p_test, e_test = _relax_full(coords_test, max_iter=100, gtol=1e-8)
                            if e_test < best_energy:
                                best_energy = e_test
                                best_pos = p_test
                        except Exception:
                            pass
            except Exception:
                continue

    # Phase 5: Final Polish
    if time.time() < t0 + DEADLINE - 1:
        try:
            p_polished, e_polished = _relax_full(best_pos, max_iter=2000, gtol=1e-14, ftol=1e-16)
            if e_polished < best_energy:
                best_energy = e_polished
                best_pos = p_polished
        except Exception:
            pass

    # Phase 6: Final Scale Opt
    if time.time() < t0 + DEADLINE - 0.5:
        try:
            p_scaled, e_scaled = _optimize_scale(best_pos)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos

if __name__ == "__main__":
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
