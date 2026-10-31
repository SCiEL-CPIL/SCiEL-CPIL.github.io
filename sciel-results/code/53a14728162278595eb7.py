import time
import numpy as np
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0  # Safety margin within 150s limit

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # Avoid division by zero
    
    r6 = r2 ** 3
    r12 = r6 ** 2

    # Energy
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient
    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _make_fcc_cluster():
    """
    Generate the 38-atom FCC truncated octahedron geometry.
    """
    # Generate FCC points
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))
    
    pts = np.array(pts, dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    
    pts = pts[:N].copy()
    pts -= np.mean(pts, axis=0)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    min_r = np.sqrt(min_r2)
    
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-10:
        scale = target_r / min_r
        pts *= scale
        
    return pts

def _make_icosahedral_cluster():
    """
    Generate a 38-atom icosahedral cluster.
    1 center + 12 (1st shell) + 20 (2nd shell) + 5 (3rd shell) = 38?
    Actually, 1 + 12 + 20 = 33. We need 5 more.
    The 38-atom icosahedron usually has a 5-atom cap or similar.
    A common 38-atom icosahedron is 1 center, 12 neighbors, 20 next, and 5 more.
    Let's use a simpler heuristic: Start with a perfect 13-atom icosahedron and add atoms.
    Or just use a random perturbation of the FCC as a secondary start if icosahedral is hard to code correctly.
    For now, I'll implement a basic icosahedral generator.
    """
    # 1 center
    pts = [[0,0,0]]
    # 12 vertices of icosahedron
    phi = (1 + np.sqrt(5)) / 2
    ico_coords = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    pts.extend(ico_coords)
    
    # 20 vertices of second shell (face centers of icosahedron)
    # These are harder to generate procedurally without geometry libraries.
    # Approximation: Place atoms at midpoints of edges of the icosahedron? No, that's not a shell.
    # Let's skip the perfect icosahedral 38 and use a "random" start as the second candidate.
    rng = np.random.default_rng(123)
    pts = np.array(pts, dtype=np.float64)
    # Add 25 random points in a sphere of radius 2.5
    for _ in range(25):
        while len(pts) < 38:
            p = rng.normal(0, 1.5, size=3)
            if np.linalg.norm(p) < 2.5:
                # Check min distance
                if len(pts) == 0 or np.min(np.linalg.norm(pts - p, axis=1)) > 0.8:
                    pts = np.vstack([pts, p])
                    break
    
    pts = pts[:38]
    pts -= np.mean(pts, axis=0)
    
    # Scale to optimal nearest neighbor distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-10:
        pts *= (target_r / min_r)
        
    return pts

def _relax(pos, max_iter=1000, gtol=1e-10, ftol=1e-14):
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
            options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol},
        )
        p_out = res.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        return p_out, e_out
    except Exception:
        e, _ = _lj_energy_and_grad(pos)
        return pos, e

def _optimize_scale(pos, n_iter=30):
    """
    1D global search over uniform scale factor s applied to centered positions.
    """
    center = np.mean(pos, axis=0)
    centered = pos - center
    best_e = np.inf
    best_s = 1.0
    
    # Coarse scan
    for s in np.linspace(0.95, 1.05, 201):
        p = centered * s + center
        e, _ = _lj_energy_and_grad(p)
        if e < best_e:
            best_e = e
            best_s = s
            
    # Fine scan around best
    for _ in range(n_iter):
        span = 0.05 / (10 ** (_ // 3))
        for s in np.linspace(best_s - span, best_s + span, 201):
            if s <= 0.5:
                continue
            p = centered * s + center
            e, _ = _lj_energy_and_grad(p)
            if e < best_e:
                best_e = e
                best_s = s
                
    return centered * best_s + center, best_e

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairs are at least min_sep apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos

def construct():
    t0 = time.time()
    
    # 1. Generate initial geometries
    starts = []
    try:
        starts.append(_make_fcc_cluster())
    except Exception:
        pass
    
    try:
        starts.append(_make_icosahedral_cluster())
    except Exception:
        pass
        
    # Add multiple random starts with different seeds and scales
    rng = np.random.default_rng(42)
    for seed_idx in range(5):
        rand_pos = rng.normal(0, 1.2 + 0.3 * seed_idx, size=(N, 3))
        rand_pos -= np.mean(rand_pos, axis=0)
        starts.append(rand_pos)
    
    # Add a "stretched" FCC variant
    try:
        fcc_stretched = _make_fcc_cluster()
        fcc_stretched *= 1.15  # Slightly larger
        starts.append(fcc_stretched)
        
        fcc_compressed = _make_fcc_cluster()
        fcc_compressed *= 0.90  # Slightly smaller
        starts.append(fcc_compressed)
    except Exception:
        pass
    
    best_pos = None
    best_energy = np.inf

    # 2. Initial relaxation of all starts
    for pos0 in starts:
        if time.time() - t0 > DEADLINE - 30.0:
            break
        try:
            p, e = _relax(pos0, max_iter=300, gtol=1e-7)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # 3. 1D scale optimization on the best
    if best_pos is not None and time.time() - t0 < DEADLINE - 25.0:
        try:
            p_scaled, e_scaled = _optimize_scale(best_pos, n_iter=15)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled
        except Exception:
            pass

    # 4. Enhanced Basin Hopping with DE-like large jumps
    if best_pos is not None:
        current_pos = best_pos.copy()
        current_energy = best_energy
        
        # Adaptive step size and temperature
        step_size = 0.6
        T = 1.5
        
        # Keep track of recent energies for adaptive control
        recent_energies = []
        no_improve_count = 0
        
        # Run for a fixed number of steps or until time
        max_bh_steps = 80
        for i in range(max_bh_steps):
            if time.time() - t0 > DEADLINE - 15.0:
                break
            
            # Every 5 steps, try a DE-like large jump from best
            if i % 5 == 2 and best_pos is not None:
                # Large random displacement to jump over barriers
                jump_scale = 1.5 + 0.5 * (i / max_bh_steps)
                noise = rng.normal(0, jump_scale, size=best_pos.shape)
                perturbed = best_pos + noise
            else:
                # Standard basin hopping perturbation
                noise = rng.normal(0, step_size, size=current_pos.shape)
                perturbed = current_pos + noise
            
            # Relax
            new_pos, new_energy = _relax(perturbed, max_iter=200, gtol=1e-5)
            
            # Check separation constraint
            new_pos = _ensure_min_sep(new_pos)
            new_energy, _ = _lj_energy_and_grad(new_pos)
            
            # Acceptance Criterion (Metropolis)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or (rng.random() < np.exp(-delta_e / T)):
                current_pos = new_pos
                current_energy = new_energy
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
                    no_improve_count = 0
                    # Reduce step size when improving
                    step_size *= 0.92
                    T *= 0.95
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            # Track recent energies for adaptive control
            recent_energies.append(new_energy)
            if len(recent_energies) > 10:
                recent_energies.pop(0)
            
            # Adaptive: if stuck, increase step size and temperature
            if no_improve_count >= 8:
                step_size = min(step_size * 1.15, 2.0)
                T = min(T * 1.1, 5.0)
                no_improve_count = 0

    # 5. Final High-Precision Polish
    if best_pos is not None and time.time() - t0 < DEADLINE - 5.0:
        try:
            # Scale optimization
            p_scaled, e_scaled = _optimize_scale(best_pos, n_iter=25)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled
            
            # Tight relaxation
            p, e = _relax(best_pos, max_iter=3000, gtol=1e-13, ftol=1e-16)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    # 6. Final separation check
    if best_pos is not None:
        best_pos = _ensure_min_sep(best_pos)
    else:
        # Fallback
        best_pos = _make_fcc_cluster()

    return best_pos
