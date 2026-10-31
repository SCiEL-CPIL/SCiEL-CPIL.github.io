import time
import numpy as np
from scipy.optimize import minimize

N = 38
DEADLINE = 115.0  # Safety margin within 120s/150s limit

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)  # Avoid division by zero
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6

    # Energy
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient
    r8 = r2 * r2 * r2 * r2
    r14 = r2 * r2 * r2 * r2 * r2 * r2 * r2
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _make_fcc_cluster():
    """
    Generate the 38-atom FCC truncated octahedron geometry.
    Uses explicit FCC lattice points and selects the 38 closest to center.
    """
    # FCC lattice: points where (i+j+k) is even, scaled by 1/sqrt(2) for unit nearest-neighbor
    max_c = 4
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))
    
    pts = np.array(pts, dtype=np.float64)
    pts /= np.sqrt(2.0)  # Scale so nearest neighbors are at distance 1
    
    pts -= np.mean(pts, axis=0)
    
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    
    pts = pts[:N].copy()
    pts -= np.mean(pts, axis=0)
    
    # Scale so nearest neighbor distance is 2^(1/6)
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
    Generate an icosahedral-like cluster: 1 center + 12 inner shell + 18 outer shell + 7 extra.
    This provides a diverse starting point to escape the icosahedral funnel.
    """
    rng = np.random.default_rng(123)
    
    # Center atom
    pts = [np.zeros(3)]
    
    # 12 vertices of regular icosahedron
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=np.float64)
    ico_verts /= np.linalg.norm(ico_verts[0])  # Normalize to unit sphere
    ico_verts *= 1.0  # Inner shell at radius 1
    
    pts.extend(ico_verts.tolist())
    
    # 18 atoms on outer shell (radius ~1.6)
    # Use points from a slightly larger icosahedron plus some random perturbations
    outer_verts = ico_verts.copy() * 1.55
    # Add 6 more points to make 18 (midpoints of edges, roughly)
    edge_midpoints = []
    for i in range(12):
        for j in range(i+1, 12):
            mid = (outer_verts[i] + outer_verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            mid *= 1.55
            edge_midpoints.append(mid)
    
    # Take 6 well-separated midpoints
    rng.shuffle(edge_midpoints)
    selected_mids = []
    for mid in edge_midpoints:
        if len(selected_mids) >= 6:
            break
        # Check if sufficiently far from already selected
        if all(np.linalg.norm(mid - s) > 0.5 for s in selected_mids):
            selected_mids.append(mid)
    
    pts.extend(selected_mids)
    
    # Add 7 more atoms to reach 38 (distributed on outer shell)
    # Use random points on sphere at radius 1.7
    for _ in range(7):
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r_xy = np.sqrt(1 - z*z)
        pt = np.array([r_xy * np.cos(theta), r_xy * np.sin(theta), z]) * 1.7
        # Check minimum separation
        valid = True
        for existing in pts:
            if np.linalg.norm(pt - existing) < 0.8:
                valid = False
                break
        if valid:
            pts.append(pt)
        else:
            # Retry with different position
            for _retry in range(10):
                theta = rng.uniform(0, 2 * np.pi)
                z = rng.uniform(-1, 1)
                r_xy = np.sqrt(1 - z*z)
                pt = np.array([r_xy * np.cos(theta), r_xy * np.sin(theta), z]) * 1.7
                valid = True
                for existing in pts:
                    if np.linalg.norm(pt - existing) < 0.8:
                        valid = False
                        break
                if valid:
                    pts.append(pt)
                    break
    
    pts = np.array(pts[:N], dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    # Scale so nearest neighbor is at 2^(1/6)
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
    for s in np.linspace(0.95, 1.05, 101):
        p = centered * s + center
        e, _ = _lj_energy_and_grad(p)
        if e < best_e:
            best_e = e
            best_s = s
            
    # Fine scan around best
    for _ in range(n_iter):
        span = 0.02 / (10 ** (_ // 3))
        for s in np.linspace(max(0.9, best_s - span), best_s + span, 101):
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

def _basin_hopping(start_pos, rng, deadline, step_size_init=0.4, T_init=0.8):
    """
    Basin hopping with adaptive step size and temperature.
    Returns (best_pos, best_energy).
    """
    current_pos = start_pos.copy()
    current_energy, _ = _lj_energy_and_grad(current_pos)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    step_size = step_size_init
    T = T_init
    consecutive_rejects = 0
    consecutive_accepts = 0
    min_step = 0.08
    max_step = 1.2
    
    while time.time() < deadline:
        # Perturb with adaptive step size
        noise = rng.normal(0, step_size, size=current_pos.shape)
        perturbed = current_pos + noise
        
        # Relax
        new_pos, new_energy = _relax(perturbed, max_iter=200, gtol=1e-5, ftol=1e-10)
        
        # Check separation constraint
        new_pos = _ensure_min_sep(new_pos)
        # Re-evaluate energy after ensuring separation
        new_energy, _ = _lj_energy_and_grad(new_pos)
        
        # Acceptance Criterion
        delta_e = new_energy - current_energy
        
        accepted = False
        if delta_e < 0:
            # Better energy, accept
            current_pos = new_pos
            current_energy = new_energy
            accepted = True
            consecutive_accepts += 1
            consecutive_rejects = 0
            
            # Update global best
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
                
        else:
            # Worse energy, accept with probability exp(-delta_e / T)
            if delta_e < 15.0:  # Don't accept very bad moves
                if rng.random() < np.exp(-delta_e / T):
                    current_pos = new_pos
                    current_energy = new_energy
                    accepted = True
                    consecutive_accepts += 1
                    consecutive_rejects = 0
            else:
                consecutive_rejects += 1
                consecutive_accepts = 0
        
        if not accepted:
            consecutive_rejects += 1
            consecutive_accepts = 0
        
        # Adapt step size
        if consecutive_rejects >= 8:
            step_size = max(min_step, step_size * 0.75)
            T = max(0.1, T * 0.9)
            consecutive_rejects = 0
        elif consecutive_accepts >= 4:
            step_size = min(max_step, step_size * 1.2)
            T = min(2.0, T * 1.05)
            consecutive_accepts = 0

    return best_pos, best_energy

def construct():
    t0 = time.time()
    
    # 1. Generate initial FCC geometry
    try:
        pos0 = _make_fcc_cluster()
    except Exception:
        side = int(np.ceil(N ** (1.0 / 3.0)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append([i, j, k])
        pos0 = np.array(pts[:N], dtype=np.float64)
        pos0 -= np.mean(pos0, axis=0)
        pos0 *= 1.12

    best_pos = pos0.copy()
    best_energy = np.inf

    # 2. Initial relaxation of FCC
    try:
        p, e = _relax(pos0, max_iter=500, gtol=1e-8)
        if e < best_energy:
            best_energy = e
            best_pos = p.copy()
    except Exception:
        pass

    # 3. 1D scale optimization on FCC result
    if time.time() - t0 < DEADLINE - 20.0:
        try:
            p_scaled, e_scaled = _optimize_scale(best_pos)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled.copy()
        except Exception:
            pass

    # 4. Try icosahedral starting point
    if time.time() - t0 < DEADLINE - 30.0:
        try:
            pos_ico = _make_icosahedral_cluster()
            p_ico, e_ico = _relax(pos_ico, max_iter=300, gtol=1e-6)
            if e_ico < best_energy:
                best_energy = e_ico
                best_pos = p_ico.copy()
        except Exception:
            pass

    # 5. Basin Hopping from the best starting point
    rng = np.random.default_rng(42)
    
    bh_deadline = DEADLINE - 15.0
    try:
        bh_pos, bh_energy = _basin_hopping(best_pos, rng, bh_deadline)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos.copy()
    except Exception:
        pass

    # 6. Final High-Precision Polish
    if time.time() - t0 < DEADLINE - 5.0:
        try:
            # First, try to optimize scale of the best found
            p_scaled, e_scaled = _optimize_scale(best_pos, n_iter=20)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled.copy()
            
            # Then, do a very tight L-BFGS relaxation
            p, e = _relax(best_pos, max_iter=3000, gtol=1e-12, ftol=1e-15)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # 7. Final separation check
    best_pos = _ensure_min_sep(best_pos)

    return best_pos
