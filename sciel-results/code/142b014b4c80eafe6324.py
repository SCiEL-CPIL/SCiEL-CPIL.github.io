"""Dzugutov cluster global minima: icosahedral-shell seeds + L-BFGS-B relaxation with kick-and-relax."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 290.0  # safety margin under the 330 s kill

# Dzugutov parameters
_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16
_MIN_SEP = 0.5


def _remaining():
    return TIME_LIMIT - (time.time() - _START)


def _energy(pos):
    """Vectorised Dzugutov energy (flat array, n*3)."""
    p = np.asarray(pos, dtype=float)
    n = p.size // 3
    p = p.reshape(n, 3)
    diff = p[:, None, :] - p[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    r_ij = r[iu]
    
    energy = 0.0
    mask1 = r_ij < _a
    if np.any(mask1):
        r1 = r_ij[mask1]
        energy += np.sum(_A * (r1 ** (-_m) - _B) * np.exp(_c / (r1 - _a)))
    mask2 = (r_ij >= _a) & (r_ij < _b)
    if np.any(mask2):
        r2 = r_ij[mask2]
        energy += np.sum(_B * np.exp(_d / (r2 - _b)))
    return energy


def _grad_fn(pos):
    """Gradient via the provided helper, reshaped to flat."""
    p = np.asarray(pos, dtype=float)
    n = p.size // 3
    p = p.reshape(n, 3)
    _, grad = dzugutov_energy_gradient(p)
    return grad.flatten()


def _min_sep_ok(pos, min_sep=_MIN_SEP):
    """Check minimum separation constraint."""
    n = pos.shape[0]
    if n < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    return np.min(r[iu]) >= min_sep


def _make_icosahedron():
    """Generate icosahedral vertices and related symmetry points."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico /= np.linalg.norm(ico[0])
    
    # 30 edge midpoints (second shell candidates)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(ico[i] - ico[j])
            if 0.9 < d < 1.1:  # adjacent vertices
                edges.append((i, j))
    
    edge_midpoints = []
    for i, j in edges:
        mid = (ico[i] + ico[j]) / 2
        mid /= np.linalg.norm(mid)
        edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints[:30])
    
    # 20 face centers (third shell candidates)
    # Find triangular faces of icosahedron
    face_centers = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                dij = np.linalg.norm(ico[i] - ico[j])
                djk = np.linalg.norm(ico[j] - ico[k])
                dik = np.linalg.norm(ico[i] - ico[k])
                if (0.9 < dij < 1.1 and 0.9 < djk < 1.1 and 0.9 < dik < 1.1):
                    fc = (ico[i] + ico[j] + ico[k]) / 3
                    fc /= np.linalg.norm(fc)
                    face_centers.append(fc)
    
    # Deduplicate face centers
    unique_fc = []
    for fc in face_centers:
        if not any(np.linalg.norm(fc - ufc) < 0.01 for ufc in unique_fc):
            unique_fc.append(fc)
    face_centers = np.array(unique_fc[:20])
    
    return ico, edge_midpoints, face_centers


def _icosahedral_seeds(n, rng, n_variants=3, scale_range=(0.9, 1.3)):
    """Generate multiple icosahedral-shell seed structures."""
    ico, edge_pts, face_pts = _make_icosahedron()
    seeds = []
    
    for variant in range(n_variants):
        scale = rng.uniform(*scale_range)
        twist = rng.uniform(-0.3, 0.3)
        
        # Build rotation matrix for twist around z-axis
        cos_t, sin_t = np.cos(twist), np.sin(twist)
        R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
        
        points = []
        remaining = n
        
        # Central atom
        if remaining > 0:
            points.append(np.array([0.0, 0.0, 0.0]))
            remaining -= 1
        
        # First shell: icosahedron vertices
        count1 = min(12, remaining)
        for i in range(count1):
            p = ico[i] * scale * 1.2
            p += rng.normal(0, 0.03, 3)
            points.append(p)
        remaining -= count1
        
        # Second shell: edge midpoints
        if remaining > 0:
            count2 = min(30, remaining)
            for i in range(count2):
                p = R @ (edge_pts[i % 30] * scale * 1.9)
                p += rng.normal(0, 0.03, 3)
                points.append(p)
            remaining -= count2
        
        # Third shell: face centers
        if remaining > 0:
            count3 = min(20, remaining)
            for i in range(count3):
                p = R @ (face_pts[i % 20] * scale * 2.6)
                p += rng.normal(0, 0.03, 3)
                points.append(p)
            remaining -= count3
        
        # Fourth shell: more icosahedron vertices at larger radius
        if remaining > 0:
            for i in range(remaining):
                idx = i % 12
                p = ico[idx] * scale * (3.2 + 0.1 * (i // 12))
                p += rng.normal(0, 0.05, 3)
                points.append(p)
        
        points = np.array(points[:n])
        
        # Enforce minimum separation via local relaxation
        for _ in range(50):
            if _min_sep_ok(points):
                break
            # Push apart overlapping atoms
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(points[i] - points[j])
                    if d < _MIN_SEP:
                        direction = (points[i] - points[j]) / max(d, 1e-10)
                        push = (_MIN_SEP - d) / 2 + 0.01
                        points[i] += direction * push
                        points[j] -= direction * push
        
        seeds.append(points)
    
    return seeds


def _relax_lbfgsb(pos_flat, maxiter=300):
    """Relax using L-BFGS-B with tight tolerances."""
    result = minimize(
        _energy,
        pos_flat,
        jac=_grad_fn,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20, 'disp': False}
    )
    return result


def _kick_and_relax(pos, rng, n_kicks=5, kick_scale=0.3):
    """Apply random kicks and relax to escape local minima."""
    best = pos.copy()
    best_e = _energy(pos.flatten())
    
    for _ in range(n_kicks):
        kicked = pos + rng.normal(0, kick_scale, pos.shape)
        # Enforce minimum separation
        for _ in range(30):
            if _min_sep_ok(kicked):
                break
            for i in range(len(kicked)):
                for j in range(i+1, len(kicked)):
                    d = np.linalg.norm(kicked[i] - kicked[j])
                    if d < _MIN_SEP:
                        direction = (kicked[i] - kicked[j]) / max(d, 1e-10)
                        push = (_MIN_SEP - d) / 2 + 0.01
                        kicked[i] += direction * push
                        kicked[j] -= direction * push
        
        result = _relax_lbfgsb(kicked.flatten(), maxiter=150)
        if result.fun < best_e:
            best_e = result.fun
            best = result.x.reshape(pos.shape)
    
    return best, best_e


def construct(instance):
    """Find low-energy Dzugutov cluster using icosahedral seeds + L-BFGS-B + kick-and-relax."""
    size = instance["n"]
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Generate diverse icosahedral seeds and relax each
    seeds = _icosahedral_seeds(size, rng, n_variants=4, scale_range=(0.85, 1.35))
    
    for seed in seeds:
        if _remaining() < 15.0:
            break
        
        try:
            result = _relax_lbfgsb(seed.flatten(), maxiter=250)
            if result.fun < best_energy and _min_sep_ok(result.x.reshape(size, 3)):
                best_energy = result.fun
                best_positions = result.x.reshape(size, 3)
        except Exception:
            continue
    
    # Phase 2: Kick-and-relax from best position
    if best_positions is not None and _remaining() > 20.0:
        kicked_best, kicked_energy = _kick_and_relax(best_positions, rng, n_kicks=8, kick_scale=0.25)
        if kicked_energy < best_energy:
            best_energy = kicked_energy
            best_positions = kicked_best
    
    # Phase 3: Additional kicks from different seeds if time permits
    if _remaining() > 15.0:
        for seed in seeds[:2]:
            if _remaining() < 10.0:
                break
            try:
                kicked, ke = _kick_and_relax(seed, rng, n_kicks=4, kick_scale=0.35)
                if ke < best_energy:
                    best_energy = ke
                    best_positions = kicked
            except Exception:
                continue
    
    # Final polish with very tight tolerances
    if best_positions is not None and _remaining() > 5.0:
        try:
            result = _relax_lbfgsb(best_positions.flatten(), maxiter=100)
            if result.fun < best_energy and _min_sep_ok(result.x.reshape(size, 3)):
                best_energy = result.fun
                best_positions = result.x.reshape(size, 3)
        except Exception:
            pass
    
    # Fallback
    if best_positions is None:
        best_positions = seeds[0]
    
    return best_positions
# EVOLVE-BLOCK-END
