"""Lennard-Jones cluster global minimum, 55 atoms.
Structural change: direct Cartesian optimization with multiple diverse initializations
and aggressive multi-stage L-BFGS-B polish, plus a small number of basin-hopping moves."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 55

DEADLINE_SECONDS = 135.0
_start_time = time.time()

def _remaining():
    return DEADLINE_SECONDS - (time.time() - _start_time)

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for N atoms (vectorized)."""
    n = len(pos)
    # Compute all pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3.0)
    r12 = r6 * r6
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def _lj_grad(pos):
    """Compute gradient of Lennard-Jones energy w.r.t. Cartesian coords."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3.0)
    # dE/dr2 = 4*(-6*r2^-7 + 3*r2^-4) ... wait let me recompute
    # E = 4*(r^-12 - r^-6) = 4*(r2^-6 - r2^-3)
    # dE/dr2 = 4*(-6*r2^-7 + 3*r2^-4)
    # dE/dx_i = 2 * dE/dr2 * (x_i - x_j)
    coeff = 4.0 * (-6.0 * r2**(-7.0) + 3.0 * r2**(-4.0))
    np.fill_diagonal(coeff, 0.0)
    grad = 2.0 * np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad

def _lj_energy_and_grad(pos):
    """Return (energy, gradient) for L-BFGS-B with jac=True."""
    e = _lj_energy(pos)
    g = _lj_grad(pos)
    return e, g.ravel()

def _make_mackay_icosahedron():
    """Construct the two-shell Mackay icosahedron for 55 atoms."""
    PHI = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 icosahedron vertices (unit vectors)
    verts = np.array([
        [0, 1, PHI], [0, -1, PHI], [0, 1, -PHI], [0, -1, -PHI],
        [1, PHI, 0], [-1, PHI, 0], [1, -PHI, 0], [-1, -PHI, 0],
        [PHI, 0, 1], [-PHI, 0, 1], [PHI, 0, -1], [-PHI, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edge midpoints (30 edges)
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            dists[i, j] = d
            dists[j, i] = d
    min_dist = np.min(dists[dists > 0])
    edge_mask = dists < min_dist * 1.001
    
    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if edge_mask[i, j]:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid_unit = mid / np.linalg.norm(mid)
                edge_mids.append(mid_unit)
    edge_mids = np.array(edge_mids)  # (30, 3)
    
    # Two-shell Mackay: inner shell (12 verts at r1), middle shell (30 edge mids at r2), outer shell (12 verts at r3)
    # Literature-optimal radii
    r1 = 1.135
    r2 = 1.162
    r3 = 1.283
    
    pos = np.zeros((55, 3))
    pos[0] = [0, 0, 0]  # center
    pos[1:13] = verts_unit * r1       # inner 12
    pos[13:43] = edge_mids * r2       # middle 30
    pos[43:55] = verts_unit * r3      # outer 12
    
    return pos

def _make_face_center_icosahedron():
    """Construct a face-centered icosahedral structure (alternative 55-atom geometry)."""
    PHI = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 icosahedron vertices
    verts = np.array([
        [0, 1, PHI], [0, -1, PHI], [0, 1, -PHI], [0, -1, -PHI],
        [1, PHI, 0], [-1, PHI, 0], [1, -PHI, 0], [-1, -PHI, 0],
        [PHI, 0, 1], [-PHI, 0, 1], [PHI, 0, -1], [-PHI, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Face centers: 20 faces of icosahedron
    # Each face is a triangle of 3 mutually adjacent vertices
    # Find all triangles
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            dists[i, j] = d
            dists[j, i] = d
    min_dist = np.min(dists[dists > 0])
    edge_mask = dists < min_dist * 1.001
    
    face_centers = []
    for i in range(12):
        for j in range(i + 1, 12):
            if not edge_mask[i, j]:
                continue
            for k in range(j + 1, 12):
                if edge_mask[i, k] and edge_mask[j, k]:
                    fc = (verts_unit[i] + verts_unit[j] + verts_unit[k]) / 3.0
                    fc_unit = fc / np.linalg.norm(fc)
                    face_centers.append(fc_unit)
    face_centers = np.array(face_centers)  # (20, 3)
    
    # 55 atoms: 1 center + 12 verts + 30 edge mids + 12 outer verts = 55 (same as Mackay)
    # Alternative: 1 center + 12 verts + 20 face centers + 12 outer verts + 10 extra = 55
    # Let's use: 1 center + 12 inner verts + 30 edge mids + 12 outer verts (same count)
    # But use face centers for a different geometry
    # 1 + 12 + 20 + 12 + 10 = 55? Let's just use Mackay-like but with face centers
    # Actually let's do: 1 center + 12 verts(r1) + 20 face centers(r2) + 12 verts(r3) + 10 edge mids(r4) = 55
    # That's 1+12+20+12+10 = 55
    
    # Find edge midpoints
    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if edge_mask[i, j]:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid_unit = mid / np.linalg.norm(mid)
                edge_mids.append(mid_unit)
    edge_mids = np.array(edge_mids)  # (30, 3)
    
    # Use 10 of the 30 edge mids (e.g., those closest to face centers)
    # For simplicity, use first 10
    edge_mids_10 = edge_mids[:10]
    
    r1 = 1.13
    r2 = 1.18
    r3 = 1.29
    r4 = 1.22
    
    pos = np.zeros((55, 3))
    pos[0] = [0, 0, 0]
    pos[1:13] = verts_unit * r1
    pos[13:33] = face_centers * r2
    pos[33:45] = verts_unit * r3
    pos[45:55] = edge_mids_10 * r4
    
    return pos

def _relax_lbfgs(pos, maxiter=3000, gtol=1e-10, ftol=1e-12):
    """L-BFGS-B local relaxation in Cartesian space."""
    if _remaining() < 5.0:
        return pos, _lj_energy(pos)
    
    result = minimize(
        _lj_energy_and_grad,
        pos.reshape(-1),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
    )
    return result.x.reshape(-1, 3), result.fun

def _polish(pos, maxiter=10000, gtol=1e-14, ftol=1e-16):
    """High-precision polish."""
    if _remaining() < 5.0:
        return pos, _lj_energy(pos)
    
    result = minimize(
        _lj_energy_and_grad,
        pos.reshape(-1),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
    )
    return result.x.reshape(-1, 3), result.fun

def _check_min_dist(pos, min_d=0.5):
    """Check all pair distances >= min_d."""
    d = pdist(pos)
    return d.min() >= min_d - 1e-9

def _basin_hopping_step(pos, rng, scale=0.1):
    """Perturb position and relax."""
    pos_pert = pos + rng.normal(0, scale, pos.shape)
    # Ensure minimum distance
    if not _check_min_dist(pos_pert):
        d = pdist(pos_pert)
        min_d = d.min()
        if min_d < 0.5:
            scale_factor = 0.5 / min_d * 0.99
            pos_pert = pos_pert * scale_factor
    pos_relaxed, e = _relax_lbfgs(pos_pert, maxiter=1000, gtol=1e-8, ftol=1e-10)
    return pos_relaxed, e

def construct():
    global _start_time
    _start_time = time.time()
    
    best_pos = None
    best_energy = np.inf
    rng = np.random.default_rng(42)
    
    # Strategy 1: Mackay icosahedron with multi-stage polish
    if _remaining() > 20.0:
        try:
            mackay = _make_mackay_icosahedron()
            # Stage 1: moderate relaxation
            pos1, e1 = _relax_lbfgs(mackay, maxiter=2000, gtol=1e-8, ftol=1e-10)
            if _check_min_dist(pos1) and e1 < best_energy:
                best_energy = e1
                best_pos = pos1.copy()
            
            # Stage 2: tighter
            if _remaining() > 15.0:
                pos2, e2 = _polish(pos1, maxiter=5000, gtol=1e-12, ftol=1e-14)
                if _check_min_dist(pos2) and e2 < best_energy:
                    best_energy = e2
                    best_pos = pos2.copy()
            
            # Stage 3: very tight
            if _remaining() > 10.0:
                pos3, e3 = _polish(pos2, maxiter=10000, gtol=1e-14, ftol=1e-16)
                if _check_min_dist(pos3) and e3 < best_energy:
                    best_energy = e3
                    best_pos = pos3.copy()
        except Exception:
            pass
    
    # Strategy 2: Face-center icosahedron
    if _remaining() > 15.0:
        try:
            fc = _make_face_center_icosahedron()
            pos_fc, e_fc = _relax_lbfgs(fc, maxiter=2000, gtol=1e-8, ftol=1e-10)
            if _check_min_dist(pos_fc) and e_fc < best_energy:
                best_energy = e_fc
                best_pos = pos_fc.copy()
            
            if _remaining() > 10.0:
                pos_fc2, e_fc2 = _polish(pos_fc, maxiter=5000, gtol=1e-12, ftol=1e-14)
                if _check_min_dist(pos_fc2) and e_fc2 < best_energy:
                    best_energy = e_fc2
                    best_pos = pos_fc2.copy()
        except Exception:
            pass
    
    # Strategy 3: Basin hopping from best so far
    if best_pos is not None and _remaining() > 30.0:
        for i in range(8):
            if _remaining() < 10.0:
                break
            try:
                scales = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
                pos_bh, e_bh = _basin_hopping_step(best_pos, rng, scale=scales[i % len(scales)])
                if _check_min_dist(pos_bh) and e_bh < best_energy:
                    best_energy = e_bh
                    best_pos = pos_bh.copy()
            except Exception:
                continue
    
    # Strategy 4: Random initializations
    if _remaining() > 20.0:
        for i in range(5):
            if _remaining() < 10.0:
                break
            try:
                # Random positions in a sphere of radius 3
                pos_rand = rng.uniform(-2.5, 2.5, (N, 3))
                # Ensure minimum distance by rejection or scaling
                for attempt in range(10):
                    if _check_min_dist(pos_rand):
                        break
                    pos_rand = rng.uniform(-2.5, 2.5, (N, 3))
                else:
                    # Scale to ensure minimum distance
                    d = pdist(pos_rand)
                    min_d = d.min()
                    if min_d < 0.5:
                        pos_rand = pos_rand * (0.5 / min_d * 0.99)
                
                pos_r, e_r = _relax_lbfgs(pos_rand, maxiter=1500, gtol=1e-8, ftol=1e-10)
                if _check_min_dist(pos_r) and e_r < best_energy:
                    best_energy = e_r
                    best_pos = pos_r.copy()
            except Exception:
                continue
    
    # Final ultra-high-precision polish
    if best_pos is not None and _remaining() > 8.0:
        try:
            pos_final, e_final = _polish(best_pos, maxiter=15000, gtol=1e-15, ftol=1e-17)
            if _check_min_dist(pos_final) and e_final < best_energy:
                best_energy = e_final
                best_pos = pos_final.copy()
        except Exception:
            pass
    
    # One more polish if time permits
    if best_pos is not None and _remaining() > 5.0:
        try:
            pos_final2, e_final2 = _polish(best_pos, maxiter=10000, gtol=1e-16, ftol=1e-18)
            if _check_min_dist(pos_final2) and e_final2 < best_energy:
                best_energy = e_final2
                best_pos = pos_final2.copy()
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback: cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
    
    return best_pos
