"""Portfolio-search program for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 290.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    if positions is None:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _scale_to_spacing(pts, spacing=0.72):
    if len(pts) < 2:
        return pts
    rmax = np.max(np.linalg.norm(pts, axis=1))
    if rmax > 0:
        target_r = spacing * len(pts) ** (1.0 / 3.0) * 0.55
        pts = pts * (target_r / rmax)
    return pts


def _make_fcc(n):
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 4
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    pts.append(np.array([i + dx, j + dy, k + dz]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    return _scale_to_spacing(pts)


def _make_bcc(n):
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 4
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                pts.append(np.array([i, j, k]))
                pts.append(np.array([i + 0.5, j + 0.5, k + 0.5]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    return _scale_to_spacing(pts)


def _make_mackay(n):
    """Generate Mackay icosahedral cluster with proper shell construction."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 0, phi], [-1, 0, phi], [1, 0, -phi], [-1, 0, -phi],
        [0, phi, 1], [0, -phi, 1], [0, phi, -1], [0, -phi, -1],
        [phi, 1, 0], [-phi, 1, 0], [phi, -1, 0], [-phi, -1, 0]], dtype=float)
    verts /= np.linalg.norm(verts[0])

    pts = [np.zeros(3)]

    # Shell 1: 12 vertices at r=1
    for v in verts:
        pts.append(1.0 * v)

    # Find edges (pairs of vertices at edge distance)
    edge_dist = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(np.linalg.norm(verts[i] - verts[j]) - edge_dist) < 0.01:
                edges.append((i, j))

    # Shell 2: 42 points at r=sqrt(2) - 30 edge midpoints + 12 face centers
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        mid /= np.linalg.norm(mid)
        pts.append(np.sqrt(2) * mid)

    # Find faces (triples of mutually connected vertices)
    face_centers = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                d_ki = np.linalg.norm(verts[k] - verts[i])
                if (abs(d_ij - edge_dist) < 0.01 and abs(d_jk - edge_dist) < 0.01 and
                        abs(d_ki - edge_dist) < 0.01):
                    center = (verts[i] + verts[j] + verts[k]) / 3
                    center /= np.linalg.norm(center)
                    face_centers.append(center)

    # Deduplicate face centers
    unique_fc = []
    for fc in face_centers:
        if all(np.linalg.norm(fc - q) > 0.1 for q in unique_fc):
            unique_fc.append(fc)

    for v in unique_fc:
        pts.append(np.sqrt(2) * v)

    # Shell 3: 92 points at r=sqrt(3)
    # Generate directions from combinations
    shell3_dirs = []
    for v in verts:
        shell3_dirs.append(v)
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        mid /= np.linalg.norm(mid)
        shell3_dirs.append(mid)
    for v in unique_fc:
        shell3_dirs.append(v)

    # Add more directions for shell 3
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                d_ki = np.linalg.norm(verts[k] - verts[i])
                if (abs(d_ij - edge_dist) < 0.01 and abs(d_jk - edge_dist) < 0.01):
                    # Perpendicular directions
                    edge1 = verts[j] - verts[i]
                    edge2 = verts[k] - verts[j]
                    perp = np.cross(edge1, edge2)
                    if np.linalg.norm(perp) > 0.01:
                        perp /= np.linalg.norm(perp)
                        shell3_dirs.append(perp)

    # Deduplicate and add to pts
    unique_s3 = []
    for d in shell3_dirs:
        d = d / np.linalg.norm(d)
        if all(np.linalg.norm(d - q) > 0.08 for q in unique_s3):
            unique_s3.append(d)

    for v in unique_s3:
        pts.append(np.sqrt(3) * v)

    # Shell 4: 156 points at r=2
    shell4_dirs = []
    for v in verts:
        shell4_dirs.append(v)
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        mid /= np.linalg.norm(mid)
        shell4_dirs.append(mid)
    for v in unique_fc:
        shell4_dirs.append(v)
    for d in unique_s3:
        shell4_dirs.append(d)

    # Add more for shell 4
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                d_ki = np.linalg.norm(verts[k] - verts[i])
                if (abs(d_ij - edge_dist) < 0.01 and abs(d_jk - edge_dist) < 0.01 and
                        abs(d_ki - edge_dist) < 0.01):
                    center = (verts[i] + verts[j] + verts[k]) / 3
                    center /= np.linalg.norm(center)
                    shell4_dirs.append(center)
                    # Add rotated versions
                    nrm = np.cross(verts[j] - verts[i], verts[k] - verts[i])
                    nrm /= np.linalg.norm(nrm)
                    for angle in np.linspace(0, 2 * np.pi, 4, endpoint=False):
                        x = np.array([1.0, 0, 0])
                        if abs(np.dot(x, nrm)) > 0.9:
                            x = np.array([0, 1.0, 0])
                        y = np.cross(nrm, x)
                        z = nrm
                        rot = center * np.cos(angle) + y * np.sin(angle)
                        rot /= np.linalg.norm(rot)
                        shell4_dirs.append(rot)

    unique_s4 = []
    for d in shell4_dirs:
        d = d / np.linalg.norm(d)
        if all(np.linalg.norm(d - q) > 0.06 for q in unique_s4):
            unique_s4.append(d)

    for v in unique_s4:
        pts.append(2.0 * v)

    # Shell 5: 252 points at r=sqrt(5)
    shell5_dirs = []
    for v in verts:
        shell5_dirs.append(v)
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        mid /= np.linalg.norm(mid)
        shell5_dirs.append(mid)
    for v in unique_fc:
        shell5_dirs.append(v)
    for d in unique_s3:
        shell5_dirs.append(d)
    for d in unique_s4:
        shell5_dirs.append(d)

    # Add more for shell 5
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                d_ki = np.linalg.norm(verts[k] - verts[i])
                if (abs(d_ij - edge_dist) < 0.01 and abs(d_jk - edge_dist) < 0.01 and
                        abs(d_ki - edge_dist) < 0.01):
                    center = (verts[i] + verts[j] + verts[k]) / 3
                    center /= np.linalg.norm(center)
                    shell5_dirs.append(center)
                    nrm = np.cross(verts[j] - verts[i], verts[k] - verts[i])
                    nrm /= np.linalg.norm(nrm)
                    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
                        x = np.array([1.0, 0, 0])
                        if abs(np.dot(x, nrm)) > 0.9:
                            x = np.array([0, 1.0, 0])
                        y = np.cross(nrm, x)
                        z = nrm
                        rot = center * np.cos(angle) + y * np.sin(angle)
                        rot /= np.linalg.norm(rot)
                        shell5_dirs.append(rot)

    unique_s5 = []
    for d in shell5_dirs:
        d = d / np.linalg.norm(d)
        if all(np.linalg.norm(d - q) > 0.05 for q in unique_s5):
            unique_s5.append(d)

    for v in unique_s5:
        pts.append(np.sqrt(5) * v)

    pts = np.array(pts)

    # Deduplicate
    unique_pts = [pts[0]]
    for p in pts[1:]:
        if all(np.linalg.norm(p - q) > 0.05 for q in unique_pts):
            unique_pts.append(p)
    pts = np.array(unique_pts)

    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    return _scale_to_spacing(pts)


def _make_decahedral(n):
    """Decahedral cluster: FCC with 5-fold twinning approximation."""
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 4
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    pts.append(np.array([i + dx, j + dy, k + dz]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    # Apply slight icosahedral distortion
    r = np.linalg.norm(pts, axis=1)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    # Small perturbation to break cubic symmetry
    pts[:, 0] += 0.05 * np.cos(5 * theta) * r
    pts[:, 1] += 0.05 * np.sin(5 * theta) * r
    return _scale_to_spacing(pts)


def _make_random_compact(n, rng, spacing=0.72):
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 20000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
    if len(points) == n:
        return np.array(points)
    return None


def _perturb(positions, rng, scale=0.1):
    return positions + rng.normal(0, scale, positions.shape)


def _basin_hop_adaptive(positions, kind, rng, n_hops=10, base_scale=0.15):
    """Adaptive basin hopping with step size adjustment."""
    current = positions.copy()
    current_energy = np.inf
    try:
        current_energy, _ = relax(current, 'sutton_chen', kind)
    except:
        pass

    scale = base_scale
    for _ in range(n_hops):
        if _time_left() < 5.0:
            break
        perturbed = current + rng.normal(0, scale, current.shape)
        if not _is_valid(perturbed):
            scale *= 0.7
            continue
        try:
            energy, relaxed = relax(perturbed, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                if energy < current_energy:
                    current = relaxed
                    current_energy = energy
                    scale = min(scale * 1.2, 0.3)
                else:
                    scale *= 0.8
        except Exception:
            scale *= 0.8
    return current, current_energy


def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(n * 12345 + 7)
    best_energy = np.inf
    best_positions = None

    starts = []

    # Generate diverse starting structures
    try:
        fcc = _make_fcc(n)
        if _is_valid(fcc):
            starts.append(fcc)
            for s in [0.03, 0.07, 0.12, 0.2]:
                pert = _perturb(fcc, rng, s)
                if _is_valid(pert):
                    starts.append(pert)
    except Exception:
        pass

    try:
        bcc = _make_bcc(n)
        if _is_valid(bcc):
            starts.append(bcc)
            for s in [0.05, 0.1, 0.15]:
                pert = _perturb(bcc, rng, s)
                if _is_valid(pert):
                    starts.append(pert)
    except Exception:
        pass

    try:
        mackay = _make_mackay(n)
        if _is_valid(mackay):
            starts.append(mackay)
            for s in [0.03, 0.07, 0.12, 0.18]:
                pert = _perturb(mackay, rng, s)
                if _is_valid(pert):
                    starts.append(pert)
    except Exception:
        pass

    try:
        deca = _make_decahedral(n)
        if _is_valid(deca):
            starts.append(deca)
            for s in [0.05, 0.1, 0.15]:
                pert = _perturb(deca, rng, s)
                if _is_valid(pert):
                    starts.append(pert)
    except Exception:
        pass

    # Random compact starts
    for _ in range(6):
        rc = _make_random_compact(n, rng)
        if rc is not None and _is_valid(rc):
            starts.append(rc)

    # Relax each start
    for i, start in enumerate(starts):
        if _time_left() < 15.0:
            break
        try:
            energy, positions = relax(start, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # If we have time, do adaptive basin hopping from the best result
    if best_positions is not None and _time_left() > 25.0:
        try:
            bh_result, bh_energy = _basin_hop_adaptive(best_positions, kind, rng, n_hops=12, base_scale=0.12)
            energy, positions = relax(bh_result, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # Additional basin hopping from perturbed best if time permits
    if best_positions is not None and _time_left() > 40.0:
        try:
            pert_best = _perturb(best_positions, rng, 0.15)
            bh2, _ = _basin_hop_adaptive(pert_best, kind, rng, n_hops=8, base_scale=0.1)
            energy, positions = relax(bh2, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # More basin hopping from second-best perturbation
    if best_positions is not None and _time_left() > 55.0:
        try:
            pert_best2 = _perturb(best_positions, rng, 0.2)
            bh3, _ = _basin_hop_adaptive(pert_best2, kind, rng, n_hops=6, base_scale=0.08)
            energy, positions = relax(bh3, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    if best_positions is None:
        try:
            best_positions = _make_fcc(n)
        except Exception:
            best_positions = rng.uniform(-1, 1, (n, 3))

    return best_positions
# EVOLVE-BLOCK-END
