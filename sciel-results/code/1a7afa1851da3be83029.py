"""Portfolio-search program for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 285.0  # safety margin below 300s


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
    """FCC-like cluster: nearest n atoms from FCC lattice."""
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 3
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
    """BCC-like cluster."""
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 3
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


def _make_icosahedral(n):
    """Icosahedral cluster with concentric shells."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 0, phi], [-1, 0, phi], [1, 0, -phi], [-1, 0, -phi],
        [0, phi, 1], [0, -phi, 1], [0, phi, -1], [0, -phi, -1],
        [phi, 1, 0], [-phi, 1, 0], [phi, -1, 0], [-phi, -1, 0]], dtype=float)
    verts /= np.linalg.norm(verts[0])

    # Collect all icosahedral directions
    all_dirs = []
    
    # Shell 1: 12 vertices
    for v in verts:
        all_dirs.append((v, 1.0))
    
    # Shell 2: edge midpoints (30) at r=sqrt(2)
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.05:
                edges.append((i, j))
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        mid /= np.linalg.norm(mid)
        all_dirs.append((mid, np.sqrt(2)))
    
    # Shell 2: face centers (20) at r=sqrt(2)
    face_centers = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                d_ki = np.linalg.norm(verts[k] - verts[i])
                if d_ij < 1.05 and d_jk < 1.05 and d_ki < 1.05:
                    center = (verts[i] + verts[j] + verts[k]) / 3
                    center /= np.linalg.norm(center)
                    face_centers.append(center)
    unique_fc = []
    for fc in face_centers:
        if all(np.linalg.norm(fc - q) > 0.1 for q in unique_fc):
            unique_fc.append(fc)
    for v in unique_fc:
        all_dirs.append((v, np.sqrt(2)))
    
    # Shell 3: more directions at r=sqrt(3)
    shell3_dirs = []
    for v in verts:
        shell3_dirs.append(v)
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        mid /= np.linalg.norm(mid)
        shell3_dirs.append(mid)
    for v in unique_fc:
        shell3_dirs.append(v)
    
    # Add more directions from icosahedral symmetry operations
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if np.linalg.norm(verts[i] - verts[j]) < 1.05 and \
                   np.linalg.norm(verts[j] - verts[k]) < 1.05:
                    edge_dir = verts[k] - verts[j]
                    edge_dir /= np.linalg.norm(edge_dir)
                    perp = np.cross(verts[i] - verts[j], edge_dir)
                    if np.linalg.norm(perp) > 0.01:
                        perp /= np.linalg.norm(perp)
                        shell3_dirs.append(perp)
    
    unique_s3 = []
    for d in shell3_dirs:
        d = d / np.linalg.norm(d)
        if all(np.linalg.norm(d - q) > 0.1 for q in unique_s3):
            unique_s3.append(d)
    for v in unique_s3:
        all_dirs.append((v, np.sqrt(3)))
    
    # Shell 4: more directions at r=2
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
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if np.linalg.norm(verts[i] - verts[j]) < 1.05 and \
                   np.linalg.norm(verts[j] - verts[k]) < 1.05 and \
                   np.linalg.norm(verts[k] - verts[i]) < 1.05:
                    center = (verts[i] + verts[j] + verts[k]) / 3
                    center /= np.linalg.norm(center)
                    shell4_dirs.append(center)
    
    unique_s4 = []
    for d in shell4_dirs:
        d = d / np.linalg.norm(d)
        if all(np.linalg.norm(d - q) > 0.08 for q in unique_s4):
            unique_s4.append(d)
    for v in unique_s4:
        all_dirs.append((v, 2.0))
    
    # Shell 5: even more at r=sqrt(5)
    shell5_dirs = []
    for d in unique_s3:
        shell5_dirs.append(d)
    for d in unique_s4:
        shell5_dirs.append(d)
    # Add cross-product directions
    for i in range(12):
        for j in range(i + 1, 12):
            cross = np.cross(verts[i], verts[j])
            if np.linalg.norm(cross) > 0.01:
                cross /= np.linalg.norm(cross)
                shell5_dirs.append(cross)
    
    unique_s5 = []
    for d in shell5_dirs:
        d = d / np.linalg.norm(d)
        if all(np.linalg.norm(d - q) > 0.07 for q in unique_s5):
            unique_s5.append(d)
    for v in unique_s5:
        all_dirs.append((v, np.sqrt(5)))
    
    # Build point list: center + shells
    pts = [np.zeros(3)]
    for d, r in all_dirs:
        pts.append(r * d)
    
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    
    # Remove duplicates
    unique_pts = [pts[0]]
    for p in pts[1:]:
        if all(np.linalg.norm(p - q) > 0.05 for q in unique_pts):
            unique_pts.append(p)
    pts = np.array(unique_pts)
    
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    return _scale_to_spacing(pts)


def _make_decahedral(n):
    """Decahedral (5-fold twinned FCC) cluster."""
    # 5-fold twinning around z-axis
    # Each twin is an FCC-like wedge
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 3
    
    # Generate FCC points in a wedge
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + dx, j + dy, k + dz])
                    # Only keep points in the first 72-degree wedge
                    angle = np.arctan2(p[1], p[0])
                    if -0.628 < angle < 0.628:  # ~72 degrees
                        pts.append(p)
    
    pts = np.array(pts)
    
    # Apply 5-fold twinning
    all_pts = []
    for p in pts:
        for k in range(5):
            angle = 2 * np.pi * k / 5
            rot = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            all_pts.append(rot @ p)
    
    all_pts = np.array(all_pts)
    all_pts -= all_pts.mean(axis=0)
    
    # Remove duplicates
    unique_pts = [all_pts[0]]
    for p in all_pts[1:]:
        if all(np.linalg.norm(p - q) > 0.05 for q in unique_pts):
            unique_pts.append(p)
    all_pts = np.array(unique_pts)
    
    dists = np.linalg.norm(all_pts, axis=1)
    order = np.argsort(dists)
    all_pts = all_pts[order[:n]]
    return _scale_to_spacing(all_pts)


def _make_random_compact(n, rng, spacing=0.72):
    """Generate random compact cluster with minimum distance constraint."""
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    attempts = 0
    min_dist = 0.55 * spacing
    while len(points) < n and attempts < 30000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
    if len(points) == n:
        return np.array(points)
    return None


def _perturb(positions, rng, scale):
    return positions + rng.normal(0, scale, positions.shape)


def _basin_hop(start, potential, kind, rng, max_kicks=15, kick_scale=0.05, time_budget=None):
    """Basin-hopping with adaptive kick scale."""
    best_energy, best_pos = np.inf, start.copy()
    current_energy, current_pos = np.inf, start.copy()
    
    for kick in range(max_kicks):
        if time_budget is not None and _time_left() < time_budget:
            break
        try:
            energy, positions = relax(current_pos, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                current_energy = energy
                current_pos = positions
                if energy < best_energy:
                    best_energy = energy
                    best_pos = positions.copy()
        except Exception:
            pass
        
        # Adaptive kick: larger if stuck, smaller if improving
        if kick > 0 and current_energy >= best_energy:
            kick_scale_current = kick_scale * (1 + 0.15 * kick)
        else:
            kick_scale_current = kick_scale * (1 + 0.05 * kick)
        
        kicked = _perturb(current_pos, rng, kick_scale_current)
        if not _is_valid(kicked):
            kicked = current_pos + rng.normal(0, kick_scale_current * 0.5, current_pos.shape)
        if not _is_valid(kicked):
            break
        current_pos = kicked
    
    return best_energy, best_pos


def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(n * 12345 + 7)
    best_energy = np.inf
    best_positions = None

    # Generate structured templates
    templates = []
    try:
        fcc = _make_fcc(n)
        if _is_valid(fcc):
            templates.append(("fcc", fcc))
    except Exception:
        pass

    try:
        bcc = _make_bcc(n)
        if _is_valid(bcc):
            templates.append(("bcc", bcc))
    except Exception:
        pass

    try:
        ico = _make_icosahedral(n)
        if _is_valid(ico):
            templates.append(("icosahedral", ico))
    except Exception:
        pass

    try:
        deca = _make_decahedral(n)
        if _is_valid(deca):
            templates.append(("decahedral", deca))
    except Exception:
        pass

    # Add random compact starts
    n_random = min(8, max(4, n // 12))
    for _ in range(n_random):
        rc = _make_random_compact(n, rng)
        if rc is not None and _is_valid(rc):
            templates.append(("random", rc))

    # Phase 1: Relax each template directly
    for name, template in templates:
        if _time_left() < 15.0:
            break
        try:
            energy, positions = relax(template, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # Phase 2: Basin-hopping from each template
    for name, template in templates:
        if _time_left() < 20.0:
            break
        try:
            energy, positions = _basin_hop(
                template, potential, kind, rng,
                max_kicks=12, kick_scale=0.04, time_budget=12.0
            )
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # Phase 3: Perturbed templates + relax
    perturbation_scales = [0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25]
    for name, template in templates[:6]:
        if _time_left() < 15.0:
            break
        for scale in perturbation_scales:
            if _time_left() < 5.0:
                break
            pert = _perturb(template, rng, scale)
            if not _is_valid(pert):
                continue
            try:
                energy, positions = relax(pert, potential, kind)
                if np.isfinite(energy) and _is_valid(positions):
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
            except Exception:
                pass

    # Phase 4: More random starts with basin hopping if time remains
    if _time_left() > 25.0:
        for _ in range(6):
            if _time_left() < 12.0:
                break
            rc = _make_random_compact(n, rng)
            if rc is not None and _is_valid(rc):
                try:
                    energy, positions = _basin_hop(
                        rc, potential, kind, rng,
                        max_kicks=10, kick_scale=0.05, time_budget=10.0
                    )
                    if np.isfinite(energy) and _is_valid(positions):
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = positions
                except Exception:
                    pass

    # Phase 5: Final refinement - basin hop from best position
    if best_positions is not None and _time_left() > 15.0:
        try:
            energy, positions = _basin_hop(
                best_positions, potential, kind, rng,
                max_kicks=15, kick_scale=0.03, time_budget=12.0
            )
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # Fallback
    if best_positions is None:
        try:
            best_positions = _make_fcc(n)
        except Exception:
            best_positions = rng.uniform(-1, 1, (n, 3))

    return best_positions
# EVOLVE-BLOCK-END
