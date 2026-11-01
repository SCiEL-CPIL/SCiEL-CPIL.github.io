"""Packing 50 equal circles in a unit circle - contact-graph force-directed layout."""
import time
import numpy as np

N = 50
_START = time.time()
_DEADLINE = _START + 140.0  # 10s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    # Pairwise distances
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)


def _hex_lattice_init(n, jitter=0.0):
    """Generate initial positions from hexagonal lattice, trimmed to n points."""
    for spacing in np.linspace(0.14, 0.35, 40):
        k = int(1.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    rows.append((x, y))
        pts = np.array(rows)
        if len(pts) >= n:
            idx = np.random.permutation(len(pts))[:n]
            pts = pts[idx]
            if jitter > 0:
                pts += np.random.randn(n, 2) * jitter
            r = np.hypot(pts[:, 0], pts[:, 1])
            mask = r > 1.0
            if np.any(mask):
                pts[mask] /= r[mask, None] * 1.001
            return pts
    theta = np.random.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(np.random.uniform(0, 1, n))
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])


def _concentric_ring_init(n):
    """Initialize with concentric rings for a structured starting point."""
    pts = []
    # Center point
    pts.append((0.0, 0.0))
    remaining = n - 1
    ring_radius = 0.15
    while remaining > 0 and len(pts) < n:
        # How many points fit on this ring?
        circumference = 2 * np.pi * ring_radius
        # Approximate max points based on desired spacing ~ 0.25
        max_pts = max(1, int(circumference / 0.25))
        # Don't exceed what's left
        pts_on_ring = min(max_pts, remaining)
        for k in range(pts_on_ring):
            angle = 2 * np.pi * k / pts_on_ring
            x = ring_radius * np.cos(angle)
            y = ring_radius * np.sin(angle)
            if x * x + y * y <= 1.0:
                pts.append((x, y))
                remaining -= 1
        ring_radius += 0.15
    pts = np.array(pts[:n])
    if len(pts) < n:
        # Pad with random
        extra = n - len(pts)
        theta = np.random.uniform(0, 2 * np.pi, extra)
        rho = np.sqrt(np.random.uniform(0, 1, extra))
        extra_pts = np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])
        pts = np.vstack([pts, extra_pts])
    return pts


def _project_to_feasible(pts, r_target):
    """Project points to satisfy disk constraint and pairwise separation >= 2*r_target."""
    pts = pts.copy()
    n = len(pts)
    
    # Enforce disk constraint
    for _ in range(10):
        r = np.hypot(pts[:, 0], pts[:, 1])
        mask = r > 1.0 - r_target
        if not np.any(mask):
            break
        # Push back to boundary
        pts[mask] *= (1.0 - r_target) / r[mask]
    
    # Enforce pairwise separation via iterative projection
    for _ in range(5):
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = pts[i_idx, 0] - pts[j_idx, 0]
        dy = pts[i_idx, 1] - pts[j_idx, 1]
        d = np.sqrt(dx * dx + dy * dy)
        d = np.maximum(d, 1e-15)
        
        # Find violating pairs
        viol = d < 2.0 * r_target
        if not np.any(viol):
            break
        
        # Separate violating pairs
        for k in range(len(i_idx)):
            if viol[k]:
                mid_x = (pts[i_idx[k], 0] + pts[j_idx[k], 0]) / 2.0
                mid_y = (pts[i_idx[k], 1] + pts[j_idx[k], 1]) / 2.0
                sep = 2.0 * r_target - d[k]
                # Move each point half the separation distance
                pts[i_idx[k], 0] -= 0.5 * sep * dx[k] / d[k]
                pts[i_idx[k], 1] -= 0.5 * sep * dy[k] / d[k]
                pts[j_idx[k], 0] += 0.5 * sep * dx[k] / d[k]
                pts[j_idx[k], 1] += 0.5 * sep * dy[k] / d[k]
        
        # Re-enforce disk
        r = np.hypot(pts[:, 0], pts[:, 1])
        mask = r > 1.0 - r_target
        if np.any(mask):
            pts[mask] *= (1.0 - r_target) / r[mask]
    
    return pts


def _contact_graph_optimize(pts, max_iters=50000, dt_init=0.02):
    """
    Contact-graph based force-directed layout with 1/d^4 repulsion.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    dt = dt_init
    best_score = _score(pts)
    best_pts = pts.copy()
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 8.0:
            break
        
        # Compute all pairwise distances
        dx = pts[i_idx, 0] - pts[j_idx, 0]
        dy = pts[i_idx, 1] - pts[j_idx, 1]
        d2 = dx * dx + dy * dy
        d = np.sqrt(d2)
        d = np.maximum(d, 1e-12)
        
        # 1/d^4 repulsive potential - creates strong "hard sphere" behavior
        # Scale to keep forces in reasonable range
        force_mag = 1.0 / (d2 * d2)  # 1/d^4
        fx = force_mag * dx / d
        fy = force_mag * dy / d
        
        # Accumulate forces
        Fx = np.zeros(n)
        Fy = np.zeros(n)
        np.add.at(Fx, i_idx, -fx)
        np.add.at(Fx, j_idx, fx)
        np.add.at(Fy, i_idx, -fy)
        np.add.at(Fy, j_idx, fy)
        
        # Weak centering force to prevent drift
        r = np.hypot(pts[:, 0], pts[:, 1])
        Fx -= 0.01 * pts[:, 0]
        Fy -= 0.01 * pts[:, 1]
        
        # Normalize forces
        fmag = np.hypot(Fx, Fy)
        fmag_max = np.max(fmag)
        if fmag_max > 1.0:
            Fx /= fmag_max
            Fy /= fmag_max
        
        # Adaptive step: use current score to set step size
        current_r = _score(pts)
        # Step proportional to gap between current and potential
        step = dt * np.minimum(1.0, 10.0 * current_r)
        
        new_pts = pts + step * np.column_stack([Fx, Fy])
        
        # Hard constraint projection: keep inside unit disk
        nr = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            new_pts[mask] /= nr[mask, None] * (1.0 + 1e-6)
        
        new_score = _score(new_pts)
        
        if new_score >= best_score - 1e-10:
            pts = new_pts
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_pts = pts.copy()
            # Slowly decay step size
            dt *= 0.9998
        else:
            dt *= 0.99
        
        if dt < 1e-8:
            break
    
    # Final refinement with projection
    final_r = _score(best_pts)
    if final_r > 0:
        projected = _project_to_feasible(best_pts, final_r * 0.999)
        proj_score = _score(projected)
        if proj_score > best_score:
            best_pts = projected
            best_score = proj_score
    
    return best_pts, best_score


def construct():
    """Construct 50 circle centers in unit circle maximizing common radius."""
    best_pts = None
    best_score = 0.0
    
    # Try multiple initialization strategies
    initializations = [
        _hex_lattice_init(N, jitter=0.0),
        _hex_lattice_init(N, jitter=0.02),
        _concentric_ring_init(N),
        _hex_lattice_init(N, jitter=0.05),
        _concentric_ring_init(N),
        _hex_lattice_init(N, jitter=0.03),
    ]
    
    for init_pts in initializations:
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Run contact-graph force optimization
        optimized_pts, score = _contact_graph_optimize(
            init_pts,
            max_iters=30000,
            dt_init=0.02
        )
        
        if score > best_score:
            best_score = score
            best_pts = optimized_pts.copy()
    
    if best_pts is None:
        best_pts = _hex_lattice_init(N, jitter=0.0)
        best_score = _score(best_pts)
    
    # Final projection refinement
    if best_score > 0.01:
        projected = _project_to_feasible(best_pts, best_score * 0.9999)
        proj_score = _score(projected)
        if proj_score > best_score:
            best_pts = projected
            best_score = proj_score
    
    # Final validation and cleanup
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] /= r[mask, None] * (1.0 + 1e-6)
    
    return best_pts
