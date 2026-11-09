"""Direct geometric construction for 300 equal circles in a unit circle."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START


def _score(points):
    """Feasible common radius using KDTree for nearest-neighbor distance."""
    if points is None or len(points) < 2:
        return 0.0
    norms = np.linalg.norm(points, axis=1)
    min_boundary = float(1.0 - np.max(norms))
    if min_boundary <= 0:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair = float(np.min(dists[:, 1])) / 2.0
    return min(min_boundary, min_pair)


def _build_hex_lattice(spacing):
    """Hex lattice points inside unit disk, pick N closest to center."""
    k = int(1.0 / spacing) + 3
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing * 0.5:
                pts.append((x, y))
    if len(pts) < N:
        return None
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]


def _build_hex_rings(spacing, ring_radii=None):
    """
    Build N points in concentric hexagonal rings with given spacing.
    """
    pts = []
    if ring_radii is None:
        n_rings = int(1.0 / spacing) + 1
        ring_radii = np.linspace(spacing * 0.8, 1.0 - spacing * 0.5, n_rings)
    counts = []
    total = 0
    for k, r in enumerate(ring_radii):
        if k == 0 and r < 0.1:
            c = 1
        else:
            c = max(6, int(2 * np.pi * r / spacing))
        counts.append(c)
        total += c
    if total < N:
        deficit = N - total
        for i in range(len(counts) - 1, -1, -1):
            add = min(deficit, max(1, counts[i] // 2))
            counts[i] += add
            deficit -= add
            if deficit <= 0:
                break
    elif total > N:
        excess = total - N
        for i in range(len(counts) - 1, -1, -1):
            rem = min(excess, counts[i] - (1 if i == 0 and ring_radii[i] < 0.1 else 6))
            counts[i] -= rem
            excess -= rem
            if excess <= 0:
                break
    for k, r in enumerate(ring_radii):
        c = counts[k]
        if c <= 0:
            continue
        if c == 1 and r < 0.1:
            pts.append([0.0, 0.0])
        else:
            angles = 2 * np.pi * np.arange(c) / c + (np.pi / c) * (k % 2)
            pts.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
    pts = np.vstack(pts)
    if len(pts) != N:
        if len(pts) < N:
            extra = N - len(pts)
            angles = 2 * np.pi * np.arange(extra) / extra
            rr = 0.9
            pts = np.vstack([pts, np.column_stack([rr * np.cos(angles), rr * np.sin(angles)])])
        else:
            pts = pts[:N]
    return pts


def _soft_min_obj(flat_pts, beta=80.0):
    """
    Soft-min objective: minimize -r where r is approximated by soft-min of
    all pairwise half-distances and boundary distances.
    Returns (value, gradient) for L-BFGS-B.
    """
    pts = flat_pts.reshape(N, 2)
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    pair_half = dists[:, 1] / 2.0
    
    all_vals = np.concatenate([pair_half, boundary])
    all_vals = np.maximum(all_vals, 1e-12)
    
    scaled = -beta * all_vals
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum / beta
    
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    grad_flat = np.zeros(2 * N)
    
    for i in range(N):
        w = weights[N + i]
        if norms[i] > 1e-10:
            grad_flat[2*i] += -w * (-pts[i, 0] / norms[i])
            grad_flat[2*i+1] += -w * (-pts[i, 1] / norms[i])
    
    _, nn_idx = tree.query(pts, k=2)
    for i in range(N):
        w = weights[i]
        j = nn_idx[i, 1]
        d = dists[i, 1]
        if d > 1e-10:
            grad_flat[2*i] += -w * (pts[i, 0] - pts[j, 0]) / (2.0 * d)
            grad_flat[2*i+1] += -w * (pts[i, 1] - pts[j, 1]) / (2.0 * d)
            grad_flat[2*j] += -w * (pts[j, 0] - pts[i, 0]) / (2.0 * d)
            grad_flat[2*j+1] += -w * (pts[j, 1] - pts[i, 1]) / (2.0 * d)
    
    return -soft_min_val, grad_flat


def _lbfgs_refine(points, deadline, max_iters=500):
    """Refine using L-BFGS-B on soft-min objective."""
    flat = points.flatten()
    best_flat = flat.copy()
    best_r = _score(points)
    
    t0 = time.time()
    remaining = deadline - t0
    if remaining < 5:
        return points, best_r
    
    for beta in [50.0, 100.0, 200.0]:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                _soft_min_obj,
                flat.copy(),
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': max_iters, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.x is not None:
                cand = res.x.reshape(N, 2)
                r = _score(cand)
                if r > best_r:
                    best_r = r
                    best_flat = res.x.copy()
        except Exception:
            pass
    
    return best_flat.reshape(N, 2), best_r


def _local_refine_fast(points, deadline, max_iters=8000, base_step=0.02):
    """
    Fast local refinement using only neighbor checks for moved point.
    Uses cached KDTree and only recomputes distances involving the moved point.
    """
    points = points.copy()
    best_points = points.copy()
    best_r = _score(points)
    rng = np.random.default_rng(42)
    step = base_step
    
    # Precompute KDTree
    tree = cKDTree(points)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        original = points[idx].copy()
        best_local = original
        best_local_r = best_r
        
        # Find current neighbors of idx (k=8 to be safe)
        _, nn_idx = tree.query(original, k=8)
        
        for attempt in range(6):
            if time.time() > deadline:
                break
            s = step * rng.uniform(0.3, 1.0)
            ang = rng.uniform(0, 2 * np.pi)
            cand = original + s * np.array([np.cos(ang), np.sin(ang)])
            
            # Check boundary
            cnorm = np.linalg.norm(cand)
            if cnorm >= 1.0:
                continue
            
            # Check against neighbors only
            min_dist = 1.0 - cnorm  # boundary constraint
            for j in nn_idx:
                if j == idx:
                    continue
                d = np.linalg.norm(cand - points[j])
                if d < min_dist:
                    min_dist = d
            # Also check a few more random points for safety
            extra_idx = rng.integers(0, N, size=5)
            for j in extra_idx:
                if j == idx:
                    continue
                d = np.linalg.norm(cand - points[j])
                if d < min_dist:
                    min_dist = d
            
            cand_r = min_dist / 2.0
            if cand_r > best_local_r:
                best_local_r = cand_r
                best_local = cand.copy()
        
        points[idx] = best_local
        if best_local_r > best_r:
            best_r = best_local_r
            best_points = points.copy()
            # Update tree periodically
            if it % 50 == 0:
                tree = cKDTree(points)
            step = max(0.003, step * 0.997)
    
    # Final exact score
    final_r = _score(best_points)
    return best_points, final_r


def _pairwise_push_apart(points, deadline, n_sweeps=50):
    """
    Iteratively push overlapping pairs apart to increase minimum distance.
    For each pair closer than 2*r, push them apart along their connecting line.
    This is a simple but effective local improvement step.
    """
    points = points.copy()
    best_points = points.copy()
    best_r = _score(points)
    
    tree = cKDTree(points)
    
    for sweep in range(n_sweeps):
        if time.time() > deadline:
            break
        
        # Find all close pairs using KDTree
        # Query each point's nearest neighbors
        dists, nn_idx = tree.query(points, k=4)
        
        moved = False
        for i in range(N):
            for k in range(1, 4):
                j = nn_idx[i, k]
                if j <= i:
                    continue
                d = dists[i, k]
                if d < 0.12:  # If too close, push apart
                    # Push apart along the line connecting them
                    diff = points[i] - points[j]
                    norm_diff = np.linalg.norm(diff)
                    if norm_diff > 1e-10:
                        push = 0.002 * diff / norm_diff
                        new_i = points[i] + push
                        new_j = points[j] - push
                        
                        # Check boundary constraints
                        if np.linalg.norm(new_i) < 1.0 and np.linalg.norm(new_j) < 1.0:
                            points[i] = new_i
                            points[j] = new_j
                            moved = True
        
        if not moved:
            break
        
        # Update tree
        tree = cKDTree(points)
        
        # Score periodically
        if sweep % 10 == 0:
            r = _score(points)
            if r > best_r:
                best_r = r
                best_points = points.copy()
    
    final_r = _score(points)
    if final_r > best_r:
        best_points = points.copy()
        best_r = final_r
    
    return best_points, best_r


def _gradient_ascent_on_radius(points, deadline, n_iters=2000, step_size=0.005):
    """
    Gradient ascent on the minimum distance using subgradient method.
    For the point involved in the minimum pair, push it away from its nearest neighbor.
    For the point closest to boundary, push it inward.
    """
    points = points.copy()
    best_points = points.copy()
    best_r = _score(points)
    rng = np.random.default_rng(123)
    
    tree = cKDTree(points)
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        # Find critical points: those involved in minimum pair or closest to boundary
        norms = np.linalg.norm(points, axis=1)
        boundary_dists = 1.0 - norms
        min_boundary_idx = np.argmin(boundary_dists)
        
        dists, nn_idx = tree.query(points, k=2)
        pair_dists = dists[:, 1] / 2.0
        min_pair_idx = np.argmin(pair_dists)
        
        # Decide which constraint is active
        if pair_dists[min_pair_idx] < boundary_dists[min_boundary_idx]:
            # Pair constraint is active
            i = min_pair_idx
            j = nn_idx[i, 1]
            d = dists[i, 1]
            if d > 1e-10:
                # Push i away from j
                direction = (points[i] - points[j]) / d
                step = step_size * rng.uniform(0.5, 1.0)
                new_pos = points[i] + step * direction
                if np.linalg.norm(new_pos) < 1.0:
                    points[i] = new_pos
                    # Also push j slightly away from i for symmetry
                    new_pos_j = points[j] - 0.5 * step * direction
                    if np.linalg.norm(new_pos_j) < 1.0:
                        points[j] = new_pos_j
        else:
            # Boundary constraint is active
            i = min_boundary_idx
            norm_i = norms[i]
            if norm_i > 1e-10:
                # Push inward
                direction = -points[i] / norm_i
                step = step_size * rng.uniform(0.5, 1.0)
                new_pos = points[i] + step * direction
                points[i] = new_pos
        
        # Occasionally update tree
        if it % 50 == 0:
            tree = cKDTree(points)
        
        # Score periodically
        if it % 100 == 0:
            r = _score(points)
            if r > best_r:
                best_r = r
                best_points = points.copy()
    
    final_r = _score(points)
    if final_r > best_r:
        best_points = points.copy()
        best_r = final_r
    
    return best_points, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s

    candidates = []

    # Hex lattice over a range of spacings
    for trial in range(30):
        spacing = 0.055 + trial * 0.004
        pts = _build_hex_lattice(spacing)
        if pts is not None:
            r = _score(pts)
            if r > 0:
                candidates.append((pts, r))

    # Hex rings with tuned radii
    for trial in range(15):
        spacing = 0.058 + trial * 0.004
        pts = _build_hex_rings(spacing)
        if pts is not None:
            r = _score(pts)
            if r > 0:
                candidates.append((pts, r))

    if not candidates:
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_points, best_r = candidates[0]

    # Phase 1: Fast local refinement on top candidates
    for cand_idx in range(min(5, len(candidates))):
        if time.time() > deadline - 60.0:
            break
        cand_pts, cand_r = candidates[cand_idx]
        refined_pts, refined_r = _local_refine_fast(cand_pts, deadline - 60.0,
                                                     max_iters=3000, base_step=0.02)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    # Phase 2: Pairwise push-apart to fix close pairs
    if time.time() < deadline - 40.0:
        pushed_pts, pushed_r = _pairwise_push_apart(best_points, deadline - 40.0, n_sweeps=30)
        if pushed_pts is not None and pushed_r > best_r:
            best_points, best_r = pushed_pts, pushed_r

    # Phase 3: L-BFGS-B refinement on the best candidate
    if time.time() < deadline - 25.0:
        lbfgs_pts, lbfgs_r = _lbfgs_refine(best_points, deadline - 20.0, max_iters=200)
        if lbfgs_pts is not None and lbfgs_r > best_r:
            best_points, best_r = lbfgs_pts, lbfgs_r

    # Phase 4: Gradient ascent on active constraints
    if time.time() < deadline - 15.0:
        grad_pts, grad_r = _gradient_ascent_on_radius(best_points, deadline - 15.0, n_iters=1000, step_size=0.003)
        if grad_pts is not None and grad_r > best_r:
            best_points, best_r = grad_pts, grad_r

    # Phase 5: Final fast local polish
    if time.time() < deadline - 5.0:
        polished_pts, polished_r = _local_refine_fast(best_points, deadline - 2.0,
                                                       max_iters=2000, base_step=0.008)
        if polished_pts is not None and polished_r > best_r:
            best_points, best_r = polished_pts, polished_r

    return best_points


# EVOLVE-BLOCK-END
