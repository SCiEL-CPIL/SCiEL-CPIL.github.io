"""Packing 50 equal circles in a unit disk - combined ring structure + multi-stage soft-min annealing + constraint projection."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 132.0  # safety margin before 150s kill


def _pairwise_distances(pts):
    """Compute pairwise distances between all points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
    """Compute the packing score: largest common radius."""
    if pts is None or len(pts) != N:
        return -1.0
    pts = np.asarray(pts, dtype=float)
    if not np.all(np.isfinite(pts)):
        return -1.0
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    val = min(min_pair, min_boundary)
    if val <= 0:
        return -1.0
    return val


def _softmin_logsumexp(values, beta):
    """Numerically-stable soft-min via log-sum-exp."""
    if len(values) == 0:
        return 0.0
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0
    exp_sum = np.sum(np.exp(neg - neg_max))
    if exp_sum <= 0.0:
        return 0.0
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta):
    """Smooth objective: minimize negative soft-min of all margins."""
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 1e10
    return -_softmin_logsumexp(all_vals, beta)


def _ring_to_points(radii, angles_list):
    """Convert ring parameters to point coordinates."""
    pts = [np.array([0.0, 0.0])]
    for i in range(1, len(radii)):
        r = radii[i]
        a = angles_list[i]
        pts.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    return np.vstack(pts)


def _hexagonal_init(scale=0.12):
    """Generate hexagonal lattice points and filter to fit in unit disk."""
    points = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = scale * (i + 0.5 * j)
            y = scale * (np.sqrt(3) / 2 * j)
            if x**2 + y**2 <= 0.81:
                points.append([x, y])
    points = np.array(points)
    if len(points) == 0:
        return None
    dists = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(dists)
    points = points[idx[:N]]
    return points


def _ring_init(n, n_rings=5):
    """Initialize with ring structure matching known optimal packing topology."""
    points = []
    points.append([0.0, 0.0])
    ring_counts = [6, 12, 18, 13]
    ring_radii = [0.25, 0.45, 0.65, 0.85]
    for i, (count, r_est) in enumerate(zip(ring_counts, ring_radii)):
        for k in range(count):
            angle = 2 * np.pi * k / count + (np.pi / count if i % 2 == 0 else 0.0)
            points.append([r_est * np.cos(angle), r_est * np.sin(angle)])
    return np.array(points[:n])


def _anneal_points(pts, deadline):
    """Multi-stage annealing with progressive beta schedule."""
    current = pts.copy()
    current_score = _score(current)
    
    # Wide beta schedule: start smooth, end very sharp
    betas = [5.0, 15.0, 40.0, 100.0, 250.0, 600.0, 1500.0, 3500.0, 8000.0, 15000.0, 30000.0, 50000.0]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                _smooth_objective,
                current.ravel(),
                args=(beta,),
                method='L-BFGS-B',
                options={
                    'maxiter': 800,
                    'ftol': 1e-16,
                    'gtol': 1e-14,
                    'maxfun': 25000
                }
            )
            new_pts = res.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            s = _score(new_pts)
            if s > current_score:
                current = new_pts
                current_score = s
        except Exception:
            pass
            
    return current, current_score


def _constraint_push(pts, r_target, deadline, max_iters=300):
    """Polish solution by iteratively pushing violating constraints."""
    pts = pts.copy()
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        
        min_pair = np.min(pair_margins)
        min_bound = np.min(boundary_margins)
        
        if min_pair < min_bound - 1e-14:
            # Pair constraint is binding
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            push = (2.0 * r_target - dist) / 2.0
            if push > 1e-15:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            # Boundary constraint is binding
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                pts[i] += 1e-4 * np.array([1.0, 0.0])
            else:
                di = pi / dist
                push = (1.0 - r_target - dist)
                if push > 1e-15:
                    pts[i] += push * di
        
        # Check convergence
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1))))
        if r >= r_target - 1e-12:
            break
    
    return pts


def _binary_search_polish(pts, deadline):
    """Binary search for maximum feasible radius with projection polish."""
    current = pts.copy()
    low = _score(current) - 1e-8
    high = _score(current) + 5e-4
    
    for _ in range(40):
        if time.time() > deadline - 2:
            break
        mid = (low + high) / 2.0
        polished = _constraint_push(current, mid, deadline, max_iters=200)
        s = _score(polished)
        if s >= mid - 1e-10:
            low = mid
            current = polished
        else:
            high = mid
    
    return current, _score(current)


def _global_search_ring_params(time_limit=30.0):
    """Global search over ring parameter space using differential evolution."""
    t_start = time.time()
    
    # Ring structure: 1 + 6 + 12 + 18 + 13
    ring_sizes = [1, 6, 12, 18, 13]
    n_rings = len(ring_sizes)
    
    # Bounds: [radius, angle_offset] for each ring (center is fixed at origin)
    bounds = [(0.0, 0.95), (0.0, 2 * np.pi)] * (n_rings - 1)
    
    def objective(x):
        if time.time() - t_start > time_limit:
            return 1e10
        radii = np.zeros(n_rings)
        radii[0] = 0.0
        angles_list = [np.array([0.0])]
        for i in range(1, n_rings):
            r = x[2*(i-1)]
            offset = x[2*(i-1)+1]
            radii[i] = r
            n_pts = ring_sizes[i]
            angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False) + offset
            angles_list.append(angles)
        
        pts = _ring_to_points(radii, angles_list)
        if len(pts) != N:
            return 1e10
        return -_score(pts)
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=80,
            tol=1e-8,
            polish=False,
            init='sobol'
        )
        
        if result.success:
            x = result.x
            radii = np.zeros(n_rings)
            radii[0] = 0.0
            angles_list = [np.array([0.0])]
            for i in range(1, n_rings):
                r = x[2*(i-1)]
                offset = x[2*(i-1)+1]
                radii[i] = r
                n_pts = ring_sizes[i]
                angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False) + offset
                angles_list.append(angles)
            
            pts = _ring_to_points(radii, angles_list)
            score = _score(pts)
            return pts, score
    except Exception:
        pass
    
    return None, 0.0


def construct():
    t_start = time.time()
    
    # Multiple initializations
    inits = []
    
    # Ring-based initialization
    ring_pts = _ring_init(N)
    ring_score = _score(ring_pts)
    if ring_score > 0:
        inits.append((ring_pts.copy(), ring_score))
    
    # Hexagonal initialization
    hex_pts = _hexagonal_init(scale=0.12)
    if hex_pts is not None and len(hex_pts) == N:
        hex_score = _score(hex_pts)
        if hex_score > 0:
            inits.append((hex_pts.copy(), hex_score))
    
    # Slightly perturbed ring
    ring_pts2 = _ring_init(N) + np.random.randn(N, 2) * 0.01
    ring_pts2_score = _score(ring_pts2)
    if ring_pts2_score > 0:
        inits.append((ring_pts2.copy(), ring_pts2_score))
    
    # Pick best initialization
    if not inits:
        return np.random.randn(N, 2) * 0.3
    
    best_points = inits[0][0].copy()
    best_score = inits[0][1]
    for pts, sc in inits[1:]:
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
    
    # Global search over ring parameters
    if time.time() < DEADLINE - 50:
        ring_search_pts, ring_search_score = _global_search_ring_params(
            time_limit=min(25.0, DEADLINE - (time.time() - t_start) - 50)
        )
        if ring_search_pts is not None and ring_search_score > best_score:
            best_score = ring_search_score
            best_points = ring_search_pts.copy()
    
    # Multi-stage soft-min annealing (the key improvement)
    if time.time() < DEADLINE - 25:
        annealed_pts, annealed_score = _anneal_points(
            best_points,
            deadline=DEADLINE - (time.time() - t_start) - 20
        )
        if annealed_score > best_score:
            best_score = annealed_score
            best_points = annealed_pts.copy()
    
    # Constraint projection + binary search polish
    if time.time() < DEADLINE - 10:
        polished_pts, polished_score = _binary_search_polish(
            best_points,
            deadline=DEADLINE - (time.time() - t_start) - 5
        )
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts.copy()
    
    # Final validity check: ensure all points are inside and finite
    c_final = best_points.reshape(N, 2)
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1.0 - best_score + 1e-10:
            scale = (1.0 - best_score) / max(r2[i], 1e-15)
            c_final[i] *= scale
    
    # Final score verification
    final_score = _score(c_final)
    if final_score <= 0:
        # Fallback: return a valid configuration
        c_final = _ring_init(N)
    
    return c_final
