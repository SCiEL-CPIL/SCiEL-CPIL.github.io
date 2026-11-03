"""Packing 50 equal circles in unit disk - Enhanced Ring-DE with Multi-Topology Search.

Strategy:
1. Multi-topology DE search: try several ring topologies and pick the best.
2. Local refinement using multi-stage annealing (increasing beta) with L-BFGS-B.
3. Final high-precision polish using iterative constraint projection.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
DEADLINE_MARGIN = 15.0  # seconds before hard limit (150s total)


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
    """Smooth objective for optimization: minimize negative soft-min of margins."""
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    # Combine all constraints: pairwise distances (half) and boundary distances
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 1e10
    return -_softmin_logsumexp(all_vals, beta)


def _ring_to_points(radii, angles_list):
    """Convert ring parameters (radii, list of angle arrays) to point coordinates."""
    if len(radii) == 0:
        return np.empty((0, 2))
    pts = [np.array([0.0, 0.0])] # Center
    for i in range(1, len(radii)):
        r = radii[i]
        a = angles_list[i]
        pts.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    return np.vstack(pts)


def _flatten_ring(radii, angles_list):
    """Flatten ring parameters into a single vector for optimization."""
    vec = list(radii)
    for i in range(1, len(angles_list)):
        vec.extend(list(angles_list[i]))
    return np.array(vec)


def _unflatten_ring(vec, ring_sizes):
    """Unflatten optimization vector back to radii and angles."""
    radii = vec[:len(ring_sizes)].copy()
    angles_list = [np.array([0.0])]
    idx = len(ring_sizes)
    for i in range(1, len(ring_sizes)):
        n = ring_sizes[i]
        angles_list.append(vec[idx:idx + n].copy())
        idx += n
    return radii, angles_list


def _constraint_push(pts, r_target, deadline, max_iters=400):
    """Polish solution by iteratively pushing violating constraints with relaxation."""
    pts = pts.copy()
    relaxation = 1.0  # Start with full push, decrease if oscillating
    prev_score = _score(pts)
    
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
            push = (2.0 * r_target - dist) / 2.0 * relaxation
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
                push = (1.0 - r_target - dist) * relaxation
                if push > 1e-15:
                    pts[i] += push * di
        
        # Check convergence and adjust relaxation
        current_score = _score(pts)
        if current_score < prev_score - 1e-10:
            # Score decreased, reduce relaxation to avoid oscillation
            relaxation *= 0.7
        elif current_score > prev_score + 1e-12:
            # Score improved, maintain or slightly increase relaxation
            relaxation = min(relaxation * 1.1, 1.0)
        prev_score = current_score
        
        # Check if target achieved
        if current_score >= r_target - 1e-12:
            break
    
    return pts


def _anneal_points(pts, deadline):
    """Multi-stage annealing optimization on full point coordinates."""
    current = pts.copy()
    current_score = _score(current)
    
    # Progressive beta values for annealing - more stages for better convergence
    betas = [3.0, 8.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0]
    
    for beta in betas:
        if time.time() > deadline - 4:
            break
        try:
            res = minimize(
                _smooth_objective,
                current.ravel(),
                args=(beta,),
                method='L-BFGS-B',
                options={
                    'maxiter': 1500, 
                    'ftol': 1e-16, 
                    'gtol': 1e-14,
                    'maxfun': 30000
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


def _de_ring_search(ring_sizes, deadline, seed=42):
    """Run DE search for a given ring topology."""
    # Initial guess for radii
    n_rings = len(ring_sizes)
    if n_rings <= 1:
        return None, -1.0
    
    # Heuristic: distribute radii based on cumulative counts
    cum_counts = np.cumsum(ring_sizes)
    total = cum_counts[-1]
    init_radii = np.zeros(n_rings)
    for i in range(n_rings):
        if i == 0:
            init_radii[i] = 0.0
        else:
            # Proportional to sqrt of cumulative fraction
            frac = cum_counts[i] / total
            init_radii[i] = np.sqrt(frac) * 0.88
    
    # Initialize angles
    angles_list = []
    for i, n in enumerate(ring_sizes):
        if i == 0:
            angles_list.append(np.array([0.0]))
        else:
            angles_list.append(np.linspace(0, 2 * np.pi, n, endpoint=False))
    
    # Build bounds - tighter for radii
    bounds = []
    for i in range(n_rings):
        if i == 0:
            bounds.append((0.0, 0.0))  # Center is fixed at origin
        elif i == n_rings - 1:
            # Outer ring: should be near boundary
            bounds.append((0.75, 0.92))
        else:
            # Inner rings: reasonable range
            bounds.append((0.1, 0.85))
    
    for i in range(1, n_rings):
        bounds.extend([(0.0, 2.0 * np.pi)] * ring_sizes[i])
    
    def ring_obj(vec):
        if time.time() > deadline - 20:
            return 1e10
        radii, al = _unflatten_ring(vec, ring_sizes)
        # Clip radii to valid range
        radii = np.clip(radii, 0.0, 0.95)
        pts = _ring_to_points(radii, al)
        if len(pts) != N:
            return 1e10
        # Use a soft-min objective with moderate beta for DE
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        pair_half = d / 2.0
        all_vals = np.concatenate([pair_half.ravel(), boundary])
        all_vals = all_vals[np.isfinite(all_vals)]
        if len(all_vals) == 0:
            return 1e10
        return -_softmin_logsumexp(all_vals, 30.0)
    
    try:
        res_de = differential_evolution(
            ring_obj, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            tol=1e-10,
            mutation=(0.4, 1.2), 
            recombination=0.8, 
            seed=seed, 
            polish=False
        )
        if time.time() < deadline - 30:
            ring_x = res_de.x
            radii_opt, al_opt = _unflatten_ring(ring_x, ring_sizes)
            radii_opt = np.clip(radii_opt, 0.0, 0.95)
            de_pts = _ring_to_points(radii_opt, al_opt)
            de_score = _score(de_pts)
            return de_pts, de_score
    except Exception:
        pass
    
    return None, -1.0


def construct():
    t0 = time.time()
    deadline = t0 + 120.0 - DEADLINE_MARGIN
    
    best_points = None
    best_score = -1.0
    
    # Try multiple ring topologies
    # Known good: 1+6+12+18+13=50
    # Alternatives: 1+6+10+14+19=50, 1+8+12+15+14=50, 1+6+14+16+13=50
    topologies = [
        [1, 6, 12, 18, 13],  # Standard
        [1, 6, 10, 14, 19],  # Alternative 1
        [1, 8, 12, 15, 14],  # Alternative 2
        [1, 6, 14, 16, 13],  # Alternative 3
        [1, 10, 12, 16, 11], # Alternative 4
    ]
    
    # ---- Stage 1: Multi-Topology Differential Evolution ----
    for idx, ring_sizes in enumerate(topologies):
        if time.time() > deadline - 40:
            break
        if sum(ring_sizes) != N:
            continue
        
        de_pts, de_score = _de_ring_search(ring_sizes, deadline, seed=42 + idx)
        if de_pts is not None and de_score > best_score:
            best_score = de_score
            best_points = de_pts.copy()
    
    # If all DE failed, use a simple ring initialization
    if best_points is None or best_score < 0.1:
        ring_sizes = [1, 6, 12, 18, 13]
        init_radii = np.array([0.0, 0.25, 0.45, 0.65, 0.85])
        angles_list = []
        for i, n in enumerate(ring_sizes):
            if i == 0:
                angles_list.append(np.array([0.0]))
            else:
                angles_list.append(np.linspace(0, 2 * np.pi, n, endpoint=False))
        init_pts = _ring_to_points(init_radii, angles_list)
        init_score = _score(init_pts)
        if init_score > best_score:
            best_score = init_score
            best_points = init_pts.copy()

    # ---- Stage 2: Multi-stage Annealing Refinement ----
    if best_points is not None and time.time() < deadline - 15:
        refined_pts, refined_score = _anneal_points(best_points, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # ---- Stage 3: Iterative Constraint Push Polish ----
    if best_points is not None and time.time() < deadline - 2:
        # Multiple rounds of constraint push with increasing target
        for round_num in range(3):
            if time.time() > deadline - 2:
                break
            current_r = _score(best_points)
            target_r = current_r + 1e-5 * (round_num + 1)
            polished_pts = _constraint_push(best_points, target_r, deadline, max_iters=300)
            polished_score = _score(polished_pts)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_pts.copy()

    # Fallback if something went wrong
    if best_points is None or not np.all(np.isfinite(best_points)):
        ring_sizes = [1, 6, 12, 18, 13]
        init_radii = np.array([0.0, 0.25, 0.45, 0.65, 0.85])
        angles_list = []
        for i, n in enumerate(ring_sizes):
            if i == 0:
                angles_list.append(np.array([0.0]))
            else:
                angles_list.append(np.linspace(0, 2 * np.pi, n, endpoint=False))
        best_points = _ring_to_points(init_radii, angles_list)
        best_score = _score(best_points)
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
