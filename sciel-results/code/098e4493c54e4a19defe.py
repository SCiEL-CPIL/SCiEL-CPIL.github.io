"""Packing 50 equal circles in unit disk - Combined Ring-DE and Annealed Refinement.

Strategy:
1. Global search using Differential Evolution on ring parameters (radii + individual angles).
   This allows the structure to deviate from perfect symmetry if beneficial.
2. Local refinement using multi-stage annealing (increasing beta) with L-BFGS-B on full points.
   This smooths the non-smooth min objective and tightens constraints.
3. Final high-precision polish using direct constraint projection.
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


def _anneal_points(pts, deadline):
    """Multi-stage annealing optimization on full point coordinates."""
    current = pts.copy()
    current_score = _score(current)
    
    # Progressive beta values for annealing
    betas = [5.0, 15.0, 40.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0]
    
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
                    'maxiter': 800, 
                    'ftol': 1e-15, 
                    'gtol': 1e-13,
                    'maxfun': 20000
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


def construct():
    t0 = time.time()
    deadline = t0 + 120.0 - DEADLINE_MARGIN
    
    best_points = None
    best_score = -1.0
    
    # Standard ring topology for 50 circles: 1 + 6 + 12 + 18 + 13
    ring_sizes = [1, 6, 12, 18, 13]
    
    # Initial guess for radii (roughly proportional to sqrt(count) or just evenly spaced)
    # Using known good approximations
    init_radii = np.array([0.0, 0.25, 0.45, 0.65, 0.85])
    
    # Initialize angles
    angles_list = []
    for i, n in enumerate(ring_sizes):
        if i == 0:
            angles_list.append(np.array([0.0]))
        else:
            # Start with uniform distribution
            angles_list.append(np.linspace(0, 2 * np.pi, n, endpoint=False))
    
    # ---- Stage 1: Differential Evolution on Ring Parameters ----
    # This searches over radii and individual angles to find a good topology
    if time.time() < deadline - 50:
        bounds = [(0.0, 0.95)] * len(ring_sizes)
        for i in range(1, len(ring_sizes)):
            bounds.extend([(0.0, 2.0 * np.pi)] * ring_sizes[i])
            
        def ring_obj(vec):
            if time.time() > deadline - 50:
                return 1e10
            radii, al = _unflatten_ring(vec, ring_sizes)
            # Clip radii to valid range
            radii = np.clip(radii, 0.0, 0.95)
            pts = _ring_to_points(radii, al)
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
                maxiter=150, 
                popsize=20, 
                tol=1e-9,
                mutation=(0.3, 1.0), 
                recombination=0.8, 
                seed=42, 
                polish=False
            )
            if time.time() < deadline - 40:
                ring_x = res_de.x
                radii_opt, al_opt = _unflatten_ring(ring_x, ring_sizes)
                radii_opt = np.clip(radii_opt, 0.0, 0.95)
                de_pts = _ring_to_points(radii_opt, al_opt)
                de_score = _score(de_pts)
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_pts.copy()
        except Exception:
            pass
    
    # If DE failed or didn't improve, use initial ring setup
    if best_points is None or best_score < 0.1:
        init_pts = _ring_to_points(init_radii, angles_list)
        init_score = _score(init_pts)
        if init_score > best_score:
            best_score = init_score
            best_points = init_pts.copy()

    # ---- Stage 2: Multi-stage Annealing Refinement ----
    if best_points is not None and time.time() < deadline - 10:
        refined_pts, refined_score = _anneal_points(best_points, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # ---- Stage 3: Final Constraint Push Polish ----
    if best_points is not None and time.time() < deadline - 2:
        current_r = _score(best_points)
        # Try to push slightly above current radius
        target_r = current_r + 1e-6
        polished_pts = _constraint_push(best_points, target_r, deadline, max_iters=200)
        polished_score = _score(polished_pts)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts.copy()

    # Fallback if something went wrong
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _ring_to_points(init_radii, angles_list)
        best_score = _score(best_points)
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
