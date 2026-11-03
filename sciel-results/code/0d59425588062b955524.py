"""Packing 300 equal circles in a unit circle using QP-style optimization."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import cKDTree


def _compute_radius(pts):
    """Compute the packing radius for given points."""
    if len(pts) < 2:
        return 0.0
    # Use KDTree for efficiency
    tree = cKDTree(pts)
    # Get nearest neighbors
    distances, _ = tree.query(pts, k=2)
    min_pair_dist = distances[:, 1].min()
    min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
    return min(min_pair_dist / 2.0, min_boundary)


def _compute_radius_full(pts):
    """Compute packing radius with full pairwise check (for verification)."""
    if len(pts) < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair_dist / 2.0, min_boundary)


def _init_hexagonal(spacing):
    """Generate hexagonal lattice points within unit disk."""
    k = int(1.0 / spacing) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - spacing / 2.0:
                points.append((x, y))
    return np.array(points) if points else np.empty((0, 2))


def _qp_optimize(pts, deadline, max_iter=2000, tol=1e-8):
    """
    QP-style optimization: maximize minimum squared distance.
    Uses SLSQP with auxiliary variable t for min squared distance.
    """
    n = len(pts)
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Target minimum squared distance (2*r)^2
    # We'll use a surrogate: maximize t where t <= min_ij ||x_i - x_j||^2 and t <= min_i (1-||x_i||)^2
    
    # SLSQP formulation:
    # Variables: x_1, ..., x_n (2n variables) + t (1 variable) = 2n+1 total
    # Maximize t
    # Subject to:
    #   ||x_i - x_j||^2 >= t for all i<j
    #   (1 - ||x_i||)^2 >= t for all i
    #   ||x_i|| <= 1 - t/2 (approximate boundary)
    
    # For SLSQP, we minimize -t
    # Constraints are nonlinear, so we use SLSQP with numerical gradients
    
    def objective(v):
        # v = [x1, y1, x2, y2, ..., xn, yn, t]
        t = v[-1]
        return -t
    
    def constraints(v):
        t = v[-1]
        pts_v = v[:-1].reshape(n, 2)
        cons = []
        
        # Pairwise constraints: ||x_i - x_j||^2 >= t
        # Only check nearby pairs for efficiency
        for i in range(n):
            for j in range(i+1, n):
                di = pts_v[i] - pts_v[j]
                cons.append(np.sum(di**2) - t)
        
        # Boundary constraints: (1 - ||x_i||)^2 >= t
        for i in range(n):
            norm_i = np.linalg.norm(pts_v[i])
            cons.append((1.0 - norm_i)**2 - t)
        
        return cons
    
    # This is too many constraints for SLSQP with 300 points
    # Instead, use a different approach: gradient ascent on a smooth surrogate
    
    # Better approach: use the soft-min surrogate with SLSQP
    # Minimize sum of exp(-eps * (dist_ij - t)) for all pairs + boundary terms
    
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Multi-start with different eps values
    eps_list = [20.0, 50.0, 100.0, 200.0]
    
    for eps in eps_list:
        if time.time() > deadline:
            break
        
        def soft_objective(v):
            pts_v = v.reshape(n, 2)
            # Pairwise terms
            total = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(pts_v[i] - pts_v[j])
                    total += np.exp(-eps * d)
            # Boundary terms
            for i in range(n):
                b = 1.0 - np.linalg.norm(pts_v[i])
                total += np.exp(-eps * b)
            return total
        
        def soft_gradient(v):
            pts_v = v.reshape(n, 2)
            grad = np.zeros_like(v)
            for i in range(n):
                for j in range(i+1, n):
                    diff = pts_v[i] - pts_v[j]
                    d = np.linalg.norm(diff)
                    if d < 1e-12:
                        continue
                    w = eps * np.exp(-eps * d)
                    unit = diff / d
                    grad[2*i:2*i+2] += w * unit
                    grad[2*j:2*j+2] -= w * unit
                # Boundary
                norm_i = np.linalg.norm(pts_v[i])
                if norm_i < 1e-12:
                    continue
                b = 1.0 - norm_i
                w = eps * np.exp(-eps * b)
                unit_r = pts_v[i] / norm_i
                grad[2*i:2*i+2] += w * unit_r
            return grad
        
        # Use L-BFGS-B for the smooth surrogate
        try:
            result = minimize(
                soft_objective,
                best_pts.flatten(),
                jac=soft_gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < soft_objective(best_pts.flatten()):
                cand_pts = result.x.reshape(n, 2)
                # Enforce boundary
                norms = np.linalg.norm(cand_pts, axis=1)
                too_far = norms > 0.9999
                if np.any(too_far):
                    cand_pts[too_far] *= 0.9999 / norms[too_far]
                cand_r = _compute_radius(cand_pts)
                if cand_r > best_r:
                    best_r = cand_r
                    best_pts = cand_pts.copy()
        except Exception:
            pass
    
    return best_pts, best_r


def _gradient_ascent(pts, deadline, lr=0.001, max_iter=3000):
    """Direct gradient ascent on the radius objective using subgradient."""
    n = len(pts)
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Find the bottleneck: min pair distance or min boundary distance
        tree = cKDTree(best_pts)
        dists, idxs = tree.query(best_pts, k=2)
        min_pair_idx = np.argmin(dists[:, 1])
        min_pair_d = dists[min_pair_idx, 1]
        
        norms = np.linalg.norm(best_pts, axis=1)
        min_boundary_idx = np.argmin(1.0 - norms)
        min_boundary_d = 1.0 - norms[min_boundary_idx]
        
        # Subgradient: push apart the closest pair, push inward the closest to boundary
        grad = np.zeros_like(best_pts)
        
        # Pair gradient
        i, j = min_pair_idx, idxs[min_pair_idx, 1]
        diff = best_pts[i] - best_pts[j]
        d = min_pair_d
        if d > 1e-12:
            unit = diff / d
            grad[i] += unit
            grad[j] -= unit
        
        # Boundary gradient
        norm_b = norms[min_boundary_idx]
        if norm_b > 1e-12:
            unit_r = best_pts[min_boundary_idx] / norm_b
            grad[min_boundary_idx] -= unit_r
        
        # Normalize and update
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad /= grad_norm
        
        candidate = best_pts + lr * grad
        
        # Enforce boundary
        norms_cand = np.linalg.norm(candidate, axis=1)
        too_far = norms_cand > 0.9999
        if np.any(too_far):
            candidate[too_far] *= 0.9999 / norms_cand[too_far]
        
        cand_r = _compute_radius(candidate)
        if cand_r > best_r:
            best_r = cand_r
            best_pts = candidate.copy()
        else:
            lr *= 0.99
    
    return best_pts, best_r


def _local_refine(pts, deadline, neighborhood_size=10):
    """Refine local neighborhoods to improve packing."""
    n = len(pts)
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Find dense regions and try to rearrange locally
    tree = cKDTree(best_pts)
    
    for center_idx in range(n):
        if time.time() > deadline:
            break
        
        # Get neighbors
        _, neighbor_idxs = tree.query(best_pts[center_idx], k=min(neighborhood_size+1, n))
        neighbor_idxs = neighbor_idxs[1:]  # Exclude self
        
        # Try small perturbations of this neighborhood
        for _ in range(20):
            perturbation = np.random.normal(0, 0.005, size=(len(neighbor_idxs), 2))
            candidate = best_pts.copy()
            candidate[neighbor_idxs] += perturbation
            
            # Enforce boundary
            norms = np.linalg.norm(candidate, axis=1)
            too_far = norms > 0.9999
            if np.any(too_far):
                candidate[too_far] *= 0.9999 / norms[too_far]
            
            cand_r = _compute_radius(candidate)
            if cand_r > best_r:
                best_r = cand_r
                best_pts = candidate.copy()
                # Update tree
                tree = cKDTree(best_pts)
    
    return best_pts, best_r


def construct():
    deadline = time.time() + 145.0  # Safety margin for 150s limit
    
    # Phase 0: Search over hexagonal spacings for good initialization
    best_pts = None
    best_r = 0.0
    
    spacings = np.linspace(0.09, 0.13, 30)
    for spacing in spacings:
        if time.time() > deadline:
            break
        pts = _init_hexagonal(spacing)
        if len(pts) < N:
            continue
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists, kind='stable')[:N]
        candidate = pts[idx]
        r = _compute_radius(candidate)
        if r > best_r:
            best_r = r
            best_pts = candidate.copy()
    
    if best_pts is None:
        # Fallback
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points = points[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                best_pts = points[np.argsort(radius, kind="stable")[:N]]
                best_r = _compute_radius(best_pts)
                break
            spacing *= 0.98
    
    # Phase 1: QP-style optimization with SLSQP/L-BFGS-B
    if time.time() < deadline - 60.0:
        best_pts, best_r = _qp_optimize(
            best_pts,
            deadline=deadline - 60.0,
            max_iter=2000,
            tol=1e-8
        )
    
    # Phase 2: Gradient ascent with subgradient
    if time.time() < deadline - 30.0:
        best_pts, best_r = _gradient_ascent(
            best_pts,
            deadline=deadline - 30.0,
            lr=0.002,
            max_iter=2000
        )
    
    # Phase 3: Local neighborhood refinement
    if time.time() < deadline - 15.0:
        best_pts, best_r = _local_refine(
            best_pts,
            deadline=deadline - 15.0,
            neighborhood_size=8
        )
    
    # Phase 4: Final polish with small steps
    if time.time() < deadline - 5.0:
        for lr in [1e-4, 5e-5, 1e-5]:
            if time.time() > deadline - 5.0:
                break
            best_pts, best_r = _gradient_ascent(
                best_pts,
                deadline=deadline - 5.0,
                lr=lr,
                max_iter=500
            )
    
    return best_pts
# EVOLVE-BLOCK-END
