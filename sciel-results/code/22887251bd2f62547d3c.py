# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import cvxpy as cp

def _evaluate(points: np.ndarray) -> float:
    """Compute the metric (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def _smooth_metric(points: np.ndarray, alpha=10.0) -> float:
    """
    Smoothed objective to maximize.
    Uses soft-min and soft-max to approximate d_min and d_max smoothly.
    We want to maximize (d_min / d_max)^2.
    Equivalent to minimizing -2 * (log(d_min) - log(d_max)).
    """
    dists = pdist(points)
    # Soft min: approximates min(dists)
    # log-sum-exp trick for numerical stability
    m = np.min(dists)
    soft_min = m + np.log(np.sum(np.exp(dists - m)) / len(dists))
    # Soft max: approximates max(dists)
    M = np.max(dists)
    soft_max = M - np.log(np.sum(np.exp(M - dists)) / len(dists))
    
    if soft_min <= 0 or soft_max <= 0:
        return 0.0
    
    # Return the squared ratio
    return (soft_min / soft_max) ** 2

def _solve_sdp(n, d, start_time, deadline):
    """
    Solve SDP relaxation to maximize d_min subject to d_max <= 1.
    Returns points or None.
    """
    # Variables
    # G is the Gram matrix of size (n+1) x (n+1)
    # We add a dummy point at the origin to center the configuration?
    # Actually, standard embedding: x_i are points in R^d.
    # G_ij = x_i . x_j.
    # dist(i,j)^2 = G_ii + G_jj - 2 G_ij.
    # We want to maximize t such that dist(i,j)^2 >= t^2 for all i!=j.
    # And dist(i,j)^2 <= 1 for all i!=j.
    # Also G must be PSD and rank(G) <= d.
    
    # To handle the rank constraint in SDP, we can't enforce it directly.
    # However, if we fix the dimension d, the optimal solution for maximizing
    # min distance with bounded max distance often lies in a lower dimensional
    # subspace or is well-approximated by the PSD relaxation if we allow
    # the Gram matrix to be any PSD matrix, then project to rank d.
    
    # Let's use a simpler formulation:
    # Maximize t
    # Subject to:
    #   G_ii + G_jj - 2 G_ij >= t^2 for all i != j
    #   G_ii + G_jj - 2 G_ij <= 1 for all i != j
    #   G >= 0 (PSD)
    #   G_ii <= 1 (implied by dist <= 1 if centered? No. dist(0,i)^2 = G_ii if 0 is origin.
    #   Let's assume points are centered at 0?
    #   If we don't center, G is not unique.
    #   Let's enforce that the points are centered at the origin: sum x_i = 0.
    #   This implies G * 1 = 0.
    
    # Let's try without centering first, but fix one point at origin?
    # If x_1 = 0, then G_1j = 0 for all j, G_11 = 0.
    # dist(i,j)^2 = G_ii + G_jj - 2 G_ij.
    # dist(1,j)^2 = G_jj.
    # So G_jj <= 1.
    # G >= 0.
    
    G = cp.Variable((n, n), symmetric=True)
    t = cp.Variable()
    
    constraints = [G >> 0]
    
    # Fix point 0 at origin
    constraints += [G[0, :] == 0]
    constraints += [G[:, 0] == 0]
    constraints += [G[0, 0] == 0]
    
    # Distances squared
    # D_ij = G_ii + G_jj - 2 G_ij
    # For i=0, D_0j = G_jj.
    
    # Maximize t
    # Minimize -t
    objective = cp.Maximize(t)
    
    # Constraints on distances
    # t^2 <= D_ij <= 1
    # Since t is variable, t^2 is non-convex if t can be negative.
    # We assume t > 0.
    # t^2 <= D_ij is convex (quadratic constraint).
    # D_ij <= 1 is linear.
    
    # Note: t^2 <= D_ij is equivalent to (t, 1, D_ij) in SOC?
    # No, t^2 <= D_ij is a second order cone constraint if D_ij is affine in variables.
    # D_ij is linear in G.
    # So t^2 <= G_ii + G_jj - 2 G_ij is a valid convex constraint.
    
    for i in range(n):
        for j in range(i + 1, n):
            d2 = G[i, i] + G[j, j] - 2 * G[i, j]
            constraints += [d2 >= t**2]
            constraints += [d2 <= 1]
            
    prob = cp.Problem(objective, constraints)
    
    # Solve
    try:
        # Use SCS or MOSEK if available. SCS is default.
        prob.solve(solver=cp.SCS, max_iters=10000, verbose=False)
        
        if time.time() > deadline - 30:
            return None
            
        if prob.status in ["optimal", "optimal_inaccurate"]:
            G_val = G.value
            if G_val is None:
                return None
                
            # Ensure PSD
            G_val = (G_val + G_val.T) / 2
            # Project to PSD
            eigvals, eigvecs = np.linalg.eigh(G_val)
            eigvals = np.maximum(eigvals, 0)
            G_val = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Project to rank d
            # Truncate eigenvalues
            eigvals, eigvecs = np.linalg.eigh(G_val)
            # Keep top d eigenvalues
            idx = np.argsort(eigvals)[::-1][:d]
            eigvals_d = np.zeros_like(eigvals)
            eigvals_d[idx] = eigvals[idx]
            G_rank_d = eigvecs @ np.diag(eigvals_d) @ eigvecs.T
            
            # Recover points
            # G_rank_d = X X^T
            # Use SVD
            U, s, Vt = np.linalg.svd(G_rank_d)
            X = U[:, :d] * np.sqrt(s[:d])
            
            # Center points (should be near 0 if we fixed one at origin? 
            # The rank projection might shift the center.
            # Let's center to be safe.
            X -= np.mean(X, axis=0, keepdims=True)
            
            # Normalize to ensure d_max <= 1?
            # The SDP enforced d2 <= 1.
            # But numerical errors or rank projection might violate this slightly.
            dists = pdist(X)
            d_max = np.max(dists)
            if d_max > 0:
                X = X / d_max
                
            return X
    except Exception:
        return None

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600
    
    best_points = None
    best_score = 0.0
    
    def update_best(pts):
        nonlocal best_points, best_score
        try:
            pts = np.asarray(pts, dtype=float)
            if pts.shape != (n, d):
                return
            if not np.all(np.isfinite(pts)):
                return
            score = _evaluate(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
        except Exception:
            pass

    # Phase 1: SDP Relaxation
    # The SDP gives a lower bound on the optimal t.
    # The resulting point set might have rank < d or slightly violate constraints.
    # We use it as a high-quality starting point.
    
    sdp_points = _solve_sdp(n, d, start_time, deadline)
    if sdp_points is not None:
        update_best(sdp_points)
        
    # Phase 2: Local Refinement
    # Start from SDP result or a good heuristic (Icosahedron + 2)
    
    # Fallback initial: Icosahedron + 2 random
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / norms
    
    np.random.seed(42)
    extra = np.random.randn(2, d)
    ico_full = np.vstack([ico, extra])
    update_best(ico_full)
    
    # Refine the best candidate found so far
    if best_points is not None:
        current_pts = best_points.copy()
        current_score = best_score
        
        # Local search using Nelder-Mead on the negative log-ratio?
        # Or just simple coordinate descent / random perturbation.
        
        # Let's use scipy.optimize.minimize with a smooth objective
        # Objective: - (d_min / d_max)^2
        # To make it smooth, we use soft-min/max.
        
        def smooth_obj(x_flat):
            pts = x_flat.reshape(n, d)
            return -_smooth_metric(pts, alpha=20.0)
            
        # Gradient-free Nelder-Mead is robust
        # But it's slow. Let's try BFGS with numerical gradient.
        
        def num_grad(x_flat):
            eps = 1e-6
            grad = np.zeros_like(x_flat)
            for i in range(len(x_flat)):
                x_plus = x_flat.copy()
                x_minus = x_flat.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                grad[i] = (smooth_obj(x_plus) - smooth_obj(x_minus)) / (2 * eps)
            return grad

        # Run refinement
        x0 = current_pts.flatten()
        
        # Multiple restarts from slightly perturbed versions of the best point
        for r in range(5):
            if time.time() > deadline - 60:
                break
                
            if r == 0:
                x_start = x0
            else:
                # Perturb
                noise = np.random.randn(n, d) * 0.1
                x_start = (current_pts + noise).flatten()
                
            try:
                res = minimize(smooth_obj, x_start, method='BFGS', jac=num_grad,
                               options={'maxiter': 500, 'gtol': 1e-8, 'disp': False})
                x_opt = res.x
                pts_opt = x_opt.reshape(n, d)
                update_best(pts_opt)
                
                if best_points is not None:
                    current_pts = best_points.copy()
            except Exception:
                pass

    # Phase 3: Final Polish with fine-grained local search
    if best_points is not None and time.time() < deadline - 10:
        fine_pts = best_points.copy()
        fine_score = _evaluate(fine_pts)
        step_size = 0.01
        np.random.seed(42)
        
        for i in range(5000):
            if time.time() > deadline - 5:
                break
            
            idx = np.random.randint(n)
            pert = np.random.randn(d) * step_size
            new_pts = fine_pts.copy()
            new_pts[idx] += pert
            
            new_score = _evaluate(new_pts)
            
            if new_score > fine_score:
                fine_pts = new_pts
                fine_score = new_score
                step_size = max(step_size * 0.98, 1e-6)
            else:
                step_size = max(step_size * 0.9, 1e-6)
                
        update_best(fine_pts)

    # Fallback
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)
        update_best(best_points)
        
    return best_points

# EVOLVE-BLOCK-END
