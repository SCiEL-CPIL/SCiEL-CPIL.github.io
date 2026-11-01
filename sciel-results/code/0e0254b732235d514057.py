"""Pack 300 equal circles in the unit disk; maximise the common radius."""
import numpy as np
from scipy.optimize import minimize
import time

N = 300
# Best known reference radius for N=300 (Lai et al. 2021)
R_REF = 0.0531542991469


def _score(pts):
    """Return the feasible common radius for a set of centres (N,2)."""
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # Container constraint: 1 - |c_i|
    rad = np.hypot(pts[:, 0], pts[:, 1])
    r_cont = 1.0 - np.max(rad)
    # Pairwise constraint: min |c_i - c_j| / 2
    # Use a binned approach or full pairwise for 300 points (44850 pairs) - fine.
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    return min(r_cont, r_pair)


def _hexagonal_init():
    """Generate a hexagonal lattice of ~N points inside the unit disk."""
    # Estimate spacing for 300 points in a disk of radius 1.
    # Area per hex point ~ (sqrt(3)/2) * s^2. Total area ~ pi.
    # N * (sqrt(3)/2) * s^2 ~ pi  => s ~ sqrt(2*pi/(N*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    # Start with a slightly larger spacing to ensure we get at least N, then adjust.
    # Actually, we want exactly N. Let's generate a grid and pick the closest N to center.
    k = int(1.0 / s) + 3
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = i * s + (0.5 * s if j % 2 != 0 else 0.0)
            y = j * s * np.sqrt(3.0) / 2.0
            if x * x + y * y < 1.0:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) < N:
        # If we didn't get enough, reduce spacing
        s *= 0.9
        return _hexagonal_init()
    # Sort by distance to center and take N
    d = np.hypot(pts[:, 0], pts[:, 1])
    idx = np.argsort(d)
    return pts[idx[:N]]


def _local_optimize(pts, max_iter=500, tol=1e-12):
    """
    Local optimization using gradient ascent on the smooth surrogate.
    We maximize f(x) = min(1 - |x_i|, min_{i<j} |x_i - x_j|/2).
    To make it smooth, we can use a log-sum-exp approximation for the min,
    or simply use a penalty method / direct gradient of the active set.
    
    A simpler and robust approach for circle packing is to use a force-directed
    relaxation or a specific optimization library. Since SciPy is available,
    let's use `minimize` with a negative objective (maximization) and a
    smooth approximation of the min function:
    min(a, b) ≈ -log(exp(-a/tau) + exp(-b/tau)) * tau, as tau -> 0.
    
    However, implementing a custom gradient for the min is tricky.
    Alternative: Use `scipy.optimize.differential_evolution`? No, too slow for 600D.
    
    Let's use a simple iterative repulsion method (like molecular dynamics)
    or a gradient-based method where we compute the gradient of the objective
    using the active constraints.
    
    Actually, a very effective method for this is to use `scipy.optimize.minimize`
    with the objective being the negative of the soft-min, or simply maximizing
    the minimum distance.
    
    Let's try a gradient ascent approach where at each step, we compute the
    current radius r, identify the "bottleneck" constraints, and move points
    in the direction that increases r.
    
    For a more robust and standard approach, let's use `scipy.optimize.minimize`
    with the method 'L-BFGS-B' or 'Powell' on the negative of the radius.
    But the radius function is non-smooth.
    
    Let's use a soft-min approach:
    r_soft = -tau * log( sum(exp(-c_i/tau)) )
    where c_i are the constraint values (1-|x_i| and |x_i-x_j|/2).
    This is smooth and differentiable.
    """
    tau = 0.01  # Temperature for soft-min
    best_pts = pts.copy()
    best_r = _score(pts)
    
    def objective(x):
        pts_flat = x.reshape(N, 2)
        # Compute constraint values
        rad = np.hypot(pts_flat[:, 0], pts_flat[:, 1])
        c_cont = 1.0 - rad  # (N,)
        
        # Pairwise distances
        diff = pts_flat[:, None, :] - pts_flat[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        c_pair = dist / 2.0  # (N, N)
        
        # Soft-min of all constraints
        # To avoid overflow, use the log-sum-exp trick
        all_c = np.concatenate([c_cont, c_pair.flatten()])
        # Filter out non-finite or very small values if necessary, but soft-min handles it.
        m = np.min(all_c)
        if m < -1000:  # Safety check
            return -m  # Return a large negative value (bad score)
        
        # r_soft = m - tau * log( sum(exp((c_i - m)/tau)) )
        # We want to maximize r_soft, so minimize -r_soft
        sum_exp = np.sum(np.exp((all_c - m) / tau))
        r_soft = m - tau * np.log(sum_exp)
        return -r_soft
    
    # Use L-BFGS-B with bounds? No explicit bounds needed if we penalize going outside,
    # but the soft-min will naturally discourage it.
    # However, L-BFGS-B requires bounds. Let's use 'BFGS' or 'Powell'.
    # 'Powell' is derivative-free and robust for non-smooth functions, but slow.
    # 'L-BFGS-B' is faster but needs smoothness. Our soft-min is smooth.
    # Let's provide bounds to keep points in the disk.
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        res = minimize(objective, pts.flatten(), method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
        if res.success and _score(res.x.reshape(N, 2)) > best_r:
            best_pts = res.x.reshape(N, 2)
            best_r = _score(best_pts)
    except Exception:
        pass
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Initial guess
    pts = _hexagonal_init()
    r = _score(pts)
    
    # 2. Local optimization
    pts, r = _local_optimize(pts)
    
    # 3. If time remains, try a few more random restarts or perturbations
    # For N=300, local opt is usually good enough if the init is good.
    # Let's add a simple perturbation and re-optimize if time permits.
    if time.time() < deadline - 20.0:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            # Perturb the current best
            noise = rng.normal(0, 0.01, size=(N, 2))
            perturbed = pts + noise
            # Ensure points are inside the disk (simple projection)
            rad = np.hypot(perturbed[:, 0], perturbed[:, 1])
            mask = rad > 0.99
            perturbed[mask] = perturbed[mask] * (0.99 / rad[mask])[:, None]
            
            p_opt, r_opt = _local_optimize(perturbed, max_iter=200)
            if r_opt > r:
                pts = p_opt
                r = r_opt
                
    # Final check
    final_r = _score(pts)
    if final_r > r:
        r = final_r
    else:
        # If the last step made it worse, keep the previous best? 
        # _local_optimize returns the best found within it.
        pass
        
    return pts


# Run a quick self-test if executed directly
if __name__ == "__main__":
    pts = construct()
    print("Radius:", _score(pts))
