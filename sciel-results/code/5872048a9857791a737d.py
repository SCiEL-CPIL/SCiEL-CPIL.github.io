import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _hex_lattice_init(n):
    """Generate initial points using a hexagonal lattice."""
    # Estimate spacing
    # Area of unit circle is pi. Area per circle ~ pi * r^2.
    # r ~ 0.125. Area ~ 0.049.
    # Total area 50 * 0.049 = 2.45. Pi = 3.14. Density ~ 0.78.
    # Hexagonal packing density is 0.906.
    # Spacing s such that hex cell area ~ s^2 * sqrt(3)/2.
    # Let's just use the seed logic which works well.
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= n:
            # Return the n points closest to center for better symmetry/start
            return points[np.argsort(radius, kind="stable")[:n]]
        spacing *= 0.98

def _objective_and_grad(x, r, eps=0.01):
    """
    Calculate smooth overlap energy and gradient.
    x: (N, 2) array of centers.
    r: target radius.
    eps: smoothing parameter for softplus.
    """
    n = len(x)
    x = x.reshape(n, 2)
    
    # Compute all pairwise distances
    # diff[i, j] = x[i] - x[j]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf) # Ignore self
    
    # Boundary distances
    dist_bound = 1.0 - np.linalg.norm(x, axis=1)
    
    # Violations
    # Pairwise: 2r - dist
    viol_pair = 2.0 * r - dist
    # Boundary: dist_bound - r  (We want dist_bound >= r, so violation is r - dist_bound)
    viol_bound = r - dist_bound
    
    # Smooth max(0, v) using softplus: (1/eps) * log(1 + exp(eps * v))
    # To avoid overflow, use np.logaddexp(0, eps*v) / eps
    def softplus(v):
        return np.logaddexp(0, eps * v) / eps
    
    # Energy is sum of softplus violations
    # We only care about violations where v > 0, but softplus handles it smoothly.
    # Note: softplus(0) = log(2)/eps which is not 0. 
    # To make it 0 at v=0, we should use: softplus(v) - softplus(0)
    # Or just accept the offset, it's a constant.
    # Better: use a function that is 0 for v<=0 and smooth for v>0.
    # f(v) = max(0, v) is kinked.
    # Let's use: (v + softplus(-v)) ? No.
    # Standard smooth max(0, v) is: v * sigmoid(k*v) ? 
    # Let's stick to minimizing the sum of squared violations but smoothed.
    # Actually, just minimizing the sum of softplus violations works.
    # The constant offset doesn't change the gradient direction.
    
    E_pair = np.sum(softplus(viol_pair))
    E_bound = np.sum(softplus(viol_bound))
    E = E_pair + E_bound
    
    # Gradient
    # d/dx softplus(eps * (2r - dist))
    # Let u = 2r - dist.
    # dE/du = sigmoid(eps * u)  (since d/dx log(1+e^x) = e^x/(1+e^x) = sigmoid(x))
    # Wait, d/dv [ (1/eps) log(1+exp(eps v)) ] = (1/eps) * (eps exp(eps v) / (1+exp(eps v))) = sigmoid(eps v).
    
    # Gradient of dist w.r.t x_i: (x_i - x_j) / dist_ij
    # Gradient of viol_pair w.r.t x_i: - (x_i - x_j) / dist_ij
    
    grad = np.zeros_like(x)
    
    # Pairwise gradient
    # sigmoid_term[i,j] = sigmoid(eps * (2r - dist[i,j]))
    sig_pair = np.exp(eps * viol_pair) / (1 + np.exp(eps * viol_pair))
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    sig_pair[~mask] = 0.0
    
    # dist is (N, N). diff is (N, N, 2).
    # grad_ij = sig_pair[i,j] * (x_i - x_j) / dist[i,j]
    # Note: viol = 2r - dist. d(viol)/d(x_i) = - (x_i - x_j)/dist.
    # So dE/dx_i = sum_j sig_pair[i,j] * (- (x_i - x_j) / dist[i,j])
    
    # To avoid division by zero on diagonal, dist is inf there, and sig_pair is 0.
    # But 0 * inf is nan. We must handle this.
    # We already zeroed sig_pair on diagonal.
    # We should also replace dist with 1 where it is 0 or inf to prevent warnings, though mask handles it.
    safe_dist = np.where(mask, dist, 1.0)
    
    # grad_pair[i] = - sum_j sig_pair[i,j] * (x_i - x_j) / safe_dist[i,j]
    # (x_i - x_j) is diff[i,j].
    # term = sig_pair[:, :, np.newaxis] * diff / safe_dist[:, :, np.newaxis]
    # grad_pair = -np.sum(term, axis=1)
    
    term = sig_pair[:, :, np.newaxis] * diff / safe_dist[:, :, np.newaxis]
    grad_pair = -np.sum(term, axis=1)
    
    # Boundary gradient
    # viol_bound = r - dist_bound
    # dist_bound = 1 - |x_i|
    # d(viol_bound)/d(x_i) = - d(dist_bound)/d(x_i) = - (-x_i/|x_i|) = x_i/|x_i|
    # dE/dx_i = sig_bound[i] * (x_i / |x_i|)
    
    sig_bound = np.exp(eps * viol_bound) / (1 + np.exp(eps * viol_bound))
    
    norm_x = np.linalg.norm(x, axis=1, keepdims=True)
    safe_norm_x = np.where(norm_x > 1e-6, norm_x, 1.0)
    unit_x = x / safe_norm_x
    
    grad_bound = sig_bound[:, np.newaxis] * unit_x
    
    grad = grad_pair + grad_bound
    
    return E, grad

def construct():
    start_time = time.time()
    timeout = 140.0  # Leave 10s margin
    
    # Initial points
    points = _hex_lattice_init(N)
    
    # Start radius slightly below known best to ensure initial validity
    # Seed score 1.0 corresponds to ~0.1258.
    # Let's start at 0.12
    r = 0.12
    best_points = points.copy()
    best_r = 0.0
    
    # Check initial validity
    def get_score(pts):
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        min_b = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_d / 2.0, min_b)
    
    initial_score = get_score(points)
    if initial_score > best_r:
        best_r = initial_score
        best_points = points.copy()
    
    r_step = 0.0005
    eps = 0.05 # Smoothing parameter
    
    while time.time() - start_time < timeout:
        # Try to optimize for current r
        # Flatten points
        x0 = points.flatten()
        
        # Define objective for scipy
        def obj(x):
            pts = x.reshape(-1, 2)
            val, _ = _objective_and_grad(pts, r, eps)
            return val
        
        def grad(x):
            pts = x.reshape(-1, 2)
            _, g = _objective_and_grad(pts, r, eps)
            return g.flatten()
        
        try:
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e-4: # Check if we found a valid configuration
                # Verify validity with hard check
                new_points = res.x.reshape(-1, 2)
                score = get_score(new_points)
                
                if score >= r - 1e-5: # If score is close to r, it's valid for radius r
                    # Update best
                    if score > best_r:
                        best_r = score
                        best_points = new_points.copy()
                    
                    # Increase r
                    r += r_step
                    points = new_points.copy()
                else:
                    # Not valid for this r, try to fix with smaller step or break
                    # If we can't even satisfy the current r, we might be stuck.
                    # Try a smaller step? Or just break if we've been stuck.
                    # For simplicity, if we fail to find a valid config for r, 
                    # we should probably not increase r. 
                    # But we might have moved points to a better local min for a slightly smaller r.
                    # Let's just break if we can't satisfy the current r, 
                    # assuming we've converged to the local optimum for this basin.
                    break
            else:
                # Optimization failed to converge, but maybe it improved things?
                # Check if the result is valid for a *smaller* radius?
                # Actually, if it didn't converge, it might be invalid.
                # Let's check the current points (res.x) score.
                new_points = res.x.reshape(-1, 2)
                score = get_score(new_points)
                if score > best_r:
                    best_r = score
                    best_points = new_points.copy()
                # If we can't satisfy current r, we likely can't go higher.
                break
                
        except Exception as e:
            # If optimization crashes, keep current best
            break
            
        # Adjust step size if we are making good progress?
        # Keep it simple for now.
        
    return best_points
