"""Circle packing in unit square: 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _initial_grid(n):
    """Generate an initial grid arrangement of n circles in the unit square."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95  # slightly smaller to avoid boundary issues
            cells.append([x, y, r])
    return np.array(cells[:n])

def _soft_min_penalty(distances, threshold, temperature=0.01):
    """
    Soft-min penalty: penalize when any distance is below threshold.
    Uses log-sum-exp for smoothness.
    """
    violations = threshold - distances
    # Only penalize positive violations (where dist < threshold)
    # Use softplus-like: log(1 + exp(x/temperature)) * temperature
    # But we want penalty only when violations > 0
    # penalty = temperature * logsumexp(violations/temperature) - threshold * (something)
    # Simpler: use sum of softplus(violations)
    # softplus(x) = log(1 + exp(x))
    # For stability:
    max_v = np.max(violations)
    if max_v < -50:
        return 0.0
    shifted = violations - max_v
    exp_shifted = np.exp(shifted)
    logsumexp_val = max_v + np.log(np.sum(exp_shifted))
    # This gives log(1+exp(v)) summed... actually this is logsumexp of all violations
    # We want sum of softplus(v_i) where softplus(v) = log(1+exp(v))
    # softplus(v) = log(1+exp(v))
    # For numerical stability with large v:
    penalties = []
    for v in violations:
        if v > 50:
            penalties.append(v)
        elif v < -50:
            penalties.append(0.0)
        else:
            penalties.append(np.log1p(np.exp(v)))
    return np.sum(penalties)

def _objective(params, n, temperature=0.01):
    """
    Objective function: minimize penalty for overlaps and boundary violations.
    params: flattened array of [x1, y1, r1, x2, y2, r2, ...]
    """
    pts = params.reshape(n, 3)
    xs = pts[:, 0]
    ys = pts[:, 1]
    rs = pts[:, 2]
    
    # Boundary constraints: circle must be inside [0,1]^2
    # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    boundary_violations = np.maximum(0, -xs + rs) + np.maximum(0, xs + rs - 1) + \
                          np.maximum(0, -ys + rs) + np.maximum(0, ys + rs - 1)
    boundary_penalty = np.sum(boundary_violations ** 2)
    
    # Overlap constraints: for all i < j, dist(i,j) >= r_i + r_j
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            required = rs[i] + rs[j]
            violation = required - dist
            if violation > 0:
                overlap_penalty += violation ** 2
    
    # We want to maximize sum of radii, so minimize negative sum of radii + penalties
    return -np.sum(rs) + 10.0 * boundary_penalty + 10.0 * overlap_penalty

def _gradient(params, n):
    """Compute gradient of the objective function."""
    pts = params.reshape(n, 3)
    xs = pts[:, 0]
    ys = pts[:, 1]
    rs = pts[:, 2]
    
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r)
    grad[2::3] = -1.0
    
    # Boundary penalty gradient
    boundary_penalty = 0.0
    for i in range(n):
        # x - r >= 0: violation = -(x - r) = r - x
        if xs[i] - rs[i] < 0:
            v = rs[i] - xs[i]
            boundary_penalty += 2 * v
            grad[3*i] += -2 * v  # d/dx of (r-x)^2 = -2(r-x)
            grad[3*i+2] += 2 * v  # d/dr of (r-x)^2 = 2(r-x)
        
        # x + r <= 1: violation = (x + r) - 1
        if xs[i] + rs[i] > 1:
            v = xs[i] + rs[i] - 1
            boundary_penalty += 2 * v
            grad[3*i] += 2 * v
            grad[3*i+2] += 2 * v
        
        # y - r >= 0
        if ys[i] - rs[i] < 0:
            v = rs[i] - ys[i]
            boundary_penalty += 2 * v
            grad[3*i+1] += -2 * v
            grad[3*i+2] += 2 * v
        
        # y + r <= 1
        if ys[i] + rs[i] > 1:
            v = ys[i] + rs[i] - 1
            boundary_penalty += 2 * v
            grad[3*i+1] += 2 * v
            grad[3*i+2] += 2 * v
    
    grad *= 10.0  # scale by penalty weight
    
    # Overlap penalty gradient
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-12:
                continue
            dist = np.sqrt(dist_sq)
            required = rs[i] + rs[j]
            violation = required - dist
            if violation > 0:
                # d/dx_i of (required - dist)^2 = 2*(required-dist) * d(dist)/d(x_i)
                # d(dist)/d(x_i) = -dx/dist
                factor = 2 * violation / dist
                grad[3*i] += -factor * dx
                grad[3*i+1] += -factor * dy
                grad[3*i+2] += 2 * violation
                grad[3*j] += factor * dx
                grad[3*j+1] += factor * dy
                grad[3*j+2] += 2 * violation
    
    grad *= 10.0  # scale by penalty weight
    
    return grad

def _is_valid(pts):
    """Check if the configuration is valid (no overlaps, all inside unit square)."""
    xs = pts[:, 0]
    ys = pts[:, 1]
    rs = pts[:, 2]
    
    # Check boundaries
    if np.any(xs - rs < -1e-6) or np.any(xs + rs > 1 + 1e-6) or \
       np.any(ys - rs < -1e-6) or np.any(ys + rs > 1 + 1e-6):
        return False
    
    # Check overlaps
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-6:
                return False
    
    return True

def _sum_radii(pts):
    return np.sum(pts[:, 2])

def construct():
    start_time = time.time()
    deadline = 140.0  # seconds, with safety margin (evaluator kills at 150s)
    
    # Start with grid initial guess
    best_pts = _initial_grid(N)
    best_score = _sum_radii(best_pts)
    
    # Try multiple random restarts
    np.random.seed(42)
    
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() - start_time > deadline - 20:
            break
        
        # Generate initial guess
        if restart == 0:
            pts = best_pts.copy()
        else:
            # Random perturbation of grid
            pts = _initial_grid(N).copy()
            pts[:, :2] += np.random.randn(N, 2) * 0.05
            pts[:, 2] *= (0.9 + 0.2 * np.random.rand(N))
            # Clip to valid range
            pts[:, 0] = np.clip(pts[:, 0], 0.01, 0.99)
            pts[:, 1] = np.clip(pts[:, 1], 0.01, 0.99)
            pts[:, 2] = np.clip(pts[:, 2], 0.01, 0.4)
        
        # Optimize positions and radii
        params = pts.flatten()
        
        # Use L-BFGS-B with bounds
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))  # x
            bounds.append((0.0, 1.0))  # y
            bounds.append((0.01, 0.4))  # r
        
        try:
            result = minimize(
                _objective,
                params,
                jac=_gradient,
                args=(N,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if time.time() - start_time > deadline - 10:
                break
            new_pts = result.x.reshape(N, 3)
            
            # Ensure validity
            if _is_valid(new_pts):
                score = _sum_radii(new_pts)
                if score > best_score:
                    best_score = score
                    best_pts = new_pts.copy()
        except Exception:
            pass
    
    # Radius growth pass: fix positions, optimize radii only
    if time.time() - start_time < deadline - 5:
        pts = best_pts.copy()
        for iteration in range(50):
            if time.time() - start_time > deadline - 5:
                break
            
            # For each circle, try to increase its radius
            improved = False
            for i in range(N):
                if time.time() - start_time > deadline - 5:
                    break
                
                # Find maximum radius for circle i given other circles fixed
                max_r = 0.4
                # Boundary limits
                max_r = min(max_r, pts[i, 0], 1 - pts[i, 0], pts[i, 1], 1 - pts[i, 1])
                
                # Overlap limits with other circles
                for j in range(N):
                    if i == j:
                        continue
                    dx = pts[i, 0] - pts[j, 0]
                    dy = pts[i, 1] - pts[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    # r_i + r_j <= dist => r_i <= dist - r_j
                    r_limit = dist - pts[j, 2]
                    if r_limit < max_r:
                        max_r = r_limit
                
                max_r = max(0.01, min(max_r, 0.4))
                
                if max_r > pts[i, 2] + 1e-8:
                    pts[i, 2] = max_r
                    improved = True
            
            if not improved:
                break
        
        if _is_valid(pts):
            score = _sum_radii(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
    
    # Final polish with SLSQP
    if time.time() - start_time < deadline - 2:
        pts = best_pts.copy()
        params = pts.flatten()
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((0.01, 0.4))
        
        try:
            result = minimize(
                _objective,
                params,
                jac=_gradient,
                args=(N,),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14}
            )
            # Check if we are still within time limits to accept the result
            if time.time() - start_time < deadline - 1:
                new_pts = result.x.reshape(N, 3)
                if _is_valid(new_pts):
                    score = _sum_radii(new_pts)
                    if score > best_score:
                        best_score = score
                        best_pts = new_pts.copy()
        except Exception:
            pass
    
    return best_pts

# EVOLVE-BLOCK-END
