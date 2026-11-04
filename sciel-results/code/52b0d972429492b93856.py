"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _check_validity(arr, tol=1e-7):
    """Check if circles are disjoint and inside unit square."""
    n = len(arr)
    for i in range(n):
        x, y, r = arr[i]
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = arr[i]
            xj, yj, rj = arr[j]
            dist = np.hypot(xi - xj, yi - yj)
            if dist < ri + rj - tol:
                return False
    return True


def _score(arr):
    """Sum of radii, but penalize invalid configurations heavily."""
    if not _check_validity(arr, tol=1e-9):
        return -1e10
    return float(np.sum(arr[:, 2]))


def _objective(params, n):
    """
    Objective for scipy optimizer: negative sum of radii + penalties for constraint violations.
    params: (3n,) array of (x0, y0, r0, x1, y1, r1, ...)
    """
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    # Ensure radii are positive
    r = np.maximum(r, 1e-6)
    
    score = -np.sum(r)
    
    # Boundary penalties
    penalty = 0.0
    # x boundaries
    penalty += np.sum(np.maximum(0, r - x)**2)
    penalty += np.sum(np.maximum(0, r - (1 - x))**2)
    # y boundaries
    penalty += np.sum(np.maximum(0, r - y)**2)
    penalty += np.sum(np.maximum(0, r - (1 - y))**2)
    
    # Pairwise distance constraints: dist >= ri + rj
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.hypot(dx, dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    
    return score + 100 * penalty


def _gradient(params, n):
    """Gradient of the objective."""
    x = params[:n].copy()
    y = params[n:2*n].copy()
    r = np.maximum(params[2*n:3*n], 1e-6)
    
    grad = np.zeros(3*n)
    
    # Gradient of -sum(r)
    grad[2*n:3*n] = -1.0
    
    # Boundary gradients
    # x - r >= 0 -> violation when r > x
    for i in range(n):
        # x boundary: max(0, r_i - x_i)^2
        if r[i] > x[i]:
            grad[i] += 200 * (r[i] - x[i])  # d/dx of (r-x)^2 = -2(r-x), but penalty is (r-x)^2, d/dx = -2(r-x)*(-1) = 2(r-x)
            # Wait, penalty term is (r - x)^2. d/dx = 2(r-x)*(-1) = -2(r-x). But we want to push x away from boundary.
            # Actually, the penalty is added to objective. We minimize. So gradient should push to reduce penalty.
            # d/dx [(r-x)^2] = -2(r-x). If r>x, this is negative, so gradient is negative, meaning increasing x reduces penalty. Good.
            grad[i] = -200 * (r[i] - x[i])  # correction: derivative w.r.t x is -2(r-x)
            # Let me redo: f = (r - x)^2. df/dx = 2(r-x)*(-1) = -2(r-x).
            # So grad[i] += 100 * (-2(r[i]-x[i])) = -200(r[i]-x[i])
        
        # 1 - x - r >= 0 -> violation when r > 1-x
        if r[i] > 1 - x[i]:
            # penalty: (r - (1-x))^2 = (r - 1 + x)^2
            # d/dx = 2(r-1+x)*(1) = 2(r-1+x)
            grad[i] += 200 * (r[i] - 1 + x[i])
        
        # y - r >= 0
        if r[i] > y[i]:
            grad[n+i] += -200 * (r[i] - y[i])
        
        # 1 - y - r >= 0
        if r[i] > 1 - y[i]:
            grad[n+i] += 200 * (r[i] - 1 + y[i])
        
        # r > 0 handled by clipping, no gradient contribution from clipping in smooth sense, 
        # but for radius, if r is small, the -sum(r) dominates.
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.hypot(dx, dy)
            min_dist = r[i] + r[j]
            if dist < min_dist and dist > 1e-12:
                # penalty: (min_dist - dist)^2
                # d(dist)/dx_i = dx/dist
                # d(penalty)/dx_i = 2(min_dist - dist) * (-dx/dist)
                factor = 200 * (min_dist - dist) / dist
                grad[i] += -factor * dx
                grad[n+i] += -factor * dy
                grad[2*n+i] += 200 * (min_dist - dist)  # d/d r_i of (min_dist - dist)^2 = 2(min_dist-dist)*1
                
                grad[j] += factor * dx
                grad[n+j] += factor * dy
                grad[2*n+j] += 200 * (min_dist - dist)
    
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_arr = None
    best_score = -1e10
    
    # Initial grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append([((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)])
    
    arr = np.array(cells[:N])
    s = _score(arr)
    if s > best_score:
        best_score = s
        best_arr = arr.copy()
    
    # Try to optimize using L-BFGS-B
    n = N
    x0 = np.concatenate([arr[:, 0], arr[:, 1], arr[:, 2]])
    
    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 0.5))
    
    # Run optimization with time check
    try:
        res = minimize(
            _objective, 
            x0, 
            method='L-BFGS-B', 
            jac=_gradient,
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if time.time() > deadline:
            return best_arr
        
        # Extract result
        res_x = res.x[:n]
        res_y = res.x[n:2*n]
        res_r = np.maximum(res.x[2*n:3*n], 1e-6)
        
        res_arr = np.column_stack([res_x, res_y, res_r])
        
        # Check validity and score
        if _check_validity(res_arr, tol=1e-7):
            s = _score(res_arr)
            if s > best_score:
                best_score = s
                best_arr = res_arr.copy()
        
        # Polish: try to increase radii slightly while maintaining constraints
        # Use a local search to push circles apart and increase radii
        if time.time() < deadline - 5:
            # Simple local optimization: for each circle, try to increase radius
            # and adjust positions to maintain constraints
            for iteration in range(100):
                if time.time() > deadline - 2:
                    break
                    
                improved = False
                for i in range(n):
                    old_r = best_arr[i, 2]
                    new_r = old_r * 1.005  # 0.5% increase
                    
                    # Check if new radius is feasible
                    feasible = True
                    for j in range(n):
                        if i == j:
                            continue
                        dist = np.hypot(best_arr[i, 0] - best_arr[j, 0], best_arr[i, 1] - best_arr[j, 1])
                        if dist < new_r + best_arr[j, 2] - 1e-9:
                            feasible = False
                            break
                    
                    if not feasible:
                        # Try to move circle i to make room
                        # Simple: push in direction away from closest neighbor
                        closest_j = -1
                        min_slack = 1e9
                        for j in range(n):
                            if i == j:
                                continue
                            dist = np.hypot(best_arr[i, 0] - best_arr[j, 0], best_arr[i, 1] - best_arr[j, 1])
                            slack = dist - (old_r + best_arr[j, 2])
                            if slack < min_slack:
                                min_slack = slack
                                closest_j = j
                        
                        if closest_j >= 0:
                            dx = best_arr[i, 0] - best_arr[closest_j, 0]
                            dy = best_arr[i, 1] - best_arr[closest_j, 1]
                            dist = np.hypot(dx, dy)
                            if dist > 1e-12:
                                # Push away
                                push_x = dx / dist * 0.001
                                push_y = dy / dist * 0.001
                                new_x = best_arr[i, 0] + push_x
                                new_y = best_arr[i, 1] + push_y
                                
                                # Check boundaries
                                if new_x - old_r >= 0 and new_x + old_r <= 1 and new_y - old_r >= 0 and new_y + old_r <= 1:
                                    # Check all constraints with new position and old radius
                                    ok = True
                                    for j in range(n):
                                        if i == j:
                                            continue
                                        dist2 = np.hypot(new_x - best_arr[j, 0], new_y - best_arr[j, 1])
                                        if dist2 < old_r + best_arr[j, 2] - 1e-9:
                                            ok = False
                                            break
                                    
                                    if ok:
                                        best_arr[i, 0] = new_x
                                        best_arr[i, 1] = new_y
                                        improved = True
                    else:
                        # Increase radius
                        best_arr[i, 2] = new_r
                        improved = True
                
                if not improved:
                    break
    
    except Exception:
        pass
    
    # Final validity check
    if best_arr is not None and _check_validity(best_arr, tol=1e-7):
        return best_arr
    
    # Fallback to initial grid
    return arr

# EVOLVE-BLOCK-END
