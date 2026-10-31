"""Circle packing in perimeter-4 rectangle: Hexagonal Init + Gradient Ascent."""
import numpy as np
import time
from scipy.optimize import minimize
import random

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _hexagonal_init(w, h, n):
    """
    Generate a hexagonal close-packed layout for n circles in w x h rectangle.
    Returns x, y, r arrays.
    """
    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)
    
    # Estimate optimal radius for hexagonal packing
    # Area per circle in hex packing is approx 2 * sqrt(3) * r^2
    # Total area = w * h
    # n * 2 * sqrt(3) * r^2 <= w * h
    r_est = np.sqrt((w * h) / (n * 2 * np.sqrt(3)))
    
    # Adjust for boundaries and aspect ratio
    # Hexagonal rows have height sqrt(3)*r
    # Number of rows approx h / (sqrt(3)*r)
    # Number of cols approx w / r
    # Let's just use a grid that is slightly compressed to allow for expansion
    
    # Start with a grid that fits in the box
    cols = int(np.ceil(np.sqrt(n * w / h)))
    rows = int(np.ceil(n / cols))
    
    # Hexagonal offset
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Hexagonal coordinates
        # x = col * r * 1.0 + (row % 2) * r * 0.5
        # y = row * r * sqrt(3)
        
        # We want to fill the box [0, w] x [0, h]
        # Scale factors
        x_scale = w / (cols * 1.0 + 0.5) # Approx width of grid
        y_scale = h / (rows * np.sqrt(3))
        
        # Use a uniform radius initially, then let optimizer expand
        # To ensure they fit initially, we pick r such that they don't overlap
        # In hex packing, min dist is r.
        # Let's pick r = min(x_scale, y_scale) * 0.5 
        # Actually, let's just place them in a grid that is guaranteed to fit, 
        # then the optimizer will expand them.
        
        # Grid placement
        x[i] = (col + 0.5 + (row % 2) * 0.5) * (w / (cols + 0.5))
        y[i] = (row + 0.5) * (h / rows)
        
    # Initial radius: half the minimum distance to boundary or neighbor
    # To be safe, start small and let optimizer grow them.
    # But starting too small is slow.
    # Let's calculate the "natural" radius for this grid.
    # Distance between neighbors in x: w / (cols + 0.5)
    # Distance between neighbors in y: h / rows
    # Min dist approx min(w/(cols+0.5), h/rows)
    # For hex, r = dist / 2 roughly.
    
    min_dx = w / (cols + 0.5)
    min_dy = h / rows
    r_init = min(min_dx, min_dy) * 0.45
    
    # Ensure r_init fits in box
    r_init = min(r_init, w/2, h/2)
    
    for i in range(n):
        # Clamp to boundary
        r[i] = min(r_init, x[i], w - x[i], y[i], h - y[i])
        
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_value = 0.0
    best_result = None
    
    def update_best(x, y, r, w, h):
        nonlocal best_value, best_result
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)
    
    # Search over aspect ratios
    # Perimeter 2(w+h) = 4 => w+h=2
    # w in [0.5, 1.5]
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.9, 1.1, 1.3, 0.75, 1.2, 1.5]
    
    for w in aspect_ratios:
        if time.time() > deadline - 30:
            break
        h = 2.0 - w
        if h < 0.2:
            continue
            
        # 1. Initialize with hexagonal/grid layout
        x, y, r = _hexagonal_init(w, h, N)
        
        # 2. Optimize using L-BFGS-B
        # Variables: x[0..N-1], y[0..N-1], r[0..N-1]
        # Objective: -sum(r)
        # Constraints:
        #   r[i] <= x[i]
        #   r[i] <= w - x[i]
        #   r[i] <= y[i]
        #   r[i] <= h - y[i]
        #   (x[i]-x[j])^2 + (y[i]-y[j])^2 >= (r[i]+r[j])^2
        
        # L-BFGS-B handles box constraints, but not general nonlinear constraints easily.
        # We can use penalty method with L-BFGS-B or use SLSQP.
        # SLSQP is slower but handles constraints natively.
        # Let's try a hybrid: Use a "soft" penalty in the objective and box constraints for r>=0, x,y in [0,w/h].
        # Actually, a better approach for this specific problem is to fix positions and optimize radii? No, positions matter.
        
        # Let's use SLSQP but with a better initial guess and fewer iterations, relying on the hexagonal start.
        
        def objective(z):
            xi = z[0::3][:N]
            yi = z[1::3][:N]
            ri = z[2::3][:N]
            
            # Box violations
            pen = 0.0
            for i in range(N):
                if xi[i] < ri[i]: pen += (ri[i] - xi[i])**2
                if xi[i] > w - ri[i]: pen += (xi[i] - (w - ri[i]))**2
                if yi[i] < ri[i]: pen += (ri[i] - yi[i])**2
                if yi[i] > h - ri[i]: pen += (yi[i] - (h - ri[i]))**2
            
            # Overlap violations
            for i in range(N):
                for j in range(i+1, N):
                    dx = xi[i] - xi[j]
                    dy = yi[i] - yi[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    overlap = ri[i] + ri[j] - d
                    if overlap > 0:
                        pen += overlap**2
            
            return -np.sum(ri) + 100 * pen
        
        def gradient(z):
            xi = z[0::3][:N]
            yi = z[1::3][:N]
            ri = z[2::3][:N]
            grad = np.zeros_like(z)
            
            # Gradient of -sum(r)
            grad[2::3][:N] = -1.0
            
            # Gradient of penalties
            # Box
            for i in range(N):
                # x[i] < r[i] => pen += (r[i]-x[i])^2 => d/dx = -2(r[i]-x[i]), d/dr = 2(r[i]-x[i])
                if xi[i] < ri[i]:
                    diff = ri[i] - xi[i]
                    grad[i] += 100 * (-2 * diff)
                    grad[2+i] += 100 * (2 * diff)
                
                if xi[i] > w - ri[i]:
                    diff = xi[i] - (w - ri[i])
                    grad[i] += 100 * (2 * diff)
                    grad[2+i] += 100 * (-2 * diff)
                
                if yi[i] < ri[i]:
                    diff = ri[i] - yi[i]
                    grad[N+i] += 100 * (-2 * diff)
                    grad[2+i] += 100 * (2 * diff)
                
                if yi[i] > h - ri[i]:
                    diff = yi[i] - (h - ri[i])
                    grad[N+i] += 100 * (2 * diff)
                    grad[2+i] += 100 * (-2 * diff)
            
            # Overlap
            for i in range(N):
                for j in range(i+1, N):
                    dx = xi[i] - xi[j]
                    dy = yi[i] - yi[j]
                    d_sq = dx*dx + dy*dy
                    if d_sq < 1e-10:
                        continue
                    d = np.sqrt(d_sq)
                    overlap = ri[i] + ri[j] - d
                    if overlap > 0:
                        # pen += overlap^2
                        # d/dx_i = 2 * overlap * d/dx_i (r_i + r_j - d)
                        # d/dx_i (-d) = -dx/d
                        grad[i] += 100 * 2 * overlap * (-dx/d)
                        grad[N+i] += 100 * 2 * overlap * (-dy/d)
                        grad[2+i] += 100 * 2 * overlap * (1)
                        
                        grad[j] += 100 * 2 * overlap * (dx/d)
                        grad[N+j] += 100 * 2 * overlap * (dy/d)
                        grad[2+j] += 100 * 2 * overlap * (1)
            
            return grad
            
        z0 = np.zeros(3*N)
        z0[0::3][:N] = x
        z0[1::3][:N] = y
        z0[2::3][:N] = r
        
        bounds = []
        for i in range(N):
            bounds.append((0, w)) # x
            bounds.append((0, h)) # y
            bounds.append((0, min(w, h)/2)) # r
        
        try:
            res = minimize(objective, z0, jac=gradient, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-12, 'maxls': 50})
            
            x_opt = res.x[0::3][:N]
            y_opt = res.x[1::3][:N]
            r_opt = res.x[2::3][:N]
            
            # Clean up: ensure valid
            # Clip radii to be non-negative
            r_opt = np.maximum(r_opt, 0)
            
            # If L-BFGS-B resulted in invalid packing due to penalty being too weak,
            # we can try to "fix" it by shrinking radii until valid, or just discard.
            # Given the gradient info, it should be close.
            
            update_best(x_opt, y_opt, r_opt, w, h)
            
            # Polish: Try to expand radii greedily
            for _ in range(10):
                if time.time() > deadline - 10:
                    break
                improved = False
                for i in range(N):
                    r_max = min(x_opt[i], w - x_opt[i], y_opt[i], h - y_opt[i])
                    for j in range(N):
                        if i == j: continue
                        dx = x_opt[i] - x_opt[j]
                        dy = y_opt[i] - y_opt[j]
                        d = np.sqrt(dx*dx + dy*dy)
                        r_allowed = d - r_opt[j]
                        if r_allowed < r_max:
                            r_max = r_allowed
                    if r_max > r_opt[i] + 1e-8:
                        r_opt[i] = r_max
                        improved = True
                if not improved:
                    break
            
            update_best(x_opt, y_opt, r_opt, w, h)
            
        except Exception as e:
            pass

    # Fallback
    if best_result is None:
        w = 1.0
        h = 1.0
        x, y, r = _hexagonal_init(w, h, N)
        best_result = (x, y, r, w, h)
        
    x, y, r, w, h = best_result
    
    # Final safety shrink
    for _ in range(5):
        r_trial = r * (1 - 1e-8)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
            
    return np.column_stack((x, y, r))

# Dummy to satisfy imports if needed, though not used directly in construct
def _force_resolve(x, y, r, w, h, max_iter=2000):
    return x, y

def _expand_radii(x, y, r, w, h, max_rounds=50, time_budget=None):
    return r

def _slsqp_optimize(x0, y0, r0, w0, h0, time_budget):
    return None

def _generate_candidate(seed, w):
    return np.zeros(N), np.zeros(N), np.zeros(N)

def _try_expand_positions(x, y, r, w, h, time_budget):
    return x, y, False
