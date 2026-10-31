"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin out of 150s

    def time_left():
        return deadline - time.time()

    def energy(v):
        """Compute energy: -sum(r) + penalty for violations."""
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        
        # Objective: maximize sum of radii
        e = -np.sum(r)
        
        # Wall violations
        wall_viol = (np.maximum(0, r - x) + np.maximum(0, r - (1.0 - x)) +
                     np.maximum(0, r - y) + np.maximum(0, r - (1.0 - y)))
        e += 10.0 * np.sum(wall_viol)
        
        # Circle-circle violations
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx*dx + dy*dy
                rsum = r[i] + r[j]
                if rsum > 0:
                    viol = max(0.0, rsum*rsum - d2) / max(rsum, 1e-10)
                    e += 10.0 * viol
        
        # Radius non-negativity
        e += 10.0 * np.sum(np.maximum(0, -r))
        
        return e

    def gradient(v):
        """Numerical gradient of energy."""
        eps = 1e-6
        g = np.zeros_like(v)
        for i in range(len(v)):
            v[i] += eps
            e_plus = energy(v)
            v[i] -= 2*eps
            e_minus = energy(v)
            v[i] += eps
            g[i] = (e_plus - e_minus) / (2*eps)
        return g

    def force_directed_step(v, lr=0.01, max_iter=5000):
        """Simple gradient descent on energy landscape."""
        for it in range(max_iter):
            if time.time() > deadline - 20:
                break
            g = gradient(v)
            v = v - lr * g
            # Clamp
            v[:N] = np.clip(v[:N], 0.0, 1.0)
            v[N:2*N] = np.clip(v[N:2*N], 0.0, 1.0)
            v[2*N:3*N] = np.clip(v[2*N:3*N], 0.0, 0.5)
        return v

    def expand_radii(x, y, r, max_iters=100):
        """Greedily expand each circle's radius as much as possible."""
        r = r.copy()
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(N):
                r_wall = min(x[i], 1.0-x[i], y[i], 1.0-y[i])
                r_max = r_wall
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d > 0:
                        r_max = min(r_max, d - r[j])
                r_max = max(0.0, r_max)
                if r_max > r[i] + 1e-8:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
        return r

    def is_valid(x, y, r, tol=1e-6):
        """Check if a packing is feasible."""
        if np.any(r < -tol) or np.any(x < -tol) or np.any(x > 1+tol) or \
           np.any(y < -tol) or np.any(y > 1+tol):
            return False
        if np.any(r > x + tol) or np.any(r > 1-x + tol) or \
           np.any(r > y + tol) or np.any(r > 1-y + tol):
            return False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx*dx + dy*dy < (r[i]+r[j])**2 - tol:
                    return False
        return True

    def slsqp_refine(x, y, r):
        """SLSQP refinement."""
        v0 = np.concatenate([x, y, r])
        bounds = [(0.0, 1.0)]*N + [(0.0, 1.0)]*N + [(0.0, 0.5)]*N
        
        try:
            result = minimize(
                energy,
                v0, method='SLSQP', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            x_r = result.x[:N]
            y_r = result.x[N:2*N]
            r_r = result.x[2*N:3*N]
            r_r = np.clip(r_r, 0, 0.5)
            x_r = np.clip(x_r, 0, 1)
            y_r = np.clip(y_r, 0, 1)
            r_r = expand_radii(x_r, y_r, r_r, max_iters=100)
            if is_valid(x_r, y_r, r_r):
                return x_r, y_r, r_r
        except Exception:
            pass
        return x, y, r

    # Generate diverse initial packings
    np.random.seed(42)
    best_value = -np.inf
    best_circles = None
    
    def try_solution(x, y, r):
        nonlocal best_value, best_circles
        if is_valid(x, y, r):
            val = np.sum(r)
            if val > best_value:
                best_value = val
                best_circles = np.column_stack([x, y, r])

    # Grid initialization
    side = int(np.ceil(np.sqrt(N)))
    grid_x = np.array([((i % side) + 0.5) / side for i in range(N)])
    grid_y = np.array([((i // side) + 0.5) / side for i in range(N)])
    grid_r = np.full(N, 0.5 / side - 0.01)
    
    # Process multiple initializations
    initializations = []
    
    # 1. Grid
    initializations.append((grid_x.copy(), grid_y.copy(), grid_r.copy()))
    
    # 2. Perturbed grids
    for k in range(4):
        if time.time() > deadline - 50:
            break
        px = grid_x + np.random.randn(N) * 0.03
        py = grid_y + np.random.randn(N) * 0.03
        pr = grid_r + np.random.randn(N) * 0.01
        px = np.clip(px, 0.02, 0.98)
        py = np.clip(py, 0.02, 0.98)
        pr = np.clip(pr, 0.01, 0.3)
        initializations.append((px, py, pr))
    
    # 3. Random scattered
    for k in range(3):
        if time.time() > deadline - 50:
            break
        rx = np.random.uniform(0.1, 0.9, N)
        ry = np.random.uniform(0.1, 0.9, N)
        rr = np.full(N, 0.05)
        initializations.append((rx, ry, rr))
    
    # Process each initialization through the pipeline
    for idx, (x, y, r) in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        # Step 1: Expand radii from initial positions
        r = expand_radii(x, y, r, max_iters=100)
        
        # Step 2: Force-directed gradient descent
        v = np.concatenate([x, y, r])
        v = force_directed_step(v, lr=0.005, max_iter=2000)
        x, y, r = v[:N], v[N:2*N], v[2*N:3*N]
        
        # Step 3: SLSQP refinement
        x, y, r = slsqp_refine(x, y, r)
        
        # Step 4: Final expansion
        r = expand_radii(x, y, r, max_iters=100)
        
        # Verify and record
        try_solution(x, y, r)
    
    # Final polish
    if best_circles is not None:
        x, y, r = best_circles[:, 0], best_circles[:, 1], best_circles[:, 2]
        r = expand_radii(x, y, r, max_iters=200)
        if is_valid(x, y, r):
            best_circles = np.column_stack([x, y, r])
    
    if best_circles is None:
        # Fallback
        best_circles = np.column_stack([grid_x, grid_y, grid_r])
    
    return best_circles

# EVOLVE-BLOCK-END
