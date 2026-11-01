"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit, leave 10s margin

    best_score = -np.inf
    best_config = None

    # Objective and Gradient with penalty weight parameter
    def objective_and_grad(params, penalty_weight=100.0):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        obj = -np.sum(r)
        grad = np.zeros(3*N)
        grad[2*N:3*N] = -1.0
        
        penalty = 0.0
        grad_pen = np.zeros(3*N)
        
        # Overlap
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    p = overlap * overlap
                    penalty += p
                    factor = 2.0 * overlap
                    inv_dist = 1.0 / dist
                    
                    grad_pen[i] += -factor * dx * inv_dist
                    grad_pen[N+j] += factor * dx * inv_dist
                    grad_pen[2*N+i] += factor
                    grad_pen[2*N+j] += factor
                    
                    grad_pen[N+i] += -factor * dy * inv_dist
                    grad_pen[N+j] += factor * dy * inv_dist
                    
        # Boundary
        for i in range(N):
            # Left: r > x
            viol = r[i] - x[i]
            if viol > 0:
                p = viol * viol
                penalty += p
                factor = 2.0 * viol
                grad_pen[i] += -factor
                grad_pen[2*N+i] += factor
            # Right: r > 1-x
            viol = r[i] - (1 - x[i])
            if viol > 0:
                p = viol * viol
                penalty += p
                factor = 2.0 * viol
                grad_pen[i] += factor
                grad_pen[2*N+i] += factor
            # Bottom: r > y
            viol = r[i] - y[i]
            if viol > 0:
                p = viol * viol
                penalty += p
                factor = 2.0 * viol
                grad_pen[N+i] += -factor
                grad_pen[2*N+i] += factor
            # Top: r > 1-y
            viol = r[i] - (1 - y[i])
            if viol > 0:
                p = viol * viol
                penalty += p
                factor = 2.0 * viol
                grad_pen[N+i] += factor
                grad_pen[2*N+i] += factor
                
        obj += penalty_weight * penalty
        grad += penalty_weight * grad_pen
        
        return obj, grad

    def is_valid(config):
        x, y, r = config[:, 0], config[:, 1], config[:, 2]
        if np.any(x < r - 1e-9) or np.any(1-x < r - 1e-9) or np.any(y < r - 1e-9) or np.any(1-y < r - 1e-9):
            return False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True

    def score(config):
        if not is_valid(config):
            return -np.inf
        return np.sum(config[:, 2])

    bounds = [(0,1)]*N + [(0,1)]*N + [(0,0.5)]*N

    # Generate Seeds
    seeds = []
    
    # 1. Random
    for _ in range(20):
        x = np.random.rand(N)
        y = np.random.rand(N)
        r = np.random.rand(N) * 0.04 + 0.01
        seeds.append(np.concatenate([x, y, r]))
        
    # 2. Hexagonal close-packing initialization
    def hex_initialization(spacing_scale=1.0):
        """Create hexagonal grid with appropriate spacing for N circles"""
        # Target: pack N circles in hex pattern
        # Hex grid: rows offset by half spacing
        # For N=32, try different spacings
        best_seed = None
        best_pen = np.inf
        
        for spacing in [0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]:
            dx = spacing * spacing_scale
            dy = dx * np.sqrt(3) / 2
            
            positions = []
            for row in range(12):
                for col in range(12):
                    x = col * dx + (dx/2 if row % 2 else 0) + 0.05
                    y = row * dy + 0.05
                    if 0.01 <= x <= 0.99 and 0.01 <= y <= 0.99:
                        positions.append((x, y))
            
            if len(positions) >= N:
                # Take first N positions (centered)
                # Sort by distance from center to get most central
                cx, cy = 0.5, 0.5
                positions.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
                selected = positions[:N]
                
                hx = np.array([p[0] for p in selected])
                hy = np.array([p[1] for p in selected])
                hr = np.full(N, spacing * 0.45)
                seed = np.concatenate([hx, hy, hr])
                
                # Check validity and compute penalty
                config = seed.reshape(N, 3)
                if is_valid(config):
                    obj, _ = objective_and_grad(seed, penalty_weight=1.0)
                    if obj < best_pen:
                        best_pen = obj
                        best_seed = seed
        
        return best_seed

    hex_seed = hex_initialization()
    if hex_seed is not None:
        seeds.append(hex_seed)
    
    # 3. Staggered grid (checkerboard)
    side = 6
    stagger_x = []
    stagger_y = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            if i % 2 == 0:
                x += 0.5/side * 0.3
            else:
                x -= 0.5/side * 0.3
            if 0.02 <= x <= 0.98:
                stagger_x.append(x)
                stagger_y.append(y)
    
    if len(stagger_x) >= N:
        sx = np.array(stagger_x[:N])
        sy = np.array(stagger_y[:N])
        sr = np.full(N, 0.03)
        seeds.append(np.concatenate([sx, sy, sr]))

    # 4. Square Grid
    grid_x = []
    grid_y = []
    for i in range(side):
        for j in range(side):
            grid_x.append((i + 0.5) / side)
            grid_y.append((j + 0.5) / side)
    gx = np.array(grid_x[:N])
    gy = np.array(grid_y[:N])
    gr = np.full(N, 0.03)
    seeds.append(np.concatenate([gx, gy, gr]))

    # 5. Corner-weighted random (circles tend to corners)
    for _ in range(10):
        x = np.random.rand(N)
        y = np.random.rand(N)
        # Bias toward corners
        x = np.where(np.random.rand(N) < 0.3, x * 0.3 + 0.05, x)
        x = np.where(np.random.rand(N) < 0.3, 1 - x * 0.3 - 0.05, x)
        y = np.where(np.random.rand(N) < 0.3, y * 0.3 + 0.05, y)
        y = np.where(np.random.rand(N) < 0.3, 1 - y * 0.3 - 0.05, y)
        x = np.clip(x, 0.02, 0.98)
        y = np.clip(y, 0.02, 0.98)
        r = np.random.rand(N) * 0.03 + 0.01
        seeds.append(np.concatenate([x, y, r]))

    # Optimization Loop with two-phase approach
    for seed in seeds:
        if time.time() > deadline:
            break
            
        try:
            # Phase 1: Coarse optimization with moderate penalty
            def obj_coarse(params):
                return objective_and_grad(params, penalty_weight=50.0)[0]
            def jac_coarse(params):
                return objective_and_grad(params, penalty_weight=50.0)[1]
            
            res1 = minimize(
                obj_coarse, 
                seed, 
                jac=jac_coarse,
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            # Phase 2: Fine optimization with high penalty
            def obj_fine(params):
                return objective_and_grad(params, penalty_weight=500.0)[0]
            def jac_fine(params):
                return objective_and_grad(params, penalty_weight=500.0)[1]
            
            res2 = minimize(
                obj_fine, 
                res1.x, 
                jac=jac_fine,
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            config = res2.x.reshape(N, 3)
            s = score(config)
            
            if s > best_score:
                best_score = s
                best_config = config
        except Exception:
            pass

    # Final polish pass on best config if time remains
    if best_config is not None and time.time() < deadline - 15:
        try:
            best_params = best_config.flatten()
            
            def obj_polish(params):
                return objective_and_grad(params, penalty_weight=1000.0)[0]
            def jac_polish(params):
                return objective_and_grad(params, penalty_weight=1000.0)[1]
            
            res_polish = minimize(
                obj_polish, 
                best_params, 
                jac=jac_polish,
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            config = res_polish.x.reshape(N, 3)
            s = score(config)
            
            if s > best_score:
                best_score = s
                best_config = config
        except Exception:
            pass

    if best_config is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
# EVOLVE-BLOCK-END
