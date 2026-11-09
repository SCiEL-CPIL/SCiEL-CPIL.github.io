"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _construct():
    """
    Circle packing in unit square: pack N circles with disjoint interiors,
    maximize sum of radii.
    
    Strategy:
    1. Generate diverse initial configurations using force-directed relaxation.
    2. For each, run a local optimization that jointly optimizes positions and radii
       with constraints enforced via penalty + gradient-based solver.
    3. Use multiple restarts and keep the best valid solution.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    
    # --- Phase 1: Force-directed relaxation to get diverse starting points ---
    def force_relax(x0, y0, r0, iters=500, dt=0.01, temp=0.05):
        """Simple force-directed relaxation to separate circles."""
        n = len(x0)
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        
        for it in range(iters):
            if time.time() > deadline - 5:
                break
            fx = np.zeros(n)
            fy = np.zeros(n)
            fr = np.zeros(n)
            
            for i in range(n):
                for j in range(i+1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 1e-10:
                        dx = random.uniform(-0.01, 0.01)
                        dy = random.uniform(-0.01, 0.01)
                        dist = np.sqrt(dx*dx + dy*dy)
                    
                    # Repulsion if overlapping
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        force = (min_dist - dist) / dist
                        fx[i] += force * dx
                        fy[i] += force * dy
                        fx[j] -= force * dx
                        fy[j] -= force * dy
                        # Grow radii slightly
                        fr[i] += 0.001
                        fr[j] += 0.001
            
            # Boundary repulsion
            for i in range(n):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 2.0
                if x[i] > 1 - r[i]:
                    fx[i] -= (x[i] - (1 - r[i])) * 2.0
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 2.0
                if y[i] > 1 - r[i]:
                    fy[i] -= (y[i] - (1 - r[i])) * 2.0
            
            # Update positions
            x += dt * fx
            y += dt * fy
            
            # Clamp to [0, 1]
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            
            # Grow radii slowly
            r = np.minimum(r + fr * 0.1, 0.5)
        
        return x, y, r
    
    # --- Phase 2: Local optimization with constraint handling ---
    def local_optimize(x, y, r, max_iters=200):
        """
        Jointly optimize positions and radii to maximize sum(r) 
        subject to non-overlap and boundary constraints.
        """
        n = len(x)
        
        def objective(v):
            """Negative sum of radii (we minimize negative)."""
            xi = v[:n]
            yi = v[n:2*n]
            ri = v[2*n:3*n]
            
            # Penalty for overlaps
            penalty = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    dx = xi[i] - xi[j]
                    dy = yi[i] - yi[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = ri[i] + ri[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
            
            # Penalty for boundary violations
            for i in range(n):
                if xi[i] < ri[i]:
                    penalty += (ri[i] - xi[i]) ** 2
                if xi[i] > 1 - ri[i]:
                    penalty += (xi[i] - (1 - ri[i])) ** 2
                if yi[i] < ri[i]:
                    penalty += (ri[i] - yi[i]) ** 2
                if yi[i] > 1 - ri[i]:
                    penalty += (yi[i] - (1 - ri[i])) ** 2
            
            return -np.sum(ri) + 100.0 * penalty
        
        def check_valid(xi, yi, ri):
            """Check if configuration is valid (no overlaps, within bounds)."""
            eps = 1e-9
            for i in range(n):
                if xi[i] < ri[i] - eps or xi[i] > 1 - ri[i] + eps:
                    return False
                if yi[i] < ri[i] - eps or yi[i] > 1 - ri[i] + eps:
                    return False
            for i in range(n):
                for j in range(i+1, n):
                    dx = xi[i] - xi[j]
                    dy = yi[i] - yi[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < ri[i] + ri[j] - eps:
                        return False
            return True
        
        v0 = np.concatenate([x, y, r])
        
        best_v = v0.copy()
        best_obj = objective(v0)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Use L-BFGS-B for optimization
            try:
                res = minimize(objective, v0, method='L-BFGS-B', 
                              options={'maxiter': 50, 'gtol': 1e-6})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_v = res.x
                v0 = res.x
            except Exception:
                break
            
            # Check convergence
            if np.std(v0) < 1e-10:
                break
        
        # Extract and validate
        xi = best_v[:n]
        yi = best_v[n:2*n]
        ri = best_v[2*n:3*n]
        
        # Final validation and repair
        if not check_valid(xi, yi, ri):
            # Try to repair by reducing radii
            for i in range(n):
                ri[i] = min(ri[i], xi[i], 1-xi[i], yi[i], 1-yi[i])
                ri[i] = max(ri[i], 0.01)
        
        return xi, yi, ri, np.sum(ri)
    
    # --- Main loop: multiple restarts ---
    np.random.seed(42)
    random.seed(42)
    
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Generate initial configuration
        if restart == 0:
            # Grid-based initial
            side = int(np.ceil(np.sqrt(N)))
            x = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)][:N])
            y = np.array([(_ + 0.5) / side for _ in range(side) for i in range(side)][:N])
            r = np.full(N, 0.5 / side - 0.01)
        else:
            # Random initial with some structure
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            r = np.random.uniform(0.05, 0.1, N)
        
        # Force relaxation
        x_rel, y_rel, r_rel = force_relax(x, y, r, iters=300, dt=0.02, temp=0.05)
        
        # Local optimization
        xi, yi, ri, value = local_optimize(x_rel, y_rel, r_rel, max_iters=100)
        
        if value > best_value:
            best_value = value
            best_result = np.column_stack([xi, yi, ri])
    
    # Final validation
    if best_result is None:
        # Fallback: trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result

def construct():
    return _construct()

# EVOLVE-BLOCK-END
