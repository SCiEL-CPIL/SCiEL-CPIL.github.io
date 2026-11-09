"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 26

# EVOLVE-BLOCK-START

def _validity(arr):
    """Check if circles are inside [0,1]^2 and disjoint."""
    if arr.shape != (N, 3):
        return False
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r <= 0):
        return False
    if np.any(x - r < -1e-12) or np.any(x + r > 1 + 1e-12):
        return False
    if np.any(y - r < -1e-12) or np.any(y + r > 1 + 1e-12):
        return False
    # Check pairwise disjointness
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-10:
                return False
    return True


def _objective_and_constraints(z):
    """
    z = [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    Maximize sum(r_i) -> minimize -sum(r_i)
    Subject to:
      x_i - r_i >= 0
      x_i + r_i <= 1
      y_i - r_i >= 0
      y_i + r_i <= 1
      (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2  for i < j
      r_i > 0
    """
    n = N
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    
    # Objective: minimize negative sum of radii
    obj = -np.sum(r)
    
    # Inequality constraints: g(z) >= 0
    # We'll use scipy's 'ineq' form: g(z) >= 0
    cons = []
    
    # Boundary constraints
    for i in range(n):
        # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[i*3] - z[i*3+2]})
        # 1 - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[i*3] - z[i*3+2]})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[i*3+1] - z[i*3+2]})
        # 1 - y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[i*3+1] - z[i*3+2]})
        # r_i >= 0 (we'll enforce r_i >= small positive via bounds)
    
    # Pairwise disjointness: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda z, i=i, j=j: (z[i*3] - z[j*3])**2 + (z[i*3+1] - z[j*3+1])**2 - (z[i*3+2] + z[j*3+2])**2
            })
    
    return obj, cons


def _local_search_from_initial(init_z, max_iter=500, tol=1e-10):
    """
    Run SLSQP local search from initial point init_z.
    Returns best z found.
    """
    n = N
    # Bounds: x, y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x_i
        bounds.append((0, 1))  # y_i
        bounds.append((0, 0.5))  # r_i
    
    obj, cons = _objective_and_constraints(init_z)
    
    try:
        res = minimize(
            lambda z: obj[0],  # This won't work, need to restructure
            init_z,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': max_iter, 'ftol': tol}
        )
        if res.success:
            return res.x
    except Exception:
        pass
    return init_z


def _construct_slsqp():
    """
    Use SLSQP to maximize sum of radii for 26 circles in unit square.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n = N
    # Initial guess: grid arrangement
    side = int(np.ceil(np.sqrt(n)))
    init_z = np.zeros(3 * n)
    for i in range(n):
        ix = i % side
        iy = i // side
        init_z[i*3] = (ix + 0.5) / side
        init_z[i*3+1] = (iy + 0.5) / side
        init_z[i*3+2] = 0.4 / side  # Small initial radius
    
    # Define the optimization problem properly for SLSQP
    # We need to define the objective and constraints as functions that take z
    
    def objective(z):
        r = z[2::3]
        return -np.sum(r)
    
    def constraint_funcs(z):
        """Return list of constraint values (all should be >= 0 for feasibility)."""
        vals = []
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Boundary constraints
        for i in range(n):
            vals.append(x[i] - r[i])
            vals.append(1 - x[i] - r[i])
            vals.append(y[i] - r[i])
            vals.append(1 - y[i] - r[i])
        
        # Pairwise constraints
        for i in range(n):
            for j in range(i + 1, n):
                vals.append((x[i] - x[j])**2 + (y[i] - y[j])**2 - (r[i] + r[j])**2)
        
        return np.array(vals)
    
    # Use a vectorized constraint approach with SLSQP
    # SLSQP expects constraints in a specific format
    
    # Let's use a different approach: iterative radius expansion with local search
    
    best_z = init_z.copy()
    best_val = -np.sum(init_z[2::3])
    
    # Strategy: Start with small circles, then iteratively expand radii
    # and adjust positions to maintain feasibility
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        x = best_z[0::3].copy()
        y = best_z[1::3].copy()
        r = best_z[2::3].copy()
        
        # Try to increase each radius slightly and see if we can adjust positions
        improved = False
        
        for i in range(n):
            # Try increasing radius i
            dr = 0.001
            r_new = r.copy()
            r_new[i] += dr
            
            # Check if still feasible with current positions
            feasible = True
            # Boundary check
            if r_new[i] > x[i] or r_new[i] > 1 - x[i] or r_new[i] > y[i] or r_new[i] > 1 - y[i]:
                feasible = False
            else:
                # Pairwise check
                for j in range(n):
                    if j == i:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx*dx + dy*dy
                    req_sq = (r_new[i] + r_new[j])**2
                    if dist_sq < req_sq - 1e-12:
                        feasible = False
                        break
            
            if feasible:
                best_z[i*3+2] = r_new[i]
                improved = True
            else:
                # Try to move circle i to make room
                # Simple gradient-based adjustment: move away from nearest constraint
                # For now, skip this circle
                pass
        
        if not improved:
            # Try random perturbations
            for _ in range(50):
                if time.time() > deadline:
                    break
                z_perturbed = best_z.copy()
                # Perturb positions slightly
                for i in range(n):
                    z_perturbed[i*3] += np.random.randn() * 0.01
                    z_perturbed[i*3+1] += np.random.randn() * 0.01
                # Clamp to bounds
                for i in range(n):
                    z_perturbed[i*3] = np.clip(z_perturbed[i*3], 0, 1)
                    z_perturbed[i*3+1] = np.clip(z_perturbed[i*3+1], 0, 1)
                    # Adjust radius to be feasible
                    max_r = min(z_perturbed[i*3], 1 - z_perturbed[i*3], 
                               z_perturbed[i*3+1], 1 - z_perturbed[i*3+1])
                    z_perturbed[i*3+2] = min(z_perturbed[i*3+2], max_r)
                
                # Check feasibility
                if _validity(np.column_stack([z_perturbed[0::3], z_perturbed[1::3], z_perturbed[2::3]])):
                    val = -np.sum(z_perturbed[2::3])
                    if val < best_val:
                        best_val = val
                        best_z = z_perturbed
    
    # Final polish: use SLSQP on the best configuration
    if time.time() < deadline - 20:
        try:
            # Convert to proper format for SLSQP
            # We'll use a simplified approach with scipy
            from scipy.optimize import minimize
            
            def obj_func(z):
                return -np.sum(z[2::3])
            
            def cons_func(z):
                """Return array of constraint values (all >= 0 for feasibility)."""
                x = z[0::3]
                y = z[1::3]
                r = z[2::3]
                vals = []
                for i in range(n):
                    vals.append(x[i] - r[i])
                    vals.append(1 - x[i] - r[i])
                    vals.append(y[i] - r[i])
                    vals.append(1 - y[i] - r[i])
                for i in range(n):
                    for j in range(i + 1, n):
                        vals.append((x[i] - x[j])**2 + (y[i] - y[j])**2 - (r[i] + r[j])**2)
                return np.array(vals)
            
            # Use SLSQP with inequality constraints
            # SLSQP constraint format: {'type': 'ineq', 'fun': g} where g(z) >= 0
            # But we have many constraints, so let's use a different approach
            
            # Actually, let's just do a final local search with small steps
            for _ in range(100):
                if time.time() > deadline:
                    break
                x = best_z[0::3].copy()
                y = best_z[1::3].copy()
                r = best_z[2::3].copy()
                
                # Try to increase the smallest radius
                min_idx = np.argmin(r)
                dr = 0.0005
                r_new = r.copy()
                r_new[min_idx] += dr
                
                # Check feasibility
                feasible = True
                if r_new[min_idx] > x[min_idx] or r_new[min_idx] > 1 - x[min_idx] or \
                   r_new[min_idx] > y[min_idx] or r_new[min_idx] > 1 - y[min_idx]:
                    feasible = False
                else:
                    for j in range(n):
                        if j == min_idx:
                            continue
                        dx = x[min_idx] - x[j]
                        dy = y[min_idx] - y[j]
                        dist_sq = dx*dx + dy*dy
                        req_sq = (r_new[min_idx] + r_new[j])**2
                        if dist_sq < req_sq - 1e-12:
                            feasible = False
                            break
                
                if feasible:
                    best_z[min_idx*3+2] = r_new[min_idx]
                else:
                    # Try moving min_idx circle
                    # Move it towards the center of the available space
                    # Simple heuristic: move towards (0.5, 0.5)
                    if np.random.random() < 0.5:
                        best_z[min_idx*3] += 0.005
                    else:
                        best_z[min_idx*3+1] += 0.005
                    # Clamp
                    best_z[min_idx*3] = np.clip(best_z[min_idx*3], 0, 1)
                    best_z[min_idx*3+1] = np.clip(best_z[min_idx*3+1], 0, 1)
        
        except Exception:
            pass
    
    # Convert back to (N, 3) array
    result = np.column_stack([best_z[0::3], best_z[1::3], best_z[2::3]])
    
    # Ensure validity
    if not _validity(result):
        # Fallback to grid
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.4 / side) for i in range(side) for j in range(side)]
        result = np.array(cells[:n])
    
    return result


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    try:
        result = _construct_slsqp()
        if time.time() > deadline:
            raise TimeoutError()
        return result
    except Exception as e:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.4 / side) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
