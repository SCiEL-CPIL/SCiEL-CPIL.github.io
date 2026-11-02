import numpy as np
from scipy.optimize import minimize
import time
from scipy.spatial import Delaunay
import sys

N = 26

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s buffer for 150s limit
    
    best_score = -np.inf
    best_result = None
    
    def _score_valid(p):
        """
        Validate and score a packing configuration.
        p: (N, 3) array of [x, y, r]
        Returns: sum of radii if valid, -inf otherwise.
        """
        x = p[:, 0]
        y = p[:, 1]
        r = p[:, 2]
        
        # Check bounds
        if np.any(x < -1e-9) or np.any(y < -1e-9) or np.any(x > 1 + 1e-9) or np.any(y > 1 + 1e-9):
            return -np.inf
        if np.any(r < 0):
            return -np.inf
            
        # Check wall constraints with tolerance
        # r <= x, r <= y, r <= 1-x, r <= 1-y
        if np.any(r > x + 1e-7) or np.any(r > y + 1e-7) or np.any(r > 1 - x + 1e-7) or np.any(r > 1 - y + 1e-7):
            return -np.inf
            
        # Check overlap
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        
        # Exclude diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # If any distance is less than sum of radii (minus tolerance), it's invalid
        if np.any(d[mask] < r_sum[mask] - 1e-7):
            return -np.inf
            
        return np.sum(r)

    def _pbd_optimize(x, y, r, target_r, max_iters=2000, time_budget=5.0):
        """
        Position Based Dynamics optimization.
        Tries to match r to target_r while resolving overlaps and walls.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        start_t = time.time()
        
        for it in range(max_iters):
            if time.time() - start_t > time_budget:
                break
                
            # 1. Try to increase radii towards target
            # We scale up radii slightly each step
            scale = 1.0 + 0.005
            new_r = np.minimum(r * scale, target_r)
            
            # 2. Resolve Wall Constraints
            # Push circles away from walls
            # Left: x >= r
            # Right: x <= 1 - r
            # Bottom: y >= r
            # Top: y <= 1 - r
            
            # Calculate penetration into walls
            pen_left = np.maximum(0, new_r - x)
            pen_right = np.maximum(0, x - (1 - new_r))
            pen_bottom = np.maximum(0, new_r - y)
            pen_top = np.maximum(0, y - (1 - new_r))
            
            # Move centers
            x += pen_left
            x -= pen_right
            y += pen_bottom
            y -= pen_top
            
            # 3. Resolve Overlaps (Gauss-Seidel style)
            # Iterate pairs
            # For better performance, we can use a simple loop or vectorized approximation
            # Since N=26, O(N^2) is fine (675 pairs)
            
            # We do a few passes to ensure stability
            for _ in range(5):
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = x[j] - x[i]
                        dy = y[j] - y[i]
                        dist = np.sqrt(dx**2 + dy**2)
                        min_dist = new_r[i] + new_r[j]
                        
                        if dist < min_dist:
                            if dist < 1e-9:
                                # Random perturbation
                                dx = np.random.uniform(-0.1, 0.1)
                                dy = np.random.uniform(-0.1, 0.1)
                                dist = np.sqrt(dx**2 + dy**2)
                                
                            # Push apart
                            # Weight by inverse mass (proportional to 1/r^2 or just equal)
                            # Let's use equal weight for simplicity first, or mass proportional to area
                            # Mass ~ r^2. Heavier circles move less.
                            m_i = new_r[i] ** 2
                            m_j = new_r[j] ** 2
                            total_m = m_i + m_j
                            
                            if total_m == 0:
                                continue
                                
                            # Correction vector
                            corr_x = (min_dist - dist) / dist * dx
                            corr_y = (min_dist - dist) / dist * dy
                            
                            # Apply correction
                            # i moves opposite to j
                            x[i] -= corr_x * (m_j / total_m)
                            y[i] -= corr_y * (m_j / total_m)
                            x[j] += corr_x * (m_i / total_m)
                            y[j] += corr_y * (m_i / total_m)
                
                # Re-check walls after overlap resolution as they might have pushed each other out
                pen_left = np.maximum(0, new_r - x)
                pen_right = np.maximum(0, x - (1 - new_r))
                pen_bottom = np.maximum(0, new_r - y)
                pen_top = np.maximum(0, y - (1 - new_r))
                
                x += pen_left
                x -= pen_right
                y += pen_bottom
                y -= pen_top
            
            # Update radii to the target for next iteration
            # Only accept if we didn't "break" too much? 
            # In PBD, we usually project positions to satisfy constraints.
            # Here, we just keep the new_r as the current radius state.
            # If the constraints are satisfied, great. If not, we continue.
            r = new_r.copy()
            
            # Check if we have converged to target
            if np.allclose(r, target_r, atol=1e-5):
                break

        return x, y, r

    def _delaunay_polish(p, time_budget=10.0):
        """
        Polish the solution using SLSQP on the Delaunay graph.
        """
        x = p[:, 0].copy()
        y = p[:, 1].copy()
        r = p[:, 2].copy()
        
        # Find Delaunay triangulation of centers
        points = np.column_stack([x, y])
        try:
            tri = Delaunay(points)
            # Get edges
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        idx1, idx2 = simplex[i], simplex[j]
                        edges.add((min(idx1, idx2), max(idx1, idx2)))
        except Exception:
            # Fallback to all pairs if Delaunay fails (e.g., collinear points)
            edges = set()
            for i in range(N):
                for j in range(i + 1, N):
                    edges.add((i, j))
        
        # Objective: Maximize sum of radii
        # Variables: x, y, r
        # Constraints:
        # 1. Bounds: 0 <= x,y <= 1
        # 2. Walls: r <= x, r <= 1-x, r <= y, r <= 1-y
        # 3. Overlaps for edges in Delaunay graph (plus maybe some neighbors)
        #    To be safe, include all pairs that are close or in Delaunay.
        #    Actually, for SLSQP, providing all constraints is safer if N is small.
        #    N=26 -> 325 pairs. 325 constraints + 26*4 bounds = ~430 constraints.
        #    SLSQP can handle this, but it might be slow.
        #    Let's try to restrict to Delaunay edges + 1-ring neighbors.
        
        # Let's stick to Delaunay edges for efficiency, but verify validity after.
        
        def objective(vars):
            # vars: [x, y, r] flattened
            r_vars = vars[2*N:]
            return -np.sum(r_vars)
            
        def grad_objective(vars):
            g = np.zeros(3*N)
            g[2*N:] = -1.0
            return g
            
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((0, 0.5)) # r
            
        # Constraints
        # Inequality constraints: g(vars) >= 0
        
        cons = []
        
        # Wall constraints
        # x - r >= 0
        # 1 - x - r >= 0
        # y - r >= 0
        # 1 - y - r >= 0
        
        def wall_cons(vars):
            x_v = vars[:N]
            y_v = vars[N:2*N]
            r_v = vars[2*N:]
            c1 = x_v - r_v
            c2 = 1 - x_v - r_v
            c3 = y_v - r_v
            c4 = 1 - y_v - r_v
            return np.concatenate([c1, c2, c3, c4])
            
        cons.append({'type': 'ineq', 'fun': wall_cons})
        
        # Overlap constraints for edges
        # d_ij - (r_i + r_j) >= 0
        edge_list = list(edges)
        if not edge_list:
            # Fallback
            for i in range(N):
                for j in range(i+1, N):
                    edge_list.append((i, j))
                    
        def overlap_cons(vars):
            x_v = vars[:N]
            y_v = vars[N:2*N]
            r_v = vars[2*N:]
            vals = []
            for i, j in edge_list:
                dx = x_v[i] - x_v[j]
                dy = y_v[i] - y_v[j]
                d = np.sqrt(dx**2 + dy**2)
                vals.append(d - (r_v[i] + r_v[j]))
            return np.array(vals)
            
        cons.append({'type': 'ineq', 'fun': overlap_cons})
        
        try:
            res = minimize(objective, p.flatten(), method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 100, 'ftol': 1e-8})
            if res.success:
                p_new = res.x.reshape(N, 3)
                # Check if it's actually valid and better
                s_new = _score_valid(p_new)
                s_old = _score_valid(p)
                if s_new > s_old:
                    return p_new
        except Exception:
            pass
            
        return p

    # Initializations
    def init_grid():
        side = int(np.ceil(np.sqrt(N)))
        coords = []
        for i in range(side):
            for j in range(side):
                if len(coords) < N:
                    coords.append(((i + 0.5) / side, (j + 0.5) / side))
        return np.array(coords)
        
    def init_hex():
        # Hexagonal packing
        points = []
        r_est = 0.1
        # Generate a grid of hex points
        # Spacing
        dx = 1.0 / 10
        dy = dx * np.sqrt(3) / 2
        for i in range(-1, 12):
            for j in range(-1, 12):
                x = (i + 0.5 * (j % 2)) * dx
                y = j * dy
                if 0 <= x <= 1 and 0 <= y <= 1:
                    points.append((x, y))
        # Sort by distance to center? Or just take first N
        # Let's sort by distance to (0.5, 0.5) to get a centered block
        points.sort(key=lambda p: (p[0]-0.5)**2 + (p[1]-0.5)**2)
        return np.array(points[:N])
        
    def init_random():
        return np.random.uniform(0.1, 0.9, (N, 2))

    # Main Loop
    np.random.seed(42)
    
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        
        # Choose init
        if restarts % 3 == 1:
            p0_xy = init_grid()
        elif restarts % 3 == 2:
            p0_xy = init_hex()
        else:
            p0_xy = init_random()
            
        # Add jitter
        p0_xy = p0_xy + np.random.normal(0, 0.01, p0_xy.shape)
        p0_xy = np.clip(p0_xy, 0.05, 0.95)
        
        # Initial radius: small
        r0 = np.full(N, 0.02)
        
        # Target radius: we want to grow as large as possible.
        # Estimate max possible radius.
        # For N=26, avg area per circle ~ 1/26. r ~ sqrt(1/26/pi) ~ 0.11.
        # But packing is not perfect. Let's target 0.15 and let PBD find the limit.
        # Actually, we can just keep growing until no more progress is made.
        
        current_x = p0_xy[:, 0]
        current_y = p0_xy[:, 1]
        current_r = r0.copy()
        
        # PBD Phase
        # We will try to increase radii in steps.
        # Start with a small target, increase it.
        
        # Find a good target radius by trial?
        # Or just use a fixed schedule.
        # Let's try to reach a radius of ~0.12 (sum ~ 3.12)
        # Wait, sum of radii. If r=0.12, sum = 26*0.12 = 3.12.
        # Is this possible?
        # Area of one circle r=0.12 is ~0.045. 26 * 0.045 = 1.17 > 1.
        # So r=0.12 is impossible for all circles.
        # Average area = 1/26. r_avg = sqrt(1/(26*pi)) = 0.1107.
        # So sum of radii ~ 26 * 0.11 = 2.86 is the theoretical max if perfect circle packing (which doesn't exist, but hex packing efficiency is 0.9069).
        # So effective area = 0.9069.
        # 26 * pi * r^2 * 0.9069 <= 1 => r^2 <= 1 / (26 * pi * 0.9069) => r <= 0.114.
        # Sum ~ 26 * 0.114 = 2.96.
        
        # So target radii should be around 0.11-0.12.
        
        # Strategy:
        # 1. Start with r=0.02.
        # 2. Iteratively increase target_r.
        # 3. Run PBD to fit.
        # 4. If PBD fails to converge (or time runs out), stop.
        
        target_r = 0.02
        max_target_r = 0.15
        
        # Run PBD for a while to stabilize initial small circles
        current_x, current_y, current_r = _pbd_optimize(current_x, current_y, current_r, target_r, max_iters=100, time_budget=1.0)
        
        # Now grow
        while time.time() - start_time < time_limit - 10: # Leave time for polish
            # Increase target
            target_r *= 1.05
            if target_r > max_target_r:
                break
                
            # Run PBD
            new_x, new_y, new_r = _pbd_optimize(current_x, current_y, current_r, target_r, max_iters=100, time_budget=2.0)
            
            # Check if we actually grew
            # If new_r is significantly smaller than target_r, we might be stuck
            # But PBD projects to constraints, so new_r should be close to target_r if feasible.
            # If infeasible, the positions will be pushed to walls/each other.
            
            # Update state
            current_x, current_y, current_r = new_x, new_y, new_r
            
            # Check score
            p_curr = np.column_stack([current_x, current_y, current_r])
            s_curr = _score_valid(p_curr)
            
            if s_curr > best_score:
                best_score = s_curr
                best_result = p_curr.copy()
                
            # If we can't grow anymore (radii didn't increase much), break
            if np.mean(current_r) < target_r * 0.9:
                # Stuck
                break
                
    # Polish the best result
    if best_result is not None:
        # Try to polish with SLSQP
        if time.time() - start_time < time_limit - 5:
            polished = _delaunay_polish(best_result, time_budget=5.0)
            s_polished = _score_valid(polished)
            if s_polished > best_score:
                best_score = s_polished
                best_result = polished

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
