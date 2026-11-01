"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    def is_valid(x, y, r, tol=1e-7):
        """Check if circles are disjoint and inside unit square."""
        n = len(x)
        for i in range(n):
            if x[i] < r[i] - tol or x[i] + r[i] > 1.0 + tol:
                return False
            if y[i] < r[i] - tol or y[i] + r[i] > 1.0 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j] - tol:
                    return False
        return True
    
    def get_sum(x, y, r):
        return np.sum(r)
    
    def build_contact_graph(x, y, r, margin=1e-4):
        """Build contact graph based on proximity."""
        n = len(x)
        contacts = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j] + margin:
                    contacts[i, j] = True
                    contacts[j, i] = True
        return contacts
    
    def sqp_optimize(x0, y0, r0, max_iter=200, tol=1e-8):
        """
        SQP-style optimization:
        - Maximize sum(r)
        - Constraints: no overlap, inside square
        - Use active set method with quadratic subproblem
        """
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        n = len(x)
        
        best_sum = get_sum(x, y, r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        # Precompute pairs
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Check validity
            if is_valid(x, y, r):
                s = get_sum(x, y, r)
                if s > best_sum:
                    best_sum = s
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            
            # Compute gradient of objective (maximize sum(r))
            # Gradient with respect to x: 0
            # Gradient with respect to y: 0  
            # Gradient with respect to r: 1 for each circle
            grad_r = np.ones(n)
            
            # Active constraints: overlapping pairs and boundary violations
            active_pairs = []
            active_bounds = []  # (i, axis, side) where axis in {x,y}, side in {lo,hi}
            
            for i, j in pairs:
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j] + 1e-6:
                    active_pairs.append((i, j, d))
            
            for i in range(n):
                if x[i] < r[i] + 1e-6:
                    active_bounds.append((i, 'x', 'lo'))
                if x[i] + r[i] > 1.0 - 1e-6:
                    active_bounds.append((i, 'x', 'hi'))
                if y[i] < r[i] + 1e-6:
                    active_bounds.append((i, 'y', 'lo'))
                if y[i] + r[i] > 1.0 - 1e-6:
                    active_bounds.append((i, 'y', 'hi'))
            
            # If no active constraints, just increase radii
            if not active_pairs and not active_bounds:
                # Can increase all radii slightly
                dr = 0.001
                for i in range(n):
                    # Check if we can increase r[i]
                    new_r = r[i] + dr
                    can_increase = True
                    for i2, j2, d2 in active_pairs:
                        pass  # Already checked
                    # Check boundaries
                    if x[i] < new_r or x[i] + new_r > 1.0:
                        can_increase = False
                    if y[i] < new_r or y[i] + new_r > 1.0:
                        can_increase = False
                    if can_increase:
                        r[i] = new_r
                continue
            
            # Solve QP subproblem:
            # Maximize sum(dr_i) 
            # Subject to: for each active pair (i,j): 
            #   (x_i + dx_i - x_j - dx_j)^2 + (y_i + dy_i - y_j - dy_j)^2 >= (r_i + dr_i + r_j + dr_j)^2
            # Linearized: 
            #   2*dx_i*(x_i-x_j) + 2*dy_i*(y_i-y_j) - 2*dx_j*(x_i-x_j) - 2*dy_j*(y_i-y_j) >= 2*(r_i+r_j)*(dr_i+dr_j) - 2*d*(dx_i-dx_j+dy_i-dy_j)
            # This is complex. Use simpler approach: projected gradient.
            
            # Use a simpler approach: coordinate descent with penalty
            # For each circle, try to increase its radius and adjust position
            
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                
                # Try to increase r[i]
                old_r = r[i]
                new_r = old_r + 0.002
                
                # Check if new radius fits with current positions
                feasible = True
                for i2, j2 in [(i, j) for j in range(n) if j != i]:
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if d < new_r + r[j]:
                        feasible = False
                        break
                if x[i] < new_r or x[i] + new_r > 1.0:
                    feasible = False
                if y[i] < new_r or y[i] + new_r > 1.0:
                    feasible = False
                
                if feasible:
                    r[i] = new_r
                    improved = True
                else:
                    # Try to move circle i to make room
                    # Compute force from overlapping circles
                    fx = 0.0
                    fy = 0.0
                    for j in range(n):
                        if j == i:
                            continue
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        min_d = r[i] + r[j]
                        if d < min_d and d > 1e-10:
                            # Repulsive force
                            overlap = min_d - d
                            fx += overlap * (x[i] - x[j]) / d
                            fy += overlap * (y[i] - y[j]) / d
                    
                    # Boundary forces
                    if x[i] < r[i]:
                        fx += (r[i] - x[i]) * 10
                    if x[i] + r[i] > 1.0:
                        fx -= (x[i] + r[i] - 1.0) * 10
                    if y[i] < r[i]:
                        fy += (r[i] - y[i]) * 10
                    if y[i] + r[i] > 1.0:
                        fy -= (y[i] + y[i] - 1.0) * 10 if False else (y[i] + r[i] - 1.0) * 10
                    
                    # Move in direction of force
                    step = 0.01
                    new_x = x[i] + step * fx
                    new_y = y[i] + step * fy
                    new_x = np.clip(new_x, r[i], 1.0 - r[i])
                    new_y = np.clip(new_y, r[i], 1.0 - r[i])
                    
                    # Check if this move helps
                    old_overlap = 0
                    for j in range(n):
                        if j == i:
                            continue
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        min_d = r[i] + r[j]
                        if d < min_d:
                            old_overlap += min_d - d
                    
                    x[i] = new_x
                    y[i] = new_y
                    
                    new_overlap = 0
                    for j in range(n):
                        if j == i:
                            continue
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        min_d = r[i] + r[j]
                        if d < min_d:
                            new_overlap += min_d - d
                    
                    if new_overlap < old_overlap:
                        improved = True
                    else:
                        # Revert
                        x[i] = x[i] - step * fx
                        y[i] = y[i] - step * fy
                        x[i] = np.clip(x[i], r[i], 1.0 - r[i])
                        y[i] = np.clip(y[i], r[i], 1.0 - r[i])
            
            if not improved and it > 50:
                break
        
        return best_x, best_y, best_r
    
    def optimize_with_penalties(x0, y0, r0, max_iter=500):
        """Penalty-based optimization with gradient descent."""
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        n = len(x)
        
        best_sum = -np.inf
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        lr = 0.01
        momentum = 0.9
        vx = np.zeros(n)
        vy = np.zeros(n)
        vr = np.zeros(n)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute gradient
            gx = np.zeros(n)
            gy = np.zeros(n)
            gr = np.ones(n)  # Maximize sum(r)
            
            # Overlap penalties
            C = 100.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < 1e-10:
                        d = 1e-10
                    r_sum = r[i] + r[j]
                    if d < r_sum:
                        V = r_sum - d
                        coeff = 2.0 * C * V / d
                        gx[i] -= coeff * dx
                        gy[i] -= coeff * dy
                        gx[j] += coeff * dx
                        gy[j] += coeff * dy
                        gr[i] += 2.0 * C * V
                        gr[j] += 2.0 * C * V
            
            # Boundary penalties
            for i in range(n):
                if x[i] < r[i]:
                    V = r[i] - x[i]
                    gx[i] += 2.0 * C * V
                    gr[i] += 2.0 * C * V
                if x[i] + r[i] > 1.0:
                    V = x[i] + r[i] - 1.0
                    gx[i] -= 2.0 * C * V
                    gr[i] += 2.0 * C * V
                if y[i] < r[i]:
                    V = r[i] - y[i]
                    gy[i] += 2.0 * C * V
                    gr[i] += 2.0 * C * V
                if y[i] + r[i] > 1.0:
                    V = y[i] + r[i] - 1.0
                    gy[i] -= 2.0 * C * V
                    gr[i] += 2.0 * C * V
            
            # Gradient descent with momentum (minimize negative objective)
            vx = momentum * vx - lr * gx
            vy = momentum * vy - lr * gy
            vr = momentum * vr - lr * (-gr)  # Maximize r
            
            x += vx
            y += vy
            r += vr
            
            # Project
            r = np.maximum(r, 0.001)
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            
            # Check validity
            if is_valid(x, y, r):
                s = get_sum(x, y, r)
                if s > best_sum:
                    best_sum = s
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            
            # Decay learning rate
            lr *= 0.999
            
            if np.max(np.abs(gx)) < 1e-7 and np.max(np.abs(gy)) < 1e-7:
                break
        
        return best_x, best_y, best_r
    
    def hex_init():
        """Hexagonal lattice initialization."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Hex grid parameters
        # For hex packing, distance between centers = 2*r
        # In a hex grid with spacing s, neighbors are at distance s
        # We want to fit 32 circles in unit square
        
        # Try different hex spacings
        best_x, best_y, best_r = None, None, None
        best_sum = -np.inf
        
        for spacing in [0.15, 0.16, 0.17, 0.18, 0.19, 0.20]:
            # Generate hex points
            points = []
            row = 0
            while len(points) < N + 10:
                for col in range(int(1.0 / spacing) + 2):
                    px = col * spacing * 0.866  # cos(30)
                    py = row * spacing
                    if col % 2 == 1:
                        py += spacing * 0.5
                    if 0 <= px <= 1 and 0 <= py <= 1:
                        points.append((px, py))
                row += 1
            
            if len(points) < N:
                continue
            
            # Sort by distance from center
            points.sort(key=lambda p: (p[0] - 0.5)**2 + (p[1] - 0.5)**2)
            points = points[:N]
            
            for i, (px, py) in enumerate(points):
                x[i] = px
                y[i] = py
                r[i] = spacing * 0.5 - 0.005
            
            # Check validity and get sum
            if is_valid(x, y, r):
                s = get_sum(x, y, r)
                if s > best_sum:
                    best_sum = s
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        if best_x is not None:
            return best_x, best_y, best_r
        
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side - 0.01
        return x, y, r
    
    def grid_init(rows, cols):
        """Grid initialization."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) / cols
                y[idx] = (i + 0.5) / rows
                r[idx] = 0.5 / max(rows, cols) - 0.005
                idx += 1
        
        while idx < N:
            x[idx] = np.random.uniform(0.1, 0.9)
            y[idx] = np.random.uniform(0.1, 0.9)
            r[idx] = 0.03
            idx += 1
        
        return x, y, r
    
    def random_init():
        """Random initialization."""
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.full(N, 0.03)
        return x, y, r
    
    # Strategy: Multiple initializations + SQP + penalty optimization
    best_result = None
    best_sum = -np.inf
    
    def evaluate_and_update(x, y, r):
        nonlocal best_result, best_sum
        if is_valid(x, y, r):
            s = get_sum(x, y, r)
            if s > best_sum:
                best_sum = s
                best_result = (x.copy(), y.copy(), r.copy())
    
    # 1. Hexagonal initialization
    x, y, r = hex_init()
    evaluate_and_update(x, y, r)
    
    # Optimize from hex
    if time.time() < deadline - 60:
        x_opt, y_opt, r_opt = sqp_optimize(x, y, r, max_iter=100)
        evaluate_and_update(x_opt, y_opt, r_opt)
    
    if time.time() < deadline - 40:
        x_opt, y_opt, r_opt = optimize_with_penalties(x, y, r, max_iter=300)
        evaluate_and_update(x_opt, y_opt, r_opt)
    
    # 2. Grid initializations
    for rows, cols in [(5, 7), (6, 6), (4, 8), (8, 4)]:
        if time.time() > deadline - 30:
            break
        x, y, r = grid_init(rows, cols)
        evaluate_and_update(x, y, r)
        
        if time.time() < deadline - 20:
            x_opt, y_opt, r_opt = optimize_with_penalties(x, y, r, max_iter=200)
            evaluate_and_update(x_opt, y_opt, r_opt)
    
    # 3. Random restarts
    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            x, y, r = random_init()
            evaluate_and_update(x, y, r)
            
            x_opt, y_opt, r_opt = optimize_with_penalties(x, y, r, max_iter=150)
            evaluate_and_update(x_opt, y_opt, r_opt)
    
    # 4. Final polish from best result
    if best_result is not None and time.time() < deadline - 10:
        x, y, r = best_result
        x_opt, y_opt, r_opt = sqp_optimize(x, y, r, max_iter=50)
        evaluate_and_update(x_opt, y_opt, r_opt)
        
        x_opt, y_opt, r_opt = optimize_with_penalties(x, y, r, max_iter=100)
        evaluate_and_update(x_opt, y_opt, r_opt)
    
    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side - 0.01
        return np.column_stack((x, y, r))
    
    x_best, y_best, r_best = best_result
    
    # Final validation
    if not is_valid(x_best, y_best, r_best):
        # Try to fix
        for i in range(N):
            r_best[i] = min(r_best[i], x_best[i], 1.0 - x_best[i], y_best[i], 1.0 - y_best[i])
            r_best[i] = max(r_best[i], 0.001)
        
        if not is_valid(x_best, y_best, r_best):
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                x[i] = (col + 0.5) / side
                y[i] = (row + 0.5) / side
                r[i] = 0.5 / side - 0.01
            return np.column_stack((x, y, r))
    
    return np.column_stack((x_best, y_best, r_best))

# EVOLVE-BLOCK-END
