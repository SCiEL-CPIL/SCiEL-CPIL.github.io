"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
def _feasible(circles, tol=1e-7):
    """Check if circles are feasible: inside rectangle and non-overlapping."""
    n = len(circles)
    if n == 0:
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs <= 0):
        return False
    W = xs.max() - xs.min() + 2 * rs.max()  # rough
    # Compute bounding box
    xmin = (xs - rs).min()
    xmax = (xs + rs).max()
    ymin = (ys - rs).min()
    ymax = (ys + rs).max()
    W = xmax - xmin
    H = ymax - ymin
    if W + H > 4 + tol:
        return False
    # Check circle-circle overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < rs[i] + rs[j] - tol:
                return False
    return True

def _sum_radii(circles):
    return circles[:, 2].sum()

def _grid_initial(n):
    """Create a grid-like initial packing of n circles in a rectangle of perimeter 4."""
    # Start with a square of side 1 (perimeter 4)
    # Place circles in a grid pattern
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    # Scale to fit in 1x1 square
    # Each cell has size 1/cols x 1/rows
    # Radius = min(1/cols, 1/rows) / 2 - epsilon
    eps = 1e-4
    r = min(1.0 / cols, 1.0 / rows) / 2 - eps
    circles = []
    for i in range(cols):
        for j in range(rows):
            if len(circles) >= n:
                break
            x = (i + 0.5) / cols
            y = (j + 0.5) / rows
            circles.append([x, y, r])
    return np.array(circles[:n])

def _push_apart_optimize(circles, max_iter=5000, lr=0.01, beta=50.0, deadline=None):
    """
    Push-apart dynamics: circles repel when overlapping (smooth potential),
    and we try to increase radii while maintaining feasibility.
    """
    circles = circles.copy()
    n = len(circles)
    best = circles.copy()
    best_sum = _sum_radii(circles)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        xs = circles[:, 0].copy()
        ys = circles[:, 1].copy()
        rs = circles[:, 2].copy()
        
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)
        
        # Circle-circle repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-10:
                    dx = 1e-8
                    dy = 0
                    d = 1e-8
                overlap = rs[i] + rs[j] - d
                if overlap > 0:
                    # Smooth potential: force proportional to overlap
                    f = beta * overlap
                    fx[i] += f * dx / d
                    fy[i] += f * dy / d
                    fx[j] -= f * dx / d
                    fy[j] -= f * dy / d
                    # Try to shrink radii slightly when overlapping
                    fr[i] -= f * 0.5
                    fr[j] -= f * 0.5
                else:
                    # Small attraction to grow radii
                    fr[i] += 0.01
                    fr[j] += 0.01
        
        # Boundary constraints: keep circles inside bounding box
        # Compute current bounding box
        xmin = (xs - rs).min()
        xmax = (xs + rs).max()
        ymin = (ys - rs).min()
        ymax = (ys + rs).max()
        W = xmax - xmin
        H = ymax - ymin
        P = W + H
        
        if P > 4:
            # Need to shrink: apply inward pressure
            scale = 4.0 / P
            # Move centers toward center of bounding box
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            for i in range(n):
                fx[i] += 0.1 * (cx - xs[i])
                fy[i] += 0.1 * (cy - ys[i])
                fr[i] -= 0.05
        
        # Update positions and radii
        xs += lr * fx
        ys += lr * fy
        rs += lr * fr
        rs = np.maximum(rs, 1e-6)
        
        circles[:, 0] = xs
        circles[:, 1] = ys
        circles[:, 2] = rs
        
        s = _sum_radii(circles)
        if s > best_sum:
            best_sum = s
            best = circles.copy()
    
    return best

def _local_polish(circles, tol=1e-10, max_iter=10000):
    """High-precision polish: gradient-based optimization of sum of radii."""
    circles = circles.copy()
    n = len(circles)
    
    for it in range(max_iter):
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Compute gradients of sum(rs) subject to constraints
        # Use simple heuristic: try to increase each radius slightly
        # and adjust positions to maintain feasibility
        
        improved = False
        for i in range(n):
            # Try increasing radius of circle i
            dr = 1e-6
            new_r = rs[i] + dr
            
            # Check if we can adjust position to maintain feasibility
            # Simple: just check if it's still feasible
            circles_test = circles.copy()
            circles_test[i, 2] = new_r
            
            # Check circle-circle
            ok = True
            for j in range(n):
                if j == i:
                    continue
                dx = circles_test[i, 0] - circles_test[j, 0]
                dy = circles_test[i, 1] - circles_test[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                if d < new_r + circles_test[j, 2] - tol:
                    ok = False
                    break
            
            if ok:
                # Check bounding box
                xmin = (circles_test[:, 0] - circles_test[:, 2]).min()
                xmax = (circles_test[:, 0] + circles_test[:, 2]).max()
                ymin = (circles_test[:, 1] - circles_test[:, 2]).min()
                ymax = (circles_test[:, 1] + circles_test[:, 2]).max()
                if xmax - xmin + ymax - ymin <= 4 + tol:
                    circles = circles_test
                    improved = True
        
        if not improved:
            break
    
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Initial grid packing
    circles = _grid_initial(N)
    best = circles.copy()
    best_sum = _sum_radii(circles)
    
    # Multiple restarts with different initial configurations
    for restart in range(10):
        if time.time() > deadline:
            break
        
        if restart == 0:
            circ = _grid_initial(N)
        else:
            # Perturbed grid
            circ = _grid_initial(N).copy()
            circ[:, 0] += np.random.uniform(-0.01, 0.01, N)
            circ[:, 1] += np.random.uniform(-0.01, 0.01, N)
        
        # Push-apart optimization
        circ = _push_apart_optimize(circ, max_iter=3000, lr=0.005, beta=20.0, deadline=deadline)
        
        s = _sum_radii(circ)
        if s > best_sum:
            best_sum = s
            best = circ.copy()
        
        if time.time() > deadline:
            break
    
    # Final polish
    if time.time() < deadline - 5:
        best = _local_polish(best, tol=1e-9, max_iter=5000)
    
    return best
# EVOLVE-BLOCK-END
