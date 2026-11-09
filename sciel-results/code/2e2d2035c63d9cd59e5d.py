import numpy as np
import time
from scipy.optimize import linprog

N = 26

def is_valid(x, y, r, tol=1e-6):
    if np.any(r < -tol):
        return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    if np.any(dist < min_dist - tol):
        return False
    return True

def compute_forces(x, y, r, k_repel, k_attract):
    """
    Compute forces on each circle.
    - Repulsion: proportional to overlap depth, pushes circles apart
    - Attraction: pulls circles toward the center of available empty space
    """
    n = len(r)
    fx = np.zeros(n)
    fy = np.zeros(n)
    
    # Repulsion forces between overlapping circles
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    
    # Avoid division by zero
    dist_safe = np.where(dist > 1e-8, dist, 1e-8)
    
    # Overlap depth (positive when overlapping)
    overlap = np.maximum(0, min_dist - dist)
    np.fill_diagonal(overlap, 0)
    
    # Unit vectors
    ux = dx / dist_safe
    uy = dy / dist_safe
    
    # Force on i from j: k_repel * overlap_ij * (x_i - x_j) / dist_ij
    # This pushes i away from j
    fx += k_repel * np.sum(overlap * ux, axis=1)
    fy += k_repel * np.sum(overlap * uy, axis=1)
    
    # Attraction toward center of available empty space
    # Simple approach: attract toward (0.5, 0.5) with weak force
    # But better: compute distance to boundaries and attract toward the side with more room
    # For now, use a weak centering force
    cx = 0.5 - x
    cy = 0.5 - y
    fx += k_attract * cx
    fy += k_attract * cy
    
    # Boundary forces: push circles away from walls
    # If x - r < 0, push right
    left_violation = np.maximum(0, r - x)
    right_violation = np.maximum(0, r - (1 - x))
    bottom_violation = np.maximum(0, r - y)
    top_violation = np.maximum(0, r - (1 - y))
    
    fx += k_repel * (left_violation - right_violation)
    fy += k_repel * (bottom_violation - top_violation)
    
    return fx, fy

def local_search(x, y, r, max_iter=1000, time_limit=20.0):
    """
    Local search: try to increase radii while maintaining validity.
    Uses a greedy approach: for each circle, try to increase its radius
    by moving it slightly if needed.
    """
    start_time = time.time()
    n = len(r)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # For each circle, try to increase its radius
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            
            # Try increasing radius by a small amount
            dr = 0.001
            new_r = r[i] + dr
            
            # Check if we can fit with the new radius
            # First, check if current position works
            if is_valid(x, y, np.where(np.arange(n) == i, new_r, r), tol=1e-7):
                r[i] = new_r
                improved = True
                continue
            
            # Try moving the circle to find a better position
            # Try small perturbations in x and y
            best_new_pos = None
            best_dr = 0
            
            for dx_try in np.linspace(-0.02, 0.02, 5):
                for dy_try in np.linspace(-0.02, 0.02, 5):
                    new_x = x[i] + dx_try
                    new_y = y[i] + dy_try
                    
                    # Check bounds
                    if new_x < new_r - 1e-8 or new_x > 1 - new_r + 1e-8:
                        continue
                    if new_y < new_r - 1e-8 or new_y > 1 - new_r + 1e-8:
                        continue
                    
                    # Check validity
                    test_x = x.copy()
                    test_y = y.copy()
                    test_r = r.copy()
                    test_x[i] = new_x
                    test_y[i] = new_y
                    test_r[i] = new_r
                    
                    if is_valid(test_x, test_y, test_r, tol=1e-7):
                        best_new_pos = (new_x, new_y)
                        best_dr = dr
                        break
                if best_new_pos is not None:
                    break
            
            if best_new_pos is not None:
                x[i] = best_new_pos[0]
                y[i] = best_new_pos[1]
                r[i] = new_r
                improved = True
        
        if not improved:
            # Try larger steps
            dr = 0.005
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                
                new_r = r[i] + dr
                if is_valid(x, y, np.where(np.arange(n) == i, new_r, r), tol=1e-7):
                    r[i] = new_r
                    improved = True
            if not improved:
                break
    
    return x, y, r

def force_directed_relax(x, y, r, max_iter=500, time_limit=30.0):
    """
    Force-directed relaxation: use physical forces to resolve overlaps
    and pack circles tightly.
    """
    start_time = time.time()
    n = len(r)
    
    # Adaptive parameters
    k_repel = 10.0
    k_attract = 0.1
    lr = 0.01
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        fx, fy = compute_forces(x, y, r, k_repel, k_attract)
        
        # Update positions
        x += lr * fx
        y += lr * fy
        
        # Keep circles within bounds
        for i in range(n):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        
        # Decay learning rate
        lr *= 0.999
        
        # Check if we've reached a valid configuration
        if is_valid(x, y, r, tol=1e-6):
            # We can stop early if we're making good progress
            pass
        
        # Increase repulsion strength as we go to enforce constraints
        if iteration % 50 == 0:
            k_repel *= 1.2
            k_attract *= 0.9
    
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_config = None
    
    # Multiple restarts with different initializations
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Grid initialization
            side = 6
            xs = []
            ys = []
            rs = []
            count = 0
            for i in range(side):
                for j in range(side):
                    if count >= N:
                        break
                    xs.append((i + 0.5) / side)
                    ys.append((j + 0.5) / side)
                    rs.append(0.5 / side * 0.8)
                    count += 1
            x = np.array(xs)
            y = np.array(ys)
            r = np.array(rs)
        elif restart == 1:
            # Random initialization
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            r = np.random.uniform(0.02, 0.06, N)
        elif restart == 2:
            # Hexagonal packing initialization
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            row = 0
            col = 0
            count = 0
            spacing = 0.18
            while count < N and row < 10:
                offset = (row % 2) * spacing * 0.5
                x[count] = spacing * 0.5 + col * spacing + offset
                y[count] = spacing * 0.5 + row * spacing * 0.866
                r[count] = spacing * 0.45
                count += 1
                col += 1
                if x[count - 1] > 1 - spacing:
                    col = 0
                    row += 1
            x = x[:N]
            y = y[:N]
            r = r[:N]
        else:
            # Perturbed previous best
            if best_config is not None:
                bx, by, br = best_config
                x = bx + np.random.normal(0, 0.02, N)
                y = by + np.random.normal(0, 0.02, N)
                r = br + np.random.normal(0, 0.005, N)
                r = np.maximum(r, 0.01)
            else:
                x = np.random.uniform(0.1, 0.9, N)
                y = np.random.uniform(0.1, 0.9, N)
                r = np.random.uniform(0.02, 0.06, N)
        
        # Clip to valid range
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        # Phase 1: Force-directed relaxation to resolve overlaps
        x, y, r = force_directed_relax(x, y, r, max_iter=300, time_limit=20.0)
        
        # Phase 2: Local search to increase radii
        x, y, r = local_search(x, y, r, max_iter=500, time_limit=15.0)
        
        # Check validity and update best
        if is_valid(x, y, r, tol=1e-6):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_config = (x.copy(), y.copy(), r.copy())
        
        # Also try a second pass of local search with smaller steps
        if time.time() < deadline - 10:
            x2, y2, r2 = local_search(x, y, r, max_iter=300, time_limit=10.0)
            if is_valid(x2, y2, r2, tol=1e-6):
                score2 = np.sum(r2)
                if score2 > best_score:
                    best_score = score2
                    best_config = (x2.copy(), y2.copy(), r2.copy())
    
    # Final check
    if best_config is None:
        # Fallback: simple grid
        side = 6
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01))
                count += 1
        return np.array(cells)
    
    bx, by, br = best_config
    return np.column_stack((bx, by, br))
