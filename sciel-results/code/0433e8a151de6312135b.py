"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np


def _force_directed_packing(N, w, h, n_iters=2000, lr_start=0.002, lr_end=0.00005,
                            repulsion_strength=100.0, spring_strength=10.0,
                            wall_strength=50.0, contact_tol=0.01):
    """
    Force-directed layout for circle packing.
    Uses repulsive forces for overlaps, spring forces for near-contacts,
    and wall repulsion. Returns (x, y, r) arrays.
    """
    # Initialize: hexagonal-like grid layout
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    for i in range(N):
        col = i % cols
        row = i // cols
        # Offset odd rows for hexagonal packing
        offset = 0.5 if (row % 2 == 1) else 0.0
        x[i] = (col + 0.5 + offset * 0.3) * w / cols
        y[i] = (row + 0.5) * h / rows
        # Initial radius based on cell size
        r[i] = min(w / (cols + 0.5), h / (rows + 0.5)) * 0.3
    
    # Force-directed relaxation
    for it in range(n_iters):
        # Learning rate schedule (exponential decay)
        t = it / n_iters
        lr = lr_start * (lr_end / lr_start) ** t
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Circle-circle interactions
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                
                if dist_sq < 1e-12:
                    # Random small perturbation
                    dx = np.random.randn() * 1e-6
                    dy = np.random.randn() * 1e-6
                    dist_sq = dx * dx + dy * dy
                
                dist = np.sqrt(dist_sq)
                
                if dist < min_dist:
                    # Overlapping: strong repulsive force (quadratic for stability)
                    overlap = min_dist - dist
                    force = repulsion_strength * overlap * overlap / (dist + 1e-8)
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
                elif dist < min_dist * (1.0 + contact_tol):
                    # Near contact: spring force to maintain exact distance
                    diff = dist - min_dist
                    force = spring_strength * diff / (dist + 1e-8)
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
        
        # Boundary forces
        for i in range(N):
            # Left wall
            if x[i] < r[i]:
                penetration = r[i] - x[i]
                fx[i] += wall_strength * penetration
            # Right wall
            if x[i] > w - r[i]:
                penetration = x[i] - (w - r[i])
                fx[i] -= wall_strength * penetration
            # Bottom wall
            if y[i] < r[i]:
                penetration = r[i] - y[i]
                fy[i] += wall_strength * penetration
            # Top wall
            if y[i] > h - r[i]:
                penetration = y[i] - (h - r[i])
                fy[i] -= wall_strength * penetration
        
        # Update positions with velocity damping
        x += lr * fx
        y += lr * fy
        
        # Clamp to valid range
        for i in range(N):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
    
    # Grow radii iteratively
    for grow_iter in range(500):
        # Try to grow radii by small amount
        growth_factor = 1.002
        r_new = r * growth_factor
        
        # Check if we can fit with slightly larger radii
        valid = True
        for i in range(N):
            if x[i] < r_new[i] or x[i] > w - r_new[i]:
                valid = False
                break
            if y[i] < r_new[i] or y[i] > h - r_new[i]:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < r_new[i] + r_new[j] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            r = r_new
        else:
            # Try smaller growth
            growth_factor = 1.001
            r_new = r * growth_factor
            valid = True
            for i in range(N):
                if x[i] < r_new[i] or x[i] > w - r_new[i]:
                    valid = False
                    break
                if y[i] < r_new[i] or y[i] > h - r_new[i]:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist < r_new[i] + r_new[j] - 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                r = r_new
            else:
                break
    
    return x, y, r


def _validate_packing(x, y, r, w, h, tol=1e-6):
    """Check if packing is valid."""
    for i in range(len(x)):
        if r[i] < -tol:
            return False
        if x[i] - r[i] < -tol or x[i] + r[i] > w + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > h + tol:
            return False
    
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - tol:
                return False
    
    return True


def construct():
    """
    Pack 21 circles in a rectangle of perimeter at most 4.
    Use force-directed contact graph layout.
    """
    best_val = -1
    best_result = None
    
    # Try multiple aspect ratios
    # Perimeter = 2(w+h) = 4, so w+h = 2
    aspect_ratios = np.linspace(0.6, 1.4, 20)
    
    for w in aspect_ratios:
        h = 2.0 - w
        
        # Run force-directed packing with multiple random restarts
        for seed in range(3):
            np.random.seed(seed * 1000 + int(w * 100))
            x, y, r = _force_directed_packing(N, w, h, n_iters=3000)
            
            # Validate
            if _validate_packing(x, y, r, w, h):
                val = np.sum(r)
                if val > best_val:
                    best_val = val
                    best_result = np.column_stack([x, y, r])
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
