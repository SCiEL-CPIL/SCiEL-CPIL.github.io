"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _build_contact_graph(x, y, r, w, h, tol=1e-4):
    """
    Build contact graph: pairs of circles that are touching or nearly touching.
    Returns list of (i, j) tuples for contacts.
    """
    contacts = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist <= min_dist + tol:
                contacts.append((i, j))
    
    # Also add wall contacts
    for i in range(len(x)):
        if abs(x[i] - r[i]) < tol:  # left wall
            contacts.append((i, -1))
        if abs(x[i] - (w - r[i])) < tol:  # right wall
            contacts.append((i, -2))
        if abs(y[i] - r[i]) < tol:  # bottom wall
            contacts.append((i, -3))
        if abs(y[i] - (h - r[i])) < tol:  # top wall
            contacts.append((i, -4))
    
    return contacts


def _force_directed_with_contacts(N, w, h, n_iters=3000, lr_start=0.005, lr_end=0.0001):
    """
    Force-directed layout with explicit contact graph.
    Returns (x, y, r) arrays.
    """
    # Initialize: grid layout
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    for i in range(N):
        col = i % cols
        row = i // cols
        x[i] = (col + 0.5) * w / cols
        y[i] = (row + 0.5) * h / rows
        r[i] = min(w / cols, h / rows) * 0.35
    
    # Force-directed relaxation with contact graph
    for it in range(n_iters):
        # Learning rate schedule
        t = it / n_iters
        lr = lr_start * (lr_end / lr_start) ** t
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Circle-circle interactions
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                
                if dist < 1e-10:
                    # Random small perturbation
                    dx = np.random.randn() * 1e-6
                    dy = np.random.randn() * 1e-6
                    dist = np.sqrt(dx * dx + dy * dy)
                
                if dist < min_dist:
                    # Overlapping: strong repulsive force
                    overlap = min_dist - dist
                    force = 50.0 * overlap  # strong repulsion
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                elif dist < min_dist * 1.02:
                    # Near contact: spring force to maintain exact distance
                    diff = dist - min_dist
                    force = 2.0 * diff  # spring force
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Boundary forces
        for i in range(N):
            # Left wall
            if x[i] < r[i]:
                penetration = r[i] - x[i]
                fx[i] += 20.0 * penetration
            # Right wall
            if x[i] > w - r[i]:
                penetration = x[i] - (w - r[i])
                fx[i] -= 20.0 * penetration
            # Bottom wall
            if y[i] < r[i]:
                penetration = r[i] - y[i]
                fy[i] += 20.0 * penetration
            # Top wall
            if y[i] > h - r[i]:
                penetration = y[i] - (h - r[i])
                fy[i] -= 20.0 * penetration
        
        # Update positions
        x += lr * fx
        y += lr * fy
        
        # Clamp to valid range
        for i in range(N):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
    
    # After relaxation, try to grow radii
    for grow_iter in range(100):
        # Try to grow radii
        r_new = r * 1.005
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
    aspect_ratios = np.linspace(0.5, 1.5, 25)
    
    for w in aspect_ratios:
        h = 2.0 - w
        
        # Run force-directed packing
        x, y, r = _force_directed_with_contacts(N, w, h, n_iters=5000)
        
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
