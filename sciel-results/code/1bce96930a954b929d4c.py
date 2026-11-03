"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    for cx, cy, r in circles:
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True


def _total_radius(circles):
    return sum(c[2] for c in circles)


def _force_relax(circles, w, h, iterations=2000, dt=0.01):
    """
    Force-directed relaxation to resolve overlaps and boundary violations.
    Uses repulsive forces between overlapping circles and boundary forces.
    """
    n = len(circles)
    pos = np.array([c[:2] for c in circles], dtype=np.float64)
    rad = np.array([c[2] for c in circles], dtype=np.float64)
    
    for it in range(iterations):
        forces = np.zeros_like(pos)
        
        # Boundary forces
        for i in range(n):
            # Left boundary
            if pos[i, 0] - rad[i] < 0:
                forces[i, 0] += (rad[i] - pos[i, 0]) * 100
            # Right boundary
            if pos[i, 0] + rad[i] > w:
                forces[i, 0] -= (pos[i, 0] + rad[i] - w) * 100
            # Bottom boundary
            if pos[i, 1] - rad[i] < 0:
                forces[i, 1] += (rad[i] - pos[i, 1]) * 100
            # Top boundary
            if pos[i, 1] + rad[i] > h:
                forces[i, 1] -= (pos[i, 1] + rad[i] - h) * 100
        
        # Repulsive forces between overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]
                
                if dist < min_dist:
                    if dist < 1e-9:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push_dir = np.array([np.cos(angle), np.sin(angle)])
                    else:
                        push_dir = np.array([dx / dist, dy / dist])
                    
                    overlap = min_dist - dist
                    force_mag = overlap * 50
                    
                    forces[i] -= push_dir * force_mag
                    forces[j] += push_dir * force_mag
        
        # Apply forces with damping
        pos += forces * dt
        # Damping
        forces *= 0.95
        
        # Hard clamp to boundaries
        for i in range(n):
            pos[i, 0] = np.clip(pos[i, 0], rad[i], max(rad[i], w - rad[i]))
            pos[i, 1] = np.clip(pos[i, 1], rad[i], max(rad[i], h - rad[i]))
    
    # Final validation and cleanup
    new_circles = [(pos[i, 0], pos[i, 1], rad[i]) for i in range(n)]
    
    # One more pass to ensure no overlaps
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = new_circles[j][0] - new_circles[i][0]
                dy = new_circles[j][1] - new_cicles[i][1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = new_circles[i][2] + new_circles[j][2]
                
                if dist < min_dist - 1e-9:
                    if dist < 1e-9:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_dist - dist) / 2.0
                        new_circles[i] = (new_circles[i][0] - np.cos(angle) * push,
                                          new_circles[i][1] - np.sin(angle) * push,
                                          new_circles[i][2])
                        new_circles[j] = (new_circles[j][0] + np.cos(angle) * push,
                                          new_circles[j][1] + np.sin(angle) * push,
                                          new_circles[j][2])
                    else:
                        push = (min_dist - dist) / 2.0
                        ux = dx / dist
                        uy = dy / dist
                        new_circles[i] = (new_circles[i][0] - ux * push,
                                          new_circles[i][1] - uy * push,
                                          new_circles[i][2])
                        new_circles[j] = (new_circles[j][0] + ux * push,
                                          new_circles[j][1] + uy * push,
                                          new_circles[j][2])
                    changed = True
        
        # Boundary clamp
        for i in range(n):
            cx, cy, r = new_circles[i]
            cx = np.clip(cx, r, max(r, w - r))
            cy = np.clip(cy, r, max(r, h - r))
            new_circles[i] = (cx, cy, r)
        
        if not changed:
            break
    
    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles
    
    return circles


def _grow_radii(circles, w, h, deadline, max_steps=50):
    """
    Incrementally grow radii while resolving overlaps.
    """
    n = len(circles)
    current = [list(c) for c in circles]
    
    for step in range(max_steps):
        if time.time() > deadline - 2.0:
            break
        
        # Try to grow all radii by a small amount
        growth = 0.002
        new_rad = [r + growth for r in [c[2] for c in current]]
        
        # Check if this growth is feasible
        test_circles = [(current[i][0], current[i][1], new_rad[i]) for i in range(n)]
        
        # Relax to resolve overlaps
        relaxed = _force_relax(test_circles, w, h, iterations=500, dt=0.005)
        
        if _validate(relaxed, w, h, tol=1e-6):
            current = [list(c) for c in relaxed]
        else:
            # Try smaller growth
            growth = 0.0005
            new_rad = [r + growth for r in [c[2] for c in current]]
            test_circles = [(current[i][0], current[i][1], new_rad[i]) for i in range(n)]
            relaxed = _force_relax(test_circles, w, h, iterations=500, dt=0.005)
            
            if _validate(relaxed, w, h, tol=1e-6):
                current = [list(c) for c in relaxed]
            else:
                break
    
    return [tuple(c) for c in current]


def _local_optimize(circles, w, h, deadline):
    """
    Local optimization using Nelder-Mead on positions only, with radii fixed.
    Then try to grow radii.
    """
    if time.time() > deadline - 5.0:
        return circles
    
    n = len(circles)
    initial_pos = np.array([c[:2] for c in circles])
    radii = np.array([c[2] for c in circles])
    
    def objective(x):
        """Minimize overlap + boundary violation (should be 0 for valid config)."""
        pos = x.reshape(n, 2)
        penalty = 0.0
        
        for i in range(n):
            # Boundary
            penalty += max(0, radii[i] - pos[i, 0]) ** 2
            penalty += max(0, pos[i, 0] + radii[i] - w) ** 2
            penalty += max(0, radii[i] - pos[i, 1]) ** 2
            penalty += max(0, pos[i, 1] + radii[i] - h) ** 2
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return penalty
    
    try:
        res = minimize(objective, initial_pos, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.fun < 1e-6:
            new_pos = res.x.reshape(n, 2)
            new_circles = [(new_pos[i, 0], new_pos[i, 1], radii[i]) for i in range(n)]
            
            if _validate(new_circles, w, h, tol=1e-6):
                # Try to grow radii from this improved position
                grown = _grow_radii(new_circles, w, h, deadline, max_steps=20)
                if _validate(grown, w, h, tol=1e-6):
                    return grown
                return new_circles
    except Exception:
        pass
    
    return circles


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_circles = None
    best_score = 0.0
    
    # Try multiple aspect ratios (w + h = 2 for perimeter 4)
    aspect_ratios = [
        (1.0, 1.0),
        (0.8, 1.2),
        (1.2, 0.8),
        (0.7, 1.3),
        (1.3, 0.7),
        (0.9, 1.1),
        (1.1, 0.9),
    ]
    
    for w, h in aspect_ratios:
        if time.time() > deadline - 15.0:
            break
        
        # Initialize with a grid pattern
        # For 21 circles, use a 5x5 grid and remove 4 corners
        cols, rows = 5, 5
        circles = []
        
        # Create 5x5 grid positions
        for i in range(rows):
            for j in range(cols):
                cx = (j + 0.5) / cols * w
                cy = (i + 0.5) / rows * h
                circles.append((cx, cy, 0.0))
        
        # Remove 4 corner circles to get 21
        # Remove (0,0), (0,4), (4,0), (4,4)
        remove_indices = [(0, 0), (0, 4), (4, 0), (4, 4)]
        keep = []
        for idx, (cx, cy, r) in enumerate(circles):
            i = idx // cols
            j = idx % cols
            if (i, j) not in remove_indices:
                keep.append(idx)
        
        circles = [circles[i] for i in keep[:N]]
        
        # Set initial radius based on grid spacing
        base_r = min(w / cols, h / rows) * 0.4
        circles = [(cx, cy, base_r) for cx, cy, r in circles]
        
        # Relax to ensure validity
        relaxed = _force_relax(circles, w, h, iterations=1000, dt=0.01)
        
        if _validate(relaxed, w, h, tol=1e-6):
            score = _total_radius(relaxed)
            
            # Try to grow radii
            if time.time() < deadline - 10.0:
                grown = _grow_radii(relaxed, w, h, deadline, max_steps=30)
                if _validate(grown, w, h, tol=1e-6):
                    grown_score = _total_radius(grown)
                    if grown_score > score:
                        relaxed = grown
                        score = grown_score
            
            # Local optimization
            if time.time() < deadline - 5.0:
                optimized = _local_optimize(relaxed, w, h, deadline)
                if _validate(optimized, w, h, tol=1e-6):
                    opt_score = _total_radius(optimized)
                    if opt_score > score:
                        relaxed = optimized
                        score = opt_score
            
            if score > best_score:
                best_score = score
                best_circles = relaxed
    
    if best_circles is not None:
        return np.array(best_circles)
    
    # Fallback: simple grid
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.05)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
