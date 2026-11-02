"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
def _valid_pack(pts, radii):
    """Check if packing is valid: all circles inside [0,1]^2 with disjoint interiors."""
    for i in range(len(pts)):
        x, y = pts[i]
        r = radii[i]
        if x < -1e-9 or x > 1 + 1e-9 or y < -1e-9 or y > 1 + 1e-9 or r < 1e-12:
            return False
        # Check boundary containment
        if x < r - 1e-9 or x > 1 - r + 1e-9 or y < r - 1e-9 or y > 1 - r + 1e-9:
            return False
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-9:
                return False
    return True


def _score(pts, radii):
    """Score: sum of radii if valid, else negative."""
    if not _valid_pack(pts, radii):
        return -np.inf
    return float(np.sum(radii))


def _pbdsolve(pts, radii, n_iter=5000, dt=0.002, beta=0.5, gamma=0.1):
    """
    Position-Based Dynamics for circle packing.
    - Circles repel each other (distance constraint: dist >= r_i + r_j)
    - Circles are pushed away from boundaries (dist from boundary >= r_i)
    - Global pressure: circles are pushed outward to maximize radii
    
    Returns updated pts, radii.
    """
    pts = np.array(pts, dtype=float).copy()
    radii = np.array(radii, dtype=float).copy()
    n = len(pts)
    masses = radii ** 2  # mass proportional to radius squared
    masses = np.maximum(masses, 1e-6)
    
    for it in range(n_iter):
        # --- Boundary constraints (Gauss-Seidel) ---
        for i in range(n):
            r = radii[i]
            # Left boundary
            if pts[i][0] < r:
                pts[i][0] = r
            # Right boundary
            if pts[i][0] > 1 - r:
                pts[i][0] = 1 - r
            # Bottom boundary
            if pts[i][1] < r:
                pts[i][1] = r
            # Top boundary
            if pts[i][1] > 1 - r:
                pts[i][1] = 1 - r
        
        # --- Pairwise distance constraints (Gauss-Seidel) ---
        for i in range(n):
            for j in range(i + 1, n):
                dx = pts[j][0] - pts[i][0]
                dy = pts[j][1] - pts[i][1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    if dist < 1e-12:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    overlap = min_dist - dist
                    # Move apart proportionally to inverse mass
                    wi = masses[j] / (masses[i] + masses[j])
                    wj = masses[i] / (masses[i] + masses[j])
                    pts[i][0] -= wi * overlap * dx
                    pts[i][1] -= wi * overlap * dy
                    pts[j][0] += wj * overlap * dx
                    pts[j][1] += wj * overlap * dy
        
        # --- Grow radii slightly (pressure) ---
        # We want to maximize sum of radii, so we nudge radii up
        # but only if the configuration has room
        growth = 0.0001
        radii += growth
        # Clamp radii to reasonable bounds
        radii = np.clip(radii, 1e-4, 0.5)
    
    # Final cleanup: ensure validity
    for i in range(n):
        radii[i] = min(radii[i], pts[i][0], 1 - pts[i][0], pts[i][1], 1 - pts[i][1])
        if radii[i] < 1e-4:
            radii[i] = 1e-4
            pts[i][0] = radii[i]
            pts[i][1] = radii[i]
    
    return pts, radii


def _local_optimize(pts, radii, n_iter=2000):
    """
    Local optimization: iteratively try to increase each radius while maintaining validity.
    Uses a simple hill-climbing approach with small perturbations.
    """
    pts = np.array(pts, dtype=float).copy()
    radii = np.array(radii, dtype=float).copy()
    n = len(pts)
    
    for _ in range(n_iter):
        improved = False
        for i in range(n):
            # Try to increase radius i slightly
            delta = 0.001
            for trial in range(10):
                new_r = radii[i] + delta
                if new_r > 0.5:
                    break
                # Check if we can increase radius i
                # Need to check all constraints involving circle i
                valid = True
                # Boundary
                if new_r > pts[i][0] or new_r > 1 - pts[i][0] or new_r > pts[i][1] or new_r > 1 - pts[i][1]:
                    valid = False
                else:
                    # Pairwise
                    for j in range(n):
                        if j == i:
                            continue
                        dx = pts[i][0] - pts[j][0]
                        dy = pts[i][1] - pts[j][1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist < new_r + radii[j]:
                            valid = False
                            break
                if valid:
                    radii[i] = new_r
                    improved = True
                    break
                else:
                    delta *= 0.5
        
        if not improved:
            break
    
    return pts, radii


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, leave margin
    
    best_score = -np.inf
    best_pts = None
    best_radii = None
    
    # Initial configuration: grid-like with some randomness
    np.random.seed(42)
    
    # Try multiple random initial configurations
    n_attempts = 0
    while time.time() < deadline - 5:
        n_attempts += 1
        
        # Generate random initial positions and radii
        # Start with a reasonable grid and add jitter
        side = 5  # 5x5 grid gives 25 spots, we need 26
        pts = []
        radii = []
        
        # Place circles in a grid pattern with some randomness
        for i in range(5):
            for j in range(5):
                if len(pts) >= N:
                    break
                x = (i + 0.5) / 5.0 + np.random.uniform(-0.02, 0.02)
                y = (j + 0.5) / 5.0 + np.random.uniform(-0.02, 0.02)
                r = 0.08 + np.random.uniform(-0.01, 0.01)
                pts.append([x, y])
                radii.append(r)
        
        # Add one more circle at a random position
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r = 0.08
        pts.append([x, y])
        radii.append(r)
        
        pts = np.array(pts[:N])
        radii = np.array(radii[:N])
        
        # Run PBD solver
        pts, radii = _pbdsolve(pts, radii, n_iter=3000)
        
        # Local optimization
        pts, radii = _local_optimize(pts, radii, n_iter=500)
        
        # Check validity and score
        sc = _score(pts, radii)
        if sc > best_score:
            best_score = sc
            best_pts = pts.copy()
            best_radii = radii.copy()
        
        # If we have a good solution, try to improve it further
        if best_score > 1.5:
            # Do more local optimization on the best solution
            pts2, radii2 = _local_optimize(best_pts.copy(), best_radii.copy(), n_iter=1000)
            sc2 = _score(pts2, radii2)
            if sc2 > best_score:
                best_score = sc2
                best_pts = pts2.copy()
                best_radii = radii2.copy()
    
    # Final result
    if best_pts is None:
        # Fallback: grid of equal circles
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_pts[i][0], best_pts[i][1], best_radii[i]]
    return result
# EVOLVE-BLOCK-END
