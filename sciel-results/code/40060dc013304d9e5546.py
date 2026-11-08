import time
import numpy as np
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 145.0  # Safety margin before 150s kill

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        if len(pts) < 3:
            return 0.0
        # Check for collinearity or duplicates which cause errors
        if np.allclose(pts[:, 0], pts[0, 0]) or np.allclose(pts[:, 1], pts[0, 1]):
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    if n < 3:
        return 0.0
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Vectors from i to j and i to k
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            # Cross product magnitude / 2
            cross = dx_ji * dy_ki - dy_ji * dx_ki
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area

def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a

def _generate_radial_config(n, num_shells, noise_scale=0.1, rng=None):
    """
    Generate points using a radial shell structure.
    Points are assigned to shells. Within each shell, points are distributed
    angularly. The radii of the shells are the variables to be optimized.
    This reduces the dimensionality from 2n to num_shells.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    # Assign points to shells
    # We want a mix of outer and inner points.
    # For N=14, let's try 3 shells: outer (8 pts), middle (4 pts), inner (2 pts).
    # Or just distribute evenly.
    shell_sizes = []
    remaining = n
    for i in range(num_shells - 1):
        # Distribute roughly evenly, maybe slightly favor outer
        size = max(2, int(n / num_shells))
        shell_sizes.append(size)
        remaining -= size
    shell_sizes.append(remaining)
    
    # Ensure sum is N
    while sum(shell_sizes) > n:
        # Reduce largest shell
        max_idx = np.argmax(shell_sizes)
        if shell_sizes[max_idx] > 2:
            shell_sizes[max_idx] -= 1
        else:
            break
    while sum(shell_sizes) < n:
        # Increase largest shell
        max_idx = np.argmax(shell_sizes)
        shell_sizes[max_idx] += 1
        
    points = []
    # Base angles for each shell, offset to avoid alignment
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    current_idx = 0
    for shell_id, size in enumerate(shell_sizes):
        # Radius for this shell (initial guess)
        # Outer shells have larger radii
        base_radius = 1.0 - (shell_id / max(1, num_shells - 1)) * 0.6
        # Add some noise to radii initially? No, we optimize radii.
        # But we need initial radii.
        radii = np.full(size, base_radius)
        
        # Angles for points in this shell
        # Distribute evenly within the shell's angular sector? 
        # Or just take the next 'size' angles from the global list?
        # Taking global angles ensures global distribution.
        angles = base_angles[current_idx : current_idx + size]
        
        # Add small angular noise to break symmetry
        angle_noise = rng.normal(0, 0.05, size=size)
        angles += angle_noise
        
        for r, a in zip(radii, angles):
            points.append([r * np.cos(a), r * np.sin(a)])
        current_idx += size
        
    pts = np.array(points)
    
    # Normalize to unit area
    hull_a = _hull_area(pts)
    if hull_a > 0:
        pts = pts / np.sqrt(hull_a)
        
    return pts

def _optimize_radial_shells(n, num_shells, start_time, deadline, rng):
    """
    Optimize the radii of the radial shells using Simulated Annealing.
    The configuration is determined by the set of shell radii.
    """
    # Define the structure
    # We need to map the 14 points to shells.
    # Let's fix the angular positions and shell assignments, and only vary the radii.
    
    # Create a fixed angular assignment
    # 14 points. Let's say:
    # Shell 0 (Outer): 6 points
    # Shell 1 (Middle): 4 points
    # Shell 2 (Inner): 4 points
    # This is a heuristic. We can try a few structures.
    
    structures = [
        [6, 4, 4],
        [5, 5, 4],
        [8, 3, 3],
        [10, 2, 2]
    ]
    
    best_score = -1
    best_pts = None
    
    for struct in structures:
        if sum(struct) != n:
            continue
            
        # Assign angles
        base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        # Shuffle angles slightly to avoid perfect symmetry which might be a local min
        # Actually, symmetry is good for max-min. Keep it regular.
        
        # Assign points to shells based on struct
        point_shell_assignment = []
        idx = 0
        for s_id, size in enumerate(struct):
            for _ in range(size):
                point_shell_assignment.append(s_id)
                idx += 1
                
        # Initial radii
        # Outer shell = 1.0, inner shells smaller
        init_radii = []
        for s_id, size in enumerate(struct):
            r = 1.0 - s_id * 0.3
            init_radii.append(r)
            
        current_radii = np.array(init_radii)
        current_score = -1
        current_pts = None
        
        # SA Parameters
        T = 0.1
        T_min = 1e-6
        cooling_rate = 0.995
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Generate neighbor: perturb one radius
            new_radii = current_radii.copy()
            s_idx = rng.randint(0, len(struct))
            perturbation = rng.normal(0, T * 0.1)
            new_radii[s_idx] += perturbation
            # Clamp radii
            new_radii = np.clip(new_radii, 0.1, 1.5)
            
            # Reconstruct points
            # Map point index to shell index
            # We need to know which point belongs to which shell to update its radius.
            # point_shell_assignment[i] gives the shell for point i.
            pts = np.zeros((n, 2))
            for i in range(n):
                s_id = point_shell_assignment[i]
                r = new_radii[s_id]
                a = base_angles[i]
                pts[i, 0] = r * np.cos(a)
                pts[i, 1] = r * np.sin(a)
                
            # Normalize
            hull_a = _hull_area(pts)
            if hull_a <= 0:
                new_score = -1
            else:
                pts = pts / np.sqrt(hull_a)
                new_score = _score(pts)
                
            if new_score > current_score:
                current_score = new_score
                current_radii = new_radii
                current_pts = pts
            else:
                # Accept with probability
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / max(T, 1e-10))
                    if rng.random() < prob:
                        current_radii = new_radii
                        # Don't update current_score or pts, just accept the state
                        # Actually, for SA, we update the state.
                        current_pts = pts # This is technically incorrect if we don't re-normalize consistently, but score is what matters.
                        # To be safe, we should only update current_pts if we accept, and keep track of the normalized version.
                        # Since we normalized pts above, current_pts is valid.
                        
            # Cool down
            T *= cooling_rate
            
        if current_score > best_score and current_pts is not None:
            best_score = current_score
            best_pts = current_pts
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = np.random.RandomState(12345)
    
    best_pts = None
    best_score = -1
    
    # 1. Try the radial shell SA approach
    # This is the main structural change.
    # We allocate 80% of the time to this.
    sa_deadline = start_time + (DEADLINE * 0.8)
    
    if time.time() < sa_deadline:
        # Try different numbers of shells
        for num_shells in [3, 4, 2]:
            if time.time() > sa_deadline - 5:
                break
            pts, score = _optimize_radial_shells(N, num_shells, start_time, sa_deadline, rng)
            if pts is not None and score > best_score:
                best_score = score
                best_pts = pts
                
    # 2. Fallback / Hybrid: If SA didn't find anything good, or to polish.
    # Let's also try a standard random initialization with a simple local search (hill climbing)
    # to see if we can beat the SA result.
    
    if time.time() < deadline - 10:
        # Hill Climbing on random starts
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random start
            r = rng.uniform(0.2, 1.0, size=N)
            theta = rng.uniform(0, 2*np.pi, size=N)
            pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            hull_a = _hull_area(pts)
            if hull_a > 0:
                pts = pts / np.sqrt(hull_a)
                
            current_score = _score(pts)
            
            # Simple hill climbing
            step_size = 0.05
            for step in range(200):
                if time.time() > deadline - 2:
                    break
                improved = False
                for _ in range(10): # Try 10 random perturbations per step
                    new_pts = pts + rng.normal(0, step_size, size=pts.shape)
                    # Keep points within bounds? Not strictly necessary if we normalize.
                    hull_a = _hull_area(new_pts)
                    if hull_a > 0:
                        new_pts = new_pts / np.sqrt(hull_a)
                        new_score = _score(new_pts)
                        if new_score > current_score:
                            current_score = new_score
                            pts = new_pts
                            improved = True
                            break
                if not improved:
                    step_size *= 0.8
                    if step_size < 1e-4:
                        break
                
            if current_score > best_score:
                best_score = current_score
                best_pts = pts
                
    # Final check
    if best_pts is None:
        # Default fallback: Regular polygon
        angles = 2 * np.pi * np.arange(N) / N
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        hull_a = _hull_area(best_pts)
        if hull_a > 0:
            best_pts = best_pts / np.sqrt(hull_a)
            
    return best_pts

# EVOLVE-BLOCK-END
