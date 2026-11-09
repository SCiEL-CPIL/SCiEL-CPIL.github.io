import numpy as np
import time
import sys

N = 50
START_TIME = time.time()
TIME_LIMIT = 148.0  # Safety margin before 150s kill

def _score(points):
    """Compute the score: largest common radius."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points inside unit circle
    norms = np.hypot(points[:, 0], points[:, 1])
    if np.any(norms > 1.0):
        return 0.0
    # Min distance from any point to boundary
    boundary_min = 1.0 - np.max(norms)
    # Min pairwise distance / 2
    if N > 1:
        # Efficient pairwise distance calculation
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
    else:
        min_pair = boundary_min
    return min(min_pair, boundary_min)

def _energy(points, r):
    """
    Calculate energy for a given configuration and target radius r.
    Energy is sum of:
    1. Boundary violations: (r - (1 - |p_i|))^2 if p_i is too close to boundary
    2. Overlap violations: (2*r - |p_i - p_j|)^2 if circles overlap
    """
    norms = np.hypot(points[:, 0], points[:, 1])
    
    # Boundary constraint: 1 - |p_i| >= r  =>  |p_i| <= 1 - r
    # Violation is max(0, r - (1 - |p_i|)) = max(0, r + |p_i| - 1)
    # Let's use soft violation: (r + |p_i| - 1)^2 if r + |p_i| > 1
    b_viol = np.maximum(0, r + norms - 1.0)
    e_boundary = np.sum(b_viol**2)
    
    # Pairwise constraint: |p_i - p_j| >= 2*r
    # Violation is max(0, 2*r - |p_i - p_j|)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(d, np.inf)
    
    # We only care about upper triangle to avoid double counting
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = d[i_idx, j_idx]
    
    p_viol = np.maximum(0, 2.0 * r - pair_dists)
    e_pair = np.sum(p_viol**2)
    
    return e_boundary + e_pair

def _hexagonal_initial(spacing):
    """Generate hexagonal lattice points inside unit disk."""
    points = []
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        return None
    # Select N closest to origin
    norms = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(norms, kind='stable')[:N]
    return points[idx]

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_points = None
    best_score = 0.0
    
    rng = np.random.default_rng(42)
    
    # Initializations
    initializations = []
    
    # Hexagonal lattices
    for spacing in [0.20, 0.22, 0.24, 0.26, 0.28]:
        pts = _hexagonal_initial(spacing)
        if pts is not None:
            initializations.append(pts)
            
    # Random
    for _ in range(3):
        points = []
        for _ in range(N):
            while True:
                x = rng.uniform(-0.9, 0.9)
                y = rng.uniform(-0.9, 0.9)
                if x * x + y * y <= 0.81:
                    points.append([x, y])
                    break
        initializations.append(np.array(points))

    # Simulated Annealing Strategy
    # We will run SA for each initialization, trying to maximize r.
    # The SA will operate on the positions, with a temperature schedule.
    # Objective: Minimize Energy(points, r_current) where r_current is a target radius we try to increase.
    # Alternatively, standard SA on positions with a penalty function that depends on a global r.
    
    # Let's use a simpler approach: 
    # 1. Start with a valid configuration (r=0).
    # 2. Gradually increase r.
    # 3. For each r, run SA to minimize violations.
    # 4. If violations < epsilon, accept new r.
    # 5. If stuck, cool down and try to escape? Or just increase r slowly.
    
    # Actually, a more robust SA for this specific problem (max-min) is:
    # Energy = sum of violations for a fixed r.
    # We want to find the max r such that min Energy(r) is close to 0.
    
    # Let's implement a "r-increasing" SA.
    # State: points
    # Parameter: r
    # Loop:
    #   - Run SA to minimize E(points, r)
    #   - If E < threshold, increase r by delta_r
    #   - Else, decrease r or keep and try different SA params?
    #   - Stop when time limit or r converges.
    
    # To be efficient, we'll use a fixed number of SA steps per r increment.
    
    for init_idx, init_points in enumerate(initializations):
        if time.time() - START_TIME > TIME_LIMIT * 0.85:
            break
            
        # Check current score
        current_score = _score(init_points)
        
        # Initialize SA state
        points = init_points.copy()
        r = current_score
        # Ensure r is small enough to start
        if r < 0.1:
            r = 0.1
            
        # If current score is already high, start from there
        # Otherwise, start from a safe radius
        # Let's start from 0.1 and go up, or from current score.
        # Starting from current score is better if it's valid.
        
        # SA Parameters
        T_start = 0.05
        T_end = 0.001
        alpha = 0.9995  # Cooling rate
        steps_per_r = 500  # SA steps for each r increment
        dr = 0.001       # Radius increment
        
        # We need to be careful with time.
        # Total time per init ~ 20s.
        # 500 steps * (cost of energy calc)
        # Energy calc is O(N^2) = 2500 ops. Vectorized in numpy, very fast.
        # 500 steps * 1000 r-increments = 500k steps.
        # 500k * 2500 flops = 1.25e9 flops. NumPy can do ~1e8-1e9 flops/sec.
        # So ~1-2 seconds per init? Should be fine.
        
        # Let's refine the SA loop.
        # Instead of fixed steps per r, let's run a long SA with adaptive r.
        
        # Actually, a common trick is to minimize a function that penalizes violations heavily.
        # Let's stick to the r-increment approach.
        
        success_count = 0
        fail_count = 0
        
        # Precompute indices for speed
        i_idx, j_idx = np.triu_indices(N, k=1)
        
        while time.time() - START_TIME < TIME_LIMIT * 0.85:
            # Run SA steps for current r
            for _ in range(steps_per_r):
                if time.time() - START_TIME > TIME_LIMIT * 0.85:
                    break
                    
                # Calculate current energy
                E_curr = _energy(points, r)
                
                # Generate move: pick random point, move random distance
                i = rng.integers(N)
                # Move size should be related to T
                move_mag = T_start * 0.1 # Simplified, let's use fixed small move
                # Better: move_mag = 0.01 * (T_start + 0.01)
                dx = rng.uniform(-0.01, 0.01)
                dy = rng.uniform(-0.01, 0.01)
                
                new_points = points.copy()
                new_points[i, 0] += dx
                new_points[i, 1] += dy
                
                # Calculate new energy
                E_new = _energy(new_points, r)
                
                # Accept if better, or with prob exp(-(E_new - E_curr)/T)
                if E_new < E_curr:
                    points = new_points
                    success_count += 1
                    fail_count = 0
                else:
                    prob = np.exp(-(E_new - E_curr) / max(T_start, 1e-6))
                    if rng.random() < prob:
                        points = new_points
                        success_count += 1
                        fail_count = 0
                    else:
                        fail_count += 1
                
                # Cool down slightly
                T_start *= alpha
                
                # If T is too low, reset? Or stop?
                # For this inner loop, let's just cool.
                
            # After steps_per_r, check if we can increase r
            E_final = _energy(points, r)
            
            # Threshold for "valid"
            # Energy is sum of squared violations.
            # If min distance is d, violation is (2r-d)^2.
            # If 50 pairs are close, energy can be large.
            # Let's use a relative threshold or absolute.
            # Absolute: 1e-4?
            # If E < 1e-3, we consider it "good enough" to increase r.
            
            if E_final < 1e-3:
                # Increase r
                r += dr
                # Reset temperature for next phase?
                T_start = max(T_start, 0.01)
                # If we keep increasing r, we might need to re-optimize more aggressively
                # But let's keep it simple.
            else:
                # Decrease r or keep?
                # If we fail, we might be stuck.
                # Let's try a small decrease to find a valid state, then increase again.
                r = max(0.1, r - dr * 0.5)
                # Increase temperature to escape local min
                T_start = min(0.1, T_start * 1.5)
                
            # Update best score
            current_score = _score(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                
            # Convergence check?
            # If r hasn't increased in many iterations, stop this init.
            # We don't track r increases explicitly, but if we keep failing, we stop.
            # Let's add a counter for "no progress"
            # For simplicity, we rely on time limit.
            
            # Break if r is very high?
            if r > 0.15:
                break

    # Final Polish
    # The SA might stop at a slightly suboptimal point.
    # Use a few iterations of local optimization (like the original) to polish.
    # But the directive says "DIFFERENT strategy".
    # However, "polishing" is standard.
    # Let's just return the best found by SA.
    
    if best_points is None:
        # Fallback
        best_points = _hexagonal_initial(0.25)
        if best_points is None:
            best_points = np.zeros((N, 2))

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
