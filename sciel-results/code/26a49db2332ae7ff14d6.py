# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Strategy: Simulated Annealing with force-directed repulsion.
    We treat the points as particles repelling each other. 
    The "force" is designed to strongly push apart the closest pairs (to increase d_min)
    while a global constraint/penalty prevents the cluster from expanding infinitely (to keep d_max small).
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    def time_exceeded():
        return time.time() > deadline

    def get_score(points):
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2

    def compute_forces(points, temp):
        """
        Compute forces on each point.
        We want to maximize d_min. We use a repulsive force that is very strong for close pairs.
        F_ij = k * (1/d_ij^p) * (r_i - r_j)/d_ij
        To keep the system bounded, we also apply a weak attractive force to the center of mass
        or a penalty if d_max exceeds a target. 
        However, a simpler approach for ratio maximization is to normalize the system 
        so that d_max is fixed, and then maximize d_min.
        
        Let's use a potential energy approach:
        E = sum_{i<j} 1/d_ij^2  (This minimizes when points are far apart, but we want to MAXIMIZE min dist)
        Actually, we want to MAXIMIZE d_min. So we minimize -d_min.
        Since d_min is non-differentiable, we use a soft-min surrogate for the force calculation 
        or simply use a strong repulsive force that dominates for the closest pairs.
        
        Let's use: Force on i from j = C * (1/d_ij^alpha) * direction_ij
        This pushes all points apart. To prevent expansion, we scale the points 
        after each step so that the maximum distance is 1.0.
        Then we just maximize the minimum distance in this normalized space.
        """
        points = points.copy()
        # Normalize so max dist is 1
        dists = pdist(points, 'euclidean')
        d_max = np.max(dists)
        if d_max > 1e-10:
            points = points / d_max
        else:
            return np.zeros_like(points)
            
        # Recompute distances after normalization
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        
        # Force calculation
        # We want to increase d_min. 
        # Apply repulsive force proportional to 1/d^2 (Coulomb-like)
        # This creates a strong push for close pairs, weaker for far ones.
        forces = np.zeros_like(points)
        for i in range(n):
            for j in range(i + 1, n):
                rij = points[i] - points[j]
                d = np.linalg.norm(rij)
                if d < 1e-6:
                    # Random jitter to avoid singularity
                    rij += np.random.randn(2) * 1e-6
                    d = np.linalg.norm(rij)
                
                # Force magnitude: strong repulsion
                # Using 1/d^3 to make it very stiff for close pairs
                # F = k / d^2 * (rij/d) = k * rij / d^3
                f_mag = 1.0 / (d ** 3)
                f_vec = f_mag * rij
                
                forces[i] += f_vec
                forces[j] -= f_vec
        
        # Add a small random noise to escape local minima
        noise = np.random.randn(n, 2) * temp * 0.1
        forces += noise
        
        return forces

    def anneal_step(points, temp):
        """
        Perform one step of simulated annealing.
        """
        forces = compute_forces(points, temp)
        # Update position
        # Use a small step size
        step_size = 0.01
        new_points = points + step_size * forces
        
        # Renormalize to keep max dist = 1
        # This ensures the ratio d_min/d_max is effectively d_min (since d_max=1)
        dists = pdist(new_points, 'euclidean')
        d_max = np.max(dists)
        if d_max > 1e-10:
            new_points = new_points / d_max
            
        return new_points

    def run_annealing(init_points, num_steps, initial_temp, final_temp):
        """
        Run a simulated annealing schedule.
        """
        current_points = init_points.copy()
        best_points = current_points.copy()
        best_score = get_score(current_points)
        
        # Normalize init
        dists = pdist(current_points, 'euclidean')
        d_max = np.max(dists)
        if d_max > 1e-10:
            current_points = current_points / d_max
            best_points = current_points.copy()
            best_score = get_score(current_points)
            
        for step in range(num_steps):
            if time_exceeded():
                break
            
            # Linear cooling
            progress = step / num_steps
            temp = initial_temp * (final_temp / initial_temp) ** progress
            
            new_points = anneal_step(current_points, temp)
            new_score = get_score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                
            # Acceptance criterion: always accept if better, 
            # but for packing, accepting worse moves is crucial to escape local minima.
            # Simple Metropolis: accept if new_score > old_score or with prob exp((new-old)/T)
            # Since we are maximizing score, and score is bounded [0,1], 
            # and our "energy" is effectively -score, 
            # we can just accept if new_score >= old_score or random < exp((new_score - old_score)/temp_factor)
            # However, since we are using forces, the move is already a "proposal".
            # Let's use a simple random restart or just accept if better to keep it simple and fast,
            # but pure hill climbing gets stuck. 
            # Let's accept with probability 1 if better, else with prob 0.1 * (1 - temp)
            # Actually, let's just do random walks that are biased by the force.
            
            # To ensure we explore, let's occasionally jump.
            if np.random.rand() < 0.01:
                # Small random perturbation
                jump = np.random.randn(n, 2) * 0.1
                new_points = current_points + jump
                dists = pdist(new_points, 'euclidean')
                d_max = np.max(dists)
                if d_max > 1e-10:
                    new_points = new_points / d_max
                    new_score = get_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
                        current_points = new_points.copy()
            
            current_points = new_points
            
        return best_points, best_score

    # Initial configurations
    inits = []
    
    # 1. Grid-like init
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    inits.append(grid_points)
    
    # 2. Circle init
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_points = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(circle_points)
    
    # 3. Hexagonal-like init
    hex_points = []
    for row in range(-3, 4):
        for col in range(-3, 4):
            x = col + (row % 2) * 0.5
            y = row * 0.866
            hex_points.append([x, y])
    hex_points = np.array(hex_points)
    centers = np.argpartition(np.sum(hex_points**2, axis=1), 16)[:16]
    inits.append(hex_points[centers])
    
    # 4. Random inits
    for seed in range(10):
        np.random.seed(seed)
        inits.append(np.random.uniform(-1, 1, (n, 2)))

    best_points = None
    best_score = 0.0

    # Run annealing on multiple inits
    # We have ~590 seconds. 
    # Each step involves O(N^2) distance calcs. N=16, so 120 pairs. Very fast.
    # We can do many steps.
    
    num_inits = len(inits)
    # Allocate time per init
    time_per_init = 580.0 / num_inits
    
    for i, init in enumerate(inits):
        if time_exceeded():
            break
            
        # Run a few annealing passes with different parameters
        # Pass 1: High temp, many steps
        # Pass 2: Low temp, refine
        
        # Estimate steps per second
        # Let's just run until time limit for this init
        init_start = time.time()
        
        # Run 5 annealing schedules with different initial temps
        for temp_start in [1.0, 0.5, 0.2, 0.1, 0.05]:
            if time.time() - init_start > time_per_init / 5.0:
                continue
                
            # Run annealing
            # Number of steps: let's say 1000 steps per schedule
            # If it takes too long, reduce steps.
            # 1000 steps * 120 pairs * simple math is very fast. 
            # Let's try 5000 steps.
            p, s = run_annealing(init, 5000, temp_start, 0.001)
            
            if s > best_score:
                best_score = s
                best_points = p

    # Final Polish: Take the best found and run a very fine-grained local search
    if best_points is not None and not time_exceeded():
        # Run a high-precision local optimization on the best point
        # Using the same annealing but with very low temp and small steps
        p, s = run_annealing(best_points, 10000, 0.01, 0.0001)
        if s > best_score:
            best_score = s
            best_points = p

    # Fallback
    if best_points is None:
        np.random.seed(0)
        best_points = np.random.randn(n, 2)
        # Normalize
        dists = pdist(best_points, 'euclidean')
        d_max = np.max(dists)
        if d_max > 1e-10:
            best_points = best_points / d_max

    return best_points

# EVOLVE-BLOCK-END
