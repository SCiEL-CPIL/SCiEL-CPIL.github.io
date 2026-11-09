"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    best_score = 0.0
    best_points = None
    
    def get_score(pts):
        # Pairwise distances
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        # Mask diagonal
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        # Boundary distance
        min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_pair, min_boundary)

    # Helper to enforce constraints via projection
    def enforce_constraints(pts):
        # 1. Enforce non-overlap (min distance 1.0 in this normalized space, but we optimize r)
        # Wait, the score is r. If we fix positions, r is determined.
        # Better approach: Optimize positions such that min_dist is maximized.
        # Standard MD approach for packing: 
        # Let positions x_i. 
        # Constraints: |x_i - x_j| >= 1 (normalized), |x_i| <= 1 - 0.5 (normalized if r=0.5? No)
        
        # Let's stick to the raw coordinate system where container is radius 1.
        # We want to maximize r.
        # Equivalent to minimizing the "strain" or maximizing the minimum clearance.
        
        # Strategy: 
        # 1. Start with a configuration.
        # 2. Define a "virtual" radius r_current.
        # 3. If circles overlap (dist < 2*r), push them apart.
        # 4. If circles are outside (dist > 1-r), pull them in.
        # 5. This defines a feasible region for a given r.
        # 6. We can increase r slowly and re-optimize.
        
        # Alternative simpler strategy: 
        # Just maximize min_distance directly using gradient ascent or MD.
        # Let's use MD with a potential energy that penalizes overlaps.
        # E = sum_{i<j} max(0, 2r - dist_ij)^2 + sum_i max(0, dist_i - (1-r))^2
        # But r is a variable.
        
        # Let's use the "Jiggling" method directly on the positions to maximize min_dist.
        # We don't fix r. We just push points apart until they hit the boundary.
        # The resulting min_dist / 2 is the score.
        
        # Algorithm:
        # 1. Initialize points.
        # 2. Iterate:
        #    a. Compute all pairwise distances.
        #    b. For each pair i,j with dist < 1.0 (assuming target min dist is 1.0 for normalization? No, let's use a threshold):
        #       Actually, we want to maximize the minimum distance.
        #       Let's use a repulsive force F = 1/d^2 for d < d_max.
        #       And a centripetal force for boundary.
        #    c. Move points by velocity * dt + random jitter.
        #    d. Project back to feasible region? No, that constrains them.
        #       We want them to spread out.
        
        # Let's use the "Repulsion + Centripetal" MD approach.
        # Force on i:
        # F_rep = sum_{j != i} (1/d_ij^2) * (x_i - x_j) / d_ij  (if d_ij < d_cut)
        # F_cent = -k * x_i (if |x_i| > 1 - margin)
        
        # This will push them to the edges and spread them.
        # The equilibrium state is a local maximum of the minimum distance.
        
        # Parameters:
        # d_cut = 1.5 (arbitrary, just to limit long-range interactions)
        # k_cent = 10.0
        
        # To improve, we can also do "annealing" of the stiffness.
        
        # Let's implement a robust MD step.
        pass

    # Better Initial Configuration: Hexagonal
    # We want to place 50 points.
    # A hexagonal grid with spacing s.
    # Area of unit circle = pi. Area of 50 circles = 50 * pi * r^2.
    # Max packing density ~ 0.9069.
    # 50 * pi * r^2 <= 0.9069 * pi => r^2 <= 0.9069 / 50 => r <= sqrt(0.018) ~ 0.134.
    # The known best is 0.1258. So we are close.
    
    # Let's generate a hexagonal lattice that fits inside the unit circle.
    # We'll generate many points and pick the best 50.
    
    def generate_hex_initial():
        # Generate points on a hex grid
        # Spacing such that ~50 points fit.
        # Let's try spacing = 0.2
        points = []
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = i * 0.2 + (j % 2) * 0.1
                y = j * 0.2 * np.sqrt(3) / 2
                if np.hypot(x, y) < 0.9: # Leave margin
                    points.append([x, y])
        
        points = np.array(points)
        # Sort by distance from center? Or random selection?
        # Random selection from the valid set, then shuffle.
        # Actually, let's just take the first 50.
        return points[:50].copy()

    # Let's try a different init: Random points inside the circle, then relax.
    def generate_random_initial():
        # Uniform distribution in circle
        r = np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))

    # MD Simulation
    def run_md(initial_pts, max_steps=5000, dt=0.01):
        pts = initial_pts.copy()
        # Velocity is implicitly handled by forces + jitter, or we can keep velocity.
        # Let's use simple gradient ascent with jitter (Stochastic Gradient Ascent).
        # Objective: Maximize min_dist.
        # Gradient of min_dist is tricky.
        # Let's use the energy function: E = sum (1/d^2) + sum (r_i^2) (to keep them in center? No, we want them at boundary)
        
        # Actually, let's use the "Jiggling" algorithm from Graham et al.
        # 1. Compute min distance d_min.
        # 2. For all pairs with d < d_min + epsilon, apply repulsive force.
        # 3. For all points with r > 1 - d_min/2 - epsilon, apply attractive force to boundary?
        #    No, the boundary is a hard constraint. The points should be pushed against the boundary if possible.
        
        # Let's use a simpler, robust method:
        # Minimize E = sum_{i<j} (1/d_ij^2) + lambda * sum_i (|x_i| - (1 - r_est))^2
        # But r_est is unknown.
        
        # Let's just maximize the minimum distance directly using a "push apart" heuristic.
        
        current_score = get_score(pts)
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            # Compute forces
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            d_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(d_sq, np.inf)
            d = np.sqrt(d_sq)
            
            # Repulsive force: 1/d^2
            # Only consider pairs closer than some cutoff to save time? 
            # For N=50, full matrix is fast.
            force_mag = np.zeros_like(d_sq)
            # Apply force only if d < 1.0 (assuming we want min dist ~1.0 in some scale?)
            # No, let's apply force if d < 2 * current_score + 0.1
            # This focuses on the "tight" pairs.
            threshold = 2 * current_score + 0.05
            mask = d < threshold
            force_mag[mask] = 1.0 / d_sq[mask]
            
            forces = np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
            
            # Boundary force: Pull towards boundary if inside, push in if outside.
            # We want to maximize the minimum distance to the boundary too?
            # No, the score is min(min_pair_dist/2, min_boundary_dist).
            # So we want to balance pair distances and boundary distances.
            
            # Let's add a term that pushes points to the boundary if they are "free".
            # Force towards boundary: (1 - |x_i|) * (x_i / |x_i|) ?
            # No, that pushes them OUT.
            # We want them to be as far from each other as possible.
            # Usually, in optimal packings, most circles touch the boundary or other circles.
            
            # Let's just use repulsion. The boundary will naturally be hit by the repulsion from the center.
            
            # Move
            step_size = 0.01
            new_pts = pts + forces * step_size + np.random.randn(N, 2) * 0.005
            
            # Enforce hard boundary: If any point is outside, project it back.
            r = np.linalg.norm(new_pts, axis=1)
            outside = r > 1.0
            if np.any(outside):
                new_pts[outside] = new_pts[outside] / r[outside, np.newaxis]
            
            # Check if score improved
            new_score = get_score(new_pts)
            if new_score > current_score:
                pts = new_pts
                current_score = new_score
            else:
                # Accept with some probability (Simulated Annealing style)
                # T decays
                T = 0.01 * (0.99 ** step)
                if np.random.rand() < np.exp((new_score - current_score) / T):
                    pts = new_pts
                    current_score = new_score
            
            # Decay step size?
            # step_size *= 0.999

        return pts, current_score

    # Multi-start
    # 1. Hexagonal
    # 2. Random
    # 3. Ring-based
    
    initializations = []
    initializations.append(generate_hex_initial())
    
    for i in range(5):
        initializations.append(generate_random_initial())
        
    # Ring initialization: Place points on concentric rings
    def generate_ring_initial():
        pts = []
        # Center
        pts.append([0, 0])
        # Ring 1: 6 points
        for i in range(6):
            theta = 2 * np.pi * i / 6
            pts.append([0.2 * np.cos(theta), 0.2 * np.sin(theta)])
        # Ring 2: 12 points
        for i in range(12):
            theta = 2 * np.pi * i / 12 + np.pi/12
            pts.append([0.4 * np.cos(theta), 0.4 * np.sin(theta)])
        # Ring 3: 18 points
        for i in range(18):
            theta = 2 * np.pi * i / 18
            pts.append([0.6 * np.cos(theta), 0.6 * np.sin(theta)])
        # Ring 4: 13 points (to reach 50: 1+6+12+18+13 = 50)
        for i in range(13):
            theta = 2 * np.pi * i / 13 + np.pi/13
            pts.append([0.8 * np.cos(theta), 0.8 * np.sin(theta)])
        return np.array(pts[:50])
        
    initializations.append(generate_ring_initial())

    # Run MD on each initialization
    for init in initializations:
        if time.time() > deadline - 10.0:
            break
            
        pts, score = run_md(init, max_steps=3000)
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    return best_points
# EVOLVE-BLOCK-END
