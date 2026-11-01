"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_score(pts):
        """Compute the packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
        # Compute min pairwise distance
        min_dist = np.inf
        for i in range(N):
            d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
            if len(d) > 0:
                min_dist = min(min_dist, np.min(d))
        r = min(min_dist / 2.0, 1.0 - np.max(dists))
        return r

    def generate_hex_lattice_with_noise(seed=42):
        """Generate points on a hexagonal lattice with random noise."""
        rng = np.random.RandomState(seed)
        
        # Estimate spacing for N points in unit circle
        # Area of unit circle = pi, area per point ~ pi/N
        # For hex packing, area per point = (sqrt(3)/2) * s^2 where s is spacing
        # So s ~ sqrt(2*pi/(N*sqrt(3)))
        s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
        
        # Generate hex lattice points
        points = []
        extent = int(1.0 / s) + 3
        for i in range(-extent, extent + 1):
            for j in range(-extent, extent + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r < 1.0 - s * 0.6:  # Leave margin
                    points.append([x, y])
        
        points = np.array(points)
        
        # If we have too many, select closest to origin
        if len(points) > N:
            radii = np.linalg.norm(points, axis=1)
            indices = np.argsort(radii)[:N]
            points = points[indices]
        elif len(points) < N:
            # Try smaller spacing
            s *= 0.95
            points = []
            extent = int(1.0 / s) + 3
            for i in range(-extent, extent + 1):
                for j in range(-extent, extent + 1):
                    x = (i + 0.5 * (j % 2)) * s
                    y = j * s * np.sqrt(3.0) / 2.0
                    r = np.hypot(x, y)
                    if r < 1.0 - s * 0.6:
                        points.append([x, y])
            points = np.array(points)
            if len(points) > N:
                radii = np.linalg.norm(points, axis=1)
                indices = np.argsort(radii)[:N]
                points = points[indices]
            elif len(points) < N:
                # Pad with random points
                while len(points) < N:
                    angle = rng.uniform(0, 2 * np.pi)
                    r = rng.uniform(0, 0.9)
                    points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
        
        # Add random noise to break symmetry
        noise_scale = s * 0.15
        points = points + rng.randn(N, 2) * noise_scale
        
        # Project back inside circle
        radii = np.linalg.norm(points, axis=1)
        for i in range(N):
            if radii[i] > 0.95:
                points[i] *= 0.95 / radii[i]
        
        return points

    def physics_simulate(pts, max_iters=5000, dt=0.003, cooling=0.9999):
        """Physics-based simulation to refine packing."""
        pts = pts.copy()
        current_temp = 1.0
        best_score = compute_score(pts)
        best_pts = pts.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute forces
            forces = np.zeros_like(pts)
            
            # Pairwise repulsion (only for nearby pairs for efficiency)
            for i in range(N):
                # Find points within some distance
                diffs = pts[i+1:] - pts[i]
                dists = np.linalg.norm(diffs, axis=1)
                mask = dists < 0.3  # Only consider nearby points
                if np.any(mask):
                    nearby_dists = dists[mask]
                    nearby_diffs = diffs[mask]
                    # Repulsive force: k / d^2 (capped)
                    forces_i = nearby_diffs / (nearby_dists[:, np.newaxis] + 1e-10)
                    forces_i *= 1.0 / (nearby_dists[:, np.newaxis] + 1e-10)
                    forces[i] -= np.sum(forces_i, axis=0)
            
            # Wall constraint: push inward if too close to boundary
            radii = np.linalg.norm(pts, axis=1)
            wall_force = np.zeros_like(pts)
            for i in range(N):
                if radii[i] > 0.9:
                    wall_force[i] = -pts[i] / radii[i] * (radii[i] - 0.9) * 5.0
            forces += wall_force
            
            # Apply forces with cooling
            velocities = forces * dt
            pts += velocities * current_temp
            
            # Project back inside circle
            radii = np.linalg.norm(pts, axis=1)
            for i in range(N):
                if radii[i] > 0.99:
                    pts[i] *= 0.99 / radii[i]
            
            # Cool down
            current_temp *= cooling
            
            # Check improvement
            if it % 100 == 0:
                score = compute_score(pts)
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
        
        return best_pts, best_score

    # Step 1: Generate initial configuration
    pts, best_score = generate_hex_lattice_with_noise(seed=42)
    
    # Step 2: Physics simulation to refine
    if time.time() < deadline - 10:
        pts, sim_score = physics_simulate(pts, max_iters=3000, dt=0.005, cooling=0.9998)
        if sim_score > best_score:
            best_score = sim_score
            pts = pts
    
    # Step 3: Try multiple restarts with different seeds if time permits
    if time.time() < deadline - 30:
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 10:
                break
            pts2, score2 = generate_hex_lattice_with_noise(seed=seed)
            if score2 > best_score:
                best_score = score2
                pts = pts2
            
            # Quick physics refinement
            if time.time() < deadline - 15:
                pts2, sim_score2 = physics_simulate(pts2, max_iters=1000, dt=0.005, cooling=0.9998)
                if sim_score2 > best_score:
                    best_score = sim_score2
                    pts = pts2
    
    # Final validation
    if pts is None or compute_score(pts) <= 0:
        # Fallback: simple grid
        pts = np.zeros((N, 2))
        n_per_row = int(np.sqrt(N))
        spacing = 0.03
        for i in range(N):
            row = i // n_per_row
            col = i % n_per_row
            pts[i, 0] = (col - n_per_row/2) * spacing
            pts[i, 1] = (row - n_per_row/2) * spacing
    
    return pts
# EVOLVE-BLOCK-END

