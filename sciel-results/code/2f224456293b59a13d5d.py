<<<<<<< SEARCH
from scipy.optimize import differential_evolution, minimize_scalar
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    N = 12

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (flat-top when angle=0)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def polygon_disjoint_sat(v1, v2):
        """Check if two convex polygons are disjoint using SAT."""
        axes = []
        for v in [v1, v2]:
            for i in range(len(v)):
                j = (i + 1) % len(v)
                edge = v[j] - v[i]
                normal = np.array([-edge[1], edge[0]])
                if np.linalg.norm(normal) > 1e-12:
                    normal /= np.linalg.norm(normal)
                    axes.append(normal)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - 1e-7 or max2 < min1 - 1e-7:
                return True
        
        return False

    def hexagons_disjoint(positions, angles):
        """Check if all hexagons are pairwise disjoint."""
        n = len(positions)
        for i in range(n):
            vi = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            for j in range(i + 1, n):
                vj = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                if not polygon_disjoint_sat(vi, vj):
                    return False
        return True

    def calc_min_outer_side(all_verts, theta):
        """Calculate minimum outer hexagon side length for a given orientation theta."""
        angles = theta + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        widths = dots.max(axis=0) - dots.min(axis=0)
        max_width = np.max(widths)
        apothem = max_width / 2.0
        side = 2 * apothem / np.sqrt(3)
        return side

    def find_best_outer_orientation(all_verts):
        """Find orientation and side length for minimum enclosing hexagon."""
        res = minimize_scalar(lambda th: calc_min_outer_side(all_verts, th), 
                              bounds=(0, np.pi/3), method='bounded', 
                              options={'xatol': 1e-4})
        return res.fun, res.x

    def get_overlap_penalty(positions, angles):
        """Calculate a smooth penalty for overlaps using SAT penetration depth approximation."""
        penalty = 0.0
        n = len(positions)
        for i in range(n):
            vi = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            for j in range(i + 1, n):
                vj = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                
                # Get axes
                axes = []
                for v in [vi, vj]:
                    for k in range(len(v)):
                        m = (k + 1) % len(v)
                        edge = v[m] - v[k]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-12:
                            normal /= norm
                            axes.append(normal)
                
                min_overlap = float('inf')
                for axis in axes:
                    proj1 = np.dot(vi, axis)
                    proj2 = np.dot(vj, axis)
                    min1, max1 = proj1.min(), proj1.max()
                    min2, max2 = proj2.min(), proj2.max()
                    
                    # If separated, no overlap on this axis
                    if max1 < min2 or max2 < min1:
                        min_overlap = 0.0
                        break
                    
                    # Calculate overlap on this axis
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap < min_overlap:
                        min_overlap = overlap
                
                # If min_overlap > 0, they are intersecting
                if min_overlap > 0:
                    penalty += min_overlap * min_overlap
        
        return penalty

    def approx_outer_side(positions, angles):
        """Fast approximation of outer side length using fixed orientations."""
        all_verts = []
        for i in range(len(positions)):
            v = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Check a few orientations
        min_side = float('inf')
        for theta in [0, np.pi/12, np.pi/6, np.pi/4, np.pi/3]:
            side = calc_min_outer_side(all_verts, theta)
            if side < min_side:
                min_side = side
        return min_side

    def full_objective(params):
        """Objective: approx outer side + penalty for overlap."""
        if time.time() > deadline:
            return 1e10
            
        n = len(params) // 3
        positions = np.array([[params[i*3], params[i*3+1]] for i in range(n)])
        angles = np.array([params[i*3+2] for i in range(n)])
        
        # Soft penalty
        penalty = get_overlap_penalty(positions, angles)
        if penalty > 0.01:
             # Heavily penalize overlaps to push towards valid solutions
             return 10.0 + 100.0 * penalty
             
        side = approx_outer_side(positions, angles)
        return side

    # Initial honeycomb configuration with small gap for validity
    d = 2.02
    positions = []
    angles = []
    
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # Ring 1
    for k in range(6):
        ang = np.radians(60 * k)
        positions.append([d * np.cos(ang), d * np.sin(ang)])
        angles.append(0.0)
    
    # Ring 2 (5 positions to make 12 total)
    # Place them in the gaps of ring 1 to be compact
    gap_angles = [30, 90, 150, 210, 270]
    for k in range(5):
        ang = np.radians(gap_angles[k])
        dist = d * np.sqrt(3)
        positions.append([dist * np.cos(ang), dist * np.sin(ang)])
        angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    
    # Ensure initial is valid
    if not hexagons_disjoint(positions, angles):
        d = 2.05
        positions = []
        angles = []
        positions.append([0.0, 0.0])
        angles.append(0.0)
        for k in range(6):
            ang = np.radians(60 * k)
            positions.append([d * np.cos(ang), d * np.sin(ang)])
            angles.append(0.0)
        gap_angles = [30, 90, 150, 210, 270]
        for k in range(5):
            ang = np.radians(gap_angles[k])
            dist = d * np.sqrt(3)
            positions.append([dist * np.cos(ang), dist * np.sin(ang)])
            angles.append(0.0)
        positions = np.array(positions)
        angles = np.array(angles)

    best_params = np.concatenate([positions.flatten(), angles]).copy()
    best_valid_side = 1e9
    best_valid_params = best_params.copy()

    bounds = []
    for i in range(12):
        bounds.append((-3.0, 3.0))
        bounds.append((-3.0, 3.0))
        bounds.append((0.0, np.pi/3))
    
    def callback(x, convergence):
        nonlocal best_valid_params, best_valid_side
        if time.time() > deadline:
            return True # Stop
        # Check if current best in DE is valid
        # x is the current best vector
        pos = np.array([[x[i*3], x[i*3+1]] for i in range(12)])
        ang = np.array([x[i*3+2] for i in range(12)])
        if hexagons_disjoint(pos, ang):
            all_verts = []
            for i in range(12):
                v = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
                all_verts.append(v)
            all_verts = np.vstack(all_verts)
            side, _ = find_best_outer_orientation(all_verts)
            if side < best_valid_side:
                best_valid_side = side
                best_valid_params = x.copy()
        return False

    try:
        result = differential_evolution(
            full_objective,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-6,
            polish=False,
            callback=callback
        )
        
        # Check final result
        if result.fun < 10.0: # Likely valid or close
             pos = np.array([[result.x[i*3], result.x[i*3+1]] for i in range(12)])
             ang = np.array([result.x[i*3+2] for i in range(12)])
             if hexagons_disjoint(pos, ang):
                 all_verts = []
                 for i in range(12):
                     v = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
                     all_verts.append(v)
                 all_verts = np.vstack(all_verts)
                 side, _ = find_best_outer_orientation(all_verts)
                 if side < best_valid_side:
                     best_valid_side = side
                     best_valid_params = result.x.copy()

    except Exception as e:
        pass

    # Use the best valid solution found
    if best_valid_side < 1e8:
        best_positions = np.array([[best_valid_params[i*3], best_valid_params[i*3+1]] for i in range(12)])
        best_angles = np.array([best_valid_params[i*3+2] for i in range(12)])
        
        all_verts = []
        for i in range(12):
            v = hexagon_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        outer_side, outer_angle_rad = find_best_outer_orientation(all_verts)
        
        # Calculate center
        angles = outer_angle_rad + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        mins = dots.min(axis=0)
        maxs = dots.max(axis=0)
        mids = (mins + maxs) / 2.0
        c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
        outer_center = tuple(c)
        outer_angle_deg = np.degrees(outer_angle_rad)
    else:
        # Fallback to initial
        outer_center = (0.0, 0.0)
        outer_side = 10.0
        outer_angle_deg = 0.0
        best_positions = positions
        best_angles = angles

    inner = [[best_positions[i, 0], best_positions[i, 1], np.degrees(best_angles[i])] for i in range(12)]
    
    return inner, outer_center, outer_side, outer_angle_deg
# EVOLVE-BLOCK-END
=======
import numpy as np
import time
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    N = 12

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (pointy-top when angle=0)."""
        angles = angle_rad + np.radians(np.arange(30, 390, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def sat_min_separation(v1, v2):
        """Calculate minimum separation distance between two convex polygons using SAT.
        Returns positive if separated, negative if overlapping (penetration depth)."""
        axes = []
        for v in [v1, v2]:
            for i in range(len(v)):
                j = (i + 1) % len(v)
                edge = v[j] - v[i]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    normal /= norm
                    axes.append(normal)
        
        max_sep = float('-inf')
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            if max1 < min2:
                sep = min2 - max1
            elif max2 < min1:
                sep = min1 - max2
            else:
                sep = -(min(max1, max2) - max(min1, min2))
            
            if sep > max_sep:
                max_sep = sep
        
        return max_sep

    def required_outer_side(all_verts, theta):
        """Calculate minimum outer hexagon side length for a given orientation theta."""
        angles = theta + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        widths = dots.max(axis=0) - dots.min(axis=0)
        max_width = np.max(widths)
        apothem = max_width / 2.0
        side = 2 * apothem / np.sqrt(3)
        return side

    def check_validity(positions, angles, eps=1e-6):
        """Check if all hexagons are pairwise disjoint with margin eps."""
        n = len(positions)
        for i in range(n):
            vi = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            for j in range(i + 1, n):
                vj = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                sep = sat_min_separation(vi, vj)
                if sep < -eps:
                    return False
        return True

    def energy(positions, angles, target_side=None):
        """Calculate total energy: overlap penalty + outer side length."""
        n = len(positions)
        overlap_energy = 0.0
        
        for i in range(n):
            vi = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            for j in range(i + 1, n):
                vj = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                sep = sat_min_separation(vi, vj)
                if sep < 0:
                    overlap_energy += sep * sep
        
        # Calculate outer side
        all_verts = []
        for i in range(n):
            v = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Quick check of a few orientations
        min_side = float('inf')
        for theta in [0, np.pi/12, np.pi/6, np.pi/4, np.pi/3]:
            side = required_outer_side(all_verts, theta)
            if side < min_side:
                min_side = side
        
        return overlap_energy * 100.0 + min_side

    def initial_config_hexagonal():
        """Create initial hexagonal lattice configuration."""
        positions = []
        angles = []
        
        # Honeycomb: center + 6 in ring 1 + 5 in ring 2
        d = 2.01  # Slightly more than 2 to avoid overlap
        positions.append([0.0, 0.0])
        angles.append(0.0)
        
        for k in range(6):
            ang = np.radians(60 * k)
            positions.append([d * np.cos(ang), d * np.sin(ang)])
            angles.append(0.0)
        
        gap_angles = [30, 90, 150, 210, 270]
        for k in range(5):
            ang = np.radians(gap_angles[k])
            dist = d * np.sqrt(3)
            positions.append([dist * np.cos(ang), dist * np.sin(ang)])
            angles.append(0.0)
        
        return np.array(positions), np.array(angles)

    def initial_config_circular():
        """Create initial circular configuration."""
        positions = []
        angles = []
        
        # Center
        positions.append([0.0, 0.0])
        angles.append(0.0)
        
        # Ring 1: 6 hexagons
        for k in range(6):
            ang = np.radians(60 * k)
            positions.append([2.01 * np.cos(ang), 2.01 * np.sin(ang)])
            angles.append(0.0)
        
        # Ring 2: 5 hexagons
        for k in range(5):
            ang = np.radians(30 + 72 * k)
            positions.append([3.5 * np.cos(ang), 3.5 * np.sin(ang)])
            angles.append(0.0)
        
        return np.array(positions), np.array(angles)

    def initial_config_random(seed=42):
        """Create random initial configuration."""
        rng = np.random.RandomState(seed)
        positions = rng.uniform(-2.5, 2.5, size=(N, 2))
        angles = rng.uniform(0, np.pi/3, size=N)
        return positions, angles

    def simulated_annealing(initial_positions, initial_angles, max_iters=50000, 
                            t_start=1.0, t_end=0.001, cooling=0.9999):
        """Simulated annealing optimization."""
        positions = initial_positions.copy()
        angles = initial_angles.copy()
        
        best_positions = positions.copy()
        best_angles = angles.copy()
        best_energy = float('inf')
        
        current_energy = energy(positions, angles)
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = positions.copy()
            best_angles = angles.copy()
        
        t = t_start
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Generate perturbation
            idx = np.random.randint(N)
            delta_pos = np.random.normal(0, 0.1, size=2)
            delta_ang = np.random.normal(0, 0.05, size=1)
            
            new_positions = positions.copy()
            new_angles = angles.copy()
            new_positions[idx] += delta_pos
            new_angles[idx] += delta_ang
            new_angles[idx] = np.mod(new_angles[idx], np.pi/3)
            
            new_energy = energy(new_positions, new_angles)
            
            # Accept if better or with probability exp(-(new-old)/t)
            if new_energy < current_energy or np.random.random() < np.exp(-(new_energy - current_energy) / t):
                positions = new_positions
                angles = new_angles
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_positions = positions.copy()
                    best_angles = angles.copy()
            
            # Cool down
            t *= cooling
            if t < t_end:
                break
        
        return best_positions, best_angles, best_energy

    # Try multiple initial configurations and annealing runs
    best_positions = None
    best_angles = None
    best_valid_side = 1e9
    
    # Initial configurations to try
    initial_configs = [
        initial_config_hexagonal(),
        initial_config_circular(),
        initial_config_random(42),
        initial_config_random(123),
    ]
    
    for init_pos, init_ang in initial_configs:
        if time.time() > deadline:
            break
        
        # Run simulated annealing
        result_pos, result_ang, result_energy = simulated_annealing(
            init_pos, init_ang, 
            max_iters=30000,
            t_start=0.5,
            t_end=0.001,
            cooling=0.99995
        )
        
        # Check validity
        if check_validity(result_pos, result_ang):
            all_verts = []
            for i in range(N):
                v = hexagon_vertices(result_pos[i, 0], result_pos[i, 1], 1.0, result_ang[i])
                all_verts.append(v)
            all_verts = np.vstack(all_verts)
            
            # Find best outer orientation
            res = minimize_scalar(lambda th: required_outer_side(all_verts, th),
                                  bounds=(0, np.pi/3), method='bounded',
                                  options={'xatol': 1e-4})
            side = res.fun
            
            if side < best_valid_side:
                best_valid_side = side
                best_positions = result_pos.copy()
                best_angles = result_ang.copy()
    
    # If no valid solution found, use the best energy solution and try to fix it
    if best_positions is None:
        # Use the hexagonal initial as fallback
        best_positions, best_angles = initial_config_hexagonal()
        # Expand slightly to ensure validity
        best_positions = best_positions * 1.02
        best_valid_side = 10.0
    
    # Final refinement: local optimization of outer hexagon orientation
    all_verts = []
    for i in range(N):
        v = hexagon_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    res = minimize_scalar(lambda th: required_outer_side(all_verts, th),
                          bounds=(0, np.pi/3), method='bounded',
                          options={'xatol': 1e-5})
    outer_side = res.fun
    outer_angle_rad = res.x
    
    # Calculate outer center
    angles = outer_angle_rad + np.radians(np.arange(0, 360, 60))
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = all_verts @ normals.T
    mins = dots.min(axis=0)
    maxs = dots.max(axis=0)
    mids = (mins + maxs) / 2.0
    c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
    outer_center = tuple(c)
    outer_angle_deg = np.degrees(outer_angle_rad)
    
    inner = [[best_positions[i, 0], best_positions[i, 1], np.degrees(best_angles[i])] for i in range(N)]
    
    return inner, outer_center, outer_side, outer_angle_deg
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
