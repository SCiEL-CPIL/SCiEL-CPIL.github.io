"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np

N = 12
START_TIME = time.time()
TIME_LIMIT = 135.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def convex_hull(points):
    """Andrew's monotone chain convex hull. Returns hull points in CCW order."""
    points = sorted(set(map(tuple, points)))
    if len(points) <= 1:
        return points
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def min_enclosing_hexagon(points, max_iters=200):
    """Find minimum side-length regular hexagon containing all points.
    Returns (center_x, center_y, side_length, angle_deg)."""
    points = np.array(points)
    best_side = float('inf')
    best_params = (0, 0, 10, 0)
    
    # Try different angles
    for angle_deg in np.arange(0, 60, 2.0):
        a = math.radians(angle_deg)
        # Hexagon support function: for a regular hexagon with side s centered at (cx,cy) rotated by a,
        # the support in direction theta is s * cos(pi/6) * |cos(theta - a - k*pi/3)| ... 
        # Actually, a regular hexagon with "radius" (center to vertex) R = s, the support function is:
        # h(theta) = R * cos(pi/6) / cos((theta - a) mod (pi/3) - pi/6)
        # Wait, let me think more carefully.
        # A regular hexagon with side length s has circumradius R = s.
        # The vertices are at angles a + k*pi/3 for k=0..5.
        # The edges are perpendicular to directions a + pi/6 + k*pi/3.
        # The apothem (inradius) is s * sqrt(3)/2.
        # The support function h(theta) = apothem / cos((theta - a - pi/6) mod (pi/3) - pi/6)
        # Actually simpler: the hexagon is the intersection of 6 half-planes.
        # Edge i has normal direction n_i = a + pi/6 + i*pi/3, and the constraint is:
        # (p - c) . n_i <= apothem = s * sqrt(3)/2
        # So for a given center c and angle a, the minimum s such that all points are inside:
        # s >= max_i ((p_i - c) . n_i) / (sqrt(3)/2) for all i
        # s >= max_i ((p_i - c) . n_i) / (sqrt(3)/2)
        
        normals = np.array([a + math.pi/6 + i*math.pi/3 for i in range(6)])
        
        # Optimize center: minimize max_i ((p_i - c).n_i) / (sqrt(3)/2)
        # This is equivalent to: minimize max_i ((p_i).n_i - c.n_i)
        # = minimize max_i (p_i.n_i) - c.n_i ... but c.n_i depends on i
        # So: for each point p_i, the constraint is c.n_i >= p_i.n_i - s*sqrt(3)/2
        # We want to minimize s, which means maximizing the minimum margin.
        
        # Use a simple iterative approach: find center that minimizes max distance
        # Start with centroid
        c = points.mean(axis=0).copy()
        
        for _ in range(max_iters):
            # Compute required s for current center
            margins = np.array([(p - c) @ np.array([math.cos(n), math.sin(n)]) for p, n in zip(points, normals)])
            # Actually we need the max over all points for each normal direction
            max_margin = 0
            for ni in range(6):
                n_vec = np.array([math.cos(normals[ni]), math.sin(normals[ni])])
                dists = points @ n_vec
                max_margin = max(max_margin, dists.max() - c @ n_vec)
            s = max_margin / (math.sqrt(3)/2)
            
            if s < best_side:
                best_side = s
                best_params = (c[0], c[1], s, angle_deg)
            
            # Update center: move toward the point that is most "out" in the worst direction
            # Find the worst constraint
            worst_val = -1
            worst_idx = 0
            for ni in range(6):
                n_vec = np.array([math.cos(normals[ni]), math.sin(normals[ni])])
                dists = points @ n_vec
                val = dists.max() - c @ n_vec
                if val > worst_val:
                    worst_val = val
                    worst_idx = ni
            
            # Move center in direction of worst normal
            n_vec = np.array([math.cos(normals[worst_idx]), math.sin(normals[worst_idx])])
            # Gradient of max_margin w.r.t. c is -n_vec (to reduce margin, move in +n_vec direction)
            c = c + 0.01 * n_vec
            # Also add small random perturbation
            c += 0.001 * np.random.randn(2)
    
    return best_params

def all_hexagon_vertices(positions, angles):
    """Get all vertices of all hexagons."""
    verts = []
    for (cx, cy), ang in zip(positions, angles):
        v = hex_vertices(cx, cy, ang)
        verts.append(v)
    return np.vstack(verts)

def physics_packing(seed=42):
    """Physics-based packing of 12 unit hexagons."""
    np.random.seed(seed)
    
    # Initial positions: rough hexagonal arrangement
    # Place in a 4x3 grid with hexagonal spacing
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # Hexagonal lattice: spacing sqrt(3) in y, 2 in x for adjacent rows
    # For 12 hexagons, try a compact arrangement
    # Row-based: 4-3-4-1 or 3-4-3-2 or similar
    # Let's use a simple initial placement
    for i in range(N):
        row = i // 4
        col = i % 4
        positions[i, 0] = col * 2.1 + (row % 2) * 1.05
        positions[i, 1] = row * 1.9
        angles[i] = 0.0
    
    # Add some randomness
    positions += 0.1 * np.random.randn(N, 2)
    angles = np.random.uniform(0, 60, N)
    
    # Physics simulation
    dt = 0.005
    friction = 0.99
    repulsion_strength = 50.0
    attraction_strength = 0.5
    target_radius = 3.5  # attract toward center
    
    for step in range(20000):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        forces = np.zeros((N, 2))
        
        for i in range(N):
            for j in range(i+1, N):
                # Compute minimum distance between two hexagons
                # Approximate using center-to-center distance minus sum of "radii"
                # For two unit hexagons, the minimum non-overlapping center distance
                # depends on relative orientation. Use a conservative estimate.
                d_vec = positions[i] - positions[j]
                d = np.linalg.norm(d_vec)
                
                # Approximate: two hexagons of side 1 have circumradius 1
                # Minimum non-overlap distance is roughly 2 * inradius = sqrt(3) ≈ 1.732
                # but can be less with rotation. Use 1.8 as threshold.
                min_dist = 1.85
                
                if d < min_dist and d > 1e-6:
                    # Repulsive force
                    overlap = min_dist - d
                    f = repulsion_strength * overlap / d
                    forces[i] += f * d_vec
                    forces[j] -= f * d_vec
                elif d <= 1e-6:
                    # Same position, push apart randomly
                    rand_dir = np.random.randn(2)
                    rand_dir /= np.linalg.norm(rand_dir)
                    forces[i] += repulsion_strength * rand_dir
                    forces[j] -= repulsion_strength * rand_dir
            
            # Attraction toward center to keep compact
            to_center = -positions[i]
            d_center = np.linalg.norm(to_center)
            if d_center > 0.1:
                forces[i] += attraction_strength * to_center / d_center
        
        # Update positions
        positions += forces * dt
        positions *= friction  # damping
        
        # Keep positions bounded
        max_pos = 15
        positions = np.clip(positions, -max_pos, max_pos)
    
    return positions, angles

def optimize_angles(positions, angles, seed=42):
    """Try to optimize angles to reduce required container size."""
    np.random.seed(seed)
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_side = float('inf')
    
    # Compute initial side
    verts = all_hexagon_vertices(positions, angles)
    _, _, side, _ = min_enclosing_hexagon(verts, max_iters=50)
    best_side = side
    
    for step in range(500):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Perturb a random angle
        idx = np.random.randint(N)
        old_angle = angles[idx]
        angles[idx] = old_angle + np.random.uniform(-5, 5)
        
        verts = all_hexagon_vertices(positions, angles)
        _, _, side, _ = min_enclosing_hexagon(verts, max_iters=30)
        
        if side < best_side:
            best_side = side
            best_angles = angles.copy()
        else:
            angles[idx] = old_angle
    
    return best_positions, best_angles, best_side

def refine_positions(positions, angles, seed=42):
    """Refine positions using local search to minimize container size."""
    np.random.seed(seed)
    best_positions = positions.copy()
    best_angles = angles.copy()
    
    # Get initial side
    verts = all_hexagon_vertices(positions, angles)
    _, _, best_side, _ = min_enclosing_hexagon(verts, max_iters=50)
    
    step_size = 0.1
    for step in range(3000):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Try moving a random hexagon
        idx = np.random.randint(N)
        old_pos = best_positions[idx].copy()
        dx = np.random.uniform(-step_size, step_size, 2)
        best_positions[idx] += dx
        
        verts = all_hexagon_vertices(best_positions, best_angles)
        _, _, side, _ = min_enclosing_hexagon(verts, max_iters=30)
        
        if side < best_side:
            best_side = side
            step_size = max(0.01, step_size * 0.99)
        else:
            best_positions[idx] = old_pos
            step_size = min(0.5, step_size * 1.02)
    
    return best_positions, best_angles, best_side

def construct():
    """Main entry point."""
    best_positions = None
    best_angles = None
    best_side = float('inf')
    best_center = (0, 0)
    best_angle_deg = 0
    
    # Multiple restarts with different seeds
    for seed in range(20):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Physics-based packing
        positions, angles = physics_packing(seed)
        
        # Optimize angles
        positions, angles, side = optimize_angles(positions, angles, seed)
        
        if side < best_side:
            best_side = side
            best_positions = positions.copy()
            best_angles = angles.copy()
        
        # Refine positions
        if time.time() - START_TIME < TIME_LIMIT - 10:
            positions, angles, side = refine_positions(positions, angles, seed)
            if side < best_side:
                best_side = side
                best_positions = positions.copy()
                best_angles = angles.copy()
    
    if best_positions is None:
        # Fallback: simple grid
        positions = np.array([[i % 4 * 2.1, (i // 4) * 1.9] for i in range(N)])
        angles = np.zeros(N)
        best_positions = positions
        best_angles = angles
        verts = all_hexagon_vertices(positions, angles)
        _, _, best_side, _ = min_enclosing_hexagon(verts, max_iters=100)
    
    # Compute final min enclosing hexagon
    verts = all_hexagon_vertices(best_positions, best_angles)
    cx, cy, side, angle_deg = min_enclosing_hexagon(verts, max_iters=200)
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([best_positions[i, 0], best_positions[i, 1], best_angles[i]])
    
    return inner, (cx, cy), side, angle_deg

# EVOLVE-BLOCK-END
