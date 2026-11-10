# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])


def _get_all_hex_vertices(positions, angles, side=1.0):
    """positions: (N,2), angles: (N,) in degrees. Returns (N, 6, 2)."""
    N = len(positions)
    verts = np.zeros((N, 6, 2))
    for i in range(N):
        verts[i] = _hex_vertices(positions[i, 0], positions[i, 1], side, angles[i])
    return verts


def _check_separation(hex_verts_a, hex_verts_b, tol=1e-6):
    """
    Separating axis test between two convex hexagons.
    Returns True if separated (valid), False if overlapping.
    hex_verts_a, hex_verts_b: (6, 2) arrays.
    """
    # Get edge normals for hex A
    for i in range(6):
        v1 = hex_verts_a[i]
        v2 = hex_verts_a[(i + 1) % 6]
        edge = v2 - v1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        
        # Project both hexagons onto this normal
        proj_a = np.dot(hex_verts_a, normal)
        proj_b = np.dot(hex_verts_b, normal)
        
        min_a, max_a = proj_a.min(), proj_a.max()
        min_b, max_b = proj_b.min(), proj_b.max()
        
        # Check if intervals are separated (gap > tol)
        if max_a < min_b - tol or max_b < min_a - tol:
            return True  # Found separating axis, they are separated
    
    # Get edge normals for hex B
    for i in range(6):
        v1 = hex_verts_b[i]
        v2 = hex_verts_b[(i + 1) % 6]
        edge = v2 - v1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        
        proj_a = np.dot(hex_verts_a, normal)
        proj_b = np.dot(hex_verts_b, normal)
        
        min_a, max_a = proj_a.min(), proj_a.max()
        min_b, max_b = proj_b.min(), proj_b.max()
        
        if max_a < min_b - tol or max_b < min_a - tol:
            return True
    
    return False  # No separating axis found, they overlap


def _check_inside_outer(hex_verts, outer_verts, tol=1e-6):
    """
    Check if all vertices of hex_verts are inside or on outer hexagon.
    hex_verts: (6,2), outer_verts: (6,2) in CCW order.
    Returns True if inside.
    """
    for v in hex_verts:
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -tol:
                return False
    return True


def _min_outer_side_for_packing(positions, angles, max_s=10.0):
    """
    Find minimum side length s of regular hexagon (centered at origin, angle 0)
    that contains all inner hexagon vertices.
    Uses binary search on s.
    """
    # Compute all inner vertices
    all_verts = _get_all_hex_vertices(positions, angles, 1.0)
    all_verts = all_verts.reshape(-1, 2)
    
    def _is_contained(s):
        outer_verts = _hex_vertices(0, 0, s, 0)
        # Check all vertices are inside
        for v in all_verts:
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False
        return True
    
    # Binary search
    lo = np.sqrt(12.0)
    hi = max_s
    for _ in range(50):
        mid = (lo + hi) / 2
        if _is_contained(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _compute_score(positions, angles):
    """Compute objective: 1/s where s is min outer side length."""
    s = _min_outer_side_for_packing(positions, angles)
    return 1.0 / s


def _is_valid_packing(positions, angles, s):
    """Check if packing is valid with given outer side length s."""
    N = len(positions)
    hex_verts = _get_all_hex_vertices(positions, angles, 1.0)
    outer_verts = _hex_vertices(0, 0, s, 0)
    
    # Check separation between all pairs
    for i in range(N):
        for j in range(i + 1, N):
            if not _check_separation(hex_verts[i], hex_verts[j]):
                return False
    
    # Check all inner hexagons are inside outer
    for i in range(N):
        if not _check_inside_outer(hex_verts[i], outer_verts):
            return False
    
    return True


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length using simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s wall-clock limit
    
    N = 12
    
    # Initial configuration: hexagonal grid pattern
    # Unit hexagons in a honeycomb-like arrangement
    # For flat-top hexagons with side 1:
    # Horizontal spacing: 2 * side * cos(30°) = sqrt(3) ≈ 1.732
    # Vertical spacing: 2 * side * sin(30°) = 1.0, but offset rows
    
    # Let's use pointy-top orientation (angle=0 means flat-top per problem spec)
    # Actually, let's just place them in a reasonable initial grid
    
    # Hexagonal lattice positions for 12 hexagons:
    # Center + 6 around + 5 more to make 12
    positions = np.zeros((N, 2))
    angles = np.zeros(N)  # all aligned
    
    # Honeycomb arrangement
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons (offset)
    
    dx = 1.5  # horizontal distance between centers in same row (for side=1)
    dy = np.sqrt(3)  # vertical distance between rows
    
    idx = 0
    # Row 0: 3 hexagons
    for col in range(3):
        positions[idx, 0] = (col - 1) * dx
        positions[idx, 1] = 0
        idx += 1
    
    # Row 1: 4 hexagons, offset by dx/2
    for col in range(4):
        positions[idx, 0] = (col - 1.5) * dx
        positions[idx, 1] = dy
        idx += 1
    
    # Row 2: 3 hexagons
    for col in range(3):
        positions[idx, 0] = (col - 1) * dx
        positions[idx, 1] = 2 * dy
        idx += 1
    
    # Row 3: 2 hexagons, offset
    for col in range(2):
        positions[idx, 0] = (col - 0.5) * dx
        positions[idx, 1] = 3 * dy
        idx += 1
    
    # Center the configuration
    positions -= positions.mean(axis=0)
    
    # Initial evaluation
    s_init = _min_outer_side_for_packing(positions, angles)
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_s = s_init
    best_score = 1.0 / best_s
    
    # Simulated annealing
    T = 0.5  # initial temperature
    cooling = 0.999
    min_T = 1e-4
    
    current_positions = positions.copy()
    current_angles = angles.copy()
    current_s = s_init
    current_score = 1.0 / current_s
    
    while T > min_T:
        if time.time() > deadline - 30:  # safety margin
            break
        
        # Generate perturbation
        trial_positions = current_positions.copy()
        trial_angles = current_angles.copy()
        
        # Randomly perturb 1-3 hexagons
        n_perturb = np.random.randint(1, 4)
        indices = np.random.choice(N, size=n_perturb, replace=False)
        
        for i in indices:
            # Small random displacement
            trial_positions[i] += np.random.normal(0, T * 0.5, size=2)
            # Small random rotation
            trial_angles[i] += np.random.normal(0, T * 10.0)
        
        # Evaluate trial
        trial_s = _min_outer_side_for_packing(trial_positions, trial_angles)
        trial_score = 1.0 / trial_s
        
        # Check if trial is valid (separation constraints)
        # For speed, we can skip full validation and rely on the score
        # But we need to ensure no overlaps for valid solutions
        valid = _is_valid_packing(trial_positions, trial_angles, trial_s)
        
        if valid:
            if trial_score > current_score:
                # Accept (improvement)
                current_positions = trial_positions
                current_angles = trial_angles
                current_s = trial_s
                current_score = trial_score
                
                if trial_score > best_score:
                    best_positions = trial_positions.copy()
                    best_angles = trial_angles.copy()
                    best_s = trial_s
                    best_score = trial_score
            else:
                # Metropolis criterion
                delta = trial_score - current_score
                if np.random.random() < np.exp(-delta / (T + 1e-10)):
                    current_positions = trial_positions
                    current_angles = trial_angles
                    current_s = trial_s
                    current_score = trial_score
        # If invalid, reject
        
        T *= cooling
    
    # Final validation of best solution
    if not _is_valid_packing(best_positions, best_angles, best_s):
        # Fallback to initial
        best_positions = positions.copy()
        best_angles = angles.copy()
        best_s = s_init
    
    inner_hex_data = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s


# EVOLVE-BLOCK-END
