# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _compute_min_side(inner, outer_center=None):
    """Compute the minimum side length of outer hexagon needed to contain all inner hexagons."""
    if outer_center is None:
        outer_center = np.array([0.0, 0.0])
    
    inner_shifted = inner.copy()
    inner_shifted[:, 0] -= outer_center[0]
    inner_shifted[:, 1] -= outer_center[1]
    
    # Get all vertices of all inner hexagons
    all_verts = []
    for i in range(11):
        verts = _hex_vertices(inner_shifted[i, 0], inner_shifted[i, 1], 1.0, inner_shifted[i, 2])
        all_verts.append(verts)
    
    all_verts = np.vstack(all_verts)
    
    # The 6 edge normals of a regular hexagon centered at origin (angle 0)
    normals = np.column_stack([np.cos(np.radians(30 + 60 * np.arange(6))), 
                                 np.sin(np.radians(30 + 60 * np.arange(6)))])
    
    # For each normal direction, compute max projection
    projections = all_verts @ normals.T  # shape (66, 6)
    max_projs = projections.max(axis=0)  # shape (6,)
    
    # The side length s must satisfy: s * sqrt(3)/2 >= max_projs
    # So s >= max_projs * 2 / sqrt(3)
    min_side = np.max(max_projs) * 2.0 / np.sqrt(3)
    
    return min_side


def _check_separation(inner, tol=1e-6):
    """Check if all pairs of inner hexagons are separated."""
    for i in range(11):
        for j in range(i+1, 11):
            vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            vj = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            
            # Get edge normals for both hexagons
            def get_edge_normals(verts):
                normals = []
                for k in range(6):
                    e = verts[(k+1) % 6] - verts[k]
                    n = np.array([-e[1], e[0]])
                    n = n / np.linalg.norm(n)
                    normals.append(n)
                return normals
            
            ni = get_edge_normals(vi)
            nj = get_edge_normals(vj)
            
            # Check on at least one of 12 edge normals the projected intervals are apart by > tol
            separated = False
            for n in ni + nj:
                proj_i = vi @ n
                proj_j = vj @ n
                max_i = proj_i.max()
                min_i = proj_i.min()
                max_j = proj_j.max()
                min_j = proj_j.min()
                
                if max_i < min_j - tol or max_j < min_i - tol:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


def _build_rigid_cluster(global_rot_deg, satellite_offsets, flower_gap=0.001):
    """
    Build the 11 hexagons from a rigid cluster topology.
    
    Topology: 7 hexagons in a flower (center + 6 around at distance 2+gap)
              + 4 satellite hexagons at specified offsets from center.
    
    All hexagons share the same angle = global_rot_deg.
    
    satellite_offsets: array of shape (4, 2) with (dx, dy) for each satellite.
    flower_gap: small gap added to flower hexagon distances to ensure separation.
    """
    sqrt3 = np.sqrt(3)
    
    # Flower positions (relative to center, before rotation)
    # Distance from center to neighbors is 2 + flower_gap to ensure separation
    d = 2.0 + flower_gap
    flower = np.array([
        [0.0, 0.0],           # center
        [d, 0.0],             # 0 deg
        [d/2, d*sqrt3/2],     # 60 deg
        [-d/2, d*sqrt3/2],    # 120 deg
        [-d, 0.0],            # 180 deg
        [-d/2, -d*sqrt3/2],   # 240 deg
        [d/2, -d*sqrt3/2],    # 300 deg
    ])
    
    # Apply global rotation to all positions
    rot = np.radians(global_rot_deg)
    c, s = np.cos(rot), np.sin(rot)
    rot_mat = np.array([[c, -s], [s, c]])
    
    all_pos = np.vstack([flower, satellite_offsets])
    all_pos = all_pos @ rot_mat.T
    
    # All hexagons have the same angle
    angles = np.full(11, global_rot_deg)
    
    return np.column_stack([all_pos[:, 0], all_pos[:, 1], angles])


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a rigid cluster topology with optimized satellite positions and global rotation.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    sqrt3 = np.sqrt(3)
    
    # Initial satellite offsets: 4 hexagons placed beyond the 6 surrounding ones
    # These extend the cluster in 4 diagonal directions
    initial_satellites = np.array([
        [3.0, sqrt3],      # beyond 60 deg neighbor
        [-3.0, sqrt3],     # beyond 120 deg neighbor  
        [3.0, -sqrt3],     # beyond 300 deg neighbor
        [-3.0, -sqrt3],    # beyond 240 deg neighbor
    ])
    
    # Initial global rotation: 0 degrees
    best_satellites = initial_satellites.copy()
    best_rot = 0.0
    best_side = None
    
    # Compute initial side
    best_inner = _build_rigid_cluster(best_rot, best_satellites)
    if _check_separation(best_inner):
        best_side = _compute_min_side(best_inner)
    else:
        # Need larger gap or different satellite positions
        # Try increasing satellite distances
        best_satellites = initial_satellites * 1.01
        best_inner = _build_rigid_cluster(best_rot, best_satellites)
        if _check_separation(best_inner):
            best_side = _compute_min_side(best_inner)
        else:
            # Use a known valid fallback
            best_inner = np.array([
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ], dtype=float)
            best_side = 8.0
            outer_hex_data = np.array([0.0, 0.0, 0.0])
            return best_inner, outer_hex_data, float(best_side)
    
    print(f"Initial side: {best_side:.6f}")
    
    # Phase 1: Optimize global rotation with fixed satellites
    # The outer hexagon has 6-fold symmetry, so we only need to search [0, 60)
    rot_grid = np.linspace(0, 60, 121, endpoint=False)
    for rot in rot_grid:
        if time.time() > deadline - 60:
            break
        inner_trial = _build_rigid_cluster(rot, best_satellites)
        if _check_separation(inner_trial):
            side_trial = _compute_min_side(inner_trial)
            if best_side is None or side_trial < best_side - 1e-10:
                best_side = side_trial
                best_rot = rot
    
    print(f"After rotation sweep: side={best_side:.6f}, rot={best_rot:.2f}")
    
    # Phase 2: Optimize satellite positions with fixed rotation
    # Use iterative local search
    rng = np.random.RandomState(42)
    current_sats = best_satellites.copy()
    current_side = best_side
    no_improve = 0
    max_no_improve = 30
    
    while time.time() < deadline - 30 and no_improve < max_no_improve:
        improved = False
        
        for trial in range(80):
            if time.time() > deadline - 30:
                break
            
            # Random perturbation of satellite positions
            step = 0.08 * (1.0 + no_improve / 50.0)
            perturbation = rng.normal(0, step, size=(4, 2))
            
            trial_sats = current_sats + perturbation
            
            inner_trial = _build_rigid_cluster(best_rot, trial_sats)
            
            if not _check_separation(inner_trial):
                continue
            
            side_trial = _compute_min_side(inner_trial)
            
            if side_trial < current_side - 1e-10:
                current_sats = trial_sats.copy()
                current_side = side_trial
                improved = True
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
    
    best_satellites = current_sats
    best_side = current_side
    print(f"After satellite optimization: side={best_side:.6f}")
    
    # Phase 3: Joint fine-tuning of rotation and satellites
    no_improve = 0
    max_no_improve = 20
    current_rot = best_rot
    current_sats = best_satellites.copy()
    current_side = best_side
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        improved = False
        
        for trial in range(60):
            if time.time() > deadline - 10:
                break
            
            # Small perturbation of both rotation and satellites
            rot_step = 0.5 * (1.0 + no_improve / 30.0)
            sat_step = 0.04 * (1.0 + no_improve / 30.0)
            
            trial_rot = (current_rot + rng.normal(0, rot_step)) % 60
            trial_sats = current_sats + rng.normal(0, sat_step, size=(4, 2))
            
            inner_trial = _build_rigid_cluster(trial_rot, trial_sats)
            
            if not _check_separation(inner_trial):
                continue
            
            side_trial = _compute_min_side(inner_trial)
            
            if side_trial < current_side - 1e-10:
                current_rot = trial_rot
                current_sats = trial_sats.copy()
                current_side = side_trial
                improved = True
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
    
    best_rot = current_rot
    best_satellites = current_sats
    best_side = current_side
    print(f"After joint optimization: side={best_side:.6f}, rot={best_rot:.3f}")
    
    # Build final result
    best_inner = _build_rigid_cluster(best_rot, best_satellites)
    
    # Final verification
    if not _check_separation(best_inner):
        # Fall back to a known valid configuration
        best_inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ], dtype=float)
        best_side = 8.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return best_inner, outer_hex_data, float(best_side)


# EVOLVE-BLOCK-END
