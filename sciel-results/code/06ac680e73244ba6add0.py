# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute the 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _get_all_vertices(positions, angles, side=1.0):
    """
    positions: (N,2), angles: (N,) in degrees. Returns (N*6, 2).
    """
    verts = []
    for i in range(len(positions)):
        verts.append(_hex_vertices(positions[i,0], positions[i,1], side, angles[i]))
    return np.vstack(verts)

def _min_outer_side(vertices):
    """
    Compute minimum side length s of a regular hexagon centered at origin (angle 0)
    that contains all given vertices.
    For a regular hex with side s and angle 0, the 6 edge inward normals point at
    angles 30, 90, 150, 210, 270, 330 degrees. The distance from center to each edge
    is s * cos(30°) = s * sqrt(3)/2.
    So we need s such that for every vertex v, the projection onto each normal is
    <= s * sqrt(3)/2.
    Thus s = max over all vertices and all normals of (dot(v, normal) / (sqrt(3)/2)).
    """
    normals = np.array([
        [np.cos(np.radians(30)), np.sin(np.radians(30))],
        [np.cos(np.radians(90)), np.sin(np.radians(90))],
        [np.cos(np.radians(150)), np.sin(np.radians(150))],
        [np.cos(np.radians(210)), np.sin(np.radians(210))],
        [np.cos(np.radians(270)), np.sin(np.radians(270))],
        [np.cos(np.radians(330)), np.sin(np.radians(330))],
    ])
    # For each vertex, compute max projection onto each normal
    # We need the max over all vertices of the max projection onto any normal
    # Then s = that max / (sqrt(3)/2)
    projections = vertices @ normals.T  # (N*6, 6)
    max_proj = np.max(np.abs(projections), axis=1)  # max abs projection per vertex
    max_overall = np.max(max_proj)
    return max_overall / (np.sqrt(3) / 2)

def _check_separation(positions, angles, side=1.0, tol=1e-6):
    """
    Check that no two inner hexagons overlap using separating axis test.
    Returns True if all pairs are separated.
    """
    n = len(positions)
    all_verts = _get_all_vertices(positions, angles, side)
    for i in range(n):
        for j in range(i+1, n):
            vi = _hex_vertices(positions[i,0], positions[i,1], side, angles[i])
            vj = _hex_vertices(positions[j,0], positions[j,1], side, angles[j])
            # Get edge normals for both hexagons
            # For hexagon vi, edges are between consecutive vertices
            edges_i = np.diff(np.vstack([vi, vi[0]]), axis=0)
            edges_j = np.diff(np.vstack([vj, vj[0]]), axis=0)
            # Normals are perpendicular to edges
            norms_i = np.column_stack([-edges_i[:,1], edges_i[:,0]])
            norms_j = np.column_stack([-edges_j[:,1], edges_j[:,0]])
            # Normalize
            norms_i = norms_i / np.linalg.norm(norms_i, axis=1, keepdims=True)
            norms_j = norms_j / np.linalg.norm(norms_j, axis=1, keepdims=True)
            all_norms = np.vstack([norms_i, norms_j])
            for norm in all_norms:
                proj_i = vi @ norm
                proj_j = vj @ norm
                if np.max(proj_i) < np.min(proj_j) - tol or np.max(proj_j) < np.min(proj_i) - tol:
                    break
            else:
                return False
    return True

def _is_valid(positions, angles, side=1.0):
    """Check if the packing is valid."""
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(angles)):
        return False
    if not _check_separation(positions, angles, side):
        return False
    all_verts = _get_all_vertices(positions, angles, side)
    s = _min_outer_side(all_verts)
    if s < np.sqrt(12) - 1e-8:
        return False
    # Check all inner vertices are inside outer hexagon
    outer_verts = _hex_vertices(0, 0, s, 0)
    for v in all_verts:
        inside = True
        for k in range(6):
            p1 = outer_verts[k]
            p2 = outer_verts[(k+1) % 6]
            cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
            if cross < -1e-6:
                inside = False
                break
        if not inside:
            return False
    return True

def _compute_score(positions, angles, side=1.0):
    """Compute 1/s for the given packing. Returns 0 if invalid."""
    if not _is_valid(positions, angles, side):
        return 0.0
    all_verts = _get_all_vertices(positions, angles, side)
    s = _min_outer_side(all_verts)
    return 1.0 / s

def _hex_contact_distance(vi, vj):
    """
    Compute the minimum distance between two hexagons vi and vj (each (6,2) array).
    Returns the minimum distance (0 if overlapping or touching).
    Uses edge-to-edge and vertex-to-edge distances.
    """
    min_dist = float('inf')
    # For each edge of vi, compute distance to each vertex of vj and vice versa
    for i in range(6):
        p1 = vi[i]
        p2 = vi[(i+1) % 6]
        edge = p2 - p1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-10:
            continue
        edge_unit = edge / edge_len
        # Distance from each vertex of vj to this edge
        for j in range(6):
            v = vj[j]
            # Project v onto the line through p1, p2
            t = np.dot(v - p1, edge_unit)
            if t < 0 or t > edge_len:
                # Closest point is an endpoint
                d1 = np.linalg.norm(v - p1)
                d2 = np.linalg.norm(v - p2)
                min_dist = min(min_dist, d1, d2)
            else:
                # Closest point is on the edge
                closest = p1 + t * edge_unit
                d = np.linalg.norm(v - closest)
                min_dist = min(min_dist, d)
    
    # Also check vertex-to-vertex (should be covered by above, but just in case)
    for i in range(6):
        for j in range(6):
            d = np.linalg.norm(vi[i] - vj[j])
            min_dist = min(min_dist, d)
    
    return min_dist

def _pressure_optimize(positions, angles, side=1.0, max_iters=5000, tol=1e-8):
    """
    Pressure-based optimization: minimize penetration by treating contacts as springs.
    Returns optimized positions and angles.
    """
    n = len(positions)
    positions = positions.copy().astype(float)
    angles = angles.copy().astype(float)
    
    # Precompute vertices
    def get_verts(pos, ang):
        return [_hex_vertices(pos[i,0], pos[i,1], side, ang[i]) for i in range(n)]
    
    current_verts = get_verts(positions, angles)
    
    # Initial s
    all_verts = np.vstack(current_verts)
    s = _min_outer_side(all_verts)
    
    # Iterative pressure-based optimization
    dt = 0.01
    repulsion_k = 100.0
    confinement_k = 1.0
    
    for it in range(max_iters):
        # Check convergence
        forces = np.zeros((n, 2))
        torque_forces = np.zeros(n)
        
        # Compute contact forces between pairs
        for i in range(n):
            for j in range(i+1, n):
                vi = current_verts[i]
                vj = current_verts[j]
                
                # Compute minimum distance and contact point
                min_dist = float('inf')
                contact_point = None
                contact_normal = None
                
                # Check all vertex-edge pairs
                for a_idx in range(6):
                    p1 = vi[a_idx]
                    p2 = vi[(a_idx+1) % 6]
                    edge = p2 - p1
                    edge_len = np.linalg.norm(edge)
                    if edge_len < 1e-10:
                        continue
                    edge_unit = edge / edge_len
                    edge_normal = np.array([-edge_unit[1], edge_unit[0]])
                    
                    for b_idx in range(6):
                        v = vj[b_idx]
                        t = np.dot(v - p1, edge_unit)
                        if t < -1e-10 or t > edge_len + 1e-10:
                            continue
                        t_clamped = np.clip(t, 0, edge_len)
                        closest_on_edge = p1 + t_clamped * edge_unit
                        d = np.linalg.norm(v - closest_on_edge)
                        if d < min_dist:
                            min_dist = d
                            if d > 1e-10:
                                contact_point = closest_on_edge
                                contact_normal = (v - closest_on_edge) / d
                            else:
                                contact_point = closest_on_edge
                                contact_normal = edge_normal
                
                # Also check vertex-vertex
                for a_idx in range(6):
                    for b_idx in range(6):
                        d = np.linalg.norm(vi[a_idx] - vj[b_idx])
                        if d < min_dist:
                            min_dist = d
                            if d > 1e-10:
                                contact_point = (vi[a_idx] + vj[b_idx]) / 2
                                contact_normal = (vi[a_idx] - vj[b_idx]) / d
                            else:
                                contact_point = (vi[a_idx] + vj[b_idx]) / 2
                                contact_normal = np.zeros(2)
                
                if min_dist < 1.0 - 1e-8 and contact_normal is not None:
                    # Repulsive force proportional to penetration
                    penetration = max(0, 1.0 - min_dist)
                    force_mag = repulsion_k * penetration
                    forces[i] += force_mag * contact_normal
                    forces[j] -= force_mag * contact_normal
                    # Torque
                    r_i = contact_point - positions[i]
                    r_j = contact_point - positions[j]
                    torque_forces[i] += np.cross(r_i, force_mag * contact_normal)
                    torque_forces[j] -= np.cross(r_j, force_mag * contact_normal)
        
        # Confinement force from outer hexagon
        all_verts = np.vstack(current_verts)
        s = _min_outer_side(all_verts)
        
        # Compute gradient of s w.r.t. each position
        # Approximate by finite differences
        for i in range(n):
            for dim in range(2):
                pos_orig = positions[i, dim]
                positions[i, dim] = pos_orig + 1e-4
                new_verts = get_verts(positions, angles)
                new_all_verts = np.vstack(new_verts)
                new_s = _min_outer_side(new_all_verts)
                positions[i, dim] = pos_orig
                ds_dx = (new_s - s) / 1e-4
                forces[i, dim] -= confinement_k * ds_dx
        
        # Update positions and angles
        for i in range(n):
            positions[i] += dt * forces[i]
            angles[i] += dt * torque_forces[i] * 0.1  # Small torque update
        
        # Recompute vertices
        current_verts = get_verts(positions, angles)
        
        # Damping
        dt *= 0.99
    
    return positions, angles

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds deadline
    
    # Initial configuration: hexagonal grid arrangement
    sqrt3 = np.sqrt(3)
    
    # Hexagonal grid positions for 12 hexagons
    # Using axial coordinates
    positions = np.array([
        [0, 0],
        [2.5, 0],
        [-2.5, 0],
        [1.25, 2.165],
        [-1.25, 2.165],
        [1.25, -2.165],
        [-1.25, -2.165],
        [3.75, 2.165],
        [-3.75, 2.165],
        [3.75, -2.165],
        [-3.75, -2.165],
        [0, -4.33],
    ])
    angles = np.zeros(12)
    
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_score = _compute_score(best_positions, best_angles)
    
    # Phase 1: Physics-based optimization
    dt = 0.01
    repulsion_k = 50.0
    attraction_k = 0.5
    damping = 0.9
    
    velocities = np.zeros_like(positions)
    
    max_iters = 50000
    for iteration in range(max_iters):
        if time.time() > deadline - 60:
            break
        
        # Compute forces
        rep_forces = np.zeros_like(positions)
        for i in range(12):
            for j in range(i+1, 12):
                d = positions[i] - positions[j]
                dist = np.linalg.norm(d)
                if dist < 1.8 and dist > 1e-10:
                    force = repulsion_k * (1.8 - dist) / dist
                    rep_forces[i] += force * d
                    rep_forces[j] -= force * d
        
        att_forces = attraction_k * (np.zeros(2) - positions)
        
        total_forces = rep_forces + att_forces
        
        # Update velocities and positions
        velocities = damping * velocities + total_forces * dt
        positions = positions + velocities * dt
        
        # Check if we've found a better configuration
        current_score = _compute_score(positions, angles)
        if current_score > best_score:
            best_score = current_score
            best_positions = positions.copy()
            best_angles = angles.copy()
        
        # Adaptive step size
        if current_score < best_score * 0.95:
            dt *= 0.5
            repulsion_k *= 0.9
        else:
            attraction_k *= 1.001
    
    # Phase 2: Pressure-based optimization
    if time.time() < deadline - 120:
        pos_opt, ang_opt = _pressure_optimize(best_positions, best_angles, max_iters=2000)
        score_opt = _compute_score(pos_opt, ang_opt)
        if score_opt > best_score:
            best_score = score_opt
            best_positions = pos_opt
            best_angles = ang_opt
    
    # Phase 3: Simulated annealing refinement
    if time.time() < deadline - 60:
        np.random.seed(42)
        T = 0.1
        cooling = 0.999
        step_size = 0.1
        
        current_positions = best_positions.copy()
        current_angles = best_angles.copy()
        current_score = best_score
        
        for it in range(20000):
            if time.time() > deadline - 30:
                break
            
            new_positions = current_positions + np.random.randn(12, 2) * step_size
            new_angles = current_angles + np.random.randn(12) * step_size * 0.1
            
            new_score = _compute_score(new_positions, new_angles)
            
            if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / T):
                current_positions = new_positions
                current_angles = new_angles
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_positions = current_positions.copy()
                    best_angles = current_angles.copy()
            
            T *= cooling
            if it % 500 == 0 and it > 0:
                step_size *= 0.95
    
    # Final polish
    if time.time() < deadline - 30:
        for _ in range(5000):
            if time.time() > deadline - 10:
                break
            trial_positions = best_positions + np.random.randn(12, 2) * 0.01
            trial_angles = best_angles + np.random.randn(12) * 0.005
            trial_score = _compute_score(trial_positions, trial_angles)
            if trial_score > best_score:
                best_score = trial_score
                best_positions = trial_positions
                best_angles = trial_angles
    
    # Use the best found configuration
    inner_hex_data = np.column_stack([best_positions, best_angles])
    all_verts = _get_all_vertices(best_positions, best_angles, side=1.0)
    s = _min_outer_side(all_verts)
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END
