"""
Combined approach:
1. Generate candidate 11-cell polyhex topologies via BFS.
2. For each topology, treat the cluster as a rigid body.
3. Use vectorized NumPy operations to compute the required container side length efficiently.
4. Optimize the global rotation and translation of the cluster relative to a fixed container orientation.
5. Perform a high-precision polish on the best result.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize
from collections import deque

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices_local(angle_rad):
    """
    Returns the 6 vertices of a unit hexagon centered at origin, rotated by angle_rad.
    Shape: (6, 2)
    """
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([np.cos(thetas), np.sin(thetas)])

def compute_required_side_vectorized(centers, angles, global_rotation, translation, container_center=(0,0,0)):
    """
    Computes the minimum side length of a regular hexagon container to enclose the given hexagons.
    
    Parameters:
    - centers: (N, 2) array of local centers of the hexagons (in the composite frame).
    - angles: (N,) array of local rotation angles (radians) of the hexagons.
    - global_rotation: Scalar, rotation of the entire composite (radians).
    - translation: (2,) array, translation of the composite center.
    - container_center: (x, y) center of the container. We assume container is centered at origin 
      and aligned with axes (normals at 30, 90, ...). The 'global_rotation' effectively 
      rotates the composite relative to this fixed container.
      
    Returns:
    - Minimum side length.
    """
    # Rotate composite centers and angles
    cos_g = np.cos(global_rotation)
    sin_g = np.sin(global_rotation)
    
    # Rotate centers: R * c + t
    # c is (N, 2). R is (2, 2).
    rot_mat = np.array([[cos_g, -sin_g], [sin_g, cos_g]])
    global_centers = centers @ rot_mat.T + translation
    
    # Global angles
    global_angles = angles + global_rotation
    
    # Pre-compute local vertices for each hexagon (6 vertices each)
    # We need to transform each vertex: v_global = c_global + R(angle_global) * v_local_unit
    # v_local_unit is fixed (unit hexagon vertices at 0 deg).
    unit_verts = get_hex_vertices_local(0.0) # (6, 2)
    
    # We want max projection onto 6 normals.
    # Normals for a hexagon with "pointy top" (vertex at 90 deg) are at 30, 90, 150, 210, 270, 330?
    # Let's define the container such that its vertices are at 0, 60, 120... (Flat top? No, pointy top usually means vertex at 90).
    # Standard regular hexagon: Vertices at 0, 60, 120, 180, 240, 300.
    # Normals (perpendicular to edges) are at 30, 90, 150, 210, 270, 330.
    # Let's use this standard definition.
    normal_angles = np.radians(30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)]) # (6, 2)
    
    max_proj = -np.inf
    
    # Iterate over hexagons. N is small (11), so this loop is fine.
    # Inside, we vectorize over the 6 vertices.
    for i in range(N):
        c_g = global_centers[i]
        a_g = global_angles[i]
        
        # Rotate unit vertices by a_g
        cos_a = np.cos(a_g)
        sin_a = np.sin(a_g)
        rot_a = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts @ rot_a.T + c_g # (6, 2)
        
        # Project onto normals
        projs = verts @ normals.T # (6, 6)
        cur_max = np.max(projs)
        
        if cur_max > max_proj:
            max_proj = cur_max
            
    # The distance from center to the farthest point along a normal is max_proj.
    # For a regular hexagon, the apothem (distance to edge) is side * sqrt(3) / 2.
    # The projection of a vertex onto the normal of the opposite edge...
    # Wait, the support function h(u) = max(v . u).
    # The container is the set of points x such that x . n_i <= apothem for all normals n_i.
    # So we need apothem >= max_proj.
    # Apothem = side * sqrt(3) / 2.
    # side = 2 * apothem / sqrt(3) = 2 * max_proj / sqrt(3).
    
    # Note: max_proj can be negative if everything is on the other side, but since container is centered at 0 
    # and we optimize translation, we effectively want the min width.
    # Actually, if max_proj < 0, the hexagons are entirely on the negative side. 
    # But the container must contain them. The distance to the negative edge is -min_proj.
    # The required apothem is max(max_proj, -min_proj)? No.
    # The container is defined by 6 inequalities: x . n_i <= A and -x . n_i <= A (which is x . (-n_i) <= A).
    # Our normals cover all 6 directions.
    # So we just need A >= max( x . n_i ) for all i and all points x.
    # Since the normals include both u and -u, max(x . n_i) effectively captures the distance to the furthest face in any direction.
    # So side = 2 * max_proj / SQRT3.
    
    if max_proj < 0:
        # This shouldn't happen if we center the composite properly, but just in case.
        # If all projections are negative, the max projection is the "least negative" one.
        # But the container must extend to cover the points.
        # If the points are at -5, the max projection onto the normal pointing to +5 is -5.
        # But the normal pointing to -5 will have projection +5.
        # So max_proj will always be >= 0 if the points are not all at the origin?
        # Actually, if a point is at (0,0), proj is 0.
        # If a point is at (1,0), proj onto (1,0) is 1, onto (-1,0) is -1. Max is 1.
        # So max_proj is always >= 0 for any non-empty set of points?
        # Yes, because for any vector v, max(v, -v) >= 0.
        pass
        
    return 2.0 * max_proj / SQRT3

def generate_polyhexes(n_cells):
    """
    Generate all unique polyhexes of size n_cells using BFS on axial coordinates.
    Returns a list of lists of (q, r) coordinates.
    """
    def axial_to_cube(q, r):
        x = q
        z = r
        y = -x - z
        return (x, y, z)
        
    def cube_to_axial(x, y, z):
        return (x, z)
        
    def cube_neighbor(x, y, z):
        return [
            (x+1, y-1, z), (x+1, y, z-1),
            (x, y+1, z-1), (x, y+1, z),
            (x-1, y+1, z), (x-1, y, z+1)
        ]
        
    def canonicalize(cells):
        """
        Find the canonical representation of a polyhex to avoid duplicates.
        1. Translate so min x, min y, min z are 0? No, just translate to origin.
        2. Rotate 6 times, reflect, take min lexicographic string.
        """
        cells = list(cells)
        # Translate to origin
        min_x = min(c[0] for c in cells)
        min_y = min(c[1] for c in cells)
        min_z = min(c[2] for c in cells)
        # Actually, just shift by (-min_x, -min_y, -min_z) doesn't work directly in axial if we don't handle constraints.
        # Easier: Shift by (-min_q, -min_r) in axial.
        min_q = min(c[0] for c in cells)
        min_r = min(c[1] for c in cells)
        shifted = [(q - min_q, r - min_r) for q, r in cells]
        
        def rotate_axial(q, r):
            # 60 degree rotation
            return (-r, q + r)
            
        def reflect_axial(q, r):
            # Reflection across x-axis (or similar)
            # (q, r) -> (q+r, -q) ? 
            # Let's use cube reflection: (x, y, z) -> (-y, -x, -z)
            x, y, z = axial_to_cube(q, r)
            nx, ny, nz = -y, -x, -z
            return cube_to_axial(nx, ny, nz)
            
        best_rep = None
        current = shifted
        for _ in range(6):
            # Check rotation
            rep = tuple(sorted(current))
            if best_rep is None or rep < best_rep:
                best_rep = rep
            # Check reflection of current
            reflected = [reflect_axial(q, r) for q, r in current]
            # Normalize reflected
            min_q_r = min(c[0] for c in reflected)
            min_r_r = min(c[1] for c in reflected)
            reflected_norm = tuple(sorted((q - min_q_r, r - min_r_r) for q, r in reflected))
            if reflected_norm < best_rep:
                best_rep = reflected_norm
                
            # Rotate for next iteration
            current = [rotate_axial(q, r) for q, r in current]
            # Re-normalize
            min_q_c = min(c[0] for c in current)
            min_r_c = min(c[1] for c in current)
            current = [(q - min_q_c, r - min_r_c) for q, r in current]
            
        return best_rep

    # BFS
    start = [(0, 0)]
    queue = deque([tuple(start)])
    seen = {tuple(start)}
    unique_polyhexes = set()
    
    while queue:
        current = queue.popleft()
        if len(current) == n_cells:
            canon = canonicalize(current)
            unique_polyhexes.add(canon)
            continue # Don't expand further
            
        if len(current) > n_cells:
            continue
            
        # Find neighbors to add
        for q, r in current:
            for nq, nr in [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]:
                if (nq, nr) not in current:
                    new_state = tuple(sorted(current + [(nq, nr)]))
                    if new_state not in seen:
                        seen.add(new_state)
                        queue.append(new_state)
                        
    return list(unique_polyhexes)

def axial_to_cartesian(q, r, scale=1.0):
    x = 1.5 * scale * q
    y = SQRT3 * scale * (r + q / 2.0)
    return x, y

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Generate topologies
    # 11 cells might take a bit to generate, but it's a one-time cost.
    # There are 1184 free polyhexes of size 11.
    # Generation should be fast enough in Python if optimized.
    try:
        topologies = generate_polyhexes(N)
    except Exception:
        # Fallback to flower topology if generation fails or is too slow
        topologies = [
            tuple(sorted([
                (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
                (2,-1), (1,1), (-2,1), (-1,-1)
            ]))
        ]
        
    # Filter topologies by bounding box size to prioritize compact ones
    def bbox_size(topo):
        qs = [q for q, r in topo]
        rs = [r for q, r in topo]
        # Convert to cartesian to get true bounding box
        pts = [axial_to_cartesian(q, r) for q, r in topo]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))
        
    topologies = sorted(topologies, key=bbox_size)
    
    # We will evaluate the top K topologies
    K_TOP = 50 
    candidate_topos = topologies[:K_TOP]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    epsilon = 1e-6
    scale = 1.0 + epsilon
    
    for topo in candidate_topos:
        if time.time() > deadline - 20:
            break
            
        # Build initial configuration for this topology
        centers = []
        for q, r in topo:
            x, y = axial_to_cartesian(q, r, scale)
            centers.append([x, y])
            
        centers_np = np.array(centers)
        angles_np = np.zeros(N)
        
        # Center the topology at origin for optimization
        cx = np.mean(centers_np[:, 0])
        cy = np.mean(centers_np[:, 1])
        centers_np -= np.array([cx, cy])
        
        def objective(params):
            # params: [rotation, tx, ty]
            rot = params[0]
            tx = params[1]
            ty = params[2]
            return compute_required_side_vectorized(centers_np, angles_np, rot, [tx, ty])
            
        # Initial guess: rotation 0, translation 0
        x0 = [0.0, 0.0, 0.0]
        
        # Multi-start optimization for this topology
        best_local_val = float('inf')
        best_local_params = None
        
        # Try a few random rotations to avoid local minima
        starts = [
            [0.0, 0.0, 0.0],
            [math.pi/6, 0.0, 0.0], # 30 deg
            [math.pi/12, 0.0, 0.0], # 15 deg
            [math.pi/4, 0.0, 0.0], # 45 deg
        ]
        
        for s in starts:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(objective, s, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_local_val:
                    best_local_val = res.fun
                    best_local_params = res.x
            except Exception:
                pass
                
        if best_local_params is not None and best_local_val < best_side:
            best_side = best_local_val
            best_local_params_final = best_local_params
            best_topo_final = topo
            best_centers_final = centers_np.copy()
            
    # If we found a better topology, reconstruct the final solution
    if best_inner is None and best_topo_final is not None:
        rot = best_local_params_final[0]
        tx = best_local_params_final[1]
        ty = best_local_params_final[2]
        
        cos_g = np.cos(rot)
        sin_g = np.sin(rot)
        rot_mat = np.array([[cos_g, -sin_g], [sin_g, cos_g]])
        
        global_centers = best_centers_final @ rot_mat.T + np.array([tx, ty])
        global_angles = np.zeros(N) + rot # All hexagons rotate with the composite
        
        # Construct inner list: (x, y, angle_deg)
        inner_list = []
        for i in range(N):
            x = global_centers[i, 0]
            y = global_centers[i, 1]
            a_deg = math.degrees(global_angles[i])
            inner_list.append([x, y, a_deg])
            
        best_inner = inner_list
        best_outer_center = (0.0, 0.0)
        # The container angle in the output should be 0 because we optimized relative to a fixed container.
        # However, the evaluator might expect the container angle to be whatever we used.
        # We used normals at 30, 90... which corresponds to a hexagon with vertices at 0, 60...
        # Let's check: Normal at 30 is perpendicular to edge between 0 and 60.
        # Yes, standard orientation.
        best_outer_angle = 0.0
        best_side = compute_required_side_vectorized(best_centers_final, np.zeros(N), rot, [tx, ty])
        
    # Final Polish:
    # If we have a valid best_inner, we can try to polish the individual angles?
    # Given the time constraint and that rigid body is usually optimal for this, 
    # we will do a high-precision polish on the rigid body parameters if we have time.
    
    if best_inner is not None and time.time() < deadline - 5:
        # Re-define objective for final polish using the specific topology found
        # We need to recover the local centers and angles from best_inner?
        # It's easier to just re-run the optimization on the best topology with tighter tolerances.
        
        # Identify which topology was best
        # (We stored best_topo_final, best_centers_final, etc.)
        
        def objective_polish(params):
            rot = params[0]
            tx = params[1]
            ty = params[2]
            return compute_required_side_vectorized(best_centers_final, np.zeros(N), rot, [tx, ty])
            
        x0 = [best_local_params_final[0], best_local_params_final[1], best_local_params_final[2]]
        
        try:
            res = minimize(objective_polish, x0, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_side:
                best_side = res.fun
                rot = res.x[0]
                tx = res.x[1]
                ty = res.x[2]
                
                cos_g = np.cos(rot)
                sin_g = np.sin(rot)
                rot_mat = np.array([[cos_g, -sin_g], [sin_g, cos_g]])
                
                global_centers = best_centers_final @ rot_mat.T + np.array([tx, ty])
                global_angles = np.zeros(N) + rot
                
                inner_list = []
                for i in range(N):
                    x = global_centers[i, 0]
                    y = global_centers[i, 1]
                    a_deg = math.degrees(global_angles[i])
                    inner_list.append([x, y, a_deg])
                    
                best_inner = inner_list
                best_outer_center = (0.0, 0.0)
                best_outer_angle = 0.0
        except Exception:
            pass

    # Fallback if nothing worked
    if best_inner is None:
        # Use the default flower topology
        topo = [
            (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
            (2,-1), (1,1), (-2,1), (-1,-1)
        ]
        centers = []
        for q, r in topo:
            x, y = axial_to_cartesian(q, r, scale)
            centers.append([x, y])
        centers_np = np.array(centers)
        cx = np.mean(centers_np[:, 0])
        cy = np.mean(centers_np[:, 1])
        centers_np -= np.array([cx, cy])
        
        # Simple rigid body optimization
        def obj_fallback(params):
            return compute_required_side_vectorized(centers_np, np.zeros(N), params[0], [params[1], params[2]])
            
        res = minimize(obj_fallback, [0,0,0], method='Nelder-Mead', options={'maxiter': 1000})
        rot, tx, ty = res.x
        
        cos_g = np.cos(rot)
        sin_g = np.sin(rot)
        rot_mat = np.array([[cos_g, -sin_g], [sin_g, cos_g]])
        global_centers = centers_np @ rot_mat.T + np.array([tx, ty])
        global_angles = np.zeros(N) + rot
        
        inner_list = []
        for i in range(N):
            x = global_centers[i, 0]
            y = global_centers[i, 1]
            a_deg = math.degrees(global_angles[i])
            inner_list.append([x, y, a_deg])
            
        best_inner = inner_list
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = res.fun

    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
