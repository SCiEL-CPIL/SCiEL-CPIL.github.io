import math
import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Unit hexagon vertices for angle 0 (flat-top)
    # Vertices at 0, 60, ..., 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals for the 6 sides of a hexagon
        # For a flat-top hexagon, side normals are at 30, 90, 150, 210, 270, 330
        normals = np.array([[math.cos(math.radians(30 + k*60)), math.sin(math.radians(30 + k*60))] for k in range(6)])
        
        # Rotate normals by outer_angle
        cos_o = math.cos(theta_rad)
        sin_o = math.sin(theta_rad)
        rot_o = np.array([[cos_o, -sin_o], [sin_o, cos_o]])
        normals = normals @ rot_o.T
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value (approximate penetration depth).
        If they are separated, returns the positive distance.
        """
        min_gap = float('inf')
        
        # Axes from poly1 edges
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        # Axes from poly2 edges
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        return min_gap

    # --- Strategy: Rigid Chain / Topological Search ---
    
    # Hexagonal lattice directions (unit vectors for moving to adjacent hexagon)
    # Angle 0 is right (1, 0). 
    # Directions: 0, 60, 120, 180, 240, 300 degrees
    lattice_dirs = []
    for k in range(6):
        ang = math.radians(60 * k)
        lattice_dirs.append([math.cos(ang), math.sin(ang)])
    lattice_dirs = np.array(lattice_dirs)
    
    # Generate all non-isomorphic Hamiltonian paths (self-avoiding walks) of length 12
    # We represent a path as a sequence of 11 turns (0, 1, 2) relative to the previous direction.
    # 0 = straight, 1 = turn right (60 deg), 2 = turn left (60 deg)
    # Note: We fix the first step direction to 0 (right) to reduce symmetry, 
    # but we must consider that the chain can start in any of the 6 directions? 
    # Actually, since we rotate the whole chain later, fixing the first step direction 
    # is sufficient for topological uniqueness if we also consider the sequence of turns.
    # However, we must avoid self-intersection.
    
    def generate_paths():
        paths = []
        def backtrack(pos, prev_dir, visited, turn_seq):
            if len(turn_seq) == 11:
                paths.append(turn_seq[:])
                return
            
            current_dir = prev_dir
            # Try 3 relative turns: straight (0), right (1), left (2)
            for turn in [0, 1, 2]:
                # Calculate new absolute direction
                # If turn 0: dir stays same
                # If turn 1: dir = (prev_dir + 1) % 6
                # If turn 2: dir = (prev_dir + 5) % 6
                new_dir = (current_dir + turn) % 6
                
                # Calculate new position
                new_pos = pos + lattice_dirs[new_dir]
                
                # Check if visited
                if tuple(new_pos) in visited:
                    continue
                
                # Backtrack
                visited.add(tuple(new_pos))
                turn_seq.append(turn)
                backtrack(new_pos, new_dir, visited, turn_seq)
                turn_seq.pop()
                visited.remove(tuple(new_pos))
                
        # Start at (0,0), first step in direction 0
        start_pos = np.array([0.0, 0.0])
        visited = {tuple(start_pos)}
        backtrack(start_pos, 0, visited, [])
        
        # Deduplicate paths by normalizing the turn sequence
        # Since the chain is rigid, a path and its reverse might be topologically equivalent 
        # but geometrically different unless we consider reflection. 
        # For now, we treat distinct turn sequences as distinct topologies.
        # We can also deduplicate by checking if the set of relative turns is a palindrome or similar, 
        # but for N=12 the number is small enough (few thousand) to just try them all.
        
        return paths

    all_paths = generate_paths()
    
    # Symmetry reduction: 
    # A path defined by turns t1...t11 is equivalent to its reverse if we consider the shape.
    # Reverse path turns: t11...t1. 
    # Also, reflecting the path (swapping left/right turns) yields a mirror image.
    # Since we can rotate the final shape, mirror images are effectively the same set of configurations 
    # if we allow full rotation. However, the specific arrangement relative to the outer hexagon 
    # might differ. But since we optimize the outer hexagon rotation too, 
    # we can just pick a representative from each symmetry class to save time.
    
    def normalize_path(turns):
        # Canonical form: lexicographically smallest of:
        # 1. Original
        # 2. Reversed
        # 3. Mirrored (swap 1 and 2)
        # 4. Reversed Mirrored
        orig = tuple(turns)
        rev = tuple(reversed(turns))
        mir = tuple(3 - t for t in orig) # 0->0, 1->2, 2->1
        rev_mir = tuple(3 - t for t in rev)
        
        return min([orig, rev, mir, rev_mir])

    unique_paths = set()
    unique_path_list = []
    for p in all_paths:
        norm = normalize_path(p)
        if norm not in unique_paths:
            unique_paths.add(norm)
            unique_path_list.append(norm)
            
    # Sort unique paths by some heuristic? e.g., bounding box size of the unrotated chain
    def path_bounding_box(turns):
        pos = np.array([0.0, 0.0])
        curr_dir = 0
        min_x, max_x = 0, 0
        min_y, max_y = 0, 0
        for t in turns:
            curr_dir = (curr_dir + t) % 6
            pos = pos + lattice_dirs[curr_dir]
            min_x = min(min_x, pos[0])
            max_x = max(max_x, pos[0])
            min_y = min(min_y, pos[1])
            max_y = max(max_y, pos[1])
        return (max_x - min_x) ** 2 + (max_y - min_y) ** 2
        
    # Sort by bounding box area to prioritize compact shapes first
    unique_path_list.sort(key=path_bounding_box)
    
    best_score = float('inf')
    best_result = None
    
    # We will test each unique topology.
    # For each topology, we have a fixed set of relative positions (in lattice units).
    # We need to map these to Cartesian coordinates.
    # The lattice vectors are unit length.
    # The angle of each hexagon in the chain:
    # The hexagon at step k is aligned with the direction it entered from? 
    # No, a "rigid chain" of hexagons usually implies they are attached edge-to-edge.
    # If hexagon i is at position P_i and hexagon i+1 is at P_i + v, 
    # then the edge of hexagon i facing P_{i+1} is parallel to the edge of hexagon i+1 facing P_i.
    # This implies that the angle of hexagon i+1 is related to hexagon i.
    # Specifically, if we move in direction d (angle 60*d), the shared edge is perpendicular to d.
    # The orientation of the hexagon is defined by its "up" direction.
    # Let's define the angle of the hexagon at step k as the angle of the vector from center k to center k+1, 
    # rotated by 90 degrees? Or simpler:
    # If we assume the chain is formed by attaching hexagons such that they share a full edge,
    # the relative rotation between consecutive hexagons is determined by the turn.
    # If we go straight (turn 0), the next hexagon has the same orientation as the previous one?
    # Let's check.
    # Hex 0 at (0,0), angle 0. Vertices at 0, 60...
    # Move to (1,0) (Dir 0).
    # Hex 1 at (1,0). To share the edge on the left of Hex 1 (which is at x=0.5) with the right edge of Hex 0 (at x=0.5),
    # Hex 1 must also have angle 0.
    # So if turn is 0 (straight), angle stays same.
    # If turn is 1 (right, 60 deg), we move to (0.5, sqrt(3)/2).
    # The vector is at 60 deg.
    # The shared edge is perpendicular to the movement vector.
    # Movement vector angle: 60 deg.
    # Edge normal angle: 60 deg.
    # For Hex 0, the edge with normal 60 deg is the top-right edge?
    # Vertices of Hex 0 (angle 0): (1,0), (0.5, .866), (-0.5, .866), (-1,0), (-0.5, -.866), (0.5, -.866).
    # Edge 0: (1,0) to (0.5, .866). Vector (-0.5, .866). Normal (0.866, 0.5) -> angle 30 deg.
    # Edge 1: (0.5, .866) to (-0.5, .866). Vector (-1, 0). Normal (0, 1) -> angle 90 deg.
    # This is getting complicated.
    
    # Simpler model:
    # The "angle" parameter in the output is the rotation of the hexagon.
    # In a rigid chain of hexagons packed edge-to-edge, the orientation of each hexagon is fixed relative to the others.
    # If we define the "global angle" of the chain, we can compute the local angle of each hexagon.
    # Let's define the angle of the first hexagon as theta.
    # The second hexagon is placed such that it shares an edge.
    # The relative rotation between hexagon i and i+1 is 0 degrees if they are "aligned" in the lattice sense?
    # Actually, if you tile the plane with hexagons, all hexagons have the same orientation.
    # So, in a rigid chain on the hexagonal lattice, ALL hexagons have the SAME orientation!
    # This is a key insight. The "turn" in the path determines the position, but not the rotation of the individual hexagons,
    # because they are all part of the same honeycomb tiling locally.
    # Therefore, the angle for all inner hexagons is the same: theta.
    
    # So the optimization is:
    # 1. Choose a topology (turn sequence).
    # 2. Choose a global rotation theta for the whole chain.
    # 3. Calculate the minimum enclosing hexagon side length.
    
    # Wait, is it true that all hexagons have the same orientation?
    # Yes, if they are disjoint and packed on the lattice, they are vertices of a subgraph of the honeycomb.
    # The honeycomb is a uniform tiling where all tiles are congruent and aligned.
    # So, relative rotations are 0.
    
    # This simplifies the problem significantly!
    # We just need to find the best topology and the best global rotation theta.
    
    for path in unique_path_list:
        if time.time() > deadline - 10:
            break
            
        # Calculate positions for this path
        # Start at (0,0), first step dir 0
        pos = np.array([0.0, 0.0])
        positions = [pos.copy()]
        curr_dir = 0
        for t in path:
            curr_dir = (curr_dir + t) % 6
            pos = pos + lattice_dirs[curr_dir]
            positions.append(pos.copy())
            
        positions = np.array(positions)
        
        # Center the chain to reduce numerical issues and symmetry
        centroid = np.mean(positions, axis=0)
        positions -= centroid
        
        # The angles are all the same: theta
        # We need to minimize the outer side length by choosing theta.
        
        def outer_side_for_theta(theta_deg):
            angles = np.full(N, theta_deg)
            return calc_min_outer_side(positions, angles, 0.0)
            
        # Minimize outer_side_for_theta over [0, 60)
        # The function is periodic with period 60 degrees.
        
        try:
            res = minimize_scalar(outer_side_for_theta, bounds=(0, 60), method='bounded', options={'xatol': 1e-8})
            if res.success:
                best_theta = res.x
                best_side = res.fun
                if best_side < best_score:
                    best_score = best_side
                    # Store the best configuration
                    angles = np.full(N, best_theta)
                    # We need to return the absolute positions. 
                    # The positions were centered. 
                    # We can return them as is, and the outer hexagon center is (0,0).
                    best_positions = positions.copy()
                    best_result = (best_positions, angles)
        except Exception:
            pass

    # Fallback if no path was processed or all failed
    if best_result is None:
        # Use the honeycomb initial guess from the previous program as fallback
        sqrt3 = math.sqrt(3)
        lattice_points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * 1.5
                y = i * sqrt3 / 2.0 + j * sqrt3
                lattice_points.append((x, y))
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_positions = np.array(lattice_points[:12])
        init_positions -= np.mean(init_positions, axis=0)
        init_angles = np.zeros(N)
        
        # Optimize rotation for this fallback
        def outer_side_for_theta(theta_deg):
            return calc_min_outer_side(init_positions, init_angles, 0.0)
            
        # Actually, just pick 0
        best_score = calc_min_outer_side(init_positions, init_angles, 0.0)
        best_result = (init_positions, init_angles)

    final_positions, final_angles = best_result
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0

if __name__ == "__main__":
    construct()
