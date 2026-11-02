import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12
EPS = 1e-9

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions, angles_deg):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions and angles.
        """
        all_verts = []
        for i in range(N):
            verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
            all_verts.append(verts)
        
        verts_flat = np.vstack(all_verts)
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        return np.max(all_L)

    def check_validity(positions, angles_deg, tol=1e-5):
        """
        Check if all hexagons are pairwise disjoint using SAT.
        """
        vertices_list = []
        for i in range(N):
            verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                p1 = vertices_list[i]
                p2 = vertices_list[j]
                
                disjoint = False
                # Check SAT for p1 edges
                for k in range(6):
                    edge = p1[(k + 1) % 6] - p1[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                        disjoint = True
                        break
                
                if not disjoint:
                    # Check SAT for p2 edges
                    for k in range(6):
                        edge = p2[(k + 1) % 6] - p2[k]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-10: continue
                        normal /= norm
                        proj1 = p1.dot(normal)
                        proj2 = p2.dot(normal)
                        if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                            disjoint = True
                            break
                    
                    if not disjoint:
                        return False
        return True

    # Define the symmetric topology: 1 center, 6 ring 1, 5 ring 2
    # Base positions in a standard hexagonal lattice (side 1 hexagons touching)
    # Center at (0,0)
    # Ring 1: 6 neighbors at distance 2*sqrt(3) in hex directions?
    # Wait, for unit hexagons (side 1), the distance between centers of touching hexagons is 2*sqrt(3) if flat-top?
    # No, distance between centers of two unit hexagons sharing an edge is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    # Let's verify: Hexagon side s=1. Apothem a = sqrt(3)/2. Distance between parallel sides is 2a = sqrt(3).
    # So center-to-center distance for touching is sqrt(3).
    
    # Let's define base coordinates in a local frame where hexagons are aligned (angle 0)
    # Angle 0 means flat top?
    # hexagon_vertices uses cos/sin. Angle 0 -> vertex at (1,0). Pointy top.
    # If pointy top, apothem is in y direction?
    # Vertices: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2).
    # This is pointy top/bottom. Flat sides are left/right? No, vertical sides are at x=0.5 and x=-0.5?
    # Side from (0.5, sqrt(3)/2) to (-0.5, sqrt(3)/2) is horizontal. Length 1.
    # So "flat top" in the sense of horizontal top edge.
    # Distance between centers of two such hexagons sharing a horizontal edge:
    # Top edge y = sqrt(3)/2. Bottom edge y = -sqrt(3)/2.
    # If we stack them vertically, center distance is 2 * (sqrt(3)/2) = sqrt(3).
    # Direction: (0, 1).
    
    # Let's use axial coordinates for the honeycomb.
    # q, r
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # This corresponds to pointy top hexagons?
    # Let's check:
    # q=0, r=0 -> (0,0)
    # q=1, r=0 -> (sqrt(3), 0). Distance sqrt(3). Correct for sharing vertical edge?
    # In pointy top hex, vertical edges are not shared. Horizontal edges are shared vertically.
    # The standard "pointy top" hex grid uses:
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    # Here size=1 (side length).
    # Distance between (0,0) and (1,0) in this coord is sqrt(3).
    # Do they share an edge?
    # Hex at (0,0) has vertices (1,0), (0.5, .866), (-0.5, .866), (-1,0), (-0.5, -.866), (0.5, -.866).
    # Hex at (sqrt(3), 0) has vertices (sqrt(3)+1, 0), ...
    # The rightmost point of hex 0 is (1,0). Leftmost of hex 1 is (sqrt(3)-1, 0).
    # sqrt(3) ~ 1.732. 1.732 - 1 = 0.732.
    # Gap between 1 and 0.732 is 0.268. They don't touch.
    
    # Actually, for pointy top hexagons of side 1:
    # Width = 2 (horizontal diagonal). Height = sqrt(3) (vertical diagonal).
    # Wait, my vertex calc:
    # (1,0) to (-1,0) is width 2.
    # (0.5, .866) to (-0.5, -.866) ...
    # Max Y is .866, Min Y is -.866. Height sqrt(3).
    # So it is "pointy" horizontally? No, vertices at left and right are points.
    # So it is "flat" top and bottom?
    # Top edge connects (0.5, .866) and (-0.5, .866). This is a flat edge.
    # So it is Flat-Top.
    
    # For Flat-Top hexagons:
    # Distance between centers sharing a vertical edge?
    # Left edge is vertical? No, vertices are at angles 0, 60, 120...
    # Edge 0-1: (1,0) to (0.5, .866). Not vertical.
    # Edge 2-3: (-0.5, .866) to (-1,0). Not vertical.
    # There are no vertical edges.
    # The flat edges are Top (1-2), Bottom (4-5), Top-Left (5-0), Top-Right (3-4)?
    # Let's list edges:
    # 0(1,0) - 1(0.5, .866)
    # 1(0.5, .866) - 2(-0.5, .866)  <- Horizontal Top
    # 2(-0.5, .866) - 3(-1,0)
    # 3(-1,0) - 4(-0.5, -.866)
    # 4(-0.5, -.866) - 5(0.5, -.866) <- Horizontal Bottom
    # 5(0.5, -.866) - 0(1,0)
    
    # So Flat Top/Bottom.
    # To tile, we can stack vertically.
    # Top edge y=.866. Bottom edge of next hex y = -0.866 + dy.
    # For touching, dy = 2 * .866 = 1.732 = sqrt(3).
    # So vertical neighbor at (0, sqrt(3)).
    # Horizontal neighbor?
    # The hex has "shoulders" at x=0.5 and x=-0.5.
    # To fit a hex to the side, it must interlock.
    # Standard hex grid spacing for Flat Top:
    # x = 3/2 * size * q
    # y = sqrt(3) * size * (r + q/2)
    # Let's use this.
    
    # q, r coords
    # (0,0) -> (0,0)
    # (1,0) -> (1.5, 0.866)
    # (0,1) -> (0, 1.732)
    # Distance (0,0) to (1,0) is sqrt(1.5^2 + 0.866^2) = sqrt(2.25 + 0.75) = sqrt(3).
    # Distance (0,0) to (0,1) is 1.732 = sqrt(3).
    # This works.
    
    # Topology:
    # 1 center at (0,0) -> q=0, r=0
    # 6 ring 1 neighbors:
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # 5 ring 2 neighbors (subset of 12 available):
    # We need 5 more.
    # Available ring 2:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # To maintain symmetry (or close to it), we pick 5.
    # Maybe break symmetry slightly to allow tighter packing?
    # Or keep 6-fold symmetry if possible? 1+6+5=12. 5 is not divisible by 6.
    # So we cannot have full 6-fold symmetry.
    # We can have 1-fold symmetry (no symmetry) or 2-fold?
    # Let's just use a fixed set of 5 and allow global rotation.
    
    base_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    base_pos = np.zeros((N, 2))
    for i, (q, r) in enumerate(base_coords):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2.0)
        base_pos[i, 0] = x
        base_pos[i, 1] = y
        
    # Parameters:
    # s: radial scale factor (compression)
    # theta: global rotation
    # twist: relative rotation of outer ring? Or just global?
    # Let's try: s, theta, phi (rotation of ring 1 relative to center? No, they are connected).
    # Maybe: s, theta, and a "shear" or "stretch" in one direction?
    # Let's stick to: s (radial compression), theta (global rotation), alpha (rotation of all hexagons relative to lattice? i.e., all hexagons rotated by alpha).
    
    # Actually, the "Centroid-Minimization with Radial Expansion" suggests:
    # Fix topology.
    # Allow global affine deformation.
    # But affine deformation of a rigid body is just rotation+translation+scaling.
    # If we scale the positions, do we scale the hexagons? No, hexagons are fixed size 1.
    # So we are compressing the *spacing* between centers.
    # This will cause overlap if compressed too much.
    # So we optimize s (scale of positions) such that no overlap occurs.
    # And we rotate the whole assembly to minimize bounding box.
    # And we can rotate the individual hexagons?
    # If we rotate the whole assembly, the hexagons rotate with it.
    # Can we rotate hexagons independently?
    # If we allow independent rotation, it's 12 extra parameters.
    # Let's try: s, theta, and a single "twist" parameter applied to the outer 11 hexagons relative to the center?
    # Or just s and theta.
    
    def get_config(s, theta, twist):
        # s: scale of positions
        # theta: global rotation
        # twist: rotation of outer hexagons relative to global?
        
        pos = base_pos * s
        
        # Apply global rotation
        c = np.cos(np.radians(theta))
        sn = np.sin(np.radians(theta))
        rot_mat = np.array([[c, -sn], [sn, c]])
        
        pos = pos @ rot_mat.T
        
        # Angles
        # Base angle is 0.
        # Global rotation adds theta to all angles.
        # Twist adds twist to angles of non-center hexagons?
        angles = np.full(N, theta)
        # Apply twist to indices 1..11
        angles[1:] += twist
        
        return pos, angles

    def objective(params):
        s, theta, twist = params
        # s should be > 0. If s is too small, overlap.
        # We want to minimize outer side.
        # Outer side increases as s increases.
        # So we want to minimize s subject to no overlap.
        # But we can also rotate to reduce bounding box.
        # So objective is outer_side(pos, angles).
        # We need to penalize overlap.
        
        pos, angles = get_config(s, theta, twist)
        
        # Check overlap penalty
        # Use smooth approximation of max(0, overlap_depth)
        # For each pair, calculate minimum distance between edges?
        # Or just use SAT violation?
        
        # Simple penalty: if any pair overlaps, add large penalty.
        # But we need gradient.
        # Let's use a soft penalty based on distance between centers?
        # No, rotation matters.
        
        # Let's compute vertices and check distance.
        # This is expensive.
        
        # Alternative:
        # If we assume the topology is "rigid" enough that overlap only happens when s is below a threshold,
        # we can find the critical s for each pair and use that.
        
        # But for now, let's just evaluate the side length and check validity in the wrapper.
        # If invalid, return large value.
        # This makes the function non-smooth.
        # Nelder-Mead can handle non-smooth functions.
        
        side = required_outer_side(pos, angles)
        
        # Add penalty for overlap
        # We'll do a quick check.
        # If overlap, return side + 100 * overlap_amount
        # We need overlap_amount.
        
        # Let's just return side for now and rely on NM to find valid region.
        # If it steps into invalid region, it will see high "cost" if we add penalty.
        
        # Let's implement a simple overlap penalty using center distances as a proxy?
        # No, that's wrong for rotated hexagons.
        
        # Let's use the SAT check to compute a "violation" score.
        # If disjoint, violation = 0.
        # If not, violation = -max_sep (where max_sep is negative).
        
        vertices_list = []
        for i in range(N):
            verts = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, angles[i])
            vertices_list.append(verts)
            
        max_violation = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                p1 = vertices_list[i]
                p2 = vertices_list[j]
                
                min_sep = np.inf
                # Check SAT
                for k in range(6):
                    edge = p1[(k + 1) % 6] - p1[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10: continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    sep = min(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
                    if sep < min_sep:
                        min_sep = sep
                        
                # If min_sep < 0, they overlap.
                # The "depth" of overlap is roughly -min_sep (not exact, but good for penalty).
                if min_sep < 0:
                    if -min_sep > max_violation:
                        max_violation = -min_sep
                    
        penalty = 10.0 * max_violation
        return side + penalty

    # Optimization
    best_side = float('inf')
    best_params = None
    
    # Initial guesses
    # s=1 is touching.
    # Try s from 0.9 to 1.2
    # theta from 0 to 60
    # twist from -30 to 30
    
    inits = []
    for s in [0.95, 1.0, 1.05]:
        for theta in [0, 15, 30, 45]:
            for twist in [-15, 0, 15]:
                inits.append([s, theta, twist])
                
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                objective,
                init,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            
            s, theta, twist = res.x
            pos, angles = get_config(s, theta, twist)
            
            if check_validity(pos, angles):
                side = required_outer_side(pos, angles)
                if side < best_side:
                    best_side = side
                    best_params = (s, theta, twist, pos, angles)
        except Exception:
            pass

    # Fallback
    if best_params is None:
        # Use the initial lattice
        pos, angles = get_config(1.0, 0.0, 0.0)
        best_side = required_outer_side(pos, angles)
        best_params = (1.0, 0.0, 0.0, pos, angles)

    final_pos = best_params[3]
    final_angles = best_params[4]
    
    # Final validation
    if not check_validity(final_pos, final_angles):
        # Try to repair by scaling up slightly
        for scale in [1.01, 1.02, 1.05]:
            scaled_pos = final_pos * scale
            if check_validity(scaled_pos, final_angles):
                final_pos = scaled_pos
                break
        else:
            # Last resort: linear arrangement
            final_pos = np.array([[i * 3.0, 0.0] for i in range(N)])
            final_angles = np.zeros(N)
            
    final_side = required_outer_side(final_pos, final_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(final_pos[i, 0]), float(final_pos[i, 1]), float(final_angles[i])])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
