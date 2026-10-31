import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Base hexagon vertices for side 1, angle 0
    # Vertices at angles 0, 60, ..., 300
    base_verts = []
    for i in range(6):
        theta = math.radians(60 * i)
        base_verts.append((math.cos(theta), math.sin(theta)))

    def transform_verts(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        verts = []
        for vx, vy in base_verts:
            rx = vx * ca - vy * sa
            ry = vx * sa + vy * ca
            verts.append((cx + S * rx, cy + S * ry))
        return verts

    def compute_outer_hex(inner, ocx, ocy, oangle):
        """
        Computes the minimum side length S of a regular hexagon 
        centered at (ocx, ocy) with rotation oangle 
        that contains all inner hexagons.
        """
        inner_verts = []
        for (ix, iy, iangle) in inner:
            inner_verts.append(transform_verts(ix, iy, 1.0, iangle))
        
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        
        # The distance from center to side of regular hexagon with side S is S * sqrt(3)/2
        # We need to check the projection of all points onto the normals of the 6 sides.
        for i in range(6):
            # Normal angle for side i. Sides are perpendicular to vertices?
            # Vertices are at 0, 60...
            # Side 0 is between vertex 0 and 1. Midpoint angle is 30.
            # Normal points outwards at 30, 90, 150...
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            
            max_proj = 0.0
            for iv in inner_verts:
                for v in iv:
                    proj = (v[0] - ocx) * nx + (v[1] - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
            
            # If max_proj is the distance from center to the furthest point in that direction,
            # and the side of the container is at distance d = S * sqrt(3)/2,
            # then we need S * sqrt(3)/2 >= max_proj.
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def check_fit(inner, ocx, ocy, S, oangle, eps=1e-9):
        """
        Checks if inner hexagons are disjoint and fit inside outer hexagon.
        """
        outer_verts = transform_verts(ocx, ocy, S, oangle)
        inner_verts = []
        for (ix, iy, iangle) in inner:
            inner_verts.append(transform_verts(ix, iy, 1.0, iangle))
        
        # Check containment
        def point_in_polygon(p, polygon):
            x, y = p
            n = len(polygon)
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = polygon[i]
                xj, yj = polygon[j]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            return inside

        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    return False

        # Check disjointness
        # For regular hexagons of same size, disjointness can be checked by center distance
        # if rotations are aligned, but here rotations might differ.
        # However, our topology generation uses consistent rotation or 0.
        # Let's use a robust check: distance between centers must be >= sqrt(3) if aligned.
        # If rotated, it's complex. 
        # But in our strategy, we fix inner rotations to 0 (or global rotation).
        # If we fix inner rotations to 0, then center distance >= sqrt(3) is sufficient and necessary.
        
        for i in range(len(inner)):
            for j in range(i + 1, len(inner)):
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Minimum distance for disjoint unit hexagons (flat-to-flat or point-to-point)
                # The minimum distance between centers of two disjoint unit hexagons is sqrt(3).
                # This occurs when they touch flat-to-flat.
                if dist < math.sqrt(3) - eps:
                    return False
        return True

    def generate_topologies():
        """
        Generates a list of rigid topologies (list of (x, y) coordinates) 
        for N hexagons on a triangular lattice with spacing sqrt(3).
        """
        topologies = []
        sp = math.sqrt(3)
        
        # Basis vectors for triangular lattice
        # a1 = (sp, 0)
        # a2 = (sp/2, sp*sqrt(3)/2)
        
        # We will generate clusters by picking N points from a small grid 
        # that are "connected" (neighbors within sp + eps).
        # To ensure compactness, we pick points with smallest radial distance from centroid,
        # but we must ensure connectivity.
        
        # Strategy: Generate all subsets of size N from a small grid (e.g., 5x5)
        # that are connected and have minimal bounding box area or moment of inertia.
        
        # Grid indices -3 to 3
        indices = range(-3, 4)
        points = []
        for i in indices:
            for j in indices:
                x = i * sp + j * sp / 2
                y = j * sp * math.sqrt(3) / 2
                points.append((i, j, x, y))
        
        # Sort by distance from origin
        points.sort(key=lambda p: p[2]**2 + p[3]**2)
        
        # We want to find good subsets. 
        # Instead of checking all C(49, 11) subsets, we use a heuristic:
        # Pick the N points closest to the origin? This gives a "blob".
        # But we want to explore different shapes.
        
        # Let's manually define a few known good shapes for N=11.
        # 1. Center + 6 neighbors + 4 more?
        # 2. 3-4-3 rows?
        
        # Let's use a random sampling approach with connectivity check.
        # Sample 1000 random connected subsets.
        
        rng = random.Random(42)
        seen_shapes = set()
        
        # Precompute adjacency
        point_map = {(p[0], p[1]): p for p in points}
        adj = {}
        for p in points:
            adj[(p[0], p[1])] = []
        
        for p in points:
            pi, pj = p[0], p[1]
            # Neighbors in triangular lattice (i, j)
            # (i+1, j), (i, j+1), (i+1, j-1) ... wait, basis depends on definition.
            # x = i*sp + j*sp/2
            # y = j*sp*sqrt(3)/2
            # Neighbor 1: i+1, j -> dx=sp, dy=0. Dist sp. OK.
            # Neighbor 2: i, j+1 -> dx=sp/2, dy=sp*sqrt(3)/2. Dist sp. OK.
            # Neighbor 3: i-1, j+1 -> dx=-sp/2, dy=sp*sqrt(3)/2. Dist sp. OK.
            # Neighbor 4: i-1, j -> dx=-sp, dy=0. Dist sp. OK.
            # Neighbor 5: i, j-1 -> dx=-sp/2, dy=-sp*sqrt(3)/2. Dist sp. OK.
            # Neighbor 6: i+1, j-1 -> dx=sp/2, dy=-sp*sqrt(3)/2. Dist sp. OK.
            
            neighbors_coords = [
                (pi+1, pj), (pi, pj+1), (pi-1, pj+1),
                (pi-1, pj), (pi, pj-1), (pi+1, pj-1)
            ]
            for nc in neighbors_coords:
                if nc in point_map:
                    adj[(pi, pj)].append(nc)
        
        # BFS to check connectivity and get shape signature
        def get_shape_signature(coords):
            # Normalize by centering and rotating to canonical form?
            # Simpler: sort coords and hash
            return tuple(sorted([(c[0], c[1]) for c in coords]))

        # Generate initial candidates
        candidates = []
        
        # Candidate 1: The "hexagon of 7" plus 4 extra.
        # Center (0,0). Neighbors (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
        # Add 4 more from the next ring.
        # Next ring coords: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
        # Pick 4 that are symmetric? e.g., (2,0), (0,2), (-2,1), (-1,-1) ?
        
        # Let's just run a quick optimization on the lattice points.
        # Pick 11 points from the 49 grid points.
        # We'll try a few heuristic selections.
        
        # Heuristic 1: Closest to origin
        # We already have points sorted by dist.
        # But taking the 11 closest might not be connected if the grid is sparse? 
        # No, the grid is dense. The 11 closest to (0,0) are:
        # (0,0)
        # 6 neighbors
        # 4 from the next ring (closest ones)
        # The next ring has 12 points. The closest ones to origin are all at same distance.
        # So any 4 from the next ring will work.
        # Let's pick 4 that form a symmetric pattern.
        
        # Let's generate 50 random connected sets of size 11 from the grid.
        for _ in range(100):
            # Start with a random point
            start = rng.choice(points)
            current_set = {start[0]: start}
            current_keys = [start[0]]
            
            # Grow by adding random neighbors
            while len(current_set) < N:
                # Pick a random existing point
                base_key = rng.choice(list(current_set.keys()))
                neighbors = adj.get(base_key, [])
                if not neighbors:
                    break
                next_key = rng.choice(neighbors)
                if next_key not in current_set:
                    current_set[next_key] = point_map[next_key]
                    current_keys.append(next_key)
            
            if len(current_set) == N:
                # Check if it's a "good" shape (compact)
                # Calculate moment of inertia or bounding box
                xs = [p[2] for p in current_set.values()]
                ys = [p[3] for p in current_set.values()]
                cx = sum(xs)/N
                cy = sum(ys)/N
                inertia = sum((x-cx)**2 + (y-cy)**2 for x,y in zip(xs, ys))
                
                sig = get_shape_signature([(p[0], p[1]) for p in current_set.values()])
                if sig not in seen_shapes:
                    seen_shapes.add(sig)
                    # Store the actual coordinates centered at (0,0)
                    coords = [(p[2]-cx, p[3]-cy) for p in current_set.values()]
                    candidates.append((inertia, coords))
        
        # Sort by inertia (more compact is generally better for fitting in hexagon)
        candidates.sort(key=lambda x: x[0])
        
        # Return top 20 unique shapes
        top_shapes = []
        seen = set()
        for inertia, coords in candidates:
            sig = get_shape_signature([(round(c[0], 5), round(c[1], 5)) for c in coords])
            if sig not in seen:
                seen.add(sig)
                top_shapes.append(coords)
            if len(top_shapes) >= 20:
                break
                
        return top_shapes

    shapes = generate_topologies()
    
    if not shapes:
        # Fallback: linear chain
        sp = math.sqrt(3)
        coords = [(i * sp - 5 * sp, 0.0) for i in range(N)]
        shapes = [coords]

    # For each shape, optimize global rotation and outer angle
    for coords in shapes:
        if time.time() > deadline:
            break
            
        # Normalize shape: center of mass is already 0.
        # We need to try rotating this entire rigid body.
        # Since the shape is on a lattice, rotating it by arbitrary angles might break the "lattice" property 
        # but that's fine, the hexagons just move.
        # However, the relative distances are fixed.
        
        # We iterate over global rotation angles.
        # Due to 6-fold symmetry of the hexagons, rotating the whole cluster by 60 deg 
        # results in an equivalent configuration relative to the outer hexagon?
        # Not exactly, because the outer hexagon is fixed.
        # But rotating the cluster by 60 deg is the same as rotating the outer hexagon by -60 deg.
        # So we can fix cluster rotation to [0, 60) and vary outer angle [0, 360).
        # Or fix outer angle to 0 and vary cluster rotation [0, 360).
        
        # Let's fix outer angle to 0 (aligned with axes) and vary cluster rotation.
        # This is efficient.
        
        oangle = 0.0
        ocx, ocy = 0.0, 0.0
        
        # Cluster rotation angles
        # Step size 1 degree is probably enough, or use golden ratio search.
        # Let's do a coarse search then fine search.
        
        best_local_S = float('inf')
        best_local_rot = 0.0
        
        # Coarse search
        for deg in range(0, 360, 2):
            rad = math.radians(deg)
            ca, sa = math.cos(rad), math.sin(rad)
            
            rotated_coords = []
            for x, y in coords:
                rx = x * ca - y * sa
                ry = x * sa + y * ca
                rotated_coords.append((rx, ry))
                
            inner = [[x, y, 0.0] for x, y in rotated_coords]
            
            S = compute_outer_hex(inner, ocx, ocy, oangle)
            
            # Check validity (disjointness is guaranteed by lattice construction if sp >= sqrt(3))
            # But we used sp = sqrt(3) exactly. 
            # If the evaluator is strict about "disjoint" (no touching), we might need sp > sqrt(3).
            # Usually "disjoint" means interiors are disjoint. Touching is allowed.
            # If not, we can scale coords by 1.001.
            
            if S < best_local_S:
                best_local_S = S
                best_local_rot = rad
                best_local_inner = inner

        # Fine search around best_local_rot
        if best_local_S < float('inf'):
            for delta_deg in np.linspace(-2, 2, 100):
                rad = best_local_rot + math.radians(delta_deg)
                ca, sa = math.cos(rad), math.sin(rad)
                
                rotated_coords = []
                for x, y in coords:
                    rx = x * ca - y * sa
                    ry = x * sa + y * ca
                    rotated_coords.append((rx, ry))
                    
                inner = [[x, y, 0.0] for x, y in rotated_coords]
                
                S = compute_outer_hex(inner, ocx, ocy, oangle)
                
                if S < best_local_S:
                    best_local_S = S
                    best_local_rot = rad
                    best_local_inner = inner
            
            # Verify validity of the best found
            if check_fit(best_local_inner, ocx, ocy, best_local_S, oangle):
                if best_local_S < best_S:
                    best_S = best_local_S
                    best_inner = best_local_inner
                    best_center = (ocx, ocy)
                    best_angle = oangle
                    best_valid = True

    # If no valid solution found, use a simple fallback
    if not best_valid:
        sp = math.sqrt(3) * 1.01 # Slightly expanded to ensure strict disjointness if needed
        coords = [(i * sp, 0.0) for i in range(N)]
        inner = [[x, y, 0.0] for x, y in coords]
        S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
        # Ensure it fits by scaling if necessary (though linear should fit)
        # Just return it
        best_inner = inner
        best_center = (0.0, 0.0)
        best_S = S
        best_angle = 0.0
        best_valid = True

    return best_inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
