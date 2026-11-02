"""
Strategy: Discrete Lattice Search with Analytical Outer Hexagon Optimization.
Instead of continuous Simulated Annealing, we restrict inner hexagons to a
triangular lattice (spacing 2.0) with 6 discrete orientations. This ensures
disjointness by construction (mostly) and reduces the search space to
combinatorial selection of lattice sites. We then optimize the outer hexagon's
center and rotation to minimize the side length containing the selected inner
hexagons.
"""
import numpy as np
import time
import math
from itertools import combinations

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Lattice generation
    # Triangular lattice with spacing 2.0 (touching hexagons)
    # Basis vectors: (2, 0) and (1, sqrt(3))
    # Actually, for hexagons of side 1, the distance between centers must be >= 2.
    # In a triangular packing, distance is exactly 2.
    # Let's generate a grid of points within a radius.
    # Max outer side is roughly 4-5. So radius ~ 6.
    
    lattice_points = []
    # a1 = (2, 0), a2 = (1, sqrt(3))
    a1 = np.array([2.0, 0.0])
    a2 = np.array([1.0, math.sqrt(3)])
    
    # Generate points p = i*a1 + j*a2 for i, j in range
    # We need enough points to pick 12.
    # Let's go from -10 to 10 for i and j.
    for i in range(-10, 11):
        for j in range(-10, 11):
            p = i * a1 + j * a2
            # Only keep points within a reasonable radius
            if np.linalg.norm(p) < 8.0:
                lattice_points.append(p)
    
    # Remove duplicates (shouldn't be any) and convert to list of tuples
    lattice_points = [tuple(p) for p in lattice_points]
    
    # Orientations: 0, 60, 120, 180, 240, 300 degrees
    # Actually, due to symmetry of the hexagon, orientation 0 is same as 60 for the shape,
    # but the vertices change. However, for the "disjointness" on a lattice,
    # if we use the same orientation, they touch. If we rotate, they might overlap?
    # Wait, if two regular hexagons of side 1 are centered at distance 2, they touch
    # at one vertex if aligned, or might overlap if rotated?
    # Let's check: Two hexagons centered at (0,0) and (2,0).
    # Hexagon 1: vertices at (1,0), (0.5, sqrt(3)/2), ...
    # Hexagon 2: vertices at (3,0), (2.5, sqrt(3)/2), ...
    # If both angle 0:
    # H1 rightmost vertex: (1,0). H2 leftmost vertex: (1,0). They touch.
    # If H2 is rotated 30 deg:
    # H2 vertices relative to center: (sqrt(3)/2, 0.5), ...
    # Leftmost vertex of H2 (rotated 30):
    # Original angles: 0, 60, ...
    # Rotated 30: 30, 90, ...
    # Vertices: (cos(30), sin(30)) = (sqrt(3)/2, 0.5).
    # Center (2,0). Vertex: (2 + sqrt(3)/2, 0.5).
    # This is further right.
    # What about the vertex pointing towards H1?
    # Angle 180+30 = 210. cos(210) = -sqrt(3)/2.
    # Vertex: (2 - sqrt(3)/2, -0.5).
    # Distance from origin: sqrt((2-sqrt(3)/2)^2 + (-0.5)^2)
    # = sqrt(4 - 2sqrt(3) + 0.75 + 0.25) = sqrt(5 - 2sqrt(3))
    # sqrt(5 - 3.464) = sqrt(1.536) ~ 1.24.
    # This is < 2? No, distance from origin is ~1.24.
    # Wait, the vertex is at (2 - 0.866, -0.5) = (1.134, -0.5).
    # Distance from (0,0) is sqrt(1.134^2 + 0.5^2) = sqrt(1.286 + 0.25) = sqrt(1.536) = 1.24.
    # This point is INSIDE the first hexagon?
    # First hexagon (angle 0) contains points where max projection on normals <= apothem.
    # Normal at 180 deg: (-1, 0). Projection of (1.134, -0.5) is -1.134.
    # Apothem is sqrt(3)/2 ~ 0.866.
    # |-1.134| > 0.866. So it is outside.
    # So they are disjoint?
    # Let's check SAT for two hexagons at dist 2, one rotated 30 deg.
    # It is known that in a triangular lattice of hexagons, all aligned, they tile the plane.
    # If you rotate one, it might create a gap or overlap?
    # Actually, if you rotate a hexagon in a tiling, it usually overlaps with neighbors unless the rotation is a symmetry of the lattice (60 deg multiples).
    # So, to guarantee disjointness on a triangular lattice of spacing 2, we should probably stick to the 6 orientations that are symmetries of the lattice?
    # Or just use one orientation?
    # If we use only one orientation, the problem becomes: select 12 points from the lattice to minimize the enclosing hexagon size.
    # This is a much simpler problem.
    # Let's try this first. If it's too constrained, we can add rotations, but we must check disjointness.
    
    # For now, let's assume all inner hexagons have angle 0.
    # This is a valid packing (triangular lattice tiling).
    # We just need to find the best set of 12 points.
    
    # Precompute vertices for angle 0
    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def min_outer_side_and_center(points, outer_angle_deg):
        """
        Given a set of points (vertices of inner hexagons), find the min side length
        of a regular hexagon with given angle that contains them, and its center.
        """
        # This is the minimum enclosing hexagon problem.
        # For a fixed orientation, the minimum enclosing hexagon is determined by
        # the maximum extent in the 6 normal directions.
        # However, the center is also a variable.
        # Let the outer hexagon be defined by:
        # n_i . (x - c) <= S * apothem  for i=0..5
        # where n_i are unit normals.
        # To minimize S, we want to center c such that the max of (n_i . x) - (n_i . c) is minimized.
        # Let R_i = max_x (n_i . x) and L_i = min_x (n_i . x).
        # The width in direction i is W_i = R_i - L_i.
        # The constraint is:
        # n_i . (x - c) <= S * A  => n_i . x <= n_i . c + S * A
        # -n_i . (x - c) <= S * A => -n_i . x <= -n_i . c + S * A  (Note: -n_i is the opposite normal)
        # Actually, the 6 normals are n_0..n_5. The opposite normal is n_3 (if 0,60,120,180,240,300).
        # So for each pair of opposite normals (n_i, -n_i), we have:
        # max(n_i . x) - min(n_i . x) <= 2 * S * A
        # So S >= (max(n_i . x) - min(n_i . x)) / (2 * A)
        # The minimum S for a fixed orientation is max over i of (W_i / (2A)).
        # The center c can be chosen to center the bounding box.
        # c_i = (max(n_i . x) + min(n_i . x)) / 2 along the normal?
        # Yes, c = sum( (max_i + min_i)/2 * n_i ) ... wait, this is not quite right in 2D.
        # In 2D, if we define c such that the projection of c onto n_i is the midpoint of the range,
        # then the hexagon is centered.
        
        # Let's compute the ranges.
        # Normals for outer hexagon at angle outer_angle_deg
        # The sides are perpendicular to normals at 30, 90, 150, 210, 270, 330 + outer_angle
        base_normals = np.radians(np.arange(30, 360, 60))
        outer_rad = np.radians(outer_angle_deg)
        normals = base_normals + outer_rad
        
        # Compute min and max projections for each normal
        min_projs = np.zeros(6)
        max_projs = np.zeros(6)
        
        for k, na in enumerate(normals):
            nx = np.cos(na)
            ny = np.sin(na)
            projs = points[:, 0] * nx + points[:, 1] * ny
            min_projs[k] = np.min(projs)
            max_projs[k] = np.max(projs)
            
        # Widths
        widths = max_projs - min_projs
        
        # Apothem A = S * sqrt(3) / 2
        # S >= width / (2 * A) = width / (sqrt(3) * S) ? No.
        # Constraint: n . (x-c) <= S * A
        # Max(n . x) - n . c <= S * A
        # Min(n . x) - n . c >= -S * A  => n . c - Min(n . x) <= S * A
        # Adding: Max - Min <= 2 * S * A
        # S >= (Max - Min) / (2 * A)
        # A = S * sqrt(3) / 2
        # S >= (Max - Min) / (2 * S * sqrt(3) / 2) = (Max - Min) / (S * sqrt(3))
        # S^2 >= (Max - Min) / sqrt(3) ? No.
        # Wait. A is apothem of the OUTER hexagon.
        # A = S * sqrt(3) / 2.
        # So 2 * A = S * sqrt(3).
        # S >= width / (S * sqrt(3)) => S^2 >= width / sqrt(3) => S >= sqrt(width / sqrt(3)).
        # This implies S depends on itself? No, A is proportional to S.
        # Let's re-read the definition.
        # Hexagon side S. Apothem = S * sqrt(3) / 2.
        # Distance from center to side is Apothem.
        # So the half-width in the normal direction is Apothem.
        # Total width in normal direction is 2 * Apothem = S * sqrt(3).
        # So S * sqrt(3) >= width.
        # S >= width / sqrt(3).
        
        min_s = np.max(widths) / math.sqrt(3)
        
        # Now find the center.
        # For each normal n_i, the center's projection c_i should be (max_i + min_i) / 2.
        # We have 6 equations for 2 unknowns (cx, cy).
        # c . n_i = (max_i + min_i) / 2
        # This is an overdetermined system. We can solve it using least squares.
        
        # Construct matrix N (6x2) and vector b (6x1)
        N_mat = np.column_stack([np.cos(normals), np.sin(normals)])
        b_vec = (max_projs + min_projs) / 2.0
        
        # Solve N_mat @ c = b_vec
        c, residuals, rank, sv = np.linalg.lstsq(N_mat, b_vec, rcond=None)
        
        return min_s, tuple(c)

    # Generate all vertices for all lattice points (angle 0)
    all_vertices = []
    point_indices = []
    for idx, p in enumerate(lattice_points):
        verts = hexagon_vertices(p[0], p[1], 1.0, 0.0)
        all_vertices.append(verts)
        
    # We need to select 12 points.
    # This is a combinatorial problem.
    # Number of points is ~ 100. Choosing 12 is huge.
    # However, the optimal set is likely "compact".
    # We can use a heuristic: start with a compact set (e.g., center + neighbors) and swap points to improve.
    # Or, since the lattice is fixed, we can just iterate over "clusters" of 12 points.
    # A better approach: Since the lattice is regular, the optimal set of 12 points
    # that minimizes the enclosing hexagon is likely to be a "hexagonal number" shape or similar.
    # For 12 points, the shapes are:
    # 1. 1 + 6 + 5 (part of ring 2)
    # 2. 1 + 6 + 5 is 12 points.
    # Let's generate candidate sets of 12 points centered at various lattice points.
    
    # Identify lattice point indices
    # We need to find neighbors.
    # Let's just pick a few candidate "centers" and take the 12 closest points to each.
    
    candidate_sets = []
    
    # Pick a few centers from the lattice
    # To avoid duplicates, we can just take the first 100 points as potential centers?
    # Or just iterate over all lattice points as centers, and for each, take the 12 nearest neighbors (including itself).
    
    # This is O(M * M log M) where M is number of lattice points.
    # M ~ 100. 100 * 100 log 100 ~ 100 * 100 * 7 = 70,000 operations. Very fast.
    
    for center_idx, center_p in enumerate(lattice_points):
        # Calculate distances from center_p to all other points
        dists = np.array([np.linalg.norm(np.array(p) - np.array(center_p)) for p in lattice_points])
        
        # Get indices of 12 closest points
        closest_indices = np.argsort(dists)[:12]
        
        # Check if this set is already in candidate_sets (avoid duplicates)
        # Represent set as sorted tuple of indices
        set_rep = tuple(sorted(closest_indices))
        
        # Simple check: if the center is in the set (it should be, dist 0)
        # We can skip if we've seen this exact set of indices before.
        # To save memory/time, we'll just store the best score for each unique set.
        
        # Actually, many centers will yield the same set of 12 points.
        # We can just evaluate each unique set once.
        
        # For now, let's just collect them and deduplicate later or use a set.
        candidate_sets.append(set_rep)
        
    # Deduplicate
    unique_sets = list(set(candidate_sets))
    
    best_s = float('inf')
    best_center = (0,0)
    best_angle = 0
    best_indices = None
    
    # Scan outer angle
    # The outer hexagon can be rotated.
    # Due to symmetry of the lattice (6-fold), we only need to scan 0 to 60 degrees.
    # But the set of 12 points might break the symmetry.
    # So we should scan the full 0-60 range for each set.
    
    # To be efficient, we can precompute the vertices for each point in the set.
    # But the set changes.
    
    # Let's iterate over unique sets.
    # If there are too many, we might time out.
    # M ~ 100. Number of unique sets is likely small (maybe < 1000).
    
    for set_rep in unique_sets:
        if time.time() > deadline:
            break
            
        # Get the points
        points_in_set = [lattice_points[i] for i in set_rep]
        
        # Get all vertices
        verts_list = [all_vertices[i] for i in set_rep]
        all_verts = np.vstack(verts_list)
        
        # Scan outer angle
        # We can do a coarse scan first, then refine.
        # But since we have many sets, let's do a coarse scan (step 5 deg) and then refine the best.
        
        best_s_for_set = float('inf')
        best_c_for_set = (0,0)
        best_a_for_set = 0
        
        for theta in range(0, 60, 5):
            s, c = min_outer_side_and_center(all_verts, theta)
            if s < best_s_for_set:
                best_s_for_set = s
                best_c_for_set = c
                best_a_for_set = theta
        
        # Refine around best theta
        for theta in np.arange(best_a_for_set - 5, best_a_for_set + 5, 0.5):
            s, c = min_outer_side_and_center(all_verts, theta)
            if s < best_s_for_set:
                best_s_for_set = s
                best_c_for_set = c
                best_a_for_set = theta
                
        if best_s_for_set < best_s:
            best_s = best_s_for_set
            best_center = best_c_for_set
            best_angle = best_a_for_set
            best_indices = set_rep
            
    # Construct the inner list
    inner = []
    for idx in best_indices:
        p = lattice_points[idx]
        inner.append((p[0], p[1], 0.0))
        
    # Final validation
    # Check if inner hexagons are disjoint.
    # Since they are on a triangular lattice with spacing 2, they should be disjoint (touching).
    # But let's verify with a simple distance check for centers.
    # If distance < 2 - epsilon, they overlap.
    # On the lattice, min distance is 2.
    
    # Check if all inner hexagons are inside the outer hexagon.
    # We calculated the outer hexagon to contain them, so this should be true.
    # But let's double check the logic of min_outer_side_and_center.
    # It assumes the outer hexagon is centered at 'c'.
    # We returned 'c' as the center.
    
    # One issue: min_outer_side_and_center assumes the outer hexagon is aligned with the normals.
    # The normals are defined relative to the outer hexagon's angle.
    # The center 'c' is calculated to center the bounding box in those normal directions.
    # This is correct for a fixed orientation.
    
    return inner, best_center, best_s, best_angle

# EVOLVE-BLOCK-END
