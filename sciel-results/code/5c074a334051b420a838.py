"""
Strategy: Lattice-Point Enumeration with Triangular Lattice Constraints.
Place 12 unit hexagons on a triangular lattice (centers at lattice points).
The lattice spacing is chosen so that adjacent hexagons just touch (distance = 2).
We enumerate subsets of 12 lattice points that form a connected cluster,
then optimize the outer hexagon's rotation angle to minimize its side length.

Key insight: On a triangular lattice with spacing 2, two unit hexagons at adjacent
lattice points are tangent (disjoint). The support function for the cluster
depends only on the outer hexagon's orientation, which we optimize in 1D.
"""
import numpy as np
import time
import math
from itertools import combinations

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Unit hexagon vertices (flat-top, side length 1)
    unit_hex_verts = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))]
        for a in range(0, 360, 60)
    ])
    
    def unit_hex_support(nx, ny):
        """Support function of unit hexagon in direction (nx, ny)."""
        dots = unit_hex_verts[:, 0] * nx + unit_hex_verts[:, 1] * ny
        return np.max(dots)
    
    # Triangular lattice basis vectors with spacing 2
    # e1 = (2, 0), e2 = (1, sqrt(3))
    e1 = np.array([2.0, 0.0])
    e2 = np.array([1.0, math.sqrt(3)])
    
    # Generate lattice points in a region large enough to contain 12-point clusters
    # We need to explore various cluster shapes
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            pt = i * e1 + j * e2
            lattice_points.append(pt)
    
    lattice_points = np.array(lattice_points)
    n_lattice = len(lattice_points)
    
    # Precompute pairwise distances for collision checking
    # Two hexagons on the lattice are disjoint if their centers are at distance >= 2
    # (since each has "radius" 1 in the relevant directions for the lattice geometry)
    # Actually, for regular hexagons of side 1, the minimum distance between centers
    # for disjointness depends on relative orientation. For same orientation,
    # adjacent lattice points (distance 2) give tangent hexagons.
    
    # For simplicity, we assume all inner hexagons have the same orientation (0 degrees).
    # This is valid because the lattice is designed for this case.
    
    def compute_min_side_for_cluster(points, outer_angle_deg):
        """
        Compute minimal outer hexagon side length for a cluster of points
        (centers of unit hexagons all with orientation 0) and given outer orientation.
        """
        outer_angle_rad = math.radians(outer_angle_deg)
        normal_angles = outer_angle_rad + math.radians(30) + math.radians(60) * np.arange(6)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_required = 0.0
        for pt in points:
            for nx, ny in normals:
                val = pt[0] * nx + pt[1] * ny + unit_hex_support(nx, ny)
                max_required = max(max_required, val)
        
        apothem = max_required
        side = apothem / (math.sqrt(3) / 2)
        return side
    
    def is_connected(points, indices):
        """Check if the selected lattice points form a connected cluster."""
        selected = set(indices)
        if len(selected) == 0:
            return False
        if len(selected) == 1:
            return True
        
        # Build adjacency: two lattice points are adjacent if distance = 2
        # (they are neighbors in the triangular lattice)
        # For triangular lattice with basis e1, e2, neighbors of (i,j) are:
        # (i+1,j), (i-1,j), (i,j+1), (i,j-1), (i+1,j-1), (i-1,j+1)
        
        # We need to map indices back to (i,j) coordinates
        # Let's precompute this
        idx_to_ij = {}
        for idx, pt in enumerate(lattice_points):
            # Solve for i, j: pt = i*e1 + j*e2
            # [2 1; 0 sqrt(3)] [i; j] = [pt[0]; pt[1]]
            # j = pt[1] / sqrt(3)
            # i = (pt[0] - j) / 2
            j = pt[1] / math.sqrt(3)
            i = (pt[0] - j) / 2
            idx_to_ij[idx] = (round(i), round(j))
        
        # BFS from first selected point
        visited = set()
        queue = [list(selected)[0]]
        visited.add(queue[0])
        
        while queue:
            curr = queue.pop(0)
            ci, cj = idx_to_ij[curr]
            # Neighbors in (i,j) space
            for di, dj in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
                ni, nj = ci + di, cj + dj
                # Find index of this neighbor
                for idx, (pi, pj) in idx_to_ij.items():
                    if pi == ni and pj == nj:
                        if idx in selected and idx not in visited:
                            visited.add(idx)
                            queue.append(idx)
                        break
        
        return len(visited) == len(selected)
    
    # Enumerate all 12-point subsets from the lattice
    # This is too many if we have too many lattice points
    # Let's reduce the lattice size and be smarter
    
    # Use a smaller lattice: points within radius ~4 from origin
    lattice_points2 = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            pt = i * e1 + j * e2
            if np.linalg.norm(pt) < 5.0:
                lattice_points2.append(pt)
    
    lattice_points2 = np.array(lattice_points2)
    n_lattice2 = len(lattice_points2)
    
    # Precompute idx_to_ij for the smaller lattice
    idx_to_ij2 = {}
    for idx, pt in enumerate(lattice_points2):
        j = pt[1] / math.sqrt(3)
        i = (pt[0] - j) / 2
        idx_to_ij2[idx] = (round(i), round(j))
    
    # Build adjacency list
    adj = {idx: set() for idx in range(n_lattice2)}
    for idx1 in range(n_lattice2):
        i1, j1 = idx_to_ij2[idx1]
        for di, dj in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
            ni, nj = i1 + di, j1 + dj
            for idx2 in range(n_lattice2):
                i2, j2 = idx_to_ij2[idx2]
                if i2 == ni and j2 == nj:
                    adj[idx1].add(idx2)
                    break
    
    def is_connected2(indices):
        """Check connectivity using precomputed adjacency."""
        selected = set(indices)
        if len(selected) <= 1:
            return True
        visited = set()
        queue = [list(selected)[0]]
        visited.add(queue[0])
        while queue:
            curr = queue.pop(0)
            for neighbor in adj[curr]:
                if neighbor in selected and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return len(visited) == len(selected)
    
    # Generate all 12-point subsets (this could be large)
    # n_lattice2 is about 25 points, C(25,12) = 5.2M which is too many
    # Let's use a heuristic: start from a seed cluster and grow it
    
    # Alternative: use known good cluster shapes
    # The hexagonal cluster of 19 points (radius 2) contains many 12-point subsets
    # Let's enumerate connected 12-point subsets using BFS-based generation
    
    # For now, let's try a simpler approach: fix the cluster shape to be
    # the 12-point subset of the 19-point hexagonal cluster that minimizes
    # the bounding hexagon
    
    # 19-point hexagonal cluster: all (i,j) with max(|i|,|j|,|i-j|) <= 2
    hex19_indices = []
    for idx in range(n_lattice2):
        i, j = idx_to_ij2[idx]
        if max(abs(i), abs(j), abs(i-j)) <= 2:
            hex19_indices.append(idx)
    
    # If we have fewer than 12, expand the lattice
    if len(hex19_indices) < 12:
        # Use all lattice points
        hex19_indices = list(range(n_lattice2))
    
    # Enumerate 12-point subsets from hex19_indices
    # C(len(hex19_indices), 12) - if hex19_indices has 19 points, C(19,12) = 50388
    # That's manageable
    
    best_side = float('inf')
    best_subset = None
    best_angle = 0.0
    
    # Precompute the support values for each lattice point in each normal direction
    # for a range of angles
    
    # First, find all connected 12-point subsets
    all_subsets = []
    if len(hex19_indices) <= 25:
        for subset in combinations(hex19_indices, 12):
            if time.time() > deadline - 10:
                break
            if is_connected2(subset):
                all_subsets.append(subset)
        if time.time() > deadline - 10:
            break
    
    # If we didn't find any, use the first 12 connected points
    if not all_subsets:
        # Greedy: start from origin, add nearest neighbors
        origin_idx = None
        for idx in range(n_lattice2):
            if idx_to_ij2[idx] == (0, 0):
                origin_idx = idx
                break
        if origin_idx is None:
            origin_idx = 0
        
        selected = [origin_idx]
        while len(selected) < 12:
            # Add the nearest unselected neighbor
            best_next = None
            best_dist = float('inf')
            for idx in selected:
                for neighbor in adj[idx]:
                    if neighbor not in selected:
                        # Compute distance from centroid of selected
                        centroid = np.mean([lattice_points2[i] for i in selected], axis=0)
                        dist = np.linalg.norm(lattice_points2[neighbor] - centroid)
                        if dist < best_dist:
                            best_dist = dist
                            best_next = neighbor
            if best_next is None:
                break
            selected.append(best_next)
        all_subsets.append(tuple(selected))
    
    # For each subset, optimize the outer angle
    # The side length as a function of angle is periodic with period 60 degrees
    # We can sample the angle finely
    
    n_subsets = len(all_subsets)
    for si, subset in enumerate(all_subsets):
        if time.time() > deadline - 5:
            break
        
        points = lattice_points2[list(subset)]
        
        # Sample angles from 0 to 60 degrees
        best_local_side = float('inf')
        best_local_angle = 0.0
        
        for angle_deg in np.linspace(0, 60, 121):
            side = compute_min_side_for_cluster(points, angle_deg)
            if side < best_local_side:
                best_local_side = side
                best_local_angle = angle_deg
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_subset = subset
            best_angle = best_local_angle
    
    # If no valid subset found, use a default configuration
    if best_subset is None:
        # Default: 12 hexagons in a 3x4 rectangular-ish arrangement
        default_indices = []
        for i in range(-1, 2):
            for j in range(-2, 2):
                for idx in range(n_lattice2):
                    if idx_to_ij2[idx] == (i, j):
                        default_indices.append(idx)
                        break
                if len(default_indices) >= 12:
                    break
            if len(default_indices) >= 12:
                break
        if len(default_indices) < 12:
            default_indices = list(range(min(12, n_lattice2)))
        best_subset = tuple(default_indices)
        points = lattice_points2[list(best_subset)]
        best_side = compute_min_side_for_cluster(points, 0.0)
        best_angle = 0.0
    
    # Refine the angle around the best found
    points = lattice_points2[list(best_subset)]
    best_side = float('inf')
    best_angle_final = best_angle
    for angle_deg in np.linspace(best_angle - 5, best_angle + 5, 201):
        side = compute_min_side_for_cluster(points, angle_deg)
        if side < best_side:
            best_side = side
            best_angle_final = angle_deg
    
    # Construct the inner hexagons
    inner = []
    for idx in best_subset:
        x, y = lattice_points2[idx]
        inner.append((float(x), float(y), 0.0))
    
    oc = (0.0, 0.0)
    os = best_side
    oa = best_angle_final % 60
    
    return inner, oc, os, oa
