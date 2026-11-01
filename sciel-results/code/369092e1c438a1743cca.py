"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Hexagonal lattice basis for unit hexagons (side length 1)
    # Adjacent hexagon centers are separated by sqrt(3)
    sqrt3 = np.sqrt(3)
    
    # Lattice vectors
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3 / 2, 1.5])
    
    # Generate all possible positions in a bounded region
    # Use axial coordinates (q, r) where position = q*v1 + r*v2
    # Search in a reasonable range
    max_coord = 4
    
    # Generate candidate sets of 11 hexagons
    # We'll use a greedy approach: start with compact clusters
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try different cluster shapes
    # Shape 1: 3-4-3-1 (rows of 3,4,3,1 hexagons)
    # Shape 2: 2-3-3-3 (rows of 2,3,3,3)
    # Shape 3: Compact blob
    
    def hexagon_vertices(cx, cy, side, rot_deg=0.0):
        """Get vertices of a regular hexagon."""
        rot = np.radians(rot_deg)
        angles = np.linspace(0, 2*np.pi, 7) + rot
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])
    
    def min_enclosing_hexagon_side(points):
        """Find minimum side length of a regular hexagon enclosing all points.
        Points are the centers of unit hexagons. We need the enclosing hexagon 
        to also contain the full extent of each unit hexagon.
        
        For a unit hexagon centered at p, its extent in direction theta is 
        0.5 * (width in that direction). For a regular hexagon of side s,
        the apothem (inradius) is s * sqrt(3)/2.
        
        So we need: for all points p_i and all 6 directions of the outer hexagon,
        the projection of p_i onto that direction + (unit hexagon extent in that dir) 
        <= outer hexagon apothem + center projection.
        
        Simplified: the outer hexagon of side S has inradius r = S * sqrt(3)/2.
        A unit hexagon has inradius r_unit = sqrt(3)/2.
        
        If both are aligned, we need: max over all unit hex centers of |projection| 
        + r_unit <= r_outer + |center_projection|
        
        Actually, let's think differently. The outer hexagon must contain each unit hexagon.
        The minimum enclosing hexagon for a set of points (centers) with each point 
        having a "radius" (inradius of unit hex) in all directions.
        
        For aligned hexagons, the constraint in each of the 6 normal directions n_k is:
        max_i (n_k · p_i) + r_unit <= n_k · c + r_outer
        min_i (n_k · p_i) + r_unit >= n_k · c - r_outer
        
        where c is the center of the outer hexagon, r_outer = S * sqrt(3)/2, r_unit = sqrt(3)/2.
        
        This gives: S >= (max_i(n_k·p_i) - min_i(n_k·p_i) + 2*r_unit) / sqrt(3)
        for each direction k, and we take the max over k.
        
        The center c can be chosen optimally to minimize S.
        """
        points = np.array(points)
        r_unit = sqrt3 / 2  # inradius of unit hexagon
        
        # 6 normal directions for a regular hexagon (at 0, 60, 120, 180, 240, 300 degrees)
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        
        min_S_for_angle = 0.0
        best_center = np.array([0.0, 0.0])
        
        for rot in np.linspace(0, np.pi/3, 13, endpoint=False):  # Try different rotations
            normals = []
            for a in angles:
                normals.append(np.array([np.cos(a + rot), np.sin(a + rot)]))
            normals = np.array(normals)
            
            # For each normal direction, compute max and min projections
            proj = points @ normals.T  # shape (N, 6)
            
            max_proj = proj.max(axis=0)
            min_proj = proj.min(axis=0)
            
            # Width in each direction
            widths = max_proj - min_proj
            
            # Required outer inradius: r_outer >= max(widths)/2 + r_unit
            # S = r_outer / (sqrt(3)/2) = 2*r_outer/sqrt(3)
            r_outer_required = widths.max() / 2 + r_unit
            S_candidate = 2 * r_outer_required / sqrt3
            
            if S_candidate < min_S_for_angle:
                min_S_for_angle = S_candidate
                # Optimal center is midpoint of the most constraining direction
                # For simplicity, use centroid
                best_center = points.mean(axis=0)
        
        return min_S_for_angle, best_center
    
    # Try various cluster configurations
    # Use axial coordinates to generate hexagon centers
    
    def axial_to_xy(q, r):
        return q * v1 + r * v2
    
    # Generate candidate clusters
    clusters = []
    
    # Cluster 1: Row-based
    # Row 0: 1 hex at (0,0)
    # Row 1: 3 hexes
    # Row 2: 4 hexes  
    # Row 3: 3 hexes
    # Total: 11
    
    # Let's try a few specific configurations
    
    configs = []
    
    # Config A: 3-4-3-1 vertical stack
    # Using axial coords, let's place them
    # This is getting complex. Let me try a simpler approach:
    # Generate random clusters and optimize
    
    # Actually, for 11 hexagons, let's try known good packings
    # A compact 11-hexagon cluster
    
    # Let me try: a hexagon of radius 2 has 7 hexagons (1+6)
    # Add 4 more to make 11
    
    # Base: center + 6 neighbors
    base = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Add 4 more: extend one direction
    # Option 1: add (2,0), (1,1), (0,2), (-1,2)
    config1 = base + [(2,0), (1,1), (0,2), (-1,2)]
    configs.append(config1)
    
    # Option 2: add (2,0), (2,-1), (1,-2), (0,-2)
    config2 = base + [(2,0), (2,-1), (1,-2), (0,-2)]
    configs.append(config2)
    
    # Option 3: more symmetric - add 4 in a ring
    config3 = base + [(2,0), (0,2), (-2,2), (-2,0)]
    configs.append(config3)
    
    # Option 4: 2-3-3-3 arrangement
    # Let me try different shapes
    
    # For each config, compute the enclosing hexagon
    for config in configs:
        points = [axial_to_xy(q, r) for q, r in config]
        S, center = min_enclosing_hexagon_side(points)
        
        if S < best_side:
            best_side = S
            best_center = (center[0], center[1])
            best_inner = []
            for q, r in config:
                xy = axial_to_xy(q, r)
                best_inner.append([xy[0], xy[1], 0.0])
            best_angle = 0.0
    
    # Also try with rotated outer hexagon
    # The min_enclosing_hexagon_side already tries rotations
    
    # Return the best found
    if best_inner is None:
        # Fallback: linear arrangement
        best_inner = [[sqrt3 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        best_side = sqrt3 * (N - 1) + sqrt3  # rough estimate
        best_center = (0.0, 0.0)
        best_angle = 0.0
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
