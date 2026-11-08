import numpy as np

N = 13

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs 13 points to maximize the score.
    Based on the analysis that the score is inversely proportional to the 
    ratio of minimum triangle area to convex hull area, we aim to minimize 
    this ratio by creating a large hull and a cluster of points with very 
    small triangle areas.
    """
    # Define vertices of a large equilateral triangle
    # Circumradius 1
    v1 = np.array([1.0, 0.0])
    v2 = np.array([-0.5, np.sqrt(3)/2])
    v3 = np.array([-0.5, -np.sqrt(3)/2])
    
    points = [v1, v2, v3]
    
    # Small epsilon for the cluster
    eps = 1e-5
    
    # Place the remaining 10 points very close to v1
    # We arrange them in a small 2D pattern to ensure non-zero area
    # but very small.
    # A small grid or spiral works.
    # Let's use a small circle of radius eps around v1?
    # No, points on a circle have min area related to chord length.
    # Better to place them in a tight cluster.
    
    # Pattern:
    # (0,0)
    # (eps, 0)
    # (0, eps)
    # (eps, eps)
    # ...
    
    for i in range(N - 3):
        # Deterministic small offsets
        # Use i to vary the position slightly
        # To minimize the MINIMUM area, we want some points very close.
        # The closest points will define the min area.
        # Let's just place them at v1 + small vectors.
        
        # To be safe against "collinear" penalties, use 2D offsets.
        # i=0: (eps, 0)
        # i=1: (0, eps)
        # i=2: (eps, eps)
        # i=3: (2*eps, 0)
        # etc.
        
        # Simple approach: place on a small line? No, area 0.
        # Place on a small triangle?
        
        # Let's place them at:
        # v1 + eps * (dx, dy)
        # where (dx, dy) are small distinct integers.
        
        dx = (i % 3) * 1.0
        dy = (i // 3) * 1.0
        
        # This creates a grid.
        # The minimum distance is eps.
        # The minimum triangle area will be ~ eps^2.
        
        offset = eps * np.array([dx, dy])
        points.append(v1 + offset)
        
    return np.array(points)
# EVOLVE-BLOCK-END
