# EVOLVE-BLOCK-START
import numpy as np


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    
    Strategy:
    Use a Superellipse (Lamé curve) with exponent p=4.
    Sample 20 points at equidistant angles.
    
    Rationale:
    A regular polygon (circle) has symmetric distance gaps, leading to k=0 
    (smallest distances) being selected, which has poor relative uniformity.
    A superellipse has varying curvature. The distance distribution from a vertex 
    may have its minimum gap at a different index, or the distances around that 
    gap may be more uniform relative to each other.
    Specifically, the 'boxy' shape of p=4 superellipse creates vertices with 
    different local geometric properties, potentially shifting the 'gap' index 
    to a region where the 4 surrounding distances are more similar in magnitude 
    relative to their mean D.
    """
    n = 20
    p = 4.0  # Exponent for superellipse
    
    # Angles
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Superellipse coordinates
    # x = cos(theta) / |cos(theta)|^(2/p)
    # y = sin(theta) / |sin(theta)|^(2/p)
    # Handle zeros: if cos(theta) == 0, x=0. Formula: sign(c) * |c|^(1 - 2/p)
    # 1 - 2/p = 1 - 0.5 = 0.5 for p=4.
    # So x = sign(cos) * |cos|^0.5
    
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    
    # Using the power form to avoid division by zero issues directly
    # |c|^(1 - 2/p)
    exp = 1.0 - 2.0 / p
    
    x = np.sign(cos_a) * np.abs(cos_a) ** exp
    y = np.sign(sin_a) * np.abs(sin_a) ** exp
    
    polygon = np.stack([x, y], axis=1)
    
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(polygon, axis=0)
    polygon = polygon - centroid
    
    distances_to_centroid = np.linalg.norm(polygon, axis=1)
    mean_dist = np.mean(distances_to_centroid)
    
    if mean_dist <= 0.01:
        # Fallback to regular polygon if normalization fails
        angles_reg = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        polygon = np.stack([np.cos(angles_reg), np.sin(angles_reg)], axis=1)
    else:
        polygon = polygon / mean_dist
        
    return polygon

# EVOLVE-BLOCK-END
