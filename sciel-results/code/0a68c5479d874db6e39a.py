import numpy as np

N = 14

def construct():
    # For points constrained to a circle, the regular polygon maximizes the
    # minimum triangle area relative to the hull area.
    # This is because the regular polygon equalizes the areas of all consecutive
    # triangles, which constitute the minimums. Any deviation from uniform
    # angular spacing reduces the area of the smallest triangle.
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
