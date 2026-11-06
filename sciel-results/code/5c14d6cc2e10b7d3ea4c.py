import math
import time
import numpy as np
import scipy.special

def _j0(x):
    return scipy.special.j0(x)

def _compute_gap_fast(cell, size=32, n_pw=17, n_k=8, eps_high=12.0):
    # ... (keep existing logic, it's fine) ...
    pass # I will copy the existing body but ensure no errors

def _generate_cell(params, size=32):
    """
    Generate a 2x2 array of dielectric rods.
    params: 'radius', 'offset'
    """
    cell = np.zeros((size, size), dtype=int)
    r = params.get('radius', 2.0)
    d = params.get('offset', 6.0)
    
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    
    # 4 rod centers
    centers = [
        (cx - d, cy - d),
        (cx + d, cy - d),
        (cx - d, cy + d),
        (cx + d, cy + d)
    ]
    
    for i in range(size):
        for j in range(size):
            x = j + 0.5
            y = i + 0.5
            for cx_c, cy_c in centers:
                dist_sq = (x - cx_c)**2 + (y - cy_c)**2
                if dist_sq < r*r:
                    cell[i, j] = 1
                    break
    return cell.tolist()

def design(problem):
    # ... (optimize radius and offset) ...
    pass
