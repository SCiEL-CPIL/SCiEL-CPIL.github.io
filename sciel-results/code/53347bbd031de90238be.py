"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

_start = time.time()
DEADLINE = 150.0
SAFETY = 10.0

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute triple indices
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)

def _is_inside(pts):
    """Check if points are inside the triangle. Returns boolean mask."""
    # y >= 0
    mask = pts[:, 1] >= -1e-9
    # y <= sqrt(3) * x
    mask &= pts[:, 1] <= np.sqrt(3) * pts[:, 0] + 1e-9
    # y <= sqrt(3) * (1 - x)
    mask &= pts[:, 1] <= np.sqrt(3) * (1 - pts[:, 0]) + 1e-9
    return mask

def _compute_areas(pts):
    """Compute all triangle areas vectorized."""
    idx = _IDX_TRIPLES
    pi = pts[idx[:, 0]]
    pj = pts[idx[:, 1]]
    pk = pts[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)

def _objective_cart(x):
    """Objective function for Cartesian coordinates. Returns negative min area."""
    pts = x.reshape(N, 2)
    
    # Check if all points are inside
    if not np.all(_is_inside(pts)):
        # Penalty for being outside
        # Calculate how far outside
        # Simple penalty: return 1.0 (which is > any valid min area)
        return 1.0
    
    areas = _compute_areas(pts)
    min_area = areas.min()
    return -min_area

def construct():
    global _start
    rng = np.random.RandomState(42)
    
    best_val = 0.0
    best_pts = None
    
    # Initial guesses
    # 1. Center
    center = np.array([0.5, np.sqrt(3)/6])
    
    # 2. Random points inside
    def random_points_inside(rng, n):
        pts = []
        while len(pts) < n:
            # Sample in bounding box
            x = rng.uniform(0, 1, n)
            y_max = np.minimum(np.sqrt(3)*x, np.sqrt(3)*(1-x))
            y = rng.uniform(0, np.max(y_max), n)
            # Filter
            valid = y <= y_max - 1e-5
            if np.any(valid):
                pts.extend(zip(x[valid], y[valid]))
        
        pts = np.array(pts[:n])
        return pts

    starts = []
    # Grid-like start
    # Row 1: y = 0.1 * h
    # Row 2: y = 0.3 * h
    # Row 3: y = 0.5 * h
    h = np.sqrt(3)/2
    # Approximate positions
    starts.append(random_points_inside(rng, N))
    starts.append(random_points_inside(rng, N))
    
    # Symmetric start
    sym_pts = []
    # Center
    sym_pts.append([0.5, h/3])
    # Ring 1
    for i in range(5):
        angle = 2 * np.pi * i / 5
        r = 0.2
        x = 0.5 + r * np.cos(angle)
        y = h/3 + r * np.sin(angle) * (1.0) # Approx
        sym_pts.append([x, y])
    # Ring 2
    for i in range(5):
        angle = 2 * np.pi * i / 5 + np.pi/5
        r = 0.35
        x = 0.5 + r * np.cos(angle)
        y = h/3 + r * np.sin(angle)
        sym_pts.append([x, y])
    sym_pts = np.array(sym_pts)
    # Check if inside, if not, clamp or skip
    if np.all(_is_inside(sym_pts)):
        starts.append(sym_pts)

    # Optimize from each start
    for i, start_pts in enumerate(starts):
        if time.time() - _start > DEADLINE - SAFETY:
            break
            
        x0 = start_pts.flatten()
        
        try:
            # Use Nelder-Mead as it is robust and doesn't require gradients
            # Maximize min area => Minimize -min area
            result = minimize(_objective_cart, x0, method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            pts = result.x.reshape(N, 2)
            
            # Verify points are inside
            if not np.all(_is_inside(pts)):
                continue
                
            areas = _compute_areas(pts)
            val = areas.min()
            
            if val > best_val:
                best_val = val
                best_pts = pts.copy()
                
        except Exception:
            pass

    # If no valid points found, return a default valid configuration
    if best_pts is None:
        # Fallback: 11 points in a grid
        pts = []
        # 3 rows
        # y = 0.1*h, 0.3*h, 0.5*h, 0.7*h?
        # Let's just use random again
        for _ in range(10):
             cand = random_points_inside(rng, N)
             if np.all(_is_inside(cand)):
                  areas = _compute_areas(cand)
                  if areas.min() > best_val:
                      best_val = areas.min()
                      best_pts = cand
        if best_pts is None:
             best_pts = random_points_inside(rng, N)
             
    return best_pts
# EVOLVE-BLOCK-END
