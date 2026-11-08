import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_verts(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _sat_overlap_depth(A, B):
    """
    Calculate the minimum translation distance (depth) to separate two convex polygons A and B.
    Returns 0 if disjoint.
    A, B are arrays of shape (6, 2).
    """
    min_depth = float('inf')
    axes = []
    
    # Get normals of edges of A and B
    for poly in (A, B):
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            edge = p2 - p1
            length = np.linalg.norm(edge)
            if length < 1e-9:
                continue
            normal = np.array([-edge[1], edge[0]]) / length
            axes.append(normal)
            
    for ax in axes:
        # Project A
        proj_a = np.dot(A, ax)
        min_a, max_a = np.min(proj_a), np.max(proj_a)
        # Project B
        proj_b = np.dot(B, ax)
        min_b, max_b = np.min(proj_b), np.max(proj_b)
        
        # Check for separation
        if max_a <= min_b or max_b <= min_a:
            return 0.0
        
        # Calculate overlap depth
        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap < min_depth:
            min_depth = overlap
            
    return min_depth

def _containment_penalty(inner_verts, outer_verts):
    """
    Calculate the sum of distances of inner vertices outside the outer hexagon.
    """
    penalty = 0.0
    n_out = len(outer_verts)
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            el = np.linalg.norm(edge)
            if el < 1e-9:
                continue
            # Normal pointing inward (assuming CCW outer)
            # For CCW, interior is to the left of edge.
            nrm = np.array([-edge[1], edge[0]]) / el
            # Signed distance
            sd = np.dot(iv - v1, nrm)
            if sd < 0:
                penalty += -sd
    return penalty

def _objective(params, outer_angle):
    """
    params: [x1, y1, a1, ..., x12, y12, a12, S]
    """
    S = params[-1]
    if S < 1.0: # Safety
        S = 1.0
    
    inner_verts_list = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner_verts_list.append(_hex_verts(x, y, 1.0, a))
    
    outer_verts = _hex_verts(0.0, 0.0, S, outer_angle)
    
    total_penalty = 0.0
    
    # Containment penalty
    for iv in inner_verts_list:
        total_penalty += _containment_penalty(iv, outer_verts)
        
    # Overlap penalty
    for i in range(N):
        for j in range(i+1, N):
            d = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if d > 0:
                total_penalty += d * 10.0 # Heavier weight for overlap
    
    # Objective: Minimize S, penalize violations
    return S + total_penalty * 100.0

def _make_grid_init(nr, nc):
    """Create initial positions for a grid."""
    pos = []
    for r in range(nr):
        for c in range(nc):
            x = c * 2.0 + (1.0 if r % 2 else 0.0)
            y = r * np.sqrt(3.0)
            pos.append([x, y])
    arr = np.array(pos)
    # Center the grid
    arr -= arr.mean(axis=0)
    return arr

def _cluster_init():
    """1 center, 6 ring 1, 5 ring 2 (spread out)."""
    pos = [[0.0, 0.0]]
    # Ring 1
    for i in range(6):
        a = np.radians(60 * i)
        pos.append([2.0 * np.cos(a), 2.0 * np.sin(a)])
    # Ring 2: Pick 5 spread out indices: 0, 2, 4, 6, 8
    indices = [0, 2, 4, 6, 8]
    for idx in indices:
        a = np.radians(60 * idx)
        pos.append([4.0 * np.cos(a), 4.0 * np.sin(a)])
    arr = np.array(pos)
    arr -= arr.mean(axis=0)
    return arr

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0
    
    best_S = float('inf')
    best_params = None
    
    # Initializations
    inits = []
    inits.append(_make_grid_init(3, 4))
    inits.append(_make_grid_init(4, 3))
    inits.append(_cluster_init())
    
    # Random inits
    rng = np.random.default_rng(42)
    for _ in range(5):
        r = rng.uniform(-2, 2, size=(N, 2))
        inits.append(r)
        
    outer_angles = [0.0, 30.0]
    
    for oa in outer_angles:
        if time.time() > DEADLINE - 10:
            break
        for init_pos in inits:
            if time.time() > DEADLINE - 5:
                break
            
            # Initialize params: x, y, angle for each hex, then S
            init_params = np.zeros(3 * N + 1)
            for i in range(N):
                init_params[3*i] = init_pos[i, 0]
                init_params[3*i+1] = init_pos[i, 1]
                init_params[3*i+2] = 0.0 # angle
            init_params[-1] = 3.5 # Start with S=3.5
            
            def obj(p, oa=oa):
                return _objective(p, oa)
                
            try:
                res = minimize(obj, init_params, method='L-BFGS-B',
                               options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res.fun < 1e-4: # Check if penalty is effectively zero
                S_opt = res.x[-1]
                if S_opt < best_S:
                    best_S = S_opt
                    best_params = res.x
                    
    if best_params is None:
        # Fallback: return a valid grid with S=3.5
        grid = _make_grid_init(3, 4)
        inner = [[float(grid[i,0]), float(grid[i,1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 3.5, 0.0
        
    S = best_params[-1]
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
        
    return inner, (0.0, 0.0), float(S), 0.0

# EVOLVE-BLOCK-END
