import numpy as np
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    return np.array([(cx + side*np.cos(angle + k*np.pi/3), cy + side*np.sin(angle + k*np.pi/3)) for k in range(6)])

def polygons_disjoint(p1, p2):
    for poly in [p1, p2]:
        for i in range(6):
            a, b = poly[i], poly[(i+1)%6]
            ex, ey = b[0]-a[0], b[1]-a[1]
            nx, ny = -ey, ex
            norm = np.sqrt(nx*nx + ny*ny)
            if norm < 1e-12:
                continue
            nx, ny = nx/norm, ny/norm
            proj1 = np.array([pt[0]*nx + pt[1]*ny for pt in p1])
            proj2 = np.array([pt[0]*nx + pt[1]*ny for pt in p2])
            if proj1.max() < proj2.min() - 1e-9 or proj2.max() < proj1.min() - 1e-9:
                return True
    return False

def min_S_for_points(points, center=None):
    """Minimize S over center for fixed orientation (flat-top)."""
    points = np.array(points)
    normals = [np.array([np.cos(np.radians(30)), np.sin(np.radians(30))]),
               np.array([0.0, 1.0]),
               np.array([np.cos(np.radians(150)), np.sin(np.radians(150))])]
    
    if center is None:
        cx0, cy0 = points.min(axis=0)
        cx1, cy1 = points.max(axis=0)
        span = max(cx1-cx0, cy1-cy0)
        step = max(0.2, span*0.1)
        best_S = float('inf')
        best_c = None
        nx = int((cx1-cx0)/step)+1
        ny = int((cy1-cy0)/step)+1
        for i in range(nx+1):
            for j in range(ny+1):
                cx, cy = cx0+i*step, cy0+j*step
                c = np.array([cx, cy])
                max_proj = 0
                for n in normals:
                    max_proj = max(max_proj, np.abs((points-c)@n).max())
                S = (2/np.sqrt(3))*max_proj
                if S < best_S:
                    best_S = S
                    best_c = (cx, cy)
        return best_S, best_c
    else:
        c = np.array(center)
        max_proj = 0
        for n in normals:
            max_proj = max(max_proj, np.abs((points-c)@n).max())
        return (2/np.sqrt(3))*max_proj, center

# EVOLVE-BLOCK-START
def construct():
    grid_spacing = 0.3
    x_vals = np.arange(-3, 3 + grid_spacing/2, grid_spacing)
    y_vals = np.arange(-3, 3 + grid_spacing/2, grid_spacing)
    grid_points = [(x, y) for x in x_vals for y in y_vals]
    n_grid = len(grid_points)
    
    hex_cache = {}
    for gi, (x, y) in enumerate(grid_points):
        hex_cache[gi] = hex_vertices(x, y, 1.0, 0.0)
    
    def is_valid(pos_indices):
        for i in range(N):
            for j in range(i+1, N):
                if not polygons_disjoint(hex_cache[pos_indices[i]], hex_cache[pos_indices[j]]):
                    return False
        return True
    
    def compute_S(pos_indices):
        all_pts = np.vstack([hex_cache[gi] for gi in pos_indices])
        S, center = min_S_for_points(all_pts)
        return S, center
    
    np.random.seed(42)
    best_S = float('inf')
    best_indices = None
    best_center = None
    
    # Initial grid pattern
    indices = []
    for i in range(N):
        col, row = i % 4, i // 4
        x, y = (col-1.5)*2.2, (row-1.0)*2.0
        gi = np.argmin(np.array([(px-x)**2+(py-y)**2 for px,py in grid_points]))
        indices.append(gi)
    
    current_S = float('inf')
    if is_valid(indices):
        S, c = compute_S(indices)
        best_S, best_indices, best_center = S, indices[:], c
        current_S = S
    else:
        for _ in range(300):
            indices = np.random.randint(0, n_grid, size=N).tolist()
            if is_valid(indices):
                S, c = compute_S(indices)
                best_S, best_indices, best_center = S, indices[:], c
                current_S = S
                break
    
    if current_S == float('inf'):
        return [[0.0,0.0,0.0]]*N, (0.0,0.0), 40.0, 0.0
    
    T, T_min, cooling = 1.0, 0.001, 0.9995
    max_iters = 15000
    start = time.time()
    
    for it in range(max_iters):
        if time.time() - start > 110:
            break
        idx = np.random.randint(0, N)
        new_gi = np.random.randint(0, n_grid)
        trial = indices[:]
        trial[idx] = new_gi
        
        ok = True
        for j in range(N):
            if j != idx and not polygons_disjoint(hex_cache[new_gi], hex_cache[trial[j]]):
                ok = False
                break
        if not ok:
            continue
        
        trial_S, trial_c = compute_S(trial)
        delta = trial_S - current_S
        if delta < 0 or np.random.rand() < np.exp(-delta/T):
            indices, current_S = trial, trial_S
            if current_S < best_S:
                best_S, best_indices, best_center = current_S, indices[:], trial_c
        T = max(T*cooling, T_min)
    
    inner = [(grid_points[gi][0], grid_points[gi][1], 0.0) for gi in best_indices]
    return inner, best_center, best_S, 0.0
# EVOLVE-BLOCK-END
