import numpy as np
import time
import math

N = 11

def _hex_verts(x, y, theta):
    """Vertices of unit regular hexagon centered at (x,y) rotated by theta (radians)."""
    angles = np.array([0.0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
    c = np.cos(theta)
    s = np.sin(theta)
    cx = np.cos(angles)
    cy = np.sin(angles)
    vx = c * cx - s * cy + x
    vy = s * cx + c * cy + y
    return np.stack([vx, vy], axis=1)

def _hex_normals(theta):
    """6 outward unit normals of unit regular hexagon rotated by theta."""
    base = np.array([
        [math.cos(math.pi/6), math.sin(math.pi/6)],
        [math.cos(math.pi/2), math.sin(math.pi/2)],
        [math.cos(5*math.pi/6), math.sin(5*math.pi/6)],
        [math.cos(7*math.pi/6), math.sin(7*math.pi/6)],
        [math.cos(3*math.pi/2), math.sin(3*math.pi/2)],
        [math.cos(11*math.pi/6), math.sin(11*math.pi/6)]
    ])
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return base @ R.T

def _min_separation(v1, n1, v2, n2):
    """Min separation between two convex hexagons. Positive = separated, negative = overlap depth."""
    min_gap = float('inf')
    all_n = np.vstack([n1, n2])
    for n in all_n:
        p1 = v1 @ n
        p2 = v2 @ n
        gap = max(p1.min(), p2.min()) - max(p1.max(), p2.max())
        if gap > 1e-9:
            return float('inf')
        if gap < min_gap:
            min_gap = gap
    return min_gap

def _side_length(params):
    """Compute outer hexagon side length needed to contain all inner hexagons.
    Outer hexagon centered at origin, flat-top orientation."""
    # 3 unique normal directions (120 deg apart)
    n1 = np.array([math.cos(math.pi/6), math.sin(math.pi/6)])
    n2 = np.array([math.cos(math.pi/2), math.sin(math.pi/2)])
    n3 = np.array([math.cos(5*math.pi/6), math.sin(5*math.pi/6)])
    
    max_proj = 0.0
    for i in range(N):
        x, y, th = params[i]
        v = _hex_verts(x, y, th)
        proj = np.abs(v @ n1)
        max_proj = max(max_proj, proj.max())
        proj = np.abs(v @ n2)
        max_proj = max(max_proj, proj.max())
        proj = np.abs(v @ n3)
        max_proj = max(max_proj, proj.max())
    
    return 2.0 * max_proj / math.sqrt(3)

def _evaluate(params):
    """Returns (side, penalty). penalty = sum of overlap depths squared."""
    side = _side_length(params)
    
    # Precompute vertices and normals
    verts = []
    normals = []
    for i in range(N):
        x, y, th = params[i]
        verts.append(_hex_verts(x, y, th))
        normals.append(_hex_normals(th))
    
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            sep = _min_separation(verts[i], normals[i], verts[j], normals[j])
            if sep < 0:
                penalty += sep * sep
    
    return side, penalty

def construct():
    start = time.time()
    deadline = start + 140.0  # 10s safety margin
    
    # Known optimal: hexagonal grid of 11 hexagons
    d = math.sqrt(3)  # center-to-center distance for adjacent unit hexagons
    centers = [
        (0, 0),
        (d, 0), (d/2, 1.5), (-d/2, 1.5),
        (-d, 0), (-d/2, -1.5), (d/2, -1.5),
        (2*d, 0),
        (1.5*d, 3.0),
        (-1.5*d, 3.0),
        (0, 4.5)
    ]
    
    best_params = np.zeros((N, 3))
    for i in range(N):
        best_params[i, 0] = centers[i][0]
        best_params[i, 1] = centers[i][1]
        best_params[i, 2] = 0.0
    
    best_side, best_pen = _evaluate(best_params)
    # Verify it's valid
    assert best_pen < 1e-6, f"Initial grid has overlap: {best_pen}"
    
    print(f"Initial side: {best_side:.6f}, penalty: {best_pen:.2e}")
    
    # Simulated annealing to try to improve
    np.random.seed(42)
    
    T = 0.05
    T_min = 0.0005
    alpha = 0.998
    num_restarts = 30
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize: perturb best or random
        if r == 0:
            params = best_params.copy()
        else:
            if np.random.rand() < 0.6:
                params = best_params + np.random.normal(0, 0.3, (N, 3))
                params[:, 2] = np.random.uniform(-0.1, 0.1, N)
            else:
                # Random restart near grid
                params = np.zeros((N, 3))
                for i in range(N):
                    params[i, 0] = centers[i][0] + np.random.normal(0, 0.5)
                    params[i, 1] = centers[i][1] + np.random.normal(0, 0.5)
                    params[i, 2] = np.random.uniform(-0.2, 0.2)
        
        cur_side, cur_pen = _evaluate(params)
        cur_cost = cur_side + 50.0 * cur_pen
        
        steps = 8000
        for step in range(steps):
            if time.time() > deadline - 3:
                break
            
            k = np.random.randint(N)
            old = params[k].copy()
            
            # Small perturbation
            dist = np.random.uniform(0.02, 0.15)
            ang = np.random.uniform(0, 2 * math.pi)
            params[k, 0] += dist * math.cos(ang)
            params[k, 1] += dist * math.sin(ang)
            params[k, 2] += np.random.uniform(-0.05, 0.05)
            
            new_side, new_pen = _evaluate(params)
            new_cost = new_side + 50.0 * new_pen
            
            delta = new_cost - cur_cost
            if delta < 0 or np.random.rand() < math.exp(-delta / T):
                cur_cost = new_cost
                cur_side = new_side
                cur_pen = new_pen
                
                if new_pen < 1e-6 and new_side < best_side:
                    best_side = new_side
                    best_params = params.copy()
            else:
                params[k] = old
            
            T *= alpha
            if T < T_min:
                break
        
        # Reheat
        T = 0.03
    
    # Final validation
    final_side, final_pen = _evaluate(best_params)
    if final_pen > 1e-4:
        # Fallback to known valid grid
        best_params = np.zeros((N, 3))
        for i in range(N):
            best_params[i, 0] = centers[i][0]
            best_params[i, 1] = centers[i][1]
            best_params[i, 2] = 0.0
        best_side = _side_length(best_params)
    
    inner = []
    for i in range(N):
        x, y, ang = best_params[i]
        inner.append([float(x), float(y), float(math.degrees(ang))])
    
    return inner, (0.0, 0.0), float(best_side), 0.0

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Final side: {side:.8f}")
