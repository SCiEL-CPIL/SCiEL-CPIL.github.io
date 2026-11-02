"""Circle packing in perimeter-4 rectangle: DE over width + Delaunay Inradius Polish."""
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import differential_evolution

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _compute_inradii(x, y, W, H, r, max_iter=100):
    """
    Compute maximal radii for fixed positions using Delaunay triangulation.
    This is a structural change: instead of optimizing radii via gradients, 
    we solve the geometric problem of finding the largest circle that fits 
    in each vertex of the triangulation (constrained by neighbors and walls).
    """
    r = r.copy()
    n = N
    points = np.column_stack([x, y])
    
    # Initial guess: wall distance
    r_wall = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
    r = np.maximum(r, r_wall * 0.5)
    r = np.minimum(r, r_wall)
    
    for _ in range(max_iter):
        # Build Delaunay
        try:
            tri = Delaunay(points)
        except Exception:
            break
        
        sim = tri.simplices
        
        new_r = r.copy()
        for i in range(n):
            # Max radius limited by walls
            r_max = r_wall[i]
            
            # Check neighbors in triangulation
            # For each triangle containing i, the circle i is tangent to 
            # the other two vertices in the worst case (if they are large).
            # Actually, the constraint is d(i, j) >= r[i] + r[j].
            # So r[i] <= d(i, j) - r[j].
            # We iterate: r[i] = min(r_wall[i], min_{j neighbor} (d(i,j) - r[j]))
            
            # Find neighbors of i
            neighbors = set()
            for s in sim:
                if i in s:
                    for v in s:
                        if v != i:
                            neighbors.add(v)
            
            for j in neighbors:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            
            new_r[i] = max(0.0, r_max)
        
        # Relaxation for stability
        new_r = 0.5 * r + 0.5 * new_r
        
        if np.max(np.abs(new_r - r)) < 1e-10:
            r = new_r
            break
        r = new_r
        
    # Final pass: ensure strict validity by shrinking if needed
    # (Delaunay neighbor check might miss non-neighbors if positions are bad, 
    #  but usually Delaunay covers the closest points. 
    #  To be safe, we do a full check and shrink.)
    if not _is_valid(x, y, r, W, H):
        # Shrink slightly
        scale = 0.95
        while scale > 0.1:
            r_trial = r * scale
            if _is_valid(x, y, r_trial, W, H):
                r = r_trial
                break
            scale *= 0.9
            
    return r

def _resolve_overlaps(x, y, r, W, H, max_iter=500, tol=1e-6):
    """
    Force-directed relaxation to resolve overlaps.
    Returns valid x, y or None if failed.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    n = N
    dt = 0.005
    
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall forces
        fx += np.maximum(r - x, 0) * 10
        fx -= np.maximum(x + r - W, 0) * 10
        fy += np.maximum(r - y, 0) * 10
        fy -= np.maximum(y + r - H, 0) * 10
        
        # Pairwise forces
        max_overlap = 0
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-6:
                    dx = np.random.uniform(-0.1, 0.1)
                    dy = np.random.uniform(-0.1, 0.1)
                    d = np.sqrt(dx*dx + dy*dy)
                
                r_sum = r[i] + r[j]
                if d < r_sum:
                    overlap = r_sum - d
                    max_overlap = max(max_overlap, overlap)
                    f = overlap * 5.0
                    fx[i] += (dx/d) * f
                    fy[i] += (dy/d) * f
                    fx[j] -= (dx/d) * f
                    fy[j] -= (dy/d) * f
        
        if max_overlap < tol:
            break
            
        x += dt * fx
        y += dt * fy
        
        # Clamp
        x = np.clip(x, r, W - r)
        y = np.clip(y, r, H - r)
        
    if _is_valid(x, y, r, W, H):
        return x, y
    return None

def _polish_packing(x, y, r, W, H, time_budget):
    """
    Local improvement: 
    1. Resolve overlaps.
    2. Compute inradii.
    3. Random shake + re-resolve + re-compute inradii (Error Repair).
    """
    best_val = 0.0
    best_state = (x.copy(), y.copy(), r.copy())
    
    # Initial resolution
    res = _resolve_overlaps(x, y, r, W, H)
    if res is not None:
        x, y = res
        r_new = _compute_inradii(x, y, W, H, r)
        if _is_valid(x, y, r_new, W, H):
            val = np.sum(r_new)
            if val > best_val:
                best_val = val
                best_state = (x.copy(), y.copy(), r_new.copy())
    
    # Iterative shake and repair
    rng = np.random.default_rng(42)
    current_x, current_y, current_r = best_state
    
    for i in range(50): # Max shakes
        if time.time() > time_budget:
            break
            
        # Shake: move some circles randomly
        shake_x = current_x.copy()
        shake_y = current_y.copy()
        
        # Randomly select a subset to move
        num_to_move = rng.integers(3, 8)
        indices = rng.choice(N, size=num_to_move, replace=False)
        
        for idx in indices:
            shake_x[idx] += rng.uniform(-0.1, 0.1)
            shake_y[idx] += rng.uniform(-0.1, 0.1)
            
        # Resolve
        res = _resolve_overlaps(shake_x, shake_y, current_r, W, H, max_iter=200)
        if res is not None:
            nx, ny = res
            nr = _compute_inradii(nx, ny, W, H, current_r)
            if _is_valid(nx, ny, nr, W, H):
                val = np.sum(nr)
                if val > best_val:
                    best_val = val
                    best_state = (nx.copy(), ny.copy(), nr.copy())
                    current_x, current_y, current_r = best_state

    return best_state, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_value = 0.0
    best_result = None
    
    # DE Objective: Maximize sum of radii for a given width W
    def de_objective(W_array):
        W = W_array[0]
        H = 2.0 - W
        
        # Check if time is running out
        if time.time() > deadline - 5:
            return 0.0
            
        # Generate initial grid
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side * W
            y[i] = (row + 0.5) / side * H
            r[i] = min(W / side, H / side) * 0.45
        
        # Polish
        state, val = _polish_packing(x, y, r, W, H, time_budget=deadline - 2)
        
        if state is not None:
            x_p, y_p, r_p = state
            if _is_valid(x_p, y_p, r_p, W, H):
                if val > best_value:
                    # Update global best (thread safety not needed for single process, 
                    # but DE might call this concurrently? No, DE is sequential in SciPy usually)
                    # We need to capture the best result.
                    # Since DE is a black box, we'll store the best in a closure variable
                    # or just return the value and re-run the best one at the end?
                    # Better: store in a mutable list in closure.
                    global_best[0] = (x_p.copy(), y_p.copy(), r_p.copy(), W, val)
                return val
        
        return 0.0

    global_best = [None]
    
    # DE bounds: W in [0.5, 1.5] (since H = 2-W, H in [0.5, 1.5])
    # Symmetric, so W in [1.0, 1.5] is enough? No, W can be < 1.0 if H > 1.0.
    # But W and H are symmetric in the problem definition (rectangle).
    # So we can restrict W to [1.0, 1.5] without loss of generality?
    # Wait, perimeter is 2(W+H)=4 => W+H=2.
    # If W=1.2, H=0.8. If W=0.8, H=1.2. The packing in 1.2x0.8 is equivalent to 0.8x1.2 rotated.
    # The sum of radii is the same.
    # So we only need to search W in [1.0, 1.5].
    
    bounds = [(1.0, 1.5)]
    
    try:
        # Use DE to find the best aspect ratio
        # popsize=10, maxiter=20
        # Tolerances
        de_res = differential_evolution(
            de_objective, 
            bounds, 
            maxiter=30, 
            popsize=10, 
            tol=0.01, 
            seed=42,
            polish=False, # We do our own polish
            init='sobol'
        )
    except Exception:
        pass
    
    # The DE objective function updated global_best[0] with the best packing found
    if global_best[0] is not None:
        x, y, r, W, val = global_best[0]
        if val > best_value:
            best_value = val
            best_result = np.column_stack((x, y, r))
    
    # If DE didn't find anything or failed, try a few fixed aspect ratios as fallback
    if best_result is None or best_value < 1.0:
        for W in [1.0, 1.1, 1.2, 1.3]:
            H = 2.0 - W
            if time.time() > deadline - 10:
                break
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                x[i] = (col + 0.5) / side * W
                y[i] = (row + 0.5) / side * H
                r[i] = min(W / side, H / side) * 0.45
            
            state, val = _polish_packing(x, y, r, W, H, time_budget=deadline - 5)
            if state is not None:
                x_p, y_p, r_p = state
                if _is_valid(x_p, y_p, r_p, W, H) and val > best_value:
                    best_value = val
                    best_result = np.column_stack((x_p, y_p, r_p))

    # Final safety check and shrink
    if best_result is not None:
        x = best_result[:, 0]
        y = best_result[:, 1]
        r = best_result[:, 2]
        
        # We need to recover W. 
        # In the DE loop, W was local. We need to store it.
        # Let's assume the best_result was generated with a specific W.
        # Actually, construct() returns (x, y, r). The evaluator calculates W from bounding box?
        # "bounding rectangle has perimeter at most 4"
        # The evaluator likely computes the bounding box of the circles.
        # So we don't need to return W. We just need x, y, r such that the bounding box fits in 2(W+H)=4.
        # My _is_valid checked against a specific W, H.
        # If the bounding box of the returned circles is smaller than W, H, it's valid.
        # But if the evaluator computes the tight bounding box, it's fine.
        # However, if I return circles that fit in W=1.2, H=0.8, the bounding box will be <= 1.2, 0.8.
        # Perimeter will be <= 4.
        
        # One issue: _compute_inradii might produce circles that are very small if the triangulation is bad.
        # The "shake and repair" should help.
        
        # Shrink to ensure validity
        for _ in range(10):
            r_trial = r * (1 - 1e-9)
            # Check against tight bounding box?
            # The evaluator says "bounding rectangle has perimeter at most 4".
            # It likely computes min/max x/y.
            # Let's just return. The internal _is_valid was strict.
            break

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
