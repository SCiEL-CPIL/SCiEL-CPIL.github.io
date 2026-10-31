import numpy as np
import scipy.optimize as opt

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt

N = 26

def _get_lattice_indices(n_circles):
    """
    Generate indices for a triangular lattice fragment.
    We want to pack n_circles.
    Rows in triangular lattice:
    Row 0: 1 circle
    Row 1: 2 circles
    Row 2: 3 circles
    ...
    Sum of first k rows: k*(k+1)/2.
    For N=26:
    k=6: 6*7/2 = 21.
    k=7: 7*8/2 = 28.
    So we take full rows 0-5 (21 circles) and 5 circles from row 6.
    """
    indices = []
    row = 0
    count = 0
    while count < n_circles:
        row_size = row + 1
        for i in range(row_size):
            if count < n_circles:
                indices.append((row, i))
                count += 1
            else:
                break
        row += 1
    return indices

def _initial_lattice_positions(indices):
    """
    Generate initial positions for a triangular lattice.
    Side length s = 1.0 / (max_row + 1) approx? 
    Let's just use a standard triangular lattice with unit distance 1.
    Then we scale it to fit.
    """
    # dx = 1.0, dy = sqrt(3)/2
    # x = i + 0.5*row
    # y = row * sqrt(3)/2
    pos = []
    for row, i in indices:
        x = i + 0.5 * row
        y = row * np.sqrt(3) / 2.0
        pos.append([x, y])
    pos = np.array(pos)
    # Center it
    pos -= np.mean(pos, axis=0)
    return pos

def _solve_packing():
    # Get lattice structure
    indices = _get_lattice_indices(N)
    init_pos = _initial_lattice_positions(indices)
    
    # Define contact pairs (neighbors in the lattice)
    # In a triangular lattice, each point (row, i) is connected to:
    # (row, i-1), (row, i+1)
    # (row-1, i-1), (row-1, i)
    # (row+1, i), (row+1, i+1)
    # We need to map (row, i) to index in the list.
    idx_map = {}
    for k, (r, i) in enumerate(indices):
        idx_map[(r, i)] = k
        
    pairs = set()
    for r, i in indices:
        # Neighbors
        neighbors = [
            (r, i-1), (r, i+1),
            (r-1, i-1), (r-1, i),
            (r+1, i), (r+1, i+1)
        ]
        for nr, ni in neighbors:
            if (nr, ni) in idx_map:
                k1 = idx_map[(r, i)]
                k2 = idx_map[(nr, ni)]
                if k1 < k2:
                    pairs.add((k1, k2))
                else:
                    pairs.add((k2, k1))
    
    pairs = list(pairs)
    n_pairs = len(pairs)
    
    # Variables: x[0:N], y[0:N], r[0:N]
    # Total 3*N variables
    # Initial guess:
    # Scale init_pos to fit in [0,1] with some margin.
    # Let's estimate radius. 
    # Area of unit square is 1. 26 circles.
    # If packed densely, area ~ 0.9.
    # Average r ~ sqrt(0.9 / (26 * pi)) ~ sqrt(0.011) ~ 0.1.
    
    # Normalize init_pos to fit in a box of size 1x1
    # Current bounding box of init_pos
    min_x, max_x = np.min(init_pos[:,0]), np.max(init_pos[:,0])
    min_y, max_y = np.min(init_pos[:,1]), np.max(init_pos[:,1])
    width = max_x - min_x
    height = max_y - min_y
    scale = min(1.0 / width, 1.0 / height) * 0.8 # 0.8 to leave room for radius
    
    init_x = (init_pos[:,0] - min_x) * scale + 0.1
    init_y = (init_pos[:,1] - min_y) * scale + 0.1
    init_r = np.full(N, 0.08)
    
    x0 = np.concatenate([init_x, init_y, init_r])
    
    # Residual function
    def residuals(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        
        res = []
        
        # 1. Wall constraints (soft, but we want them tight for maximal packing)
        # To maximize sum of radii, we should push circles to walls.
        # However, least_squares minimizes sum of squares.
        # If we just minimize overlap, we might shrink circles.
        # Strategy: 
        # Minimize overlap + minimize "slack" from walls?
        # Actually, if we enforce contact between neighbors, the system is rigid.
        # We can fix the scale by enforcing that the "bounding box" of the lattice fits in [0,1].
        # Or, simpler: 
        # Minimize overlap. Then scale up?
        # No, let's add terms that encourage touching walls.
        # If a circle is not touching a wall, it can potentially be moved to allow others to grow?
        # In a rigid lattice, if it's not touching walls, it's "floating".
        # We want the whole assembly to touch the boundary.
        
        # Let's enforce that the min/max x/y of the circles + radii fit in [0,1].
        # This is a non-differentiable constraint (min/max).
        # Instead, let's use a penalty for being strictly inside.
        # But which circles touch walls?
        # In a triangular lattice in a square, the "corners" touch corners, "edges" touch walls.
        
        # Let's identify which circles are likely to touch walls.
        # Based on the lattice indices:
        # Row 0, i=0 -> Top-Leftish
        # Last row, last i -> Bottom-Rightish
        # 
        # Let's just enforce that the *bounding box* of the centers +/- radii is within [0,1].
        # This is hard to do with differentiable residuals directly for the whole set.
        
        # Alternative: 
        # Minimize overlap.
        # Then, in a post-processing step, scale the whole configuration?
        # No, the positions are coupled.
        
        # Let's try a different objective:
        # Minimize the sum of overlaps.
        # AND
        # Minimize the distance from specific "anchor" circles to the walls.
        # We can guess which circles touch walls.
        # For a triangular lattice of 26:
        # It looks like a triangle with side 7 (28 circles) minus 2.
        # Or 5 rows of 5? No, triangular is better.
        
        # Let's assume the "extreme" circles touch the walls.
        # Circle at min x, max x, min y, max y.
        
        # Let's just minimize overlap for now, and see if we get a valid packing.
        # Then we can try to expand it.
        
        # Actually, a common trick: 
        # Minimize overlap.
        # If overlap is 0, we have a valid packing.
        # To maximize radii, we should have started with larger radii or scale up.
        
        # Let's parameterize the lattice scale and position?
        # No, the directive says "parameterized by a small set of free variables... solving kinematic closure".
        
        # Let's stick to minimizing overlap.
        # To ensure we don't just shrink everything to 0, we need a lower bound on radius?
        # Or we can maximize sum of radii subject to no overlap.
        # `scipy.optimize.minimize` can do this.
        # Objective: -sum(r)
        # Constraints: dist(i,j) >= r_i + r_j, and walls.
        
        # Let's use `scipy.optimize.minimize` with SLSQP or COBYLA.
        # N=26, 78 variables.
        
        pass # Placeholder, will implement below

    # Let's implement the optimization properly
    def objective(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        return -np.sum(r)

    def constraints(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        
        cons = []
        
        # Wall constraints
        # x[i] >= r[i]
        # x[i] <= 1 - r[i]
        # y[i] >= r[i]
        # y[i] <= 1 - r[i]
        # We can combine these into inequalities.
        # But SLSQP handles 'ineq' where f(x) >= 0.
        
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]}) # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i] - v[2*N+i]}) # 1 - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]}) # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[N+i] - v[2*N+i]}) # 1 - y_i - r_i >= 0
            
        # Pairwise constraints
        # dist(i,j) >= r_i + r_j
        # sqrt((xi-xj)^2 + (yi-yj)^2) - ri - rj >= 0
        # This is non-linear and non-smooth at 0.
        # Better: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
        # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        
        for i, j in pairs:
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: 
                    (v[i] - v[j])**2 + (v[N+i] - v[N+j])**2 - (v[2*N+i] + v[2*N+j])**2
            })
            
        return cons

    # Initial guess
    # We need a good initial guess for SLSQP.
    # The lattice positions scaled to fit.
    # Let's calculate a feasible starting point.
    # If we set r=0, positions are free.
    # If we set r > 0, we need to satisfy constraints.
    
    # Let's use a heuristic:
    # Start with small radii, optimize positions to minimize overlap?
    # No, maximize radii.
    
    # Let's try multiple initializations.
    
    best_val = -1e10
    best_v = None
    
    # Initialization 1: Triangular lattice
    init_pos = _initial_lattice_positions(indices)
    # Scale to fit in [0.1, 0.9]
    min_x, max_x = np.min(init_pos[:,0]), np.max(init_pos[:,0])
    min_y, max_y = np.min(init_pos[:,1]), np.max(init_pos[:,1])
    scale = min(0.8 / (max_x - min_x), 0.8 / (max_y - min_y))
    init_x = (init_pos[:,0] - min_x) * scale + 0.1
    init_y = (init_pos[:,1] - min_y) * scale + 0.1
    init_r = np.full(N, 0.05)
    
    v0 = np.concatenate([init_x, init_y, init_r])
    
    # Check if v0 is feasible (approximately)
    # If not, scale down r until feasible.
    def is_feasible(v, tol=1e-6):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        for i in range(N):
            if x[i] < r[i] - tol or x[i] > 1 - r[i] + tol or y[i] < r[i] - tol or y[i] > 1 - r[i] + tol:
                return False
        for i, j in pairs:
            d2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
            r_sum = r[i] + r[j]
            if d2 < r_sum**2 - tol:
                return False
        return True

    if not is_feasible(v0):
        # Scale down r
        while not is_feasible(v0) and np.min(v0[2*N:]) > 0.01:
            v0[2*N:] *= 0.9
        if not is_feasible(v0):
            # Fallback to grid
            pass

    try:
        res = opt.minimize(objective, v0, method='SLSQP', constraints=constraints(v0), bounds=[(0,1) for _ in range(2*N)] + [(0.01, 0.5) for _ in range(N)], options={'maxiter': 1000, 'ftol': 1e-9})
        if res.success or res.fun < best_val:
            if is_feasible(res.x):
                best_val = res.fun
                best_v = res.x
    except Exception as e:
        print(f"Optimization failed: {e}")

    if best_v is None:
        # Fallback: Grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_v[0:N]
    y = best_v[N:2*N]
    r = best_v[2*N:3*N]
    
    # Final polish: ensure strict validity
    # The optimizer might leave small violations.
    # We can do a quick iterative projection to fix small overlaps.
    
    for _ in range(50):
        moved = False
        # Walls
        for i in range(N):
            if x[i] < r[i]:
                x[i] = r[i]
                moved = True
            elif x[i] > 1 - r[i]:
                x[i] = 1 - r[i]
                moved = True
            if y[i] < r[i]:
                y[i] = r[i]
                moved = True
            elif y[i] > 1 - r[i]:
                y[i] = 1 - r[i]
                moved = True
        
        # Pairs
        dx = x[None, :] - x[:, None]
        dy = y[None, :] - y[:, None]
        dist = np.sqrt(dx * dx + dy * dy)
        req = r[None, :] + r[:, None]
        overlap = req - dist
        overlap = np.maximum(overlap, 0)
        
        if np.max(overlap) < 1e-9:
            break
            
        for i in range(N):
            for j in range(i+1, N):
                ov = overlap[i, j]
                if ov > 1e-9:
                    d = dist[i, j]
                    if d < 1e-9:
                        # Random nudge
                        x[i] += 0.001
                        moved = True
                        continue
                    half = ov / 2 + 1e-8
                    ux = dx[i, j] / d
                    uy = dy[i, j] / d
                    x[i] -= half * ux
                    y[i] -= half * uy
                    x[j] += half * ux
                    y[j] += half * uy
                    moved = True
        
        if not moved:
            break
            
    # Verify
    if not is_feasible(np.concatenate([x, y, r]), tol=1e-6):
        # If still invalid, shrink radii slightly
        r *= 0.99
        # Re-project
        for _ in range(10):
             for i in range(N):
                x[i] = min(max(x[i], r[i]), 1 - r[i])
                y[i] = min(max(y[i], r[i]), 1 - r[i])
        if not is_feasible(np.concatenate([x, y, r]), tol=1e-6):
             side = int(np.ceil(np.sqrt(N)))
             cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
             return np.array(cells[:N])

    return np.column_stack((x, y, r))

def construct():
    return _solve_packing()

# EVOLVE-BLOCK-END
