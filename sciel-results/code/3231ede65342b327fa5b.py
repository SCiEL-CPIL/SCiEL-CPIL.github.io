import numpy as np
import time
from scipy.optimize import linprog

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial positions: 6x6 grid, take 32
    # 6x6 = 36 points.
    # Centers at (i+0.5)/6, (j+0.5)/6
    side = 6
    pts = []
    for i in range(side):
        for j in range(side):
            pts.append(((i + 0.5) / side, (j + 0.5) / side))
    # Take first 32
    centers = np.array(pts[:N])
    
    best_score = -1.0
    best_centers = centers.copy()
    
    # Pre-calculate boundary constraints for LP
    # r_i <= min(c_ix, c_iy, 1-c_ix, 1-c_iy)
    # This changes with positions, so we rebuild A_ub each iteration.
    
    # LP setup helpers
    def solve_lp(centers):
        # Variables: r_0 ... r_31
        # Maximize sum(r) -> Minimize -sum(r)
        c_obj = -np.ones(N)
        
        A_ub = []
        b_ub = []
        
        # Boundary constraints: r_i <= dist_to_boundary_i
        for i in range(N):
            x, y = centers[i]
            d_bound = min(x, y, 1-x, 1-y)
            # r_i <= d_bound
            row = np.zeros(N)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(d_bound)
            
        # Pair constraints: r_i + r_j <= dist_ij
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d_ij = np.sqrt(dx*dx + dy*dy)
                row = np.zeros(N)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(d_ij)
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(0, None) for _ in range(N)]
        
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0:
            r = res.x
            score = np.sum(r)
            # Get multipliers
            # res.ineqlin contains multipliers for A_ub rows
            # Order: 
            # 0..N-1: Boundary constraints
            # N..N+M-1: Pair constraints
            mults = res.ineqlin
            if mults is None:
                mults = np.zeros(len(b_ub))
            
            # Extract boundary multipliers
            b_mults = mults[:N]
            
            # Extract pair multipliers
            # Need to map index to pair
            # Index k in pair section corresponds to pair (i,j)
            # We need to reconstruct the pair order
            pair_mults = {}
            idx = N
            for i in range(N):
                for j in range(i+1, N):
                    pair_mults[(i,j)] = mults[idx]
                    idx += 1
                    
            return score, r, b_mults, pair_mults
        else:
            return -1.0, None, None, None

    # Optimization Loop
    eta = 0.005 # Step size
    iters_per_restart = 200
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # If not first restart, perturb centers
        if restart > 0:
            centers += np.random.normal(0, 0.05, centers.shape)
            centers = np.clip(centers, 0.0, 1.0)
            
        for it in range(iters_per_restart):
            if time.time() > deadline:
                break
                
            score, r, b_mults, pair_mults = solve_lp(centers)
            
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                
            if r is None:
                break
                
            # Calculate forces
            forces = np.zeros_like(centers)
            
            # Boundary forces
            for i in range(N):
                m = b_mults[i]
                if m > 1e-6:
                    x, y = centers[i]
                    # Find closest boundary
                    d_left = x
                    d_right = 1-x
                    d_bottom = y
                    d_top = 1-y
                    
                    min_d = min(d_left, d_right, d_bottom, d_top)
                    
                    if min_d == d_left:
                        forces[i] += m * np.array([1.0, 0.0])
                    elif min_d == d_right:
                        forces[i] += m * np.array([-1.0, 0.0])
                    elif min_d == d_bottom:
                        forces[i] += m * np.array([0.0, 1.0])
                    else:
                        forces[i] += m * np.array([0.0, -1.0])
                        
            # Pair forces
            for (i, j), m in pair_mults.items():
                if m > 1e-6:
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < 1e-9:
                        # Random nudge if overlapping exactly
                        fx, fy = np.random.normal(0, 1, 2)
                    else:
                        fx, fy = dx/d, dy/d
                    forces[i] += m * np.array([fx, fy])
                    forces[j] -= m * np.array([fx, fy])
                    
            # Update centers
            centers += eta * forces
            centers = np.clip(centers, 0.0, 1.0)
            
    # Final LP solve for best centers to get exact radii
    score, r, _, _ = solve_lp(best_centers)
    if r is None:
        # Fallback to seed if something went wrong
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Construct result array
    result = np.zeros((N, 3))
    for i in range(N):
        result[i][0] = best_centers[i][0]
        result[i][1] = best_centers[i][1]
        result[i][2] = r[i]
        
    return result

