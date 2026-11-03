import numpy as np
import time
from scipy.optimize import minimize
import math

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_score = -1
    best_result = None
    
    # Strategy: Contact Graph Optimization
    # 1. Define a set of candidate contact graphs (adjacency lists) for 21 circles.
    # 2. For each graph, optimize the radii vector r such that:
    #    - For touching pairs (i,j): dist(i,j) = r_i + r_j
    #    - For non-touching pairs (i,j): dist(i,j) >= r_i + r_j + epsilon
    #    - Circles are within a rectangle of perimeter 4.
    #    
    #    Since positions are coupled, we can't just optimize r.
    #    However, we can use a "rigid body" or "bar-joint" framework.
    #    Alternative: Optimize positions AND radii, but initialize positions 
    #    based on the graph structure (e.g., spring embedding or known hex coords)
    #    and use a solver that strongly penalizes overlaps while encouraging 
    #    touching of graph edges.
    
    # A more robust structural approach for this specific task:
    # Use a "Force Directed" layout with repulsive forces for non-edges 
    # and attractive forces (springs) for edges, while expanding radii.
    # This is effectively a soft-constraint optimization but guided by topology.
    
    # Let's try a hybrid: 
    # 1. Generate a few good topological seeds (hexagonal clusters).
    # 2. For each seed, run a local optimization on (x, y, r) 
    #    but with a very strong penalty for any violation of non-edge distances
    #    and a reward for edge distances being equal to r_i + r_j.
    
    # Actually, the most distinct "structural" change is to fix the rectangle 
    # aspect ratio to a few highly likely candidates and use a more aggressive 
    # multi-start local search with a better objective function that handles 
    # the non-convexity via "soft" contacts that become "hard" over iterations.
    
    # Let's implement a "Successive Linear Programming" or "Quadratic Programming" 
    # based local refinement, but the key structural change is the INITIALIZATION 
    # and the CONSTRAINT HANDLING.
    
    # New Strategy: 
    # Use a known dense packing configuration (e.g., rows of hexagonal packing) 
    # as the initial guess. 
    # Then, use a gradient-based optimizer that maximizes sum(r) 
    # subject to:
    # 1. x_i - r_i >= 0, x_i + r_i <= W
    # 2. y_i - r_i >= 0, y_i + r_i <= H
    # 3. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    
    # The current program does this but with random init. 
    # The structural change: Use DETERMINISTIC HEXAGONAL INITIALIZATION 
    # tuned for the specific N=21.
    # 21 circles fit well in a hexagonal arrangement.
    # e.g., rows of 3, 4, 5, 4, 3? Sum = 19. Need 21.
    # Rows of 4, 5, 6, 5, 4? Sum = 24. Too many.
    # Rows of 3, 4, 5, 4, 3 is 19. Add 2?
    # Maybe a 5x5 grid is better? 25 spots, remove 4 corners?
    
    # Let's define specific topological seeds.
    
    def get_hex_seed(w, h):
        """
        Generates positions for a hexagonal-like packing.
        We'll try to fit 21 circles in a hexagonal cluster.
        A hexagonal number 21 doesn't exist (1, 7, 19, 37...).
        19 is close. We can add 2 to the center or edges.
        Let's use a 5x5 grid as a fallback, but try to distort it into hex.
        """
        # 5x5 grid positions
        xs = []
        ys = []
        side = 5
        spacing = 1.0 # Will be scaled
        for i in range(side):
            for j in range(side):
                # Hexagonal offset for odd rows
                offset = 0.5 * spacing if i % 2 == 1 else 0
                x = (j + 0.5) * spacing + offset
                y = (i + 0.5) * spacing * 0.866 # sqrt(3)/2
                xs.append(x)
                ys.append(y)
        
        # We have 25 points. We need 21.
        # Remove 4 corners?
        # Indices: 
        # 0 1 2 3 4
        # 5 6 7 8 9
        # 10 11 12 13 14
        # 15 16 17 18 19
        # 20 21 22 23 24
        
        # Remove corners 0, 4, 20, 24
        remove = [0, 4, 20, 24]
        keep = [i for i in range(25) if i not in remove]
        
        cx = np.array(xs[keep])
        cy = np.array(ys[keep])
        
        # Normalize to fit within w, h
        # Center
        cx -= np.mean(cx)
        cy -= np.mean(cy)
        
        # Scale to fit
        max_x = np.max(cx) - np.min(cx)
        max_y = np.max(cy) - np.min(cy)
        
        if max_x > 0 and max_y > 0:
            scale = min(w / max_x, h / max_y) * 0.95 # 95% to allow for radii
            cx *= scale
            cy *= scale
        
        # Re-center
        cx += w / 2.0
        cy += h / 2.0
        
        # Initial radii: small
        r = np.full(21, 0.1)
        
        return cx, cy, r

    def get_grid_seed(w, h):
        side = 5
        xs, ys = [], []
        for i in range(side):
            for j in range(side):
                xs.append((j + 0.5) * w / side)
                ys.append((i + 0.5) * h / side)
        
        remove = [0, 4, 20, 24] # Remove corners
        keep = [i for i in range(25) if i not in remove]
        
        cx = np.array(xs[keep])
        cy = np.array(ys[keep])
        r = np.full(21, w / side * 0.45)
        return cx, cy, r

    def get_staggered_seed(w, h):
        # 3 rows of 7? 3*7=21
        # Row 0: 7 circles
        # Row 1: 7 circles (shifted)
        # Row 2: 7 circles
        xs, ys = []
        n_per_row = 7
        rows = 3
        # Aspect ratio needs to be wide enough for 7 circles.
        # If W=1, H=1, 7 circles in a row means r ~ 1/14 = 0.07.
        # 3 rows of 0.07 radius circles in H=1?
        # Height used = 3 * 2r * 0.866? No, staggered height is r*sqrt(3).
        # Total height ~ 3 * r * 1.732 + 2r?
        
        # Let's just place them in a 7x3 grid
        for i in range(rows):
            for j in range(n_per_row):
                x = (j + 0.5) * w / n_per_row
                y = (i + 0.5) * h / rows
                xs.append(x)
                ys.append(y)
        
        cx = np.array(xs)
        cy = np.array(ys)
        r = np.full(21, w / n_per_row * 0.45)
        return cx, cy, r

    seeds = [
        ("hex", get_hex_seed),
        ("grid", get_grid_seed),
        ("staggered", get_staggered_seed)
    ]

    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 0.7, 1.4, 0.6, 1.5, 0.5]

    # Pre-compute pairs
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((i, j))
    pairs = np.array(pairs)
    i_idx, j_idx = pairs[:, 0], pairs[:, 1]

    def validate_and_score(cx, cy, r, w, h):
        # Check bounds
        if np.any(cx - r < -1e-7) or np.any(cx + r - w > 1e-7):
            return -1, None
        if np.any(cy - r < -1e-7) or np.any(cy + r - h > 1e-7):
            return -1, None
        
        # Check overlaps
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        dist_sq = dx*dx + dy*dy
        r_sum = r[:, None] + r[None, :]
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        viol_sq = dist_sq[mask] - r_sum[mask]**2
        if np.any(viol_sq < -1e-7):
            return -1, None
        
        score = np.sum(r)
        return score, np.column_stack((cx, cy, r))

    for w in aspect_ratios:
        if time.time() > deadline:
            break
        h = 2.0 - w
        if h <= 0:
            continue
            
        for seed_name, seed_func in seeds:
            if time.time() > deadline - 10:
                break
                
            cx, cy, r = seed_func(w, h)
            
            # Optimization
            x0 = np.concatenate([cx, cy, r])
            
            r_max = min(w, h) / 2.0
            bounds = [(0.0, w)] * N + [(0.0, h)] * N + [(0.0, r_max)] * N
            
            def objective_and_grad(x):
                cx_v = x[:N]
                cy_v = x[N:2*N]
                r_v = x[2*N:]
                
                obj = -np.sum(r_v)
                grad = np.zeros(3*N)
                grad[2*N:] = -1.0
                
                # Overlap penalty
                dx = cx_v[:, None] - cx_v[None, :]
                dy = cy_v[:, None] - cy_v[None, :]
                dist_sq = dx*dx + dy*dy
                dist_sq[dist_sq < 1e-12] = 1e-12
                dist = np.sqrt(dist_sq)
                
                r_sum = r_v[:, None] + r_v[None, :]
                
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                d_pairs = dist[mask]
                rs_pairs = r_sum[mask]
                
                viol = rs_pairs - d_pairs
                viol_pos = np.maximum(0, viol)
                
                # High penalty to enforce disjointness
                penalty = 50.0 
                obj += penalty * np.sum(viol_pos**2)
                
                ii, jj = np.where(mask)
                active = viol_pos > 0
                ii_act = ii[active]
                jj_act = jj[active]
                viol_act = viol_pos[active]
                d_act = d_pairs[active]
                
                if len(ii_act) > 0:
                    factor = -2.0 * penalty * viol_act / d_act
                    dx_act = cx_v[ii_act] - cx_v[jj_act]
                    dy_act = cy_v[ii_act] - cy_v[jj_act]
                    
                    np.add.at(grad, ii_act, factor * dx_act)
                    np.add.at(grad, jj_act, -factor * dx_act)
                    np.add.at(grad, N + ii_act, factor * dy_act)
                    np.add.at(grad, N + jj_act, -factor * dy_act)
                    
                    np.add.at(grad, 2*N + ii_act, 2.0 * penalty * viol_act)
                    np.add.at(grad, 2*N + jj_act, 2.0 * penalty * viol_act)
                
                return obj, grad
            
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                x_opt = res.x
                cx_opt = x_opt[:N]
                cy_opt = x_opt[N:2*N]
                r_opt = x_opt[2*N:]
                
                score_opt, res_opt = validate_and_score(cx_opt, cy_opt, r_opt, w, h)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_result = res_opt
                    
            except Exception as e:
                pass

    # Fallback
    if best_result is None:
        # Simple grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append((cx, cy, r))
        best_result = np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
