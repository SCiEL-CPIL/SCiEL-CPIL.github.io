import math
import numpy as np
import time
import itertools
import random
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define the 6 normal vectors for the Minkowski sum of two unit hexagons (same orientation)
    # The condition for non-overlap is |d . n| >= sqrt(3) for all 6 normals
    # This defines a hexagon of "radius" sqrt(3) in the dual space (or rather, the forbidden region is a hexagon of apothem sqrt(3))
    # Actually, the forbidden region for the difference vector d is the Minkowski sum of the hexagon and its reflection.
    # For a regular hexagon of side 1, the Minkowski sum with its reflection is a regular hexagon of side 2, with apothem 2 * (sqrt(3)/2) = sqrt(3).
    # So the constraint is that the distance from d to the origin, measured in the hexagonal norm defined by these normals, is >= 1.
    # Or simply: min_k |d . n_k| >= sqrt(3).
    
    normals = np.array([
        [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
        for k in range(6)
    ])
    
    # The "lattice" of valid relative positions is not a simple lattice, but we can generate candidate relative positions
    # that are "just touching".
    # A relative position d is valid if min_k |d . n_k| >= sqrt(3).
    # We can generate a set of "primitive" valid relative positions.
    # For example, moving by 1 unit in the direction of one of the normals? No.
    # The vertices of the forbidden hexagon are at distance 2 from the center.
    # The edges are at distance sqrt(3).
    # So a point d is on the boundary if it lies on one of the 6 lines d . n_k = +/- sqrt(3).
    
    # Let's generate a set of candidate relative displacement vectors.
    # We can use a grid of points and filter for validity.
    # Or, we can use the fact that the valid region is the complement of the interior of the hexagon.
    # The "minimal" valid vectors are those on the boundary.
    # Let's generate vectors that are sums of basis vectors.
    
    # A better approach: Generate all possible contact graphs.
    # A contact graph is a graph G=(V,E) where V={1..11} and E is a set of edges representing "contact" or "near-contact".
    # For each edge (i,j), we assume that hexagon i and j are touching.
    # This means that the relative position d_ij satisfies min_k |d_ij . n_k| = sqrt(3).
    # This implies that d_ij lies on one of the 12 lines (6 normals, 2 signs).
    # So for each edge, we have 12 possible directions for d_ij.
    # This is a combinatorial explosion.
    
    # Instead, let's use a heuristic search over positions.
    # We will use a simulated annealing approach to find a good configuration.
    # The objective function is the side length of the minimal enclosing hexagon.
    # The constraints are the non-overlap conditions.
    
    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals_outer = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals_outer:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def check_overlap(pts):
        n_pts = len(pts)
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                d = pts[i] - pts[j]
                projs = np.abs(d @ normals.T)
                if np.min(projs) < math.sqrt(3) - 1e-9:
                    return False
        return True

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.05):
        normals_outer = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals_outer.T
        
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        
        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf'), float('inf'), cx, cy
        
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        
        S = (D + APOTHEM) / APOTHEM
        return S, D, cx, cy

    # Initialize positions with a random hexagonal lattice
    def init_positions():
        # Start with a compact hexagonal cluster
        # Center at (0,0)
        pts = [(0.0, 0.0)]
        # Ring 1
        for k in range(6):
            angle = math.pi/6 + k*math.pi/3
            x = 2 * math.cos(angle)
            y = 2 * math.sin(angle)
            pts.append((x, y))
        # Ring 2 (partial)
        # We need 11 points. We have 7. Need 4 more.
        # Let's place them in the gaps or extend the lattice.
        # A better init: use a random perturbation of a known good config
        # or just a random scatter and let the optimizer work.
        # Let's try a random scatter within a radius of 10.
        pts = []
        for _ in range(N):
            angle = random.uniform(0, 2*math.pi)
            r = random.uniform(0, 5)
            pts.append((r*math.cos(angle), r*math.sin(angle)))
        return np.array(pts)

    best_S = float('inf')
    best_pts = None
    best_beta = 0.0
    best_center = None
    
    # Run multiple restarts
    for restart in range(20):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = init_positions()
        
        # Simulated Annealing
        T = 1.0
        T_min = 1e-6
        cooling = 0.99
        
        current_S = float('inf')
        current_pts = pts.copy()
        current_beta = random.uniform(0, math.pi/6)
        current_center = (0,0)
        
        for step in range(2000):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            # Propose a new position
            new_pts = current_pts.copy()
            i = random.randint(0, N-1)
            dx = random.gauss(0, 0.5)
            dy = random.gauss(0, 0.5)
            new_pts[i, 0] += dx
            new_pts[i, 1] += dy
            
            # Propose a new beta
            new_beta = current_beta + random.gauss(0, 0.01)
            new_beta = ((new_beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            
            # Calculate new S
            new_S, new_center, _ = calc_S_for_beta(new_pts, new_beta)
            
            if new_S == float('inf'):
                continue
            
            if not check_overlap(new_pts):
                # Penalty for overlap
                penalty = 0.0
                for a in range(N):
                    for b in range(a+1, N):
                        d = new_pts[a] - new_pts[b]
                        projs = np.abs(d @ normals.T)
                        min_proj = np.min(projs)
                        if min_proj < math.sqrt(3):
                            penalty += (math.sqrt(3) - min_proj)
                new_S += penalty * 10.0
            
            if new_S < current_S:
                current_S = new_S
                current_pts = new_pts
                current_beta = new_beta
                current_center = new_center
            else:
                # Accept with probability exp(-(new_S - current_S)/T)
                delta = new_S - current_S
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_S = new_S
                    current_pts = new_pts
                    current_beta = new_beta
                    current_center = new_center
            
            T *= cooling
            
            if T < T_min:
                break
        
        if current_S < best_S and check_overlap(current_pts):
            best_S = current_S
            best_pts = current_pts
            best_beta = current_beta
            best_center = current_center
    
    # Final polish with local optimization
    if best_pts is not None:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def smooth_obj(params):
                n_pts = len(best_pts)
                pts = params[:2*n_pts].reshape(n_pts, 2)
                cx = params[2*n_pts]
                cy = params[2*n_pts+1]
                beta = params[2*n_pts+2]
                beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                
                S_smooth, D, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=0.01)
                if S_smooth == float('inf'):
                    return 1e10
                
                penalty = 0.0
                for i in range(n_pts):
                    for j in range(i+1, n_pts):
                        d = pts[i] - pts[j]
                        projs = np.abs(d @ normals.T)
                        min_proj = np.min(projs)
                        if min_proj < math.sqrt(3):
                            violation = math.sqrt(3) - min_proj
                            penalty += violation * 100.0
                
                return S_smooth + penalty

            init_params = np.concatenate([best_pts.ravel(), [best_center[0], best_center[1], best_beta]])
            
            try:
                res = minimize(smooth_obj, init_params, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
                
                if res.fun < best_S + 0.1:
                    n_pts = len(best_pts)
                    candidate_pts = res.x[:2*n_pts].reshape(n_pts, 2)
                    candidate_center = (res.x[2*n_pts], res.x[2*n_pts+1])
                    candidate_beta = ((res.x[2*n_pts+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    
                    S_exact, c_exact, _ = calc_S_for_beta(candidate_pts, candidate_beta)
                    if S_exact < best_S and check_overlap(candidate_pts):
                        best_S = S_exact
                        best_pts = candidate_pts
                        best_center = c_exact
                        best_beta = candidate_beta
            except Exception:
                pass

    if best_center is None or best_S == float('inf'):
        # Fallback
        best_pts = np.array([[0.0, 0.0]] * N)
        best_center = (0.0, 0.0)
        best_beta = 0.0
        best_S = 10.0

    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    # Safety scale
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    max_proj = 0
    for k in range(6):
        ang = best_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
