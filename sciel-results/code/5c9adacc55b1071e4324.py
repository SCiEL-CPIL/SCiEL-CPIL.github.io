import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    sqrt3 = math.sqrt(3)
    
    # Precompute unit hexagon vertices
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    
    def calc_S(positions, alpha_deg, beta_deg):
        """Vectorized: positions is (N,2) array, returns scalar S."""
        alpha_rad = math.radians(alpha_deg)
        beta_rad = math.radians(beta_deg)
        
        # Outer normals: 6 directions at 30+beta+k*60
        angles = math.radians(30) + beta_rad + np.arange(6) * math.radians(60)
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6,2)
        
        # For each normal, compute support of all inner hexagons
        # Support_i = P_i . n + h_unit(R_{-alpha} n)
        c = math.cos(-alpha_rad)
        s = math.sin(-alpha_rad)
        
        # Rotate all normals by -alpha: (6,2)
        n_rot = np.column_stack([
            c * normals[:,0] - s * normals[:,1],
            s * normals[:,0] + c * normals[:,1]
        ])
        
        # h_unit for each rotated normal: max of dot with unit_verts
        # unit_verts: (6,2), n_rot: (6,2) -> need all pairs? No, each n_rot[k] dot each unit_vert
        # h_unit[k] = max_j (unit_verts[j] . n_rot[k])
        # = max over j of (unit_verts @ n_rot[k])
        # Compute all: (6 normals) x (6 verts) = 36 dots
        dots = n_rot @ unit_verts.T  # (6,6): dots[k,j] = n_rot[k] . unit_verts[j]
        h_vals = np.max(dots, axis=1)  # (6,)
        
        # positions: (N,2), normals: (6,2)
        # proj[i,k] = positions[i] . normals[k] + h_vals[k]
        pos_projs = positions @ normals.T  # (N, 6)
        max_proj_per_normal = np.max(pos_projs, axis=0) + h_vals  # (6,)
        max_proj = np.max(max_proj_per_normal)
        
        return max_proj * 2.0 / sqrt3
    
    def get_positions(topo, scale=2.0):
        pts = []
        for q, r in topo:
            x = scale * (q + r/2.0)
            y = scale * (r * sqrt3/2.0)
            pts.append((x, y))
        # Center
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        return np.array([(x-cx, y-cy) for x, y in pts])
        
    topologies = []
    
    # Topo 1: 3x4 Parallelogram
    t1 = [(q, r) for q in range(3) for r in range(4)]
    topologies.append(t1)
    
    # Topo 2: 4x3 Parallelogram
    t2 = [(q, r) for q in range(4) for r in range(3)]
    topologies.append(t2)
    
    # Topo 3: 2x6 Parallelogram
    t3 = [(q, r) for q in range(2) for r in range(6)]
    topologies.append(t3)
    
    # Topo 4: 6x2 Parallelogram
    t4 = [(q, r) for q in range(6) for r in range(2)]
    topologies.append(t4)
    
    # Topo 5: Compact "Cross-like"
    t5 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-2,0), (0,2), (0,-2), (2,-1)
    ]
    if len(set(t5)) == 12:
        topologies.append(t5)
        
    # Topo 6: Another variation
    t6 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    if len(set(t6)) == 12:
        topologies.append(t6)

    best_S = float('inf')
    best_positions = None
    best_alpha = 0.0
    best_beta = 0.0
    
    for topo in topologies:
        if time.time() > deadline:
            break
        positions = get_positions(topo, scale=2.0)
        
        # Phase 1: Coarse grid search (step 2 degrees)
        step1 = 2.0
        best_local_S = float('inf')
        best_local_alpha = 0.0
        best_local_beta = 0.0
        
        for a in np.arange(0, 60, step1):
            if time.time() > deadline:
                break
            for b in np.arange(0, 60, step1):
                S = calc_S(positions, a, b)
                if S < best_local_S:
                    best_local_S = S
                    best_local_alpha = a
                    best_local_beta = b
        
        # Phase 2: Fine grid search around best (step 0.1 degrees, range ±2)
        step2 = 0.1
        range2 = 2.0
        for a in np.arange(best_local_alpha - range2, best_local_alpha + range2 + step2, step2):
            if time.time() > deadline:
                break
            for b in np.arange(best_local_beta - range2, best_local_beta + range2 + step2, step2):
                S = calc_S(positions, a, b)
                if S < best_local_S:
                    best_local_S = S
                    best_local_alpha = a
                    best_local_beta = b
        
        if best_local_S < best_S:
            best_S = best_local_S
            best_positions = positions.copy()
            best_alpha = best_local_alpha
            best_beta = best_local_beta
            
    # Phase 3: Nelder-Mead refinement on alpha and beta
    if best_positions is not None and time.time() < deadline - 15:
        positions = best_positions
        
        def obj_ab(params):
            a, b = params
            return calc_S(positions, a, b)
            
        try:
            res = minimize(obj_ab, [best_alpha, best_beta], method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_S:
                best_S = res.fun
                best_alpha = res.x[0]
                best_beta = res.x[1]
        except:
            pass
        
        # Phase 4: Optimize positions too (all 2N + 2 = 26 parameters)
        if time.time() < deadline - 5:
            pos_flat = best_positions.flatten()
            init_params = np.concatenate([pos_flat, [best_alpha, best_beta]])
            
            def obj_full(params):
                pos = params[:2*N].reshape(N, 2)
                a = params[2*N]
                b = params[2*N+1]
                return calc_S(pos, a, b)
            
            try:
                res2 = minimize(obj_full, init_params, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                if res2.fun < best_S:
                    best_S = res2.fun
                    best_positions = res2.x[:2*N].reshape(N, 2)
                    best_alpha = res2.x[2*N]
                    best_beta = res2.x[2*N+1]
            except:
                pass

    # Extract final result
    inner = []
    for i in range(N):
        x, y = best_positions[i]
        inner.append([x, y, best_alpha])
        
    return inner, (0.0, 0.0), best_S, best_beta
