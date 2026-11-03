import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
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
    
    def support_unit(n_vec):
        # n_vec is a numpy array [nx, ny]
        # Returns max dot product of unit_verts with n_vec
        return np.max(unit_verts @ n_vec)
        
    def calc_S(positions, alpha_deg, beta_deg):
        # positions: list of (x,y)
        # alpha_deg: rotation of inner hexagons
        # beta_deg: rotation of outer hexagon
        alpha_rad = math.radians(alpha_deg)
        beta_rad = math.radians(beta_deg)
        
        # Outer normals
        # Outer hexagon angle beta.
        # Normals are at 30 + beta + k*60
        max_proj = 0.0
        for k in range(6):
            normal_angle = math.radians(30) + beta_rad + k * math.radians(60)
            n = np.array([math.cos(normal_angle), math.sin(normal_angle)])
            
            # Support of inner hexagons in direction n
            # Inner hex i is at P_i, rotated by alpha.
            # Support = P_i . n + h_unit(R_-alpha n)
            
            # Rotate n by -alpha
            c = math.cos(-alpha_rad)
            s = math.sin(-alpha_rad)
            n_rot = np.array([c*n[0] - s*n[1], s*n[0] + c*n[1]])
            
            h_unit_val = support_unit(n_rot)
            
            max_p_proj = 0.0
            for x, y in positions:
                proj = x*n[0] + y*n[1] + h_unit_val
                if proj > max_p_proj:
                    max_p_proj = proj
            
            if max_p_proj > max_proj:
                max_proj = max_p_proj
                
        return max_proj * 2.0 / sqrt3

    # Define topologies
    # Axial to Cartesian: x = s(q+r/2), y = s(r*sqrt3/2)
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
        return [(x-cx, y-cy) for x, y in pts]
        
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
    
    # Topo 5: Compact "Hexagon-like"
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: Pick 5. Let's pick the ones that extend the shape evenly.
    # (2,0), (1,1), (0,2), (-1,2), (-2,1) -- this is a bit asymmetric.
    # Let's try a "Cross"
    t5 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-2,0), (0,2), (0,-2), (2,-1) # 5 more
    ]
    # Check uniqueness
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
    best_params = None
    
    for topo in topologies:
        if time.time() > deadline:
            break
        positions = get_positions(topo, scale=2.0)
        
        # Grid search for alpha, beta
        # Symmetry: period 60 degrees
        # Step 0.5 degrees
        step = 0.5
        best_local_S = float('inf')
        best_local_alpha = 0.0
        best_local_beta = 0.0
        
        for a in np.arange(0, 60, step):
            if time.time() > deadline:
                break
            for b in np.arange(0, 60, step):
                S = calc_S(positions, a, b)
                if S < best_local_S:
                    best_local_S = S
                    best_local_alpha = a
                    best_local_beta = b
        
        if best_local_S < best_S:
            best_S = best_local_S
            best_params = (positions, best_local_alpha, best_local_beta)
            
    # Refine with Nelder-Mead if time permits
    if best_params is not None and time.time() < deadline - 10:
        positions, init_a, init_b = best_params
        
        def obj(params):
            a, b = params
            return calc_S(positions, a, b)
            
        try:
            res = minimize(obj, [init_a, init_b], method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_S:
                best_S = res.fun
                best_params = (positions, res.x[0], res.x[1])
        except:
            pass

    # Extract final result
    positions, alpha, beta = best_params
    
    inner = []
    for x, y in positions:
        inner.append([x, y, alpha])
        
    return inner, (0.0, 0.0), best_S, beta
