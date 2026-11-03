import numpy as np
import scipy.optimize as opt
import time

N = 12

def hex_vertices(cx, cy, angle, side=1.0):
    """
    Returns vertices of a regular hexagon.
    angle is in radians.
    Vertices are at angles angle, angle+pi/3, ...
    """
    angles = angle + np.arange(6) * np.pi / 3
    return np.array([[cx + side * np.cos(a), cy + side * np.sin(a)] for a in angles])

def support_hex(alpha, angle=0.0):
    """
    Support function of a unit regular hexagon rotated by 'angle'.
    alpha is the direction angle.
    """
    # Normals to edges are at angle + pi/6 + k*pi/3
    # But h(alpha) = max cos(alpha - vertex_angle)
    # Vertex angles are angle + k*pi/3
    vertex_angles = angle + np.arange(6) * np.pi / 3
    return np.max(np.cos(alpha - vertex_angles))

def compute_D(phi):
    """
    Compute the minimum distance from origin to the boundary of the Minkowski sum
    of two unit hexagons with relative angle phi.
    This is the touching distance.
    """
    # Normalize phi to [0, pi/6] due to symmetry
    phi = np.mod(phi, np.pi / 3)
    if phi > np.pi / 6:
        phi = np.pi / 3 - phi
        
    # Normals of H0: pi/6 + k*pi/3
    # Normals of Hphi: pi/6 + phi + k*pi/3
    # Union of these 12 normals
    normals = np.unique(np.concatenate([
        np.pi/6 + np.arange(6)*np.pi/3,
        np.pi/6 + phi + np.arange(6)*np.pi/3
    ]))
    
    min_dist = float('inf')
    for n in normals:
        h0 = support_hex(n, 0.0)
        hphi = support_hex(n, phi)
        dist = h0 + hphi
        if dist < min_dist:
            min_dist = dist
    return min_dist

# Precompute D for a range of angles to speed up constraint evaluation?
# Or just compute on the fly. It's fast enough.
# Let's create a lookup table for speed in the constraint function.
PHI_RES = 180 # 1 degree resolution
PHI_MAX = np.pi / 6
PHI_TABLE = np.linspace(0, PHI_MAX, PHI_RES + 1)
D_TABLE = np.array([compute_D(p) for p in PHI_TABLE])

def get_D(phi):
    phi = np.mod(phi, np.pi / 3)
    if phi > np.pi / 6:
        phi = np.pi / 3 - phi
    # Interpolate
    idx = np.searchsorted(PHI_TABLE, phi)
    if idx == 0:
        return D_TABLE[0]
    if idx == len(PHI_TABLE):
        return D_TABLE[-1]
    # Linear interpolation
    p0, p1 = PHI_TABLE[idx-1], PHI_TABLE[idx]
    d0, d1 = D_TABLE[idx-1], D_TABLE[idx]
    if p1 == p0:
        return d0
    return d0 + (d1 - d0) * (phi - p0) / (p1 - p0)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0 # 120s limit, leave margin for 150s kill
    
    # Initial guess
    # 1 center, 6 around, 5 outer
    x0 = [0.0]
    y0 = [0.0]
    th0 = [0.0]
    
    for k in range(6):
        angle = k * np.pi / 3
        x0.append(np.sqrt(3) * np.cos(angle))
        y0.append(np.sqrt(3) * np.sin(angle))
        th0.append(0.0)
        
    # 5 more, placed further out
    for k in range(5):
        angle = (k + 0.5) * np.pi / 3
        x0.append(2.5 * np.cos(angle))
        y0.append(2.5 * np.sin(angle))
        th0.append(0.0)
        
    W0 = 4.0
    x_init = x0 + y0 + th0 + [W0]
    
    # Variables: x1, y1, th1, ..., x12, y12, th12, W
    # Indices:
    # 0..35: inner hexes
    # 36: W
    
    def objective(vars):
        return vars[36]
        
    def constraints(vars):
        cons = []
        W = vars[36]
        if W < 1.0:
            # Penalize small W
            cons.append({'type': 'ineq', 'fun': lambda v: v[36] - 1.0})
            return cons
        
        # Extract positions and angles
        pos = []
        angles = []
        for i in range(N):
            x = vars[3*i]
            y = vars[3*i+1]
            th = vars[3*i+2]
            pos.append((x, y))
            angles.append(th)
            
        # Pairwise constraints
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                d_req = get_D(angles[i] - angles[j])
                # dist >= d_req
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: np.sqrt((v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2) - get_D(v[3*i+2]-v[3*j+2])
                })
                
        # Containment constraints
        # Outer hex normals
        outer_normals = [np.pi/6 + k*np.pi/3 for k in range(6)]
        outer_dist = W * np.sqrt(3) / 2
        
        for i in range(N):
            x = vars[3*i]
            y = vars[3*i+1]
            th = vars[3*i+2]
            verts = hex_vertices(x, y, th)
            for v in verts:
                vx, vy = v
                for n in outer_normals:
                    # vx*cos(n) + vy*sin(n) <= outer_dist
                    # -vx*cos(n) - vy*sin(n) <= outer_dist
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda v, i=i, vx=vx, vy=vy, n=n: (v[36] * np.sqrt(3) / 2) - (vx*np.cos(n) + vy*np.sin(n))
                    })
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda v, i=i, vx=vx, vy=vy, n=n: (v[36] * np.sqrt(3) / 2) - (-vx*np.cos(n) - vy*np.sin(n))
                    })
                    
        return cons

    # Run optimization
    best_result = None
    best_W = W0
    
    # Try SLSQP
    try:
        result = opt.minimize(objective, x_init, method='SLSQP', constraints=constraints, options={'maxiter': 200, 'ftol': 1e-6})
        if result.success and result.fun < best_W:
            best_W = result.fun
            best_result = result.x
    except Exception as e:
        pass
        
    # If time remains, try COBYLA or restart
    if time.time() < deadline - 30.0 and best_W > 3.5:
        # Try with a different initial guess
        x_init2 = x_init.copy()
        # Perturb
        np.random.seed(42)
        for i in range(36):
            x_init2[i] += np.random.normal(0, 0.1)
        try:
            result2 = opt.minimize(objective, x_init2, method='SLSQP', constraints=constraints, options={'maxiter': 200, 'ftol': 1e-6})
            if result2.success and result2.fun < best_W:
                best_W = result2.fun
                best_result = result2.x
        except Exception:
            pass

    if best_result is None:
        best_result = x_init
        best_W = W0

    # Extract final values
    W_final = best_result[36]
    inner = []
    for i in range(N):
        x = best_result[3*i]
        y = best_result[3*i+1]
        th_deg = np.degrees(best_result[3*i+2])
        inner.append([x, y, th_deg])
        
    return inner, (0.0, 0.0), W_final, 0.0

# EVOLVE-BLOCK-END
