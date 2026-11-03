"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, a=V0, b=V1, c=V2, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                  (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    rng = np.random.RandomState(42)
    
    # Map from 2D parameter vector z (a, b) to point p in triangle
    # p = (exp(a)*V0 + exp(b)*V1 + 1*V2) / (exp(a) + exp(b) + 1)
    def z_to_points(z):
        # z is (N, 2)
        a = z[:, 0]
        b = z[:, 1]
        ea = np.exp(a)
        eb = np.exp(b)
        s = ea + eb + 1.0
        u = ea / s
        v = eb / s
        w = 1.0 / s
        # p = u*V0 + v*V1 + w*V2
        # V0 is (0,0), so u*V0 is 0
        # p = v*V1 + w*V2
        # V1 = (1, 0), V2 = (0.5, sqrt(3)/2)
        px = v * 1.0 + w * 0.5
        py = v * 0.0 + w * (np.sqrt(3) / 2)
        return np.column_stack((px, py))

    def objective_and_grad(z):
        points = z_to_points(z)
        n = len(points)
        
        # Calculate all triangle areas and find min
        min_area = float('inf')
        min_idx = None
        # We store signed area for gradient calculation
        # Area(i,j,k) = 0.5 * det(pj-pi, pk-pi)
        
        # Precompute for speed? N=11 is small, 165 triangles.
        # We just loop.
        
        # Gradients of p w.r.t z
        # dp/da = exp(a)/S * (V0 - p)
        # dp/db = exp(b)/S * (V1 - p)
        # We need these for the chain rule.
        
        a = z[:, 0]
        b = z[:, 1]
        ea = np.exp(a)
        eb = np.exp(b)
        s = ea + eb + 1.0
        
        # V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
        # p is computed inside z_to_points, but we can recompute or return it.
        # Let's just recompute u,v,w here for gradient calc to avoid overhead of separate function calls if we inline.
        u = ea / s
        v = eb / s
        w = 1.0 / s
        px = v * 1.0 + w * 0.5
        py = w * (np.sqrt(3) / 2)
        points = np.column_stack((px, py))
        
        dp_da = np.zeros_like(points)
        dp_db = np.zeros_like(points)
        
        # V0 - p = -p
        # V1 - p = (1 - px, -py)
        
        # dp/da = (ea/s) * (V0 - p) = u * (-px, -py)
        dp_da[:, 0] = -u * px
        dp_da[:, 1] = -u * py
        
        # dp/db = (eb/s) * (V1 - p) = v * (1 - px, -py)
        dp_db[:, 0] = v * (1.0 - px)
        dp_db[:, 1] = -v * py
        
        grad_z = np.zeros_like(z)
        
        # Find min area triangle
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    # Signed area
                    # 0.5 * ( (xj - xi)*(yk - yi) - (yj - yi)*(xk - xi) )
                    dx_ji = points[j, 0] - points[i, 0]
                    dy_ji = points[j, 1] - points[i, 1]
                    dx_ki = points[k, 0] - points[i, 0]
                    dy_ki = points[k, 1] - points[i, 1]
                    
                    signed_area = 0.5 * (dx_ji * dy_ki - dy_ji * dx_ki)
                    area = abs(signed_area)
                    
                    if area < min_area:
                        min_area = area
                        min_idx = (i, j, k)
                        # Store sign for gradient
                        min_sign = np.sign(signed_area)
                        if min_sign == 0:
                            min_sign = 1.0 # Degenerate, pick arbitrary direction
        
        if min_idx is None:
            return min_area, grad_z # Should not happen
        
        i, j, k = min_idx
        pi = points[i]
        pj = points[j]
        pk = points[k]
        
        # Gradient of signed area S w.r.t p
        # dS/dpi = 0.5 * (yj - yk, xj - xk)
        # dS/dpj = 0.5 * (yk - yi, xk - xi)
        # dS/dpk = 0.5 * (yi - yj, xi - xj)
        
        dS_dpi = np.array([0.5 * (pj[1] - pk[1]), 0.5 * (pj[0] - pk[0])])
        dS_dpj = np.array([0.5 * (pk[1] - pi[1]), 0.5 * (pk[0] - pi[0])])
        dS_dpk = np.array([0.5 * (pi[1] - pj[1]), 0.5 * (pi[0] - pj[0])])
        
        # We want to minimize -Area = -|S| = -sign(S)*S
        # Grad(-|S|) = -sign(S) * Grad(S)
        
        grad_p = np.zeros_like(points)
        grad_p[i] = -min_sign * dS_dpi
        grad_p[j] = -min_sign * dS_dpj
        grad_p[k] = -min_sign * dS_dpk
        
        # Chain rule: grad_z = grad_p . J
        # J is block diagonal.
        # dL/da_m = grad_p[m] . dp_da[m]
        # dL/db_m = grad_p[m] . dp_db[m]
        
        for m in range(n):
            grad_z[m, 0] = np.dot(grad_p[m], dp_da[m])
            grad_z[m, 1] = np.dot(grad_p[m], dp_db[m])
            
        return -min_area, grad_z

    def initial_z(seed_offset=0):
        # Generate random a, b
        # To get points roughly uniformly distributed, we can sample barycentric coords
        # and invert the softmax map? 
        # Inverse: u = ea/s, v = eb/s, w = 1/s
        # u/w = ea, v/w = eb
        # a = ln(u/w), b = ln(v/w)
        
        # Sample u, v such that u>=0, v>=0, u+v<=1
        # This gives uniform distribution in barycentric space, which is uniform in area?
        # Yes, uniform in (u,v) triangle is uniform in Cartesian area.
        
        pts = []
        for _ in range(N):
            u = rng.random()
            v = rng.random() * (1 - u)
            w = 1 - u - v
            # Avoid 0 for log
            u = max(u, 1e-6)
            v = max(v, 1e-6)
            w = max(w, 1e-6)
            a = np.log(u / w)
            b = np.log(v / w)
            pts.append([a, b])
        return np.array(pts)

    best_points = None
    best_score = -float('inf')
    
    num_restarts = 100 # L-BFGS-B is fast, can do more
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        z0 = initial_z(restart)
        
        # Add some noise to z0 to explore different regions
        if restart > 0:
             z0 += rng.normal(0, 0.5, z0.shape)
             
        try:
            res = minimize(
                objective_and_grad,
                z0,
                jac=True, # Passes gradient
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
        except Exception:
            continue
            
        if time.time() > deadline - 2.0:
            break
            
        z_opt = res.x
        points_opt = z_to_points(z_opt)
        
        # Verify validity (should be inside by construction, but check just in case)
        valid = all(point_in_triangle(p) for p in points_opt)
        
        if valid:
            score = min_triangle_area(points_opt)
            if score > best_score:
                best_score = score
                best_points = points_opt
                
    if best_points is None:
        # Fallback
        z0 = initial_z(0)
        best_points = z_to_points(z0)
        
    return best_points
# EVOLVE-BLOCK-END
