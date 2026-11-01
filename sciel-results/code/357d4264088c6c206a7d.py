import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Precompute for aligned hexagons (angle 0)
# h(phi) for unit hex
# g(phi) for unit side hex shape (support of container / S)
# Both are the same function!
# h(phi) = sqrt(3)/2 * cos(delta)
# g(phi) = sqrt(3)/2 * cos(delta)
# So R(phi) = 1.
# Wait, if R(phi) = 1, then S_i = max (1 + c_i dot u_phi / g(phi)).
# This means the container must be at least as large as the hexagon's "radius" in that direction plus the projection of the center.

def construct():
    # Precompute phi points
    n_phi = 36
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    u_phi = np.column_stack((np.cos(phi), np.sin(phi)))
    
    # Support function of unit hex (angle 0)
    # delta = (phi - pi/6) % (pi/3) - pi/6
    delta = (phi - np.pi/6) % (np.pi/3) - np.pi/6
    h_hex = (np.sqrt(3)/2) * np.cos(delta)
    
    # g(phi) is the same as h_hex for a unit side hexagon?
    # Container side S. h_cont = S * h_hex.
    # So S >= (h_hex + c dot u) / h_hex = 1 + (c dot u) / h_hex.
    
    # Precompute 1/h_hex
    inv_h_hex = 1.0 / h_hex
    
    def get_S(centers):
        # centers: (N, 2)
        # c dot u: (N, n_phi)
        c_dot_u = centers @ u_phi.T
        # term: c_dot_u / h_hex
        terms = c_dot_u * inv_h_hex
        # max over phi for each center
        S_per_center = 1.0 + np.max(terms, axis=1)
        return np.max(S_per_center)

    def objective(x):
        centers = x.reshape(N, 2)
        S = get_S(centers)
        
        # Overlap penalty
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = centers[j] - centers[i]
                dist = np.linalg.norm(d)
                if dist < 2.0:
                    pen += (2.0 - dist)**2
        return S + pen

    # Initial guess
    r = np.sqrt(3)
    x0 = np.zeros(N*2)
    x0[0:2] = [0,0]
    for k in range(6):
        angle = k * np.pi / 3
        x0[(k+1)*2:(k+2)*2] = [r * np.cos(angle), r * np.sin(angle)]
    # 4 more
    x0[14:16] = [2*r, 0]
    x0[16:18] = [2*r * np.cos(np.pi/3), 2*r * np.sin(np.pi/3)]
    x0[18:20] = [2*r * np.cos(2*np.pi/3), 2*r * np.sin(2*np.pi/3)]
    x0[20:22] = [-2*r, 0]
    
    start_time = time.time()
    
    # Optimize
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-3, 'fatol': 1e-3})
    
    best_x = res.x
    best_score = res.fun
    
    # Check time
    if time.time() - start_time > 100:
        pass
        
    centers_opt = best_x.reshape(N, 2)
    S_final = get_S(centers_opt)
    
    inner = []
    for i in range(N):
        inner.append([float(centers_opt[i][0]), float(centers_opt[i][1]), 0.0])
        
    return inner, (0.0, 0.0), float(S_final), 0.0

# EVOLVE-BLOCK-START
# (Code above)
# EVOLVE-BLOCK-END
