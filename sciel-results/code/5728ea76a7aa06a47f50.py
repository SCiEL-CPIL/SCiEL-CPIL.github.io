import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = time.time() + 140.0  # safety margin before 150s

def hex_support(p, theta_o):
    """Max over 6 directions of |p . u_k| for container normals."""
    ks = np.arange(6)
    angles = theta_o + ks * np.pi / 3
    u = np.column_stack([np.cos(angles), np.sin(angles)])
    vals = np.abs(p @ u.T)
    return np.max(vals)

def container_side_for_config(centers, theta_o):
    """Min S such that all unit hexagons (angle 0) fit in container of side S, angle theta_o."""
    # |p_i . u_k| <= (S-1)*sqrt(3)/2 for all i,k
    # => S >= 1 + max_i,k |p_i . u_k| / (sqrt(3)/2) = 1 + 2/sqrt(3) * max
    max_val = 0.0
    for p in centers:
        max_val = max(max_val, hex_support(p, theta_o))
    return 1.0 + 2.0 / np.sqrt(3.0) * max_val

def total_side(centers, theta_o):
    return container_side_for_config(centers, theta_o)

def nonoverlap_penalty(centers, min_dist=2.0):
    pen = 0.0
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                pen += (min_dist - d) ** 2
    return pen

def objective(params):
    centers = params.reshape(-1, 2)
    theta_o = params[-1]
    side = total_side(centers, theta_o)
    pen = nonoverlap_penalty(centers)
    return side + 1000.0 * pen

def construct():
    best_side = float('inf')
    best_centers = None
    best_theta = 0.0
    
    # Initial guess: hexagonal packing
    # Center + 6 around + 4 more
    r = 2.0  # min distance
    init = np.array([[0,0]])
    for k in range(6):
        a = k * np.pi / 3
        init = np.vstack([init, [r*np.cos(a), r*np.sin(a)]])
    # Add 4 more in second ring
    for k in range(6):
        a = k * np.pi / 3 + np.pi/6
        init = np.vstack([init, [2*r*np.cos(a), 2*r*np.sin(a)]])
    init = init[:N]
    
    # Try multiple theta_o values
    theta_candidates = np.linspace(0, np.pi/6, 7)
    
    for theta0 in theta_candidates:
        if time.time() > DEADLINE:
            break
        x0 = np.hstack([init.flatten(), [theta0]])
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6},
                          bounds=[(-20,20)]*2*N + [(0, np.pi/6)])
            if res.success or res.fun < best_side:
                centers = res.x[:2*N].reshape(-1, 2)
                theta_o = res.x[-1]
                side = total_side(centers, theta_o)
                pen = nonoverlap_penalty(centers)
                if pen < 1e-4:
                    if side < best_side:
                        best_side = side
                        best_centers = centers
                        best_theta = theta_o
        except:
            pass
    
    if best_centers is None:
        # Fallback: linear arrangement
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Convert to output format
    inner = []
    cx, cy = best_centers.mean(axis=0)
    for i in range(N):
        x = best_centers[i][0] - cx
        y = best_centers[i][1] - cy
        inner.append([x, y, 0.0])
    
    return inner, (0.0, 0.0), best_side, best_theta * 180.0 / np.pi
