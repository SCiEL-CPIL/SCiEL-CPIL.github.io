import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = time.time() + 140.0

def hex_vertices(cx, cy, s, angle_deg):
    angle = np.radians(angle_deg)
    return np.array([(cx + s * np.cos(angle + i * np.pi / 3),
                      cy + s * np.sin(angle + i * np.pi / 3)) for i in range(6)])

def min_enclosing_hex_side(inner, center, angle_deg):
    cx, cy = center
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    max_inr = 0.0
    for hx, hy, hangle in inner:
        verts = hex_vertices(hx, hy, 1.0, hangle)
        for v in verts:
            dx, dy = v[0] - cx, v[1] - cy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for i in range(6):
                theta = i * np.pi / 3
                d = lx * np.cos(theta) + ly * np.sin(theta)
                if d > max_inr:
                    max_inr = d
    return max_inr * 2.0 / np.sqrt(3)

def objective(params):
    n = N
    inner = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(n)]
    cx, cy, outer_angle = params[3*n], params[3*n+1], params[3*n+2]
    side = min_enclosing_hex_side(inner, (cx, cy), outer_angle)
    penalty = 0.0
    min_dist = np.sqrt(3)
    for i in range(n):
        for j in range(i + 1, n):
            dx = inner[j][0] - inner[i][0]
            dy = inner[j][1] - inner[i][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    return side + 10.0 * penalty

def construct():
    start_time = time.time()
    sqrt3 = np.sqrt(3)
    inner_init = [[0.0, 0.0, 0.0]]
    for i in range(6):
        theta = i * np.pi / 3
        inner_init.append([sqrt3 * np.cos(theta), sqrt3 * np.sin(theta), 0.0])
    ring2_axial = [(2,0),(2,-1),(1,-2),(0,-2),(-1,-1),(-2,0),(-2,1),(-1,2),(0,2),(1,1)]
    ring2_cart = [(sqrt3*(q+r/2), 1.5*r) for q, r in ring2_axial]
    for idx in [0, 1, 8, 9]:
        x, y = ring2_cart[idx]
        inner_init.append([x, y, 0.0])
    x0 = []
    for h in inner_init:
        x0.extend(h)
    cx0 = np.mean([h[0] for h in inner_init])
    cy0 = np.mean([h[1] for h in inner_init])
    x0.extend([cx0, cy0, 0.0])
    best_params = np.array(x0)
    best_obj = objective(best_params)
    if time.time() < DEADLINE - 60:
        try:
            result = minimize(objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-7})
            if result.fun < best_obj:
                best_params = result.x
                best_obj = result.fun
        except Exception:
            pass
    final_inner = [[best_params[3*i], best_params[3*i+1], best_params[3*i+2]] for i in range(N)]
    cx, cy, outer_angle = best_params[3*N], best_params[3*N+1], best_params[3*N+2]
    side = min_enclosing_hex_side(final_inner, (cx, cy), outer_angle)
    min_dist = sqrt3
    for i in range(N):
        for j in range(i + 1, N):
            dx = final_inner[j][0] - final_inner[i][0]
            dy = final_inner[j][1] - final_inner[i][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_dist - 0.01:
                inner_fb = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
                return inner_fb, (0.0, 0.0), 40.0, 0.0
    return final_inner, (cx, cy), side, outer_angle
