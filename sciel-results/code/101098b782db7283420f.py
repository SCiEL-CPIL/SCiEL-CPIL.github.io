"""Rewritten: co-optimization of 11 hexagons + outer hexagon orientation."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(side, cx, cy, angle_rad):
    """Return (6,2) vertices of a regular hexagon with given side, center, rotation."""
    angles = angle_rad + np.arange(6) * (math.pi / 3.0)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _hex_in_hex_halfplane(inner_verts, outer_side, outer_cx, outer_cy, outer_angle):
    """
    Check if inner hexagon (given as 6x2 vertices) is inside outer hexagon.
    Returns max violation (negative = inside with margin, positive = violates).
    Outer hexagon: side length S, center (cx,cy), rotation angle.
    Constraint: for each of the 6 edge-normals of outer hex,
      max_v (v . n_k) <= outer_cx_apothem_offset ... 
    Actually: outer hexagon defined by {p : |p - c|_hex <= S} where hex norm uses 6 normals.
    Apothem of outer = S * sqrt(3)/2.
    Inner hex has apothem = 1 * sqrt(3)/2.
    For containment: for each normal direction n_k of outer hex (pointing outward from center),
      max over inner vertices v of (v - c_outer) . n_k <= apothem_outer - apothem_inner
    This is the Minkowski difference condition.
    """
    apothem_outer = outer_side * math.sqrt(3) / 2.0
    apothem_inner = math.sqrt(3) / 2.0  # side=1
    margin = apothem_outer - apothem_inner
    if margin <= 0:
        return float('inf')  # outer too small
    
    # 6 outward normals of outer hexagon
    k_angles = outer_angle + np.arange(6) * (math.pi / 3.0)
    normals = np.column_stack([np.cos(k_angles), np.sin(k_angles)])  # (6,2)
    
    # Relative positions
    rel = inner_verts - np.array([outer_cx, outer_cy])  # (6,2)
    
    # Projections: (6,6)
    projs = rel @ normals.T
    max_proj = np.max(projs, axis=0)  # (6,)
    
    # Violation: max_proj[k] - margin; we want all <= 0
    violations = max_proj - margin
    return float(np.max(violations))

def _objective(params):
    """
    params: [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10, cx, cy, S, phi]
    34 parameters.
    Objective: S (outer side length), with penalty if any inner hex not inside.
    """
    # Parse
    inner = params[:3*N].reshape(N, 3)
    cx, cy, S, phi = params[3*N], params[3*N+1], params[3*N+2], params[3*N+3]
    
    if S < 1.0:
        return 1e6  # S must be at least 1 (one hexagon)
    
    # Compute containment violation
    max_violation = 0.0
    for i in range(N):
        xi, yi, ai = inner[i]
        verts = _hex_vertices(1.0, xi, yi, ai)
        viol = _hex_in_hex_halfplane(verts, S, cx, cy, phi)
        if viol > max_violation:
            max_violation = viol
    
    if max_violation > 0:
        # Penalty: S + large * violation
        return S + 100.0 * max_violation
    return S

def _make_initial_config():
    """
    Generate initial honeycomb-like configuration.
    11 hexagons: 1 center + 6 ring1 + 4 from ring2.
    """
    d = math.sqrt(3)  # distance between adjacent hex centers in honeycomb
    centers = [(0.0, 0.0)]
    
    # Ring 1: 6 neighbors
    for i in range(6):
        angle = i * math.pi / 3
        centers.append((d * math.cos(angle), d * math.sin(angle)))
    
    # Ring 2: pick 4 from the 12
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2.append((x, y))
    
    # Pick 4 symmetrically: e.g., at angles 0, 90, 180, 270 equivalent
    # ring2 has 12 points; pick every 3rd for symmetry
    picks = [0, 3, 6, 9]
    for idx in picks:
        centers.append(ring2[idx])
    
    # All angles = 0 initially
    params = []
    for cx, cy in centers:
        params.extend([cx, cy, 0.0])
    
    # Outer hex: estimate S
    # Max extent from origin
    max_r = max(math.sqrt(cx**2 + cy**2) for cx, cy in centers)
    S_est = (2.0 / math.sqrt(3)) * (max_r + 1.0)  # rough
    params.extend([0.0, 0.0, S_est, 0.0])
    
    return np.array(params)

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 15.0
    
    # Initial configuration
    x0 = _make_initial_config()
    
    # Evaluate initial
    best_S = _objective(x0)
    best_params = x0.copy()
    
    # Multi-start Nelder-Mead
    n_starts = 12
    np.random.seed(42)
    
    for start_idx in range(n_starts):
        if time.time() > deadline - safety_margin:
            break
        
        # Perturb initial config
        if start_idx == 0:
            x_try = x0.copy()
        else:
            x_try = x0.copy()
            # Random perturbations
            noise_pos = 0.5
            noise_angle = math.pi / 12
            for i in range(N):
                x_try[3*i] += np.random.uniform(-noise_pos, noise_pos)
                x_try[3*i+1] += np.random.uniform(-noise_pos, noise_pos)
                x_try[3*i+2] += np.random.uniform(-noise_angle, noise_angle)
            # Perturb outer
            x_try[3*N+0] += np.random.uniform(-0.3, 0.3)  # cx
            x_try[3*N+1] += np.random.uniform(-0.3, 0.3)  # cy
            x_try[3*N+2] *= (1.0 + np.random.uniform(-0.1, 0.1))  # S
            x_try[3*N+3] += np.random.uniform(-math.pi/6, math.pi/6)  # phi
        
        try:
            res = minimize(_objective, x_try, method='Nelder-Mead',
                           options={'xatol': 1e-6, 'fatol': 1e-6, 
                                    'maxiter': 5000, 'maxfev': 10000})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Refine best with tighter tolerances
    if time.time() < deadline - safety_margin:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-8,
                                    'maxiter': 10000, 'maxfev': 20000})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Extract final configuration
    inner_params = best_params[:3*N].reshape(N, 3)
    cx, cy, S, phi = best_params[3*N], best_params[3*N+1], best_params[3*N+2], best_params[3*N+3]
    
    # Build output
    inner = []
    for i in range(N):
        xi, yi, ai = inner_params[i]
        inner.append([float(xi), float(yi), float(math.degrees(ai))])
    
    return inner, (float(cx), float(cy)), float(S), float(math.degrees(phi))

# EVOLVE-BLOCK-END
