# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Return (6,2) array of vertices for a regular hexagon with side d."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _hex_edge_normals(vertices):
    """Return (6,2) outward unit edge normals for a CCW hexagon."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return normals / norms


def _min_enclosing_hex(params):
    """
    Compute minimum enclosing regular hexagon side length and optimal angle.
    params: (36,) = [x1..x12, y1..y12, a1..a12(deg)]
    Returns (s, outer_angle_deg)
    """
    n = 12
    positions = np.column_stack([params[0:n], params[n:2*n]])
    angles = params[2*n:3*n]
    
    all_verts = np.vstack([_hex_vertices(positions[i,0], positions[i,1], 1.0, angles[i]) for i in range(n)])
    
    best_s = np.inf
    best_theta = 0.0
    for theta_deg in np.linspace(0, 60, 181):
        theta = np.radians(theta_deg)
        normal_angles = theta + np.pi/6 + np.arange(6) * np.pi/3
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        projections = all_verts @ normals.T
        max_proj = np.max(projections, axis=0)
        s_candidate = (2.0 / np.sqrt(3.0)) * np.max(max_proj)
        if s_candidate < best_s:
            best_s = s_candidate
            best_theta = theta_deg
    
    return best_s, best_theta


def _separation_violation(params):
    """Compute max overlap between any pair of inner hexagons using SAT."""
    n = 12
    positions = np.column_stack([params[0:n], params[n:2*n]])
    angles = params[2*n:3*n]
    
    verts_list = [_hex_vertices(positions[i,0], positions[i,1], 1.0, angles[i]) for i in range(n)]
    normals_list = [_hex_edge_normals(v) for v in verts_list]
    
    max_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            axes = np.vstack([normals_list[i], normals_list[j]])
            proj_i = verts_list[i] @ axes.T
            proj_j = verts_list[j] @ axes.T
            overlap = np.minimum(np.max(proj_i, axis=0), np.max(proj_j, axis=0)) - \
                      np.maximum(np.min(proj_i, axis=0), np.min(proj_j, axis=0))
            pos_overlap = np.maximum(overlap, 0.0)
            max_viol = max(max_viol, np.max(pos_overlap))
    
    return max_viol


def _objective(params):
    """Minimize enclosing hex side + penalty for overlap."""
    s, _ = _min_enclosing_hex(params)
    sep = _separation_violation(params)
    return s + 100.0 * sep * sep


def _honeycomb_initial():
    """Generate initial 3-4-3 honeycomb positions."""
    a = 1.0
    positions = []
    angles = []
    
    # 3-4-3 arrangement (flat-top)
    row_data = [
        (0, [0, 3, 6]),
        (1, [1.5, 4.5, 7.5, 10.5]),
        (2, [0, 3, 6]),
    ]
    for row_idx, (row, xs) in enumerate(row_data):
        y = row * a * np.sqrt(3)
        for x in xs:
            positions.append([x, y])
            angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    center = np.mean(positions, axis=0)
    positions = positions - center
    
    return np.concatenate([positions.flatten(), angles])


def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    n = 12
    params0 = _honeycomb_initial()
    
    best_params = params0.copy()
    best_s, _ = _min_enclosing_hex(params0)
    best_obj = _objective(params0)
    
    np.random.seed(42)
    
    # Phase 1: Multi-start Nelder-Mead
    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            p = params0.copy()
        else:
            p = best_params + np.random.normal(0, 0.3, size=36)
        
        result = minimize(
            _objective, p, method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x.copy()
    
    # Phase 2: Local refinement with smaller steps
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        p = best_params + np.random.normal(0, 0.05, size=36)
        result = minimize(
            _objective, p, method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x.copy()
    
    # Extract results
    best_s, best_theta = _min_enclosing_hex(best_params)
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_params[i]
        inner_hex_data[i, 1] = best_params[n + i]
        inner_hex_data[i, 2] = best_params[2*n + i]
    
    outer_hex_data = np.array([0.0, 0.0, best_theta])
    
    return inner_hex_data, outer_hex_data, best_s


# EVOLVE-BLOCK-END
