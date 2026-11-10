# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _get_all_inner_vertices(positions):
    """Get all 72 vertices of 12 unit hexagons at given positions (angle 0)."""
    verts_list = []
    for i in range(len(positions)):
        cx, cy = positions[i]
        verts = _hex_vertices(cx, cy, 1.0, 0)
        verts_list.append(verts)
    return np.vstack(verts_list)

def _check_separation_fast(positions):
    """Fast check if all pairs of unit hexagons are separated using SAT."""
    n = len(positions)
    all_verts = _get_all_inner_vertices(positions)
    
    for i in range(n):
        vi = all_verts[i*6:(i+1)*6]
        for j in range(i+1, n):
            vj = all_verts[j*6:(j+1)*6]
            
            # Get edge normals for both hexagons
            for verts in [vi, vj]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k+1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    
                    proj_i = vi @ normal
                    proj_j = vj @ normal
                    
                    if proj_i.max() < proj_j.min() - 1e-6 or proj_j.max() < proj_i.min() - 1e-6:
                        break
                else:
                    continue
                break
            else:
                return False
    return True

def _min_containing_hex_side(positions, outer_center, outer_angle_deg):
    """
    Compute minimum side length s of outer hexagon (centered at outer_center, angle outer_angle_deg)
    that contains all unit hexagons at given positions.
    """
    all_verts = _get_all_inner_vertices(positions)
    
    a_rad = np.radians(outer_angle_deg)
    
    def _is_inside(s):
        outer_verts = _hex_vertices(outer_center[0], outer_center[1], s, outer_angle_deg)
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i+1) % 6]
            edge = p2 - p1
            diffs = all_verts - p1
            crosses = edge[0] * diffs[:, 1] - edge[1] * diffs[:, 0]
            if np.any(crosses < -1e-6):
                return False
        return True
    
    lo = np.sqrt(12)
    hi = 25.0
    
    for _ in range(50):
        mid = (lo + hi) / 2
        if _is_inside(mid):
            hi = mid
        else:
            lo = mid
    
    return hi

def _objective(params):
    """
    Objective: minimize outer hexagon side length.
    params: [x1,y1,...,x12,y12, cx, cy, angle_deg]
    """
    positions = params[:24].reshape(12, 2)
    outer_center = params[24:26]
    outer_angle_deg = params[26]
    
    # Check separation
    if not _check_separation_fast(positions):
        return 1e10
    
    s = _min_containing_hex_side(positions, outer_center, outer_angle_deg)
    return s

def _hex_lattice_positions():
    """Return 12 positions on a hexagonal lattice forming a compact cluster."""
    sqrt3 = np.sqrt(3)
    # Hexagonal lattice with spacing 2.5 horizontal, sqrt(3)*1.25 vertical
    # A compact arrangement of 12 hexagons
    positions = np.array([
        [0.0, 0.0],
        [-2.5, 0.0],
        [2.5, 0.0],
        [-1.25, 2.165],
        [1.25, 2.165],
        [-1.25, -2.165],
        [1.25, -2.165],
        [-3.75, 2.165],
        [3.75, 2.165],
        [-3.75, -2.165],
        [3.75, -2.165],
        [0.0, -4.330],
    ])
    return positions

def _circular_positions():
    """Return 12 positions arranged in a circle around origin."""
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    r = 2.8
    positions = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    return positions

def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, maximizing 1/outer_side_length.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    # Initial configurations to try
    init_configs = [
        _hex_lattice_positions(),
        _circular_positions(),
    ]
    
    best_s = np.inf
    best_params = None
    
    np.random.seed(42)
    
    for config_idx, init_pos in enumerate(init_configs):
        if time.time() > deadline - 60:
            break
        
        # Try multiple random restarts
        n_restarts = 8 if config_idx == 0 else 4
        
        for restart in range(n_restarts):
            if time.time() > deadline - 30:
                break
            
            if restart == 0:
                positions = init_pos.copy()
            else:
                positions = init_pos + np.random.randn(12, 2) * 0.3
            
            # Optimize outer center and angle
            outer_center = np.zeros(2)
            outer_angle_deg = 0.0
            
            # Build initial parameter vector
            params = np.concatenate([positions.flatten(), outer_center, [outer_angle_deg]])
            
            # Use Nelder-Mead optimization
            best_local_s = _objective(params)
            
            # Try Nelder-Mead with multiple initial step sizes
            for step_scale in [0.1, 0.05, 0.02]:
                if time.time() > deadline - 15:
                    break
                
                result = minimize(
                    _objective,
                    params,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'maxiter': 300,
                        'adaptive': True,
                    }
                )
                
                if result.fun < best_local_s:
                    best_local_s = result.fun
                    params = result.x
            
            # Final polish with tighter tolerances
            if time.time() > deadline - 10:
                break
                
            result = minimize(
                _objective,
                params,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'maxiter': 200,
                    'adaptive': True,
                }
            )
            
            if result.fun < best_local_s:
                best_local_s = result.fun
                params = result.x
            
            if best_local_s < best_s:
                best_s = best_local_s
                best_params = params.copy()
    
    # Extract final positions
    if best_params is None:
        # Fallback to initial lattice
        positions = _hex_lattice_positions()
        outer_center = np.zeros(2)
        outer_angle_deg = 0.0
        best_s = _min_containing_hex_side(positions, outer_center, outer_angle_deg)
    else:
        positions = best_params[:24].reshape(12, 2)
        outer_center = best_params[24:26]
        outer_angle_deg = best_params[26]
        best_s = _min_containing_hex_side(positions, outer_center, outer_angle_deg)
    
    # Construct output
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = positions[:, 0]
    inner_hex_data[:, 1] = positions[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle_deg])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
