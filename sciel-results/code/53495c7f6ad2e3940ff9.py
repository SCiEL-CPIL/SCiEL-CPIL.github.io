"""
Hexagon packing: 12 unit hexagons in a regular hexagon, minimize outer side length.
Strategy: Use a "caging" formulation where we optimize the convex hull of all inner
hexagon vertices. Use Nelder-Mead on a compact initial configuration, plus simulated
annealing for refinement.
"""
import numpy as np
import time
from itertools import combinations

N = 12
DEADLINE = 135.0  # Safety margin before 150s kill

# Unit hexagon vertices (flat-top, side length 1)
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

# Outward normals for the 6 edges of a regular hexagon
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])


def get_all_vertices(params):
    """
    params: (N, 3) array of [x, y, angle_deg]
    Returns: (N, 6, 2) array of vertices
    """
    n = len(params)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        x, y, a_deg = params[i]
        a_rad = np.radians(a_deg)
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        verts[i] = UNIT_VERTS @ R.T + np.array([x, y])
    return verts


def required_outer_side(params, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    verts = get_all_vertices(np.array(params, dtype=float))
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    for i in range(len(params)):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        max_extent = max(max_extent, np.max(max_proj))
    
    return max_extent / (np.sqrt(3) / 2)


def min_separation(params):
    """
    Compute the minimum separation between any pair of inner hexagons using SAT.
    Returns the minimum clearance (positive means no overlap, negative means overlap).
    """
    verts = get_all_vertices(np.array(params, dtype=float))
    n = verts.shape[0]
    min_sep = float('inf')
    
    for i in range(n):
        v1 = verts[i]
        for j in range(i + 1, n):
            v2 = verts[j]
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            norms = np.linalg.norm(axes, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            axes = axes / norms
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            min_sep = min(min_sep, np.max(sep))
    
    return min_sep


def check_validity(params, outer_center, outer_side, outer_angle, tol=1e-9):
    """Check if packing is valid."""
    verts = get_all_vertices(np.array(params, dtype=float))
    n = verts.shape[0]
    
    # Check pairwise separation
    for i in range(n):
        v1 = verts[i]
        for j in range(i + 1, n):
            v2 = verts[j]
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            norms = np.linalg.norm(axes, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            axes = axes / norms
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            if np.max(sep) < -tol:
                return False
    
    # Check containment
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    for i in range(n):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        
        if np.max(max_proj) > apothem + tol:
            return False
    
    return True


def initial_config_hexagonal():
    """
    Create a compact 3x4 honeycomb-like arrangement.
    For flat-top hexagons, adjacent centers are at distance sqrt(3).
    Lattice vectors: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 3/2)
    """
    a1 = np.array([np.sqrt(3), 0])
    a2 = np.array([np.sqrt(3) / 2, 1.5])
    
    # 3 columns x 4 rows staggered pattern
    positions = []
    for row in range(4):
        for col in range(3):
            # Stagger odd rows
            offset_x = 0 if row % 2 == 0 else np.sqrt(3) / 2
            pos = col * a1 + row * a2 + np.array([offset_x, 0])
            positions.append(pos)
    
    # Center the configuration
    centers = np.array(positions)
    centroid = centers.mean(axis=0)
    centers -= centroid
    
    params = np.zeros((N, 3))
    for i in range(N):
        params[i, 0] = centers[i, 0]
        params[i, 1] = centers[i, 1]
        params[i, 2] = 0.0
    
    return params


def initial_config_circular():
    """Circular arrangement of 12 hexagons."""
    params = np.zeros((N, 3))
    radius = 2.0
    for i in range(N):
        theta = 2 * np.pi * i / N
        params[i, 0] = radius * np.cos(theta)
        params[i, 1] = radius * np.sin(theta)
        params[i, 2] = 0.0
    return params


def nelder_mead_optimize(params, max_iter=500, tol=1e-6):
    """
    Nelder-Mead optimization to minimize outer side length subject to non-overlap.
    Uses a penalty function for overlaps.
    """
    def objective(x):
        x = np.array(x).reshape(N, 3)
        side = required_outer_side(x)
        sep = min_separation(x)
        if sep < 0:
            penalty = -sep * 100.0
            return side + penalty
        return side
    
    # Nelder-Mead implementation
    n_params = N * 3
    x0 = params.flatten()
    
    # Initial simplex
    simplex = [x0.copy()]
    step = 0.1
    for i in range(n_params):
        xi = x0.copy()
        xi[i] += step
        simplex.append(xi)
    
    # Objective values
    f_values = [objective(s) for s in simplex]
    
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5   # contraction
    sigma = 0.5 # shrink
    
    for iteration in range(max_iter):
        # Sort simplex
        order = np.argsort(f_values)
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]
        
        # Check convergence
        if abs(f_values[-1] - f_values[0]) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflect
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < f_values[0]:
            # Try expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                f_values[-1] = fe
            else:
                simplex[-1] = xr
                f_values[-1] = fr
        elif fr < f_values[-2]:
            simplex[-1] = xr
            f_values[-1] = fr
        else:
            # Try contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < f_values[-1]:
                simplex[-1] = xc
                f_values[-1] = fc
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_values[i] = objective(simplex[i])
    
    # Return best point
    best_idx = np.argmin(f_values)
    best_params = np.array(simplex[best_idx]).reshape(N, 3)
    best_side = required_outer_side(best_params)
    
    return best_params, best_side


def simulated_annealing(params, time_budget=30.0):
    """
    Simulated annealing to refine the packing.
    """
    start_time = time.time()
    best_params = params.copy()
    best_side = required_outer_side(params)
    
    current_params = params.copy()
    current_side = best_side
    
    T = 0.5  # initial temperature
    cooling_rate = 0.995
    
    np.random.seed(42)
    
    while time.time() - start_time < time_budget:
        # Generate a random move
        move_type = np.random.choice(['translate', 'rotate', 'both'])
        
        if move_type == 'translate':
            idx = np.random.randint(N)
            dx = np.random.normal(0, 0.1)
            dy = np.random.normal(0, 0.1)
            new_params = current_params.copy()
            new_params[idx, 0] += dx
            new_params[idx, 1] += dy
        elif move_type == 'rotate':
            idx = np.random.randint(N)
            da = np.random.normal(0, 2.0)
            new_params = current_params.copy()
            new_params[idx, 2] += da
        else:
            idx1 = np.random.randint(N)
            idx2 = np.random.randint(N)
            if idx1 == idx2:
                continue
            dx1 = np.random.normal(0, 0.05)
            dy1 = np.random.normal(0, 0.05)
            dx2 = np.random.normal(0, 0.05)
            dy2 = np.random.normal(0, 0.05)
            new_params = current_params.copy()
            new_params[idx1, 0] += dx1
            new_params[idx1, 1] += dy1
            new_params[idx2, 0] += dx2
            new_params[idx2, 1] += dy2
        
        new_side = required_outer_side(new_params)
        new_sep = min_separation(new_params)
        
        if new_sep < 0:
            # Overlap - penalize
            new_score = new_side + abs(new_sep) * 50.0
        else:
            new_score = new_side
        
        # Acceptance criterion
        if new_score < current_side or np.random.random() < np.exp(-(new_score - current_side) / max(T, 1e-10)):
            current_params = new_params
            current_side = new_side
            
            if new_sep >= 0 and new_side < best_side:
                best_params = new_params.copy()
                best_side = new_side
        
        T *= cooling_rate
    
    return best_params, best_side


def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    # Try multiple initial configurations
    initial_configs = [
        initial_config_hexagonal(),
        initial_config_circular(),
    ]
    
    # Add some random configurations
    np.random.seed(42)
    for _ in range(3):
        params = np.zeros((N, 3))
        for i in range(N):
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0.5, 3.0)
            params[i, 0] = r * np.cos(theta)
            params[i, 1] = r * np.sin(theta)
            params[i, 2] = np.random.uniform(-30, 30)
        initial_configs.append(params)
    
    for config_idx, init_params in enumerate(initial_configs):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        # Check if initial config is valid
        init_side = required_outer_side(init_params)
        init_sep = min_separation(init_params)
        
        if init_sep < 0:
            # Try to fix overlaps with a quick SA
            init_params, init_side = simulated_annealing(init_params, time_budget=5.0)
            init_sep = min_separation(init_params)
        
        # Nelder-Mead optimization
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 20:
            nm_params, nm_side = nelder_mead_optimize(init_params, max_iter=300)
            nm_sep = min_separation(nm_params)
            
            if nm_sep >= 0 and nm_side < best_score:
                best_score = nm_side
                best_params = nm_params.copy()
        
        # Simulated annealing refinement
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 10:
            sa_params, sa_side = simulated_annealing(init_params, time_budget=min(15.0, DEADLINE - 10 - elapsed))
            sa_sep = min_separation(sa_params)
            
            if sa_sep >= 0 and sa_side < best_score:
                best_score = sa_side
                best_params = sa_params.copy()
        
        # Check initial config itself
        if init_sep >= 0 and init_side < best_score:
            best_score = init_side
            best_params = init_params.copy()
    
    # Final refinement with more SA time if available
    elapsed = time.time() - start_time
    if best_score < float('inf') and elapsed < DEADLINE - 5:
        remaining = min(20.0, DEADLINE - 5 - elapsed)
        if remaining > 2:
            final_params, final_side = simulated_annealing(best_params, time_budget=remaining)
            final_sep = min_separation(final_params)
            if final_sep >= 0 and final_side < best_score:
                best_score = final_side
                best_params = final_params.copy()
    
    # Construct result
    if best_score < float('inf'):
        inner_list = [list(best_params[i]) for i in range(N)]
        # Validate
        if check_validity(inner_list, (0.0, 0.0), best_score, 0.0):
            return inner_list, (0.0, 0.0), best_score, 0.0
        else:
            # Recompute exact side length
            exact_side = required_outer_side(inner_list)
            if check_validity(inner_list, (0.0, 0.0), exact_side, 0.0):
                return inner_list, (0.0, 0.0), exact_side, 0.0
    
    # Fallback
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = required_outer_side(inner_default)
    return inner_default, (0.0, 0.0), required_side, 0.0


if __name__ == "__main__":
    result = construct()
    print(f"Best outer side length: {result[2]:.6f}")
    print(f"Number of inner hexagons: {len(result[0])}")
