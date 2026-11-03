import numpy as np
import time
import math
import random

N = 12
START_TIME = time.time()
DEADLINE = 145.0  # Safety margin before 150s

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side, rotation."""
    angle = np.radians(angle_deg)
    vertices = []
    for k in range(6):
        a = angle + k * np.pi / 3
        x = cx + side * np.cos(a)
        y = cy + side * np.sin(a)
        vertices.append((x, y))
    return np.array(vertices)

def get_all_vertices(params):
    """Vectorized computation of all 72 vertices from params."""
    n_inner = N
    inner_params = params[:n_inner * 3]
    
    all_vertices = np.zeros((n_inner * 6, 2))
    for i in range(n_inner):
        x = inner_params[i * 3]
        y = inner_params[i * 3 + 1]
        a = inner_params[i * 3 + 2]
        verts = hexagon_vertices(x, y, 1.0, a)
        all_vertices[i*6:(i+1)*6] = verts
    
    return all_vertices

def outer_hexagon_constraint(vertices, center, side, angle_deg):
    """
    Return the minimum side length of outer hexagon (with given center, rotation)
    that contains all given vertices.
    """
    cx, cy = center
    theta = np.radians(angle_deg)
    # Edge normals are at theta + 30° + k*60°
    normals = []
    for k in range(6):
        a = theta + np.pi/6 + k * np.pi/3
        normals.append(np.array([np.cos(a), np.sin(a)]))
    normals = np.array(normals)  # shape (6, 2)
    
    # For each vertex, compute max projection onto each normal
    rel = vertices - np.array(center)
    proj = rel @ normals.T  # (M, 6)
    max_proj = proj.max()
    
    return max_proj / (np.sqrt(3) / 2)

def compute_overlap_penalty(params, beta=50.0):
    """
    Compute a smooth penalty for overlapping inner hexagons.
    Uses minimum distance between hexagon edges as a proxy.
    """
    n_inner = N
    inner_params = params[:n_inner * 3]
    
    penalty = 0.0
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            xi = inner_params[i * 3]
            yi = inner_params[i * 3 + 1]
            ai = inner_params[i * 3 + 2]
            xj = inner_params[j * 3]
            yj = inner_params[j * 3 + 1]
            aj = inner_params[j * 3 + 2]
            
            # Distance between centers
            dx = xj - xi
            dy = yj - yi
            dist = math.sqrt(dx*dx + dy*dy)
            
            # For two unit hexagons, they overlap if dist < ~1.732 (sqrt(3))
            # Use a smooth penalty that activates when dist < threshold
            threshold = 1.732  # sqrt(3)
            if dist < threshold:
                # Smooth penalty: use log-sum-exp style
                overlap = threshold - dist
                penalty += math.exp(beta * overlap)
    
    return penalty

def objective(params):
    """
    params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, cx, cy, outer_angle]
    Returns the outer hexagon side length.
    """
    n_inner = N
    inner_params = params[:n_inner * 3]
    cx = params[n_inner * 3]
    cy = params[n_inner * 3 + 1]
    outer_angle = params[n_inner * 3 + 2]
    
    all_vertices = get_all_vertices(params)
    
    return outer_hexagon_constraint(all_vertices, (cx, cy), 1.0, outer_angle)

def smooth_objective(params, beta=10.0):
    """
    Smooth surrogate using log-sum-exp for the max projection.
    """
    n_inner = N
    inner_params = params[:n_inner * 3]
    cx = params[n_inner * 3]
    cy = params[n_inner * 3 + 1]
    outer_angle = params[n_inner * 3 + 2]
    
    theta = np.radians(outer_angle)
    normals = []
    for k in range(6):
        a = theta + np.pi/6 + k * np.pi/3
        normals.append(np.array([np.cos(a), np.sin(a)]))
    normals = np.array(normals)  # shape (6, 2)
    
    all_vertices = get_all_vertices(params)
    rel = all_vertices - np.array([cx, cy])
    proj = rel @ normals.T  # (72, 6)
    
    # Log-sum-exp for smooth max
    proj_shifted = proj * beta
    max_proj_shifted = np.max(proj_shifted, axis=0)
    log_sum_exp = max_proj_shifted + np.log(np.sum(np.exp(proj_shifted - max_proj_shifted), axis=0))
    smooth_max_proj = log_sum_exp / beta
    
    # Add overlap penalty
    overlap_pen = compute_overlap_penalty(params, beta=20.0)
    
    side = smooth_max_proj / (np.sqrt(3) / 2)
    
    return side + 0.1 * overlap_pen

def initial_config_grid():
    """Create a grid-like initial configuration."""
    params = []
    dx = np.sqrt(3)
    dy = 1.5
    
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            positions.append((x, y, 0.0))
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    for i, (x, y, a) in enumerate(positions):
        params.append(x - cx)
        params.append(y - cy)
        params.append(a)
    
    params.append(0.0)
    params.append(0.0)
    params.append(0.0)
    
    return np.array(params)

def initial_config_circular():
    """Create a circular initial configuration."""
    circ_params = []
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 3.0
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        circ_params.extend([x, y, 0.0])
    circ_params.extend([0.0, 0.0, 0.0])
    return np.array(circ_params)

def initial_config_rings():
    """Create a two-ring initial configuration."""
    compact_params = []
    for i in range(6):
        angle = 2 * np.pi * i / 6
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        compact_params.extend([x, y, 0.0])
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        compact_params.extend([x, y, 0.0])
    compact_params.extend([0.0, 0.0, 0.0])
    return np.array(compact_params)

def initial_config_flower():
    """Create a flower-like configuration: 1 center, 6 around, 5 more."""
    params = []
    # Center hexagon
    params.extend([0.0, 0.0, 0.0])
    
    # 6 hexagons around the center
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = np.sqrt(3)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        params.extend([x, y, 0.0])
    
    # 5 more hexagons in a larger ring
    for i in range(5):
        angle = 2 * np.pi * i / 5 + np.pi / 5
        r = 2 * np.sqrt(3)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        params.extend([x, y, 0.0])
    
    params.extend([0.0, 0.0, 0.0])
    return np.array(params)

def initial_config_compact_hex():
    """Create a compact hexagonal cluster configuration."""
    params = []
    # Hexagonal close-packed: 1 center, 6 first ring, 5 second ring
    positions = [(0.0, 0.0)]
    
    # First ring: 6 hexagons
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = np.sqrt(3)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        positions.append((x, y))
    
    # Second ring: 5 hexagons (to make 12 total)
    for i in range(5):
        angle = 2 * np.pi * i / 5 + np.pi / 5
        r = 2 * np.sqrt(3)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        positions.append((x, y))
    
    for x, y in positions:
        params.extend([x, y, 0.0])
    
    params.extend([0.0, 0.0, 0.0])
    return np.array(params)

def construct():
    from scipy.optimize import minimize
    
    best_params = None
    best_score = float('inf')
    
    # Generate multiple initial configurations
    configs = [
        initial_config_grid(),
        initial_config_circular(),
        initial_config_rings(),
        initial_config_flower(),
        initial_config_compact_hex()
    ]
    
    # Add some random perturbations
    for base_config in configs[:2]:
        if time.time() > DEADLINE:
            break
        for _ in range(3):
            perturbed = base_config + np.random.randn(len(base_config)) * 0.5
            configs.append(perturbed)
    
    # Stage 1: Coarse optimization with smooth objective
    for config in configs:
        if time.time() > DEADLINE:
            break
        
        try:
            result = minimize(
                smooth_objective,
                config,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 1000, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            # Evaluate with true objective
            true_score = objective(result.x)
            if true_score < best_score:
                best_score = true_score
                best_params = result.x.copy()
                
        except Exception as e:
            pass
    
    # Stage 2: Fine optimization with true objective from best results
    if best_params is not None and time.time() < DEADLINE:
        try:
            result = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            score = result.fun
            if score < best_score:
                best_score = score
                best_params = result.x.copy()
                
        except Exception as e:
            pass
    
    # Stage 3: Local search with random restarts from best
    if best_params is not None and time.time() < DEADLINE:
        for _ in range(5):
            if time.time() > DEADLINE:
                break
            
            perturbed = best_params + np.random.randn(len(best_params)) * 0.1
            try:
                result = minimize(
                    objective,
                    perturbed,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                score = result.fun
                if score < best_score:
                    best_score = score
                    best_params = result.x.copy()
                    
            except Exception as e:
                pass
    
    if best_params is None:
        # Fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
    
    # Extract results
    inner = []
    for i in range(N):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        a = np.degrees(best_params[i * 3 + 2])
        inner.append([x, y, a])
    
    cx = best_params[N * 3]
    cy = best_params[N * 3 + 1]
    outer_angle = np.degrees(best_params[N * 3 + 2])
    
    side = best_score
    
    # Verify and clamp side to be reasonable
    if side < 1.0:
        side = 1.0
    
    return inner, (cx, cy), side, outer_angle
