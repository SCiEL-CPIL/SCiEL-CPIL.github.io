# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _min_outer_hex_side(inner_hex_data):
    max_side = 0.0
    sqrt3_over_2 = np.sqrt(3) / 2
    for i in range(len(inner_hex_data)):
        cx, cy, a_deg = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1.0, a_deg)
        for j in range(6):
            px, py = verts[j]
            for k in range(6):
                angle_k = np.radians(60 * k)
                nx, ny = np.cos(angle_k), np.sin(angle_k)
                dist = px * nx + py * ny
                if dist > max_side * sqrt3_over_2:
                    required_s = dist / sqrt3_over_2
                    if required_s > max_side:
                        max_side = required_s
    return max_side


def _separation(i_hex, j_hex, tol=1e-6):
    ci = _hex_vertices(i_hex[0], i_hex[1], 1.0, i_hex[2])
    cj = _hex_vertices(j_hex[0], j_hex[1], 1.0, j_hex[2])
    for poly in [ci, cj]:
        for k in range(6):
            p1 = poly[k]
            p2 = poly[(k + 1) % 6]
            ex, ey = p2[0] - p1[0], p2[1] - p1[1]
            nx, ny = -ey, ex
            length = np.sqrt(nx ** 2 + ny ** 2)
            if length < 1e-10:
                continue
            nx, ny = nx / length, ny / length
            proj_i = ci[:, 0] * nx + ci[:, 1] * ny
            proj_j = cj[:, 0] * nx + cj[:, 1] * ny
            if np.max(proj_i) + tol < np.min(proj_j) or np.max(proj_j) + tol < np.min(proj_i):
                return True
    return False


def _all_separated(inner_hex_data, tol=1e-6):
    n = len(inner_hex_data)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separation(inner_hex_data[i], inner_hex_data[j], tol):
                return False
    return True


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (11,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    deadline = time.time() + 580  # Safety margin

    sqrt3 = np.sqrt(3)
    # Start with a valid configuration with a small gap
    d = sqrt3 * 1.02
    
    initial = np.array([
        [0, 0, 0],
        [d, 0, 0],
        [d * 0.5, d * sqrt3 * 0.5, 0],
        [-d * 0.5, d * sqrt3 * 0.5, 0],
        [-d, 0, 0],
        [-d * 0.5, -d * sqrt3 * 0.5, 0],
        [d * 0.5, -d * sqrt3 * 0.5, 0],
        [1.5 * d, 0, 0],
        [0.75 * d, 1.5 * d * sqrt3 * 0.5, 0],
        [-0.75 * d, 1.5 * d * sqrt3 * 0.5, 0],
        [-1.5 * d, 0, 0],
    ])

    best_x = initial.copy()
    best_side = _min_outer_hex_side(initial)
    
    # Local search to minimize side length while maintaining separation
    # We use a simple hill climbing with random perturbations
    step_size = 0.1
    max_iter = 5000
    rng = np.random.default_rng(42)
    
    current_x = best_x.copy()
    current_side = best_side
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Pick a random hexagon to move
        idx = rng.integers(0, 11)
        
        # Generate a random perturbation
        dx = rng.normal(0, step_size)
        dy = rng.normal(0, step_size)
        da = rng.normal(0, step_size * 10) # Angle in degrees
        
        new_x = current_x.copy()
        new_x[idx, 0] += dx
        new_x[idx, 1] += dy
        new_x[idx, 2] += da
        
        # Check validity
        if _all_separated(new_x):
            new_side = _min_outer_hex_side(new_x)
            if new_side < current_side:
                current_x = new_x
                current_side = new_side
                if current_side < best_side:
                    best_x = current_x
                    best_side = current_side
            # Occasionally accept worse moves to escape local minima? 
            # For now, strict hill climbing.
        else:
            # If invalid, we could try a smaller step or just skip.
            pass
        
        # Adaptive step size: reduce if no improvement recently?
        # Simple decay
        if i % 100 == 0 and i > 0:
             step_size *= 0.9

    # Final check
    if not _all_separated(best_x):
        # Fallback to initial valid config
        best_x = initial
        best_side = _min_outer_hex_side(initial)

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    return best_x, outer_hex_data, float(best_side)


# EVOLVE-BLOCK-END
