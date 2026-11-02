"""
Hexagon packing: 12 unit hexagons in a regular hexagon, minimize outer side length.
Strategy: Conformal map packing via Möbius transformation + simulated annealing refinement.
"""
import numpy as np
import time
import random
import math

N = 12
DEADLINE = 148.0  # Safety margin before 150s kill

# Unit hexagon vertices (flat-top, side length 1)
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

# Outward normals for the 6 edges of a regular hexagon
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])

# Precompute edge vectors for SAT
EDGE_VECTORS = np.roll(UNIT_VERTS, -1, axis=0) - UNIT_VERTS
EDGE_NORMALS = np.array([[-e[1], e[0]] for e in EDGE_VECTORS])
EDGE_NORMALS = EDGE_NORMALS / (np.linalg.norm(EDGE_NORMALS, axis=1, keepdims=True) + 1e-15)


def get_rotated_verts(center, angle_deg):
    """Get vertices of a unit hexagon at given center and rotation."""
    a_rad = np.radians(angle_deg)
    c = np.cos(a_rad)
    s = np.sin(a_rad)
    R = np.array([[c, -s], [s, c]])
    return UNIT_VERTS @ R.T + np.array(center)


def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    for i in range(len(centers)):
        x, y, a_deg = centers[i]
        verts = get_rotated_verts((x, y), a_deg)
        v_rel = verts - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        max_extent = max(max_extent, np.max(max_proj))
    
    return max_extent / (np.sqrt(3) / 2)


def check_validity(centers, outer_center, outer_side, outer_angle, tol=1e-9):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    n = len(centers)
    
    # Precompute all vertices
    all_verts = []
    all_normals = []
    for i in range(n):
        x, y, a_deg = centers[i]
        verts = get_rotated_verts((x, y), a_deg)
        all_verts.append(verts)
        # Compute edge normals for this hexagon
        edges = np.roll(verts, -1, axis=0) - verts
        norms = np.array([[-e[1], e[0]] for e in edges])
        norms = norms / (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-15)
        all_normals.append(norms)
    
    # Check pairwise separation using SAT
    for i in range(n):
        v1 = all_verts[i]
        n1 = all_normals[i]
        for j in range(i + 1, n):
            v2 = all_verts[j]
            n2 = all_normals[j]
            
            # Check all 12 axes
            axes = np.vstack([n1, n2])
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            if np.max(sep) < -tol:
                return False
    
    # Check containment in outer hexagon
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    for i in range(n):
        v = all_verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        
        if np.max(max_proj) > apothem + tol:
            return False
    
    return True


def mobius_transform(z, a, b, c, d):
    """Apply Möbius transformation (az+b)/(cz+d) to complex number z."""
    return (a * z + b) / (c * z + d)


def mobius_inverse(w, a, b, c, d):
    """Inverse Möbius transformation: (dw-b)/(-cw+a)"""
    return (d * w - b) / (-c * w + a)


def generate_conformal_initial_config():
    """
    Generate initial configuration using Möbius transformation.
    Map 12 points in a disk to a configuration that approximates hexagon packing.
    
    Strategy: Place 12 "circles" (representing hexagons) in a disk using circle packing,
    then use Möbius transformation to create non-uniform spacing that may fit better
    in a hexagonal container.
    """
    configs = []
    
    # Base configuration: honeycomb-like arrangement in a disk
    # Use Möbius transformation to warp the standard hexagonal lattice
    
    # Standard hexagonal lattice positions for 12 hexagons
    # Center + 6 in first ring + 5 in second ring (partial)
    a1 = np.array([np.sqrt(3), 0])
    a2 = np.array([np.sqrt(3) / 2, 1.5])
    
    # Pattern: compact hexagonal cluster
    hex_coords = [
        (0, 0),      # center
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),  # ring 1
        (2, 0), (-2, 0), (0, 2), (2, 2), (-2, 2)  # ring 2 partial
    ]
    
    base_positions = []
    for (i, j) in hex_coords:
        pos = i * a1 + j * a2
        base_positions.append(pos)
    
    base_positions = np.array(base_positions)
    
    # Try several Möbius transformations
    # A Möbius transformation maps circles/lines to circles/lines
    # We use it to create asymmetric packings that might fit hexagonal boundaries better
    
    mobius_params_list = [
        # (a, b, c, d) for (az+b)/(cz+d)
        (1, 0, 0, 1),           # Identity
        (1, 0.3, 0, 1),         # Translation
        (1, 0, 0.05, 1),        # Mild distortion
        (1, 0, 0.1, 1),         # More distortion
        (1.1, 0, 0, 1),         # Scaling
        (1, 0.2, 0.05, 1),      # Combined
        (1, -0.2, 0.05, 1),     # Combined reverse
        (0.9, 0.3, 0.03, 1),    # Another combo
    ]
    
    for (a, b, c, d) in mobius_params_list:
        # Apply Möbius transform to each position (treating as complex)
        transformed = []
        for pos in base_positions:
            z = complex(pos[0], pos[1])
            w = mobius_transform(z, a, b, c, d)
            transformed.append(np.array([w.real, w.imag]))
        
        transformed = np.array(transformed)
        
        # Center the configuration
        centroid = np.mean(transformed, axis=0)
        centered = transformed - centroid
        
        # Scale to reasonable size
        max_r = np.max(np.linalg.norm(centered, axis=1))
        if max_r > 0:
            scale = 3.0 / max_r
            centered = centered * scale
        
        # Create centers with angle 0
        centers = [[float(centered[i][0]), float(centered[i][1]), 0.0] for i in range(N)]
        configs.append(centers)
    
    # Also add the standard honeycomb without transformation
    centroid = np.mean(base_positions, axis=0)
    centered = base_positions - centroid
    max_r = np.max(np.linalg.norm(centered, axis=1))
    if max_r > 0:
        scale = 3.0 / max_r
        centered = centered * scale
    centers = [[float(centered[i][0]), float(centered[i][1]), 0.0] for i in range(N)]
    configs.append(centers)
    
    return configs


def simulated_annealing(centers, max_time=10.0, seed=42):
    """
    Refine configuration using simulated annealing.
    Objective: minimize outer side length while maintaining validity.
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    # Current state
    curr = [list(c) for c in centers]
    curr_side = required_outer_side(curr, (0.0, 0.0), 0.0)
    
    best = [list(c) for c in curr]
    best_side = curr_side
    
    # Temperature schedule
    T0 = 0.5
    T_end = 0.001
    cooling_rate = 0.995
    
    T = T0
    
    while T > T_end:
        if time.time() - start_time > max_time:
            break
        
        # Generate neighbor: perturb one or two hexagons
        n_perturb = 1 if rng.random() < 0.7 else 2
        indices = rng.sample(range(N), n_perturb)
        
        neighbor = [list(c) for c in curr]
        for idx in indices:
            dx = rng.gauss(0, T * 0.3)
            dy = rng.gauss(0, T * 0.3)
            da = rng.gauss(0, T * 5.0)
            neighbor[idx][0] += dx
            neighbor[idx][1] += dy
            neighbor[idx][2] += da
        
        # Compute new side length
        new_side = required_outer_side(neighbor, (0.0, 0.0), 0.0)
        
        # Check validity (only if side improved or significantly)
        delta = new_side - curr_side
        
        if delta < 0:
            # Accept if valid
            if check_validity(neighbor, (0.0, 0.0), new_side, 0.0):
                curr = neighbor
                curr_side = new_side
                if new_side < best_side:
                    best = [list(c) for c in curr]
                    best_side = new_side
        else:
            # Metropolis acceptance
            if rng.random() < math.exp(-delta / T):
                if check_validity(neighbor, (0.0, 0.0), new_side, 0.0):
                    curr = neighbor
                    curr_side = new_side
        
        T *= cooling_rate
    
    return best, best_side


def nelder_mead_polish(centers, max_iter=50):
    """
    Final polish using Nelder-Mead on a subset of parameters.
    We optimize the global rotation and translation.
    """
    start_time = time.time()
    
    # Parameters: [dx, dy, angle] - global translation and rotation
    params = [0.0, 0.0, 0.0]
    
    def objective(p):
        dx, dy, angle = p
        shifted_rotated = [[c[0] + dx, c[1] + dy, c[2] + angle] for c in centers]
        side = required_outer_side(shifted_rotated, (0.0, 0.0), 0.0)
        
        # Check validity
        if not check_validity(shifted_rotated, (0.0, 0.0), side, 0.0):
            return side + 1000  # Penalty for invalid
        
        return side
    
    # Simple Nelder-Mead implementation
    n = len(params)
    simplex = [list(params)]
    for i in range(n):
        point = list(params)
        point[i] += 0.1
        simplex.append(point)
    
    for _ in range(max_iter):
        if time.time() - start_time > 5:
            break
        
        # Evaluate all vertices
        values = [objective(v) for v in simplex]
        indices = sorted(range(n + 1), key=lambda i: values[i])
        
        # Reorder
        simplex = [simplex[i] for i in indices]
        values = [values[i] for i in indices]
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        
        # Reflection
        alpha = 1.0
        reflected = centroid + alpha * (centroid - worst)
        ref_val = objective(reflected)
        
        if ref_val < values[0]:
            # Expansion
            gamma = 2.0
            expanded = centroid + gamma * (centroid - worst)
            exp_val = objective(expanded)
            if exp_val < ref_val:
                simplex[-1] = expanded
                values[-1] = exp_val
            else:
                simplex[-1] = reflected
                values[-1] = ref_val
        elif ref_val < values[-2]:
            simplex[-1] = reflected
            values[-1] = ref_val
        else:
            # Contraction
            rho = 0.5
            contracted = centroid + rho * (worst - centroid)
            con_val = objective(contracted)
            if con_val < values[-1]:
                simplex[-1] = contracted
                values[-1] = con_val
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    values[i] = objective(simplex[i])
    
    # Return best
    best_idx = np.argmin(values)
    best_params = simplex[best_idx]
    
    # Apply best params
    dx, dy, angle = best_params
    final_centers = [[c[0] + dx, c[1] + dy, c[2] + angle] for c in centers]
    final_side = required_outer_side(final_centers, (0.0, 0.0), 0.0)
    
    return final_centers, final_side


def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    # Generate initial configurations using conformal mapping
    configs = generate_conformal_initial_config()
    
    for cfg_idx, centers in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        # Quick evaluation
        side = required_outer_side(centers, (0.0, 0.0), 0.0)
        
        if side < best_score:
            if check_validity(centers, (0.0, 0.0), side, 0.0):
                best_score = side
                best_result = ([list(c) for c in centers], (0.0, 0.0), side, 0.0)
        
        # Refine with simulated annealing
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
        
        sa_time = min(8.0, DEADLINE - elapsed - 5)
        if sa_time > 2.0:
            refined, refined_side = simulated_annealing(centers, max_time=sa_time, seed=42 + cfg_idx)
            
            if refined_side < best_score:
                if check_validity(refined, (0.0, 0.0), refined_side, 0.0):
                    best_score = refined_side
                    best_result = ([list(c) for c in refined], (0.0, 0.0), refined_side, 0.0)
        
        # Nelder-Mead polish on the best found so far
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        if best_result is not None:
            polished, polished_side = nelder_mead_polish(best_result[0], max_iter=30)
            
            if polished_side < best_score:
                if check_validity(polished, (0.0, 0.0), polished_side, 0.0):
                    best_score = polished_side
                    best_result = ([list(c) for c in polished], (0.0, 0.0), polished_side, 0.0)
    
    # If we still have time, do additional SA on the best result
    elapsed = time.time() - start_time
    if best_result is not None and elapsed < DEADLINE - 10:
        sa_time = min(15.0, DEADLINE - elapsed - 3)
        if sa_time > 3.0:
            refined, refined_side = simulated_annealing(best_result[0], max_time=sa_time, seed=999)
            
            if refined_side < best_score:
                if check_validity(refined, (0.0, 0.0), refined_side, 0.0):
                    best_score = refined_side
                    best_result = ([list(c) for c in refined], (0.0, 0.0), refined_side, 0.0)
        
        # Final polish
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 3:
            polished, polished_side = nelder_mead_polish(best_result[0], max_iter=40)
            
            if polished_side < best_score:
                if check_validity(polished, (0.0, 0.0), polished_side, 0.0):
                    best_score = polished_side
                    best_result = ([list(c) for c in polished], (0.0, 0.0), polished_side, 0.0)
    
    if best_result is not None:
        return best_result
    
    # Fallback: circular arrangement
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = required_outer_side(inner_default, (0.0, 0.0), 0.0)
    return inner_default, (0.0, 0.0), required_side, 0.0


if __name__ == "__main__":
    result = construct()
    print(f"Best outer side length: {result[2]:.6f}")
    print(f"Number of inner hexagons: {len(result[0])}")
