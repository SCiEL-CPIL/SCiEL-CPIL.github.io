"""
Hexagon packing: 12 unit hexagons in a regular hexagon, minimize outer side length.
Strategy: Soft-body relaxation with smooth overlap penalties + gradient descent + local refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0  # Safety margin before 150s kill

# Unit hexagon vertices (pointy-top, side length 1)
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])

# Edge normals (outward) for containment check
EDGE_NORMALS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

def hexagon_vertices(center, angle_deg):
    """Compute vertices of a unit hexagon at given center and angle."""
    a_rad = np.radians(angle_deg)
    c = np.cos(a_rad)
    s = np.sin(a_rad)
    R = np.array([[c, -s], [s, c]])
    return UNIT_VERTS @ R.T + center

def smooth_overlap_penalty(v1, v2, sigma=0.1):
    """
    Compute smooth approximation of overlap between two convex polygons.
    Uses signed distance-like penalty that is smooth everywhere.
    """
    # For each edge of poly1, check if any vertex of poly2 is inside
    # and vice versa. Use smooth max of penetration depths.
    
    # Edges of v1
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    
    # Normal directions (outward)
    def get_outward_normals(edges, verts):
        """Compute outward normals for polygon edges."""
        normals = np.zeros_like(edges)
        for i in range(len(edges)):
            e = edges[i]
            n = np.array([-e[1], e[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            # Ensure outward: normal should point away from centroid
            centroid = np.mean(verts, axis=0)
            edge_mid = (verts[i] + verts[(i+1) % len(verts)]) / 2
            if np.dot(n, edge_mid - centroid) < 0:
                n = -n
            normals[i] = n
        return normals
    
    n1 = get_outward_normals(edges1, v1)
    n2 = get_outward_normals(edges2, v2)
    
    # For each edge of v1, compute max penetration of v2 vertices
    # Penetration = max(0, dot(v2_vert - v1_edge_start, n1_edge))
    pen1 = np.zeros(6)
    for i in range(6):
        # Project all v2 vertices onto normal of edge i of v1
        rel = v2 - v1[i]
        projs = rel @ n1[i]
        # Max penetration depth (how far v2 vertex goes past edge i)
        pen1[i] = np.max(projs)
    
    # Similarly for v2 edges
    pen2 = np.zeros(6)
    for i in range(6):
        rel = v1 - v2[i]
        projs = rel @ n2[i]
        pen2[i] = np.max(projs)
    
    # Smooth max of all penetration values
    all_pen = np.concatenate([pen1, pen2])
    # Use softplus-like smooth max: sigma * log(sum(exp(pen/sigma)))
    # Clip to avoid overflow
    clipped = np.clip(all_pen, -50, 50)
    smooth_max = sigma * np.log(np.sum(np.exp(clipped / sigma)))
    
    # Only count positive penetration (overlap)
    return max(0.0, smooth_max)

def compute_overlap_penalty_all(centers, angles, sigma=0.1):
    """Compute total smooth overlap penalty for all pairs."""
    total_pen = 0.0
    for i in range(N):
        v1 = hexagon_vertices(centers[i], angles[i])
        for j in range(i + 1, N):
            v2 = hexagon_vertices(centers[j], angles[j])
            total_pen += smooth_overlap_penalty(v1, v2, sigma)
    return total_pen

def compute_required_side(centers, angles, outer_center, outer_angle_deg):
    """Compute minimum outer side length to contain all hexagons."""
    oa_rad = np.radians(outer_angle_deg)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    all_verts = []
    for i in range(N):
        v = hexagon_vertices(centers[i], angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)  # (72, 2)
    
    v_rel = all_verts - outer_center
    v_rot = v_rel @ R_o.T
    projs = v_rot @ EDGE_NORMALS.T
    abs_projs = np.abs(projs)
    max_proj = np.max(abs_projs, axis=1)
    max_extent = np.max(max_proj)
    
    return max_extent / (np.sqrt(3) / 2)

def objective_soft(params, outer_center, outer_angle_deg, sigma=0.1, weight=100.0):
    """
    Objective function for soft-body relaxation.
    params: (N*3,) array of [x0, y0, a0, x1, y1, a1, ...]
    Returns: side_length + weight * overlap_penalty
    """
    centers = params[:N*2].reshape(N, 2)
    angles = params[N*2:]
    
    side = compute_required_side(centers, angles, outer_center, outer_angle_deg)
    pen = compute_overlap_penalty_all(centers, angles, sigma)
    
    return side + weight * pen

def generate_initial_layout(layout_type='honeycomb'):
    """Generate initial non-overlapping layout."""
    if layout_type == 'honeycomb':
        # Central hexagon + 6 neighbors + 5 from next ring
        coords = [
            (0, 0),
            (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
            (2, -1), (-2, 1), (0, 2), (0, -2), (2, 0)
        ]
    elif layout_type == 'lattice':
        # 3x4 rectangular sub-lattice
        coords = []
        for i in range(3):
            for j in range(4):
                coords.append((i, j))
    else:
        coords = [
            (0, 0),
            (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
            (2, -1), (-2, 1), (0, 2), (0, -2), (2, 0)
        ]
    
    local_centers = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        local_centers.append([x, y])
    
    local_centers = np.array(local_centers)
    
    # Center around origin
    cx = np.mean(local_centers[:, 0])
    cy = np.mean(local_centers[:, 1])
    local_centers -= np.array([cx, cy])
    
    return local_centers

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    best_side = float('inf')
    best_params = None
    
    # Try multiple initial layouts and random perturbations
    layouts = ['honeycomb', 'lattice']
    num_restarts = 20
    
    for layout_idx, layout_type in enumerate(layouts):
        for restart in range(num_restarts):
            if time.time() > DEADLINE - 30:
                break
            
            # Generate initial layout with random perturbation
            base_centers = generate_initial_layout(layout_type)
            
            # Add random perturbation
            scale = 1.0 if restart == 0 else 0.5 * (1 + 0.1 * restart)
            centers = base_centers + np.random.normal(0, scale * 0.3, (N, 2))
            angles = np.random.uniform(-10, 10, N)
            
            # Parameterize: [x0, y0, a0, x1, y1, a1, ...]
            params = np.concatenate([centers.flatten(), angles])
            
            # Phase 1: Gradient descent with decreasing sigma
            current_sigma = 0.5
            current_weight = 50.0
            
            for sigma_step in range(10):
                if time.time() > DEADLINE - 25:
                    break
                
                # Run several gradient descent iterations
                for gd_iter in range(50):
                    if time.time() > DEADLINE - 25:
                        break
                    
                    # Compute objective
                    obj = objective_soft(params, outer_center, outer_angle, current_sigma, current_weight)
                    
                    # Numerical gradient
                    grad = np.zeros_like(params)
                    eps = 1e-4
                    for i in range(len(params)):
                        params_plus = params.copy()
                        params_plus[i] += eps
                        obj_plus = objective_soft(params_plus, outer_center, outer_angle, current_sigma, current_weight)
                        grad[i] = (obj_plus - obj) / eps
                    
                    # Gradient descent step
                    lr = 0.01 / (1 + 0.1 * sigma_step)
                    params = params - lr * grad
                    
                    # Optional: clamp angles to reasonable range
                    angle_indices = np.arange(N*2, len(params))
                    params[angle_indices] = np.clip(params[angle_indices], -30, 30)
                
                # Decrease sigma and increase weight
                current_sigma *= 0.7
                current_weight *= 1.3
            
            # Phase 2: Local refinement with Nelder-Mead on the side length only
            # (assuming overlaps are now minimal)
            def side_only_obj(p):
                c = p[:N*2].reshape(N, 2)
                a = p[N*2:]
                return compute_required_side(c, a, outer_center, outer_angle)
            
            res = minimize(
                side_only_obj,
                params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            
            if res.fun < best_side:
                # Verify no overlaps
                c = res.x[:N*2].reshape(N, 2)
                a = res.x[N*2:]
                pen = compute_overlap_penalty_all(c, a, sigma=0.05)
                if pen < 0.01:  # Very small overlap tolerance
                    best_side = res.fun
                    best_params = res.x.copy()
    
    # If no good result, use a simple fallback
    if best_params is None:
        base_centers = generate_initial_layout('honeycomb')
        angles = np.zeros(N)
        best_params = np.concatenate([base_centers.flatten(), angles])
        best_side = compute_required_side(base_centers, angles, outer_center, outer_angle)
    
    # Extract final solution
    centers = best_params[:N*2].reshape(N, 2)
    angles = best_params[N*2:]
    
    inner_list = []
    for i in range(N):
        inner_list.append([float(centers[i, 0]), float(centers[i, 1]), float(angles[i])])
    
    # Final validation
    final_side = compute_required_side(centers, angles, outer_center, outer_angle)
    
    return inner_list, (0.0, 0.0), final_side, outer_angle

if __name__ == "__main__":
    result = construct()
    print(f"Best outer side length: {result[2]:.6f}")
    print(f"Number of inner hexagons: {len(result[0])}")
