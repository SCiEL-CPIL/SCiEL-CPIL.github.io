from scipy.optimize import minimize, minimize_scalar
import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, side, angle_rad):
        """Returns 6x2 array of vertices for a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        """verts: 6x2. Returns 6x2 array of outward unit normals."""
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([-edges[:, 1], edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        norms = np.maximum(norms, 1e-15)
        normals = normals / norms[:, np.newaxis]
        return normals

    def penetration_depth(v1, v2):
        """Returns max overlap depth between two convex hexagons (0 if disjoint)."""
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        
        overlaps = np.minimum(max1, max2) - np.maximum(min1, max2)
        min_overlap = np.min(overlaps)
        
        return max(0.0, min_overlap)

    def containment_violation(inner_verts, outer_verts):
        """Returns how much inner_verts stick out of outer_verts (0 if contained)."""
        outer_normals = get_normals(outer_verts)
        
        p_inner = inner_verts @ outer_normals.T
        p_outer_max = (outer_verts @ outer_normals.T).max(axis=0)
        
        p_inner_max = p_inner.max(axis=0)
        
        viol = p_inner_max - p_outer_max
        return max(0.0, np.max(viol))

    def make_objective(alpha_rad, theta_rad):
        """Create objective function for given fixed rotations."""
        def objective(params):
            if time.time() > deadline:
                return 1e10
                
            positions = params[:2 * N].reshape(N, 2)
            s = params[2 * N]
            
            if s < 1.0:
                return 1e6

            outer_verts = hex_vertices(0, 0, s, theta_rad)
            
            total_penalty = 0.0
            inner_verts_list = []
            
            for i in range(N):
                x, y = positions[i]
                verts = hex_vertices(x, y, 1.0, alpha_rad)
                inner_verts_list.append(verts)
                
                viol = containment_violation(verts, outer_verts)
                total_penalty += viol * 50.0
            
            for i in range(N):
                for j in range(i + 1, N):
                    pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
                    total_penalty += pd * 50.0
            
            return s + total_penalty
        
        return objective, inner_verts_list

    def compute_min_enclosing_hex(all_verts):
        """Find minimum side length and orientation of enclosing regular hexagon."""
        def side_for_theta(theta):
            angles = theta + np.radians(np.arange(0, 360, 60))
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            dots = all_verts @ normals.T
            widths = dots.max(axis=0) - dots.min(axis=0)
            max_width = np.max(widths)
            apothem = max_width / 2.0
            side = 2 * apothem / np.sqrt(3)
            return side

        res = minimize_scalar(side_for_theta, bounds=(0, np.pi/3), method='bounded',
                              options={'xatol': 1e-8})
        best_theta = res.x
        best_side = res.fun

        # Compute center
        angles = best_theta + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        mins = dots.min(axis=0)
        maxs = dots.max(axis=0)
        mids = (mins + maxs) / 2.0
        c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
        center = tuple(c)

        return best_side, best_theta, center

    def verify_solution(positions, alpha_rad, theta_rad, s):
        """Check if solution is valid (no overlaps, all contained)."""
        outer_verts = hex_vertices(0, 0, s, theta_rad)
        inner_verts_list = []
        for i in range(N):
            verts = hex_vertices(positions[i, 0], positions[i, 1], 1.0, alpha_rad)
            inner_verts_list.append(verts)
            if containment_violation(verts, outer_verts) > 1e-4:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                if penetration_depth(inner_verts_list[i], inner_verts_list[j]) > 1e-4:
                    return False
        return True

    # Initial configurations to try
    def make_flower_init():
        """Standard honeycomb flower: 1 center, 6 ring, 5 outer."""
        d1 = np.sqrt(3)
        d2 = 2 * np.sqrt(3)
        pos = [[0.0, 0.0]]
        for k in range(6):
            a = np.radians(60 * k)
            pos.append([d1 * np.cos(a), d1 * np.sin(a)])
        for k in range(5):
            a = np.radians(60 * k)
            pos.append([d2 * np.cos(a), d2 * np.sin(a)])
        return np.array(pos)

    def make_tight_flower_init():
        """Tighter flower with slightly compressed distances."""
        d1 = np.sqrt(3) * 0.98
        d2 = 2 * np.sqrt(3) * 0.97
        pos = [[0.0, 0.0]]
        for k in range(6):
            a = np.radians(60 * k)
            pos.append([d1 * np.cos(a), d1 * np.sin(a)])
        for k in range(5):
            a = np.radians(60 * k)
            pos.append([d2 * np.cos(a), d2 * np.sin(a)])
        return np.array(pos)

    def make_ring_init():
        """Two concentric rings: 6 inner, 6 outer."""
        r1 = np.sqrt(3) * 0.9
        r2 = np.sqrt(3) * 1.8
        pos = []
        for k in range(6):
            a = np.radians(60 * k)
            pos.append([r1 * np.cos(a), r1 * np.sin(a)])
        for k in range(6):
            a = np.radians(60 * k + 30)
            pos.append([r2 * np.cos(a), r2 * np.sin(a)])
        return np.array(pos)

    inits = [make_flower_init(), make_tight_flower_init(), make_ring_init()]

    best_s = np.inf
    best_positions = None
    best_alpha = 0.0
    best_theta = 0.0

    # Try multiple rotation combinations
    alphas = np.radians([0, 10, 20, 30, 40, 50])
    thetas = np.radians([0, 10, 20, 30, 40, 50])

    attempts = 0
    max_attempts = 20

    for init_pos in inits:
        if time.time() > deadline - 15:
            break
        for alpha in alphas:
            if time.time() > deadline - 10:
                break
            for theta in thetas:
                if time.time() > deadline - 5:
                    break
                if attempts >= max_attempts:
                    break
                attempts += 1

                obj_func, _ = make_objective(alpha, theta)
                x0 = np.concatenate([init_pos.flatten(), [4.5]])

                try:
                    res = minimize(
                        obj_func,
                        x0,
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50}
                    )

                    pos = res.x[:2 * N].reshape(N, 2)
                    s = res.x[2 * N]

                    if s < 1.0:
                        continue

                    # Verify validity
                    if verify_solution(pos, alpha, theta, s):
                        if s < best_s:
                            best_s = s
                            best_positions = pos.copy()
                            best_alpha = alpha
                            best_theta = theta

                except Exception:
                    pass

    # If no valid solution found, use a safe fallback
    if best_positions is None:
        best_positions = inits[0]
        best_alpha = 0.0
        best_theta = 0.0
        # Find enclosing hexagon for the initial config
        all_verts = []
        for i in range(N):
            v = hex_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_alpha)
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        best_s, best_theta, _ = compute_min_enclosing_hex(all_verts)

    # Refine: compute exact minimum enclosing hexagon for the best positions
    all_verts = []
    for i in range(N):
        v = hex_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_alpha)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)

    refined_side, refined_theta, center = compute_min_enclosing_hex(all_verts)

    # Use the refined values
    final_side = refined_side
    final_theta = refined_theta
    final_center = center

    inner = [[best_positions[i, 0], best_positions[i, 1], np.degrees(best_alpha)] for i in range(N)]
    outer_center = final_center
    outer_side = final_side
    outer_angle_deg = np.degrees(final_theta)

    return inner, outer_center, outer_side, outer_angle_deg
