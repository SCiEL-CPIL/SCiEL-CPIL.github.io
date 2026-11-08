import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute unit hexagon vertices (flat-top, side=1, centered at origin)
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.deg2rad(np.arange(6) * 60)
    unit_verts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normals for the 6 faces of the outer hexagon (flat-top)
    normal_angles = np.deg2rad(30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    def get_vertices(cx, cy, rot_rad):
        cos_r = np.cos(rot_rad)
        sin_r = np.sin(rot_rad)
        rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        return unit_verts @ rot_mat.T + np.array([cx, cy])

    def min_dist_polygons(v1, v2):
        """
        Compute minimum distance between two convex polygons (hexagons).
        Returns 0 if they intersect (or touch).
        v1, v2: (6, 2) arrays of vertices.
        """
        min_d = float('inf')
        # Check vertex of v1 to edges of v2
        for i in range(6):
            p1 = v1[i]
            for j in range(6):
                p2 = v2[j]
                p3 = v2[(j+1)%6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        # Check vertex of v2 to edges of v1
        for i in range(6):
            p1 = v2[i]
            for j in range(6):
                p2 = v1[j]
                p3 = v1[(j+1)%6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        return min_d

    def check_validity(verts_list, R):
        """
        Check if all hexagons are inside the outer hexagon of inradius R
        and do not overlap.
        """
        # Check boundary
        for v in verts_list:
            for p in v:
                # Max support function in any of the 6 normal directions must be <= R
                # Actually, for a point to be inside a regular hexagon defined by inradius R,
                # its projection on each of the 6 normals must be <= R.
                # Note: The hexagon is the intersection of 6 half-planes.
                dots = np.dot(normals, p)
                if np.any(dots > R + 1e-9):
                    return False
        
        # Check overlaps
        eps = 1e-4
        for i in range(N):
            for j in range(i+1, N):
                # Quick center distance check
                c1 = verts_list[i].mean(axis=0)
                c2 = verts_list[j].mean(axis=0)
                if np.linalg.norm(c1 - c2) < 2.0:
                    d = min_dist_polygons(verts_list[i], verts_list[j])
                    if d < eps:
                        return False
        return True

    def calc_outer_side(verts_list):
        """
        Calculate the minimum outer side length S such that all hexagons fit inside.
        Returns S.
        """
        max_support = 0.0
        for v in verts_list:
            for p in v:
                dots = np.dot(normals, p)
                m = np.max(dots)
                if m > max_support:
                    max_support = m
        return 2.0 * max_support / np.sqrt(3)

    def init_grid():
        """
        Initialize a valid configuration using a hexagonal grid.
        We want to place 12 hexagons.
        A hexagonal grid with spacing slightly larger than sqrt(3) ensures no overlap.
        Let's try a 4x3 grid or similar.
        """
        # Spacing for hexagons of side 1 to be disjoint:
        # If aligned, horizontal dist > 2, vertical dist > sqrt(3).
        # But we can interlock them.
        # Simple valid start: 3 rows of 4.
        # x spacing: 2.1, y spacing: 1.8
        xs = np.zeros(N)
        ys = np.zeros(N)
        rots = np.zeros(N)
        
        # 4 columns, 3 rows
        cols = 4
        rows = 3
        x_step = 2.1
        y_step = 1.8
        
        # Center the grid
        x_offset = -(cols - 1) * x_step / 2.0
        y_offset = -(rows - 1) * y_step / 2.0
        
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= N: break
                xs[idx] = x_offset + c * x_step
                ys[idx] = y_offset + r * y_step
                rots[idx] = 0.0
                idx += 1
        
        return xs, ys, rots

    def local_search(xs, ys, rots, max_iter=5000, step_size=0.05):
        """
        Steepest descent with hard constraints.
        """
        current_S = calc_outer_side([get_vertices(xs[i], ys[i], rots[i]) for i in range(N)])
        best_S = current_S
        best_xs, best_ys, best_rots = xs.copy(), ys.copy(), rots.copy()
        
        # Precompute vertices for current state
        verts_list = [get_vertices(xs[i], ys[i], rots[i]) for i in range(N)]
        R = current_S * np.sqrt(3) / 2.0
        
        no_improve_count = 0
        current_step = step_size
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Calculate gradient numerically? Or just try random small moves?
            # Given the hard constraint, gradient is tricky if we are on the boundary.
            # Let's use a "jitter and project" or simple random walk with acceptance.
            # Since we want to minimize S, we try to move hexagons inward or closer.
            
            # Pick a random hexagon and a random direction
            i = np.random.randint(N)
            
            # Try moving x, y, or rotating
            move_type = np.random.randint(3)
            
            orig_x, orig_y, orig_rot = xs[i], ys[i], rots[i]
            new_xs, new_ys, new_rots = xs.copy(), ys.copy(), rots.copy()
            
            if move_type == 0:
                # Move x
                new_xs[i] += np.random.uniform(-current_step, current_step)
            elif move_type == 1:
                # Move y
                new_ys[i] += np.random.uniform(-current_step, current_step)
            else:
                # Rotate
                new_rots[i] += np.random.uniform(-current_step, current_step)
            
            # Update vertices for the moved hexagon
            new_verts_list = verts_list.copy()
            new_verts_list[i] = get_vertices(new_xs[i], new_ys[i], new_rots[i])
            
            # Check validity
            # We can optimize this check by only checking against the moved hexagon and boundary
            valid = True
            
            # Check boundary for moved hexagon
            for p in new_verts_list[i]:
                dots = np.dot(normals, p)
                if np.any(dots > R + 1e-9):
                    valid = False
                    break
            
            # Check overlap with others
            if valid:
                for j in range(N):
                    if i == j: continue
                    # Quick check
                    c1 = new_verts_list[i].mean(axis=0)
                    c2 = new_verts_list[j].mean(axis=0)
                    if np.linalg.norm(c1 - c2) < 2.0:
                        d = min_dist_polygons(new_verts_list[i], new_verts_list[j])
                        if d < 1e-4:
                            valid = False
                            break
            
            if valid:
                # Calculate new S
                new_S = calc_outer_side(new_verts_list)
                if new_S < best_S - 1e-6:
                    # Accept
                    xs, ys, rots = new_xs, new_ys, new_rots
                    verts_list = new_verts_list
                    best_S = new_S
                    best_xs, best_ys, best_rots = xs.copy(), ys.copy(), rots.copy()
                    R = best_S * np.sqrt(3) / 2.0
                    no_improve_count = 0
                    # Optionally increase step size if stuck? No, decrease it.
                    current_step = max(0.001, current_step * 0.99)
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            if no_improve_count > 200:
                # Shrink step size and reset counter
                current_step *= 0.5
                no_improve_count = 0
                if current_step < 0.001:
                    break

        return best_xs, best_ys, best_rots, best_S

    # Main logic
    best_overall_S = float('inf')
    best_overall_params = None
    
    # Run multiple restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize with some randomness
        xs, ys, rots = init_grid()
        
        # Add random jitter to break symmetry
        if r > 0:
            xs += np.random.uniform(-0.5, 0.5, N)
            ys += np.random.uniform(-0.5, 0.5, N)
            rots += np.random.uniform(-np.pi/6, np.pi/6, N)
            
        # Check if initial is valid. If not, we might need to expand?
        # The init_grid is designed to be valid. Jitter might break it.
        # If invalid, we can try to "relax" it or skip.
        # For simplicity, if invalid, we try to fix by scaling positions away from center?
        # Or just run local search which will reject invalid moves.
        # But if start is invalid, local search might get stuck.
        # Let's ensure start is valid.
        verts_list = [get_vertices(xs[i], ys[i], rots[i]) for i in range(N)]
        S_init = calc_outer_side(verts_list)
        R_init = S_init * np.sqrt(3) / 2.0
        
        # Check validity of start
        valid_start = check_validity(verts_list, R_init)
        if not valid_start:
            # Try to fix by scaling out
            scale = 1.5
            for k in range(5):
                xs_scaled = xs * scale
                ys_scaled = ys * scale
                verts_list_scaled = [get_vertices(xs_scaled[i], ys_scaled[i], rots[i]) for i in range(N)]
                S_scaled = calc_outer_side(verts_list_scaled)
                R_scaled = S_scaled * np.sqrt(3) / 2.0
                if check_validity(verts_list_scaled, R_scaled):
                    xs, ys = xs_scaled, ys_scaled
                    break
                scale *= 1.2
            
            # Re-check
            verts_list = [get_vertices(xs[i], ys[i], rots[i]) for i in range(N)]
            S_init = calc_outer_side(verts_list)
            R_init = S_init * np.sqrt(3) / 2.0
            if not check_validity(verts_list, R_init):
                continue # Skip this restart if we can't make it valid
        
        # Run local search
        xs_opt, ys_opt, rots_opt, S_opt = local_search(xs, ys, rots)
        
        if S_opt < best_overall_S:
            best_overall_S = S_opt
            best_overall_params = (xs_opt, ys_opt, rots_opt)
            
        # If we are getting close to a theoretical lower bound, we might stop early?
        # Lower bound: Area of 12 hexagons = 12 * (3*sqrt(3)/2) = 18*sqrt(3) ~ 31.18
        # Area of outer hexagon with side S is (3*sqrt(3)/2) * S^2.
        # (3*sqrt(3)/2) * S^2 >= 31.18 => S^2 >= 31.18 / (3*sqrt(3)/2) = 31.18 / 2.598 = 12.0
        # S >= sqrt(12) = 3.464.
        # So 3.464 is a strict lower bound based on area.
        # If best_overall_S < 3.6, we can probably stop.
        if best_overall_S < 3.6:
            break

    if best_overall_params is None:
        # Fallback to safe grid
        xs, ys, rots = init_grid()
        best_overall_S = calc_outer_side([get_vertices(xs[i], ys[i], rots[i]) for i in range(N)])
    else:
        xs, ys, rots = best_overall_params

    # Final extraction
    rots_deg = np.rad2deg(rots)
    inner = []
    for i in range(N):
        inner.append([xs[i], ys[i], rots_deg[i]])
        
    return inner, (0.0, 0.0), best_overall_S, 0.0

if __name__ == "__main__":
    construct()
