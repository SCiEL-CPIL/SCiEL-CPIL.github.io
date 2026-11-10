# EVOLVE-BLOCK-START
import numpy as np
import time


def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.

    The shape is specified IMPLICITLY by a motion "path". Each path point is a triple
    (x, y, angle_degrees) giving the position of the corridor's inner corner and the
    corridor's rotation at one instant of the move. In the reference frame of the sofa
    the corridor sweeps over the plane; the sofa is then the intersection of every
    corridor placement along the path, so by construction it always clears the corridor.
    The objective is to maximize the area of that intersection.

    The evaluator prepends the start state (0, 0, 0) and appends the end state
    (-2, 0, 90), then densifies the path to <=1-degree rotation steps.

    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees) for the
            corridor's inner corner.
    """
    start_time = time.time()
    deadline = 625.0  # evaluator kills at 630s, keep safety margin

    # Sofa grid
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))  # (N, 2)
    N = P.shape[0]

    # Start and end states (added by evaluator)
    start_state = np.array([0.0, 0.0, 0.0])
    end_state = np.array([-2.0, 0.0, 90.0])

    def densify_path(path):
        """
        Densify the path by adding linear interpolations between consecutive rows.
        Between consecutive rows (x1,y1,a1) and (x2,y2,a2), add m = max(1, ceil(|a2-a1|))
        equal steps of interpolation.
        """
        if len(path) == 0:
            return np.array([start_state, end_state])
        
        full_path = [start_state] + list(path) + [end_state]
        
        densified = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(1, m + 1):
                t = k / m
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                densified.append([x, y, a])
        
        return np.array(densified)

    def check_point_in_corridor_vectorized(P, x, y, a_deg, cos_a=None, sin_a=None):
        """
        Vectorized check: which points in P are inside the corridor placement at (x, y, a_deg).
        Returns a boolean array of shape (N,).
        If cos_a and sin_a are provided (precomputed), they are used instead of recomputing.
        """
        if cos_a is None or sin_a is None:
            a_rad = np.radians(a_deg)
            cos_a = np.cos(-a_rad)
            sin_a = np.sin(-a_rad)
        dx = P[:, 0] - x
        dy = P[:, 1] - y
        u = cos_a * dx - sin_a * dy
        v = sin_a * dx + cos_a * dy
        
        eps = 1e-9
        in_r1 = (u >= -eps) & (u <= 1 + eps) & (v >= -3 - eps) & (v <= 1 + eps)
        in_r2 = (u >= -3 - eps) & (u <= 1 + eps) & (v >= -eps) & (v <= 1 + eps)
        
        return in_r1 | in_r2

    def evaluate_path(path):
        """
        Evaluate the area of the sofa for a given path.
        Precomputes rotation matrices for all unique angles to avoid redundant trig calls.
        """
        densified = densify_path(path)
        
        # Precompute cos/sin for all unique angles
        unique_angles = np.unique(densified[:, 2])
        rot_cache = {}
        for a_deg in unique_angles:
            a_rad = np.radians(a_deg)
            rot_cache[a_deg] = (np.cos(-a_rad), np.sin(-a_rad))
        
        # Initialize all points as potentially in the sofa
        in_sofa = np.ones(N, dtype=bool)
        
        for x, y, a in densified:
            cos_a, sin_a = rot_cache[a]
            mask = check_point_in_corridor_vectorized(P, x, y, a, cos_a, sin_a)
            in_sofa &= mask
            if not in_sofa.any():
                break  # Early exit if no points remain
        
        return in_sofa.sum() * 0.0001

    # Initial path - richer starting point with 9 control points
    path = np.array([
        [-0.05, 0.05, 10.0],
        [-0.15, 0.15, 20.0],
        [-0.25, 0.25, 30.0],
        [-0.35, 0.35, 40.0],
        [-0.45, 0.35, 50.0],
        [-0.55, 0.25, 60.0],
        [-0.65, 0.20, 70.0],
        [-0.75, 0.25, 80.0],
        [-0.85, 0.30, 86.0],
    ], dtype=np.float64)

    best_path = path.copy()
    best_score = evaluate_path(path)

    np.random.seed(42)
    
    def coordinate_descent_optimize(path, max_iterations=500, base_step=0.1):
        """
        Coordinate descent optimization with adaptive step sizes.
        """
        current_path = path.copy()
        current_score = evaluate_path(current_path)
        n_points = len(current_path)
        step = base_step
        
        for iteration in range(max_iterations):
            if (time.time() - start_time) > deadline - 30:
                break
            
            improved = False
            for i in range(n_points):
                if (time.time() - start_time) > deadline - 30:
                    break
                
                for coord in range(3):
                    for direction in [1, -1]:
                        new_path = current_path.copy()
                        if coord < 2:
                            new_path[i, coord] += direction * step
                        else:
                            new_path[i, coord] += direction * step * 2
                            new_path[i, coord] = np.clip(new_path[i, coord], 5, 85)
                        
                        new_score = evaluate_path(new_path)
                        if new_score > current_score + 1e-10:
                            current_path = new_path
                            current_score = new_score
                            improved = True
            
            if not improved:
                step *= 0.7
                if step < 0.001:
                    break
        
        return current_path, current_score

    def simulated_annealing_optimize(path, max_iterations=200, start_temp=0.05, end_temp=0.001):
        """
        Simulated annealing to escape local optima.
        """
        current_path = path.copy()
        current_score = evaluate_path(current_path)
        best_path_sa = current_path.copy()
        best_score_sa = current_score
        n_points = len(current_path)
        
        for iteration in range(max_iterations):
            if (time.time() - start_time) > deadline - 30:
                break
            
            t = iteration / max_iterations
            temp = start_temp * (end_temp / start_temp) ** t
            
            i = np.random.randint(n_points)
            coord = np.random.randint(3)
            
            new_path = current_path.copy()
            perturbation = np.random.uniform(-0.1, 0.1)
            if coord < 2:
                new_path[i, coord] += perturbation
            else:
                new_path[i, coord] += perturbation * 2
                new_path[i, coord] = np.clip(new_path[i, coord], 5, 85)
            
            new_score = evaluate_path(new_path)
            
            if new_score >= current_score or np.random.random() < np.exp(-(current_score - new_score) / max(temp, 1e-10)):
                current_path = new_path
                current_score = new_score
                if new_score > best_score_sa:
                    best_path_sa = new_path.copy()
                    best_score_sa = new_score
        
        return best_path_sa, best_score_sa

    # Phase 1: Initial coordinate descent from the starting path
    optimized_path, optimized_score = coordinate_descent_optimize(
        path, max_iterations=200, base_step=0.1
    )
    if optimized_score > best_score:
        best_score = optimized_score
        best_path = optimized_path.copy()

    # Phase 2: Simulated annealing to explore the landscape
    if (time.time() - start_time) < deadline - 80:
        sa_path, sa_score = simulated_annealing_optimize(
            best_path, max_iterations=150, start_temp=0.04, end_temp=0.002
        )
        if sa_score > best_score:
            best_score = sa_score
            best_path = sa_path.copy()

    # Phase 3: Coordinate descent polish after SA
    if (time.time() - start_time) < deadline - 60:
        cd_path, cd_score = coordinate_descent_optimize(
            best_path, max_iterations=200, base_step=0.05
        )
        if cd_score > best_score:
            best_score = cd_score
            best_path = cd_path.copy()

    # Phase 4: Another SA round with lower temperature
    if (time.time() - start_time) < deadline - 50:
        sa2_path, sa2_score = simulated_annealing_optimize(
            best_path, max_iterations=120, start_temp=0.02, end_temp=0.001
        )
        if sa2_score > best_score:
            best_score = sa2_score
            best_path = sa2_path.copy()

    # Phase 5: Fine coordinate descent
    if (time.time() - start_time) < deadline - 40:
        fine_path, fine_score = coordinate_descent_optimize(
            best_path, max_iterations=150, base_step=0.02
        )
        if fine_score > best_score:
            best_score = fine_score
            best_path = fine_path.copy()

    # Phase 6: Very fine coordinate descent
    if (time.time() - start_time) < deadline - 25:
        vf_path, vf_score = coordinate_descent_optimize(
            best_path, max_iterations=150, base_step=0.01
        )
        if vf_score > best_score:
            best_score = vf_score
            best_path = vf_path.copy()

    # Phase 7: Ultra-fine polish
    if (time.time() - start_time) < deadline - 15:
        uf_path, uf_score = coordinate_descent_optimize(
            best_path, max_iterations=100, base_step=0.005
        )
        if uf_score > best_score:
            best_score = uf_score
            best_path = uf_path.copy()

    # Phase 8: Micro polish
    if (time.time() - start_time) < deadline - 8:
        micro_path, micro_score = coordinate_descent_optimize(
            best_path, max_iterations=80, base_step=0.002
        )
        if micro_score > best_score:
            best_score = micro_score
            best_path = micro_path.copy()

    # Phase 9: Nano polish
    if (time.time() - start_time) < deadline - 4:
        nano_path, nano_score = coordinate_descent_optimize(
            best_path, max_iterations=60, base_step=0.001
        )
        if nano_score > best_score:
            best_score = nano_score
            best_path = nano_path.copy()

    return best_path

# EVOLVE-BLOCK-END
