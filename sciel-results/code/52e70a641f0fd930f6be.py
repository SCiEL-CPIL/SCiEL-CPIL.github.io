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
    (-2, 0, 90), then densifies the path to <=1-degree rotation steps, exactly as in the
    AlphaEvolve "single turn" corridor notebook.

    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees) for the
            corridor's inner corner.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep margin

    # --- Fast evaluation of a path's score (sofa area) ---
    # Grid points
    u = np.arange(-3.0, 1.5, 0.01)
    v = np.arange(-1.5, 1.5, 0.01)
    gu, gv = np.meshgrid(u, v, indexing='ij')
    grid_points = np.column_stack((gu.ravel(), gv.ravel()))
    N = grid_points.shape[0]

    def densify_path(path):
        """
        Prepend (0,0,0), append (-2,0,90), then insert linear interpolations
        between consecutive rows so that each segment spans <= 1 degree of rotation.
        """
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([-2.0, 0.0, 90.0])
        # Build full sequence: start, *path, end
        full = np.vstack(([start], path, [end]))
        
        # Ensure angles are monotonically non-decreasing? The problem says a in degrees.
        # The evaluator says: "Between consecutive rows it adds linear interpolations of (x, y, a) in m = max(1, ceil(|a2 - a1|)) equal steps"
        # This implies we just need to handle the segments.
        
        dense_points = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i + 1]
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            # If m == 1, we just add p2.
            # If m > 1, we interpolate.
            # The evaluator adds m equal steps. This usually means m-1 intermediate points + the endpoint?
            # "adds linear interpolations ... in m ... equal steps"
            # Let's assume it creates m-1 new points between p1 and p2, or m points total including p2?
            # Standard interpretation: divide the segment into m parts.
            # Points: p1, p1 + (p2-p1)/m, ..., p2.
            # The loop already added p1 (or the previous endpoint).
            # We need to add the new points.
            # If we add m-1 points, the last one is p2.
            # Let's check the wording: "adds linear interpolations ... in m ... equal steps".
            # If I have 10 points and I add 1 step, do I get 11 or 12?
            # Usually "interpolate in m steps" means create m intervals.
            # So we add m-1 points between p1 and p2, and p2 itself is the end of the interval.
            # But p2 is the start of the next interval.
            # So we should add points at t = 1/m, 2/m, ..., (m-1)/m.
            # And the next iteration will start at p2.
            # Wait, if m=1, we add 0 intermediate points. The next start is p2.
            # This seems consistent.
            
            if m > 1:
                ts = np.linspace(1.0 / m, 1.0 - 1.0 / m, m - 1)
                new_pts = p1 + ts[:, np.newaxis] * (p2 - p1)
                dense_points.extend(new_pts)
            # p2 will be added in the next iteration's start or at the very end?
            # The loop structure:
            # dense_points starts with full[0].
            # Iteration i=0: adds intermediates between full[0] and full[1].
            # Next iteration i=1: starts with full[1].
            # So we need to ensure full[1] is in the list before iteration 1?
            # No, the list is just a list of points.
            # Let's just append p2 at the end of each segment processing?
            # No, p2 is the start of the next segment.
            # If we don't add p2 now, it will be the 'p1' of the next iteration.
            # But the list needs to contain the points in order.
            # So:
            # 1. Add intermediates.
            # 2. Add p2.
            # Then next iteration uses p2 as p1.
            
            dense_points.append(p2)
            
        return np.array(dense_points)

    def score_path(path):
        """
        Compute the area of the sofa defined by the path.
        """
        if path.shape[0] == 0:
            # If empty, evaluator adds start and end.
            # But we should handle this.
            pass
            
        dense = densify_path(path)
        
        # Check inside all placements
        # For each placement (x, y, a), check if grid point is inside.
        # Inside if R(-a)(w - (x,y)) is in base corridor.
        # Base corridor: Union of two rectangles.
        # Rect 1: 0 <= u <= 1, -3 <= v <= 1
        # Rect 2: -3 <= u <= 1, 0 <= v <= 1
        
        # Vectorized check:
        # For each grid point w, and each placement p, compute local coords.
        # This is N * M operations. N=45000, M can be large.
        # M is roughly 90 (for 90 deg rotation) + number of path points.
        # If path has 100 points, M ~ 190.
        # 45000 * 190 = 8.5M checks.
        # Each check involves rotation and comparison.
        # This is doable in numpy.
        
        # However, we can optimize by checking if the point is in the INTERSECTION.
        # A point is in the sofa if it is in ALL placements.
        # So we can iterate over placements and mask out points that are NOT in that placement.
        
        inside_all = np.ones(N, dtype=bool)
        
        for x, y, a_deg in dense:
            a_rad = np.deg2rad(a_deg)
            c = np.cos(a_rad)
            s = np.sin(a_rad)
            
            # R(-a) = [[c, s], [-s, c]]
            # w_local = R(-a) * (w - p)
            # u_local = c*(w_x - x) + s*(w_y - y)
            # v_local = -s*(w_x - x) + c*(w_y - y)
            
            wx = grid_points[:, 0] - x
            wy = grid_points[:, 1] - y
            
            ul = c * wx + s * wy
            vl = -s * wx + c * wy
            
            # Check base corridor
            # Rect 1: 0 <= ul <= 1 AND -3 <= vl <= 1
            # Rect 2: -3 <= ul <= 1 AND 0 <= vl <= 1
            # Union: (Rect 1) OR (Rect 2)
            
            in_r1 = (ul >= 0.0) & (ul <= 1.0) & (vl >= -3.0) & (vl <= 1.0)
            in_r2 = (ul >= -3.0) & (ul <= 1.0) & (vl >= 0.0) & (vl <= 1.0)
            
            in_base = in_r1 | in_r2
            
            inside_all &= in_base
            
            # Early exit if no points left
            if not np.any(inside_all):
                break
        
        return np.sum(inside_all) * 0.0001

    # --- Search Strategy ---
    # 1. Start with the baseline path from the current program (score 1.0).
    # 2. Try the astroid-like path from inspiration (score 0.1, but maybe with tweaks?).
    #    The inspiration used x = 2cos^3(t)-2, y=sin^3(t).
    #    This path goes from (0,0) to (-2,1). The end state is (-2,0,90).
    #    The evaluator interpolates to (-2,0,90).
    #    The jump from y=1 to y=0 might be bad.
    #    Let's try to fit the astroid to the end point (-2,0).
    #    Standard astroid for moving sofa (Hammersley) is x = c cos^3(t), y = c sin^3(t).
    #    Here the corridor moves.
    #    The "sofa" is the intersection of corridors.
    #    The optimal path for the inner corner is known to be related to an astroid.
    #    Specifically, the locus of the inner corner for the maximal sofa is an astroid.
    #    Equation: (x+1)^(2/3) + y^(2/3) = (something)^(2/3)?
    #    Let's check the baseline path.
    #    Baseline:
    #    [-0.1, 0.1, 15.0]
    #    ...
    #    [-0.7, 0.3, 86.0]
    #    
    #    Let's try to optimize the baseline path using local search.
    #    We have ~600s.
    #    score_path takes ~0.1s?
    #    Let's time it.
    
    # Baseline path
    best_path = np.array(
        [
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ],
        dtype=np.float64,
    )
    
    best_score = score_path(best_path)
    
    # Time check
    elapsed = time.time() - start_time
    if elapsed > 580.0:
        return best_path
    
    # Try astroid variations
    # The inspiration's astroid was bad. Why?
    # Maybe the y-scale was too big.
    # Let's try scaling y.
    # x = 2 * cos^3(t) - 2
    # y = k * sin^3(t)
    # We want y to end at 0.
    # So y(t) = k * sin^3(t) - k? No, y(0)=0, y(pi/2)=0.
    # So y(t) = k * sin(t) * (1 - sin(t))? Or just sin(t)?
    # If we use y = k * sin(t), it ends at k.
    # We need it to end at 0.
    # So maybe y = k * sin(2t)?
    # Or just use the baseline and perturb it.
    
    # Local search: perturb each point slightly.
    np.random.seed(42)
    
    n_points = best_path.shape[0]
    
    # Define a perturbation function
    def perturb_path(path, sigma_xy=0.05, sigma_a=2.0):
        new_path = path.copy()
        # Don't change angles too much? The angles define the sequence.
        # Actually, the angles in the path are just waypoints.
        # The evaluator densifies based on angle difference.
        # So we can change angles, but they should be monotonically increasing?
        # The problem doesn't explicitly say they must be monotonic, but the physical
        # motion implies a continuous rotation.
        # If we go 15 -> 20 -> 15, the evaluator will handle the back-and-forth.
        # But it might create a weird shape.
        # Let's assume monotonic increase is good.
        
        # Perturb x, y
        new_path[:, 0] += np.random.normal(0, sigma_xy, n_points)
        new_path[:, 1] += np.random.normal(0, sigma_xy, n_points)
        # Perturb angles
        # Keep them roughly in order
        new_path[:, 2] += np.random.normal(0, sigma_a, n_points)
        
        # Enforce monotonicity?
        # Sort by angle?
        # If we sort, we change the path order.
        # The path order matters for the "between consecutive rows" interpolation.
        # If we sort, we might create a different path.
        # Let's just clamp angles to [0, 90] and ensure they are increasing.
        # Simplest: don't perturb angles, or perturb very little.
        # Or, perturb angles and then sort the path by angle.
        # Sorting by angle changes the spatial path.
        # Let's try sorting by angle.
        
        # Actually, the baseline angles are: 15, 20, 34, 40, 50, 55, 75, 86.
        # They are increasing.
        # Let's just perturb x,y and keep angles fixed first.
        # Then try perturbing angles.
        
        return new_path

    # Hill climbing
    # Try 1000 steps if time permits.
    max_steps = 2000
    step_count = 0
    
    current_path = best_path.copy()
    current_score = best_score
    
    # Adaptive sigma
    sigma_xy = 0.05
    sigma_a = 1.0
    
    while step_count < max_steps:
        if time.time() > deadline - 20.0:
            break
            
        new_path = current_path.copy()
        
        # Randomly choose to perturb x, y or angles
        if np.random.rand() < 0.7:
            # Perturb x, y
            idx = np.random.randint(n_points)
            new_path[idx, 0] += np.random.normal(0, sigma_xy)
            new_path[idx, 1] += np.random.normal(0, sigma_xy)
        else:
            # Perturb angle
            idx = np.random.randint(n_points)
            new_path[idx, 2] += np.random.normal(0, sigma_a)
            # Clamp angle to [0, 90]
            new_path[idx, 2] = np.clip(new_path[idx, 2], 0.0, 90.0)
            
        new_score = score_path(new_path)
        
        if new_score > current_score:
            current_path = new_path
            current_score = new_score
            if current_score > best_score:
                best_path = current_path.copy()
                best_score = current_score
            # Increase sigma slightly? Or keep same?
            # Let's keep same to explore locally.
        else:
            # Decrease sigma
            sigma_xy *= 0.95
            sigma_a *= 0.95
            
        step_count += 1
        
        # Print progress every 100 steps? No, keep quiet.
        
    return best_path


# EVOLVE-BLOCK-END
