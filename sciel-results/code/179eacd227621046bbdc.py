# EVOLVE-BLOCK-START
import numpy as np
import time


def _base_corridor_mask(grid_x, grid_y):
    """Return a boolean mask of the base corridor (a=0) for the given grid points."""
    # Base corridor: union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
    mask1 = (grid_x >= 0) & (grid_x <= 1) & (grid_y >= -3) & (grid_y <= 1)
    mask2 = (grid_x >= -3) & (grid_x <= 1) & (grid_y >= 0) & (grid_y <= 1)
    return mask1 | mask2


def _evaluate_area(path, sofa_x, sofa_y, base_mask_flat, base_x, base_y):
    """
    Compute the number of sofa points inside all placements.
    
    path: (n, 3) array of (x, y, angle_degrees)
    sofa_x, sofa_y: 1D arrays of sofa point coordinates
    base_mask_flat: 1D boolean mask of base corridor for the grid
    base_x, base_y: 1D arrays of grid point coordinates (same shape as base_mask_flat)
    
    Returns: number of sofa points that are inside ALL placements.
    """
    n_placements = len(path)
    n_sofa = len(sofa_x)
    
    # We need to check each sofa point against each placement.
    # For each placement (x, y, a), a sofa point (sx, sy) is inside if
    # R(-a)(sofa - (x,y)) is in the base corridor.
    
    # To be efficient, for each placement, transform all sofa points into the
    # base corridor frame and check against the base mask.
    
    # But the base mask is on a fixed grid. We need to check if the transformed
    # point lies in the base corridor region (which is defined analytically).
    
    # Actually, the base corridor is a union of two axis-aligned rectangles, so
    # we can check analytically without a mask.
    
    total_inside = np.ones(n_sofa, dtype=bool)
    
    for i in range(n_placements):
        px, py, pa = path[i]
        a_rad = np.deg2rad(pa)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # Transform sofa points: w' = R(-a)(w - (px, py))
        # R(-a) = [[cos_a, sin_a], [-sin_a, cos_a]]
        dx = sofa_x - px
        dy = sofa_y - py
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        # Check if (u, v) is in base corridor
        in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        in_base = in_rect1 | in_rect2
        
        total_inside &= in_base
    
    return np.sum(total_inside)


def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path (n, 3) array for the moving sofa problem.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Sofa grid points
    sofa_x = np.arange(-3, 1.5, 0.01)
    sofa_y = np.arange(-1.5, 1.5, 0.01)
    sofa_X, sofa_Y = np.meshgrid(sofa_x, sofa_y, indexing='ij')
    sofa_x_flat = sofa_X.ravel()
    sofa_y_flat = sofa_Y.ravel()
    n_sofa = len(sofa_x_flat)
    
    def _arc_path(h, n_points=9):
        """
        Generate a path following a circular arc.
        h: y-coordinate of the arc center (center is at (-1, h))
        n_points: number of intermediate points (excluding endpoints)
        
        Returns: (n_points+2, 3) array with start, intermediate, end points.
        """
        R = np.sqrt(1.0 + h * h)
        cx, cy = -1.0, h
        
        # Angles from center to start and end points
        angle_start = np.arctan2(0 - cy, 0 - cx)  # angle to (0,0)
        angle_end = np.arctan2(0 - cy, -2 - cx)   # angle to (-2,0)
        
        # We want the arc to go from (0,0) to (-2,0) via the upper region.
        # For h > 0, the center is above, so the arc bulges upward.
        # angle_start is in the 4th quadrant (positive x, negative y relative to center)
        # angle_end is in the 3rd quadrant (negative x, negative y relative to center)
        
        # The arc should sweep from angle_start to angle_end going counterclockwise
        # (through the top). Let's compute the sweep.
        
        # Actually, let's parametrize by the fraction t in [0, 1]
        # and interpolate the angle along the arc.
        
        # Determine the correct arc direction
        # The angle from center to (0,0): atan2(-h, 1)
        # The angle from center to (-2,0): atan2(-h, -1)
        
        # For h > 0:
        # atan2(-h, 1) is negative (4th quadrant)
        # atan2(-h, -1) is in 3rd quadrant, around -pi + something
        
        # We want to go from (0,0) to (-2,0) via the top, which means
        # increasing the angle (counterclockwise) from the 4th quadrant
        # through 0, 1st, 2nd, to the 3rd quadrant.
        
        # Let's compute the angular positions
        a0 = np.arctan2(-h, 1.0)   # angle to (0,0)
        a1 = np.arctan2(-h, -1.0)  # angle to (-2,0)
        
        # For h > 0, a0 is in (-pi/2, 0) and a1 is in (-pi, -pi/2)
        # Going counterclockwise from a0 to a1 means going the long way:
        # a0 -> 0 -> pi/2 -> pi -> -pi -> a1
        # That's a sweep of (2*pi - (a0 - a1)) if a0 > a1
        
        # Actually, the short arc from a0 to a1 going counterclockwise:
        # sweep = (a1 - a0) mod 2*pi
        sweep = (a1 - a0) % (2 * np.pi)
        
        # If sweep is close to 2*pi, we're going the long way.
        # For h > 0, we want the arc that goes through the top, which should
        # be the one with sweep < pi (the short arc) if the center is above.
        
        # Let's just use the short arc for now and see.
        
        # Generate n_points+1 intermediate fractions (excluding t=0 and t=1)
        if n_points <= 0:
            return np.array([[0, 0, 0], [-2, 0, 90]], dtype=np.float64)
        
        # t values for intermediate points
        t_vals = np.linspace(0, 1, n_points + 2)[1:-1]  # exclude endpoints
        
        path = []
        
        # Start point
        path.append([0.0, 0.0, 0.0])
        
        for t in t_vals:
            angle = a0 + t * sweep
            x = cx + R * np.cos(angle)
            y = cy + R * np.sin(angle)
            a_deg = 90.0 * t
            path.append([x, y, a_deg])
        
        # End point
        path.append([-2.0, 0.0, 90.0])
        
        return np.array(path, dtype=np.float64)
    
    def _score(h, n_points=9):
        """Evaluate the area for a given arc parameter h."""
        path = _arc_path(h, n_points)
        count = _evaluate_area(path, sofa_x_flat, sofa_y_flat, None, None, None)
        return count * 0.0001
    
    # Initial search over h
    # h is the y-coordinate of the arc center. Reasonable range: [-5, 5]
    best_h = 1.0
    best_score = -1
    
    # Coarse search
    h_vals = np.linspace(-3, 3, 61)
    for h in h_vals:
        if time.time() > deadline - 60:
            break
        s = _score(h, n_points=9)
        if s > best_score:
            best_score = s
            best_h = h
    
    # Fine search around best_h
    if time.time() < deadline - 30:
        h_fine = np.linspace(best_h - 0.5, best_h + 0.5, 51)
        for h in h_fine:
            if time.time() > deadline - 30:
                break
            s = _score(h, n_points=9)
            if s > best_score:
                best_score = s
                best_h = h
    
    # Even finer search
    if time.time() < deadline - 15:
        h_fine2 = np.linspace(best_h - 0.1, best_h + 0.1, 41)
        for h in h_fine2:
            if time.time() > deadline - 15:
                break
            s = _score(h, n_points=9)
            if s > best_score:
                best_score = s
                best_h = h
    
    # Generate the final path
    final_path = _arc_path(best_h, n_points=9)
    
    # Ensure the path is valid (n >= 1, all finite)
    if not np.all(np.isfinite(final_path)):
        final_path = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    
    return final_path


# EVOLVE-BLOCK-END
