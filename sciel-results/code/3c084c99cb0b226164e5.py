import numpy as np
from scipy.optimize import minimize

# Pre-compute grid
u_grid = np.arange(-3, 1.5, 0.01)
v_grid = np.arange(-1.5, 1.5, 0.01)
U, V = np.meshgrid(u_grid, v_grid)
grid_points = np.column_stack((U.ravel(), V.ravel()))

def compute_area(path_params):
    # path_params: flattened array of x, y for each waypoint
    # Waypoints angles are fixed: [15, 30, 45, 60, 75]
    angles = np.array([15, 30, 45, 60, 75])
    x = path_params[0::2]
    y = path_params[1::2]
    
    # Construct full path with start and end
    # Start: (0, 0, 0), End: (-2, 0, 90)
    all_x = np.concatenate(([0.0], x, [-2.0]))
    all_y = np.concatenate(([0.0], y, [0.0]))
    all_a = np.concatenate(([0.0], angles, [90.0]))
    
    # Interpolate to 1 degree steps
    # We need to generate points such that delta a <= 1
    # We can use np.interp for x and y based on a
    # But the evaluator does linear interpolation in (x,y,a) between consecutive rows.
    # So if we have rows (x1,y1,a1) and (x2,y2,a2), it creates intermediate points.
    # The number of steps is m = max(1, ceil(|a2-a1|)).
    # So we can simulate this.
    
    placements = []
    for i in range(len(all_x) - 1):
        x1, y1, a1 = all_x[i], all_y[i], all_a[i]
        x2, y2, a2 = all_x[i+1], all_y[i+1], all_a[i+1]
        m = max(1, int(np.ceil(np.abs(a2 - a1))))
        # Steps: a1, a1 + (a2-a1)/m, ..., a2
        # Actually, the evaluator adds linear interpolations in m equal steps.
        # This implies m+1 points including endpoints? Or m intervals?
        # "adds linear interpolations ... in m = ... equal steps"
        # Usually this means m intervals, so m+1 points. But the endpoints are already in the list.
        # So we add m-1 intermediate points.
        if m > 1:
            ts = np.linspace(0, 1, m+1)[1:-1] # intermediate points
            for t in ts:
                xi = x1 + t * (x2 - x1)
                yi = y1 + t * (y2 - y1)
                ai = a1 + t * (a2 - a1)
                placements.append((xi, yi, ai))
        # Add the end point of this segment (which is start of next)
        # We will add all unique points at the end or just append.
        # Let's just append the end point for now, and remove duplicates later.
        placements.append((x2, y2, a2))
        
    # Remove duplicates
    unique_placements = []
    prev = None
    for p in placements:
        if prev is None or np.abs(p[0]-prev[0])>1e-9 or np.abs(p[1]-prev[1])>1e-9 or np.abs(p[2]-prev[2])>1e-9:
            unique_placements.append(p)
            prev = p
            
    # Compute intersection
    # For each point in grid, check if it is inside all placements
    # inside = True for all points initially
    inside = np.ones(len(grid_points), dtype=bool)
    
    for x, y, a in unique_placements:
        # Rotate grid points by -a
        rad = -np.radians(a)
        c = np.cos(rad)
        s = np.sin(rad)
        # w - (x,y)
        wx = grid_points[:, 0] - x
        wy = grid_points[:, 1] - y
        # R(-a) * (w - (x,y))
        # [ c, -s; s, c ] * [wx; wy]
        u = c * wx - s * wy
        v = s * wx + c * wy
        
        # Check if in base L-shape
        # Base: (0<=u<=1 and -3<=v<=1) or (-3<=u<=1 and 0<=v<=1)
        in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        in_base = in_rect1 | in_rect2
        
        inside &= in_base
        
        # Early exit if no points left
        if not np.any(inside):
            return 0.0
            
    area = np.sum(inside) * 0.0001
    return area

def moving_sofa_2d() -> np.ndarray:
    # Initial guess
    # Let's try a simple arc
    # x = -0.5 * sin(a), y = 0.5 * (1 - cos(a)) ?
    # Let's just start with the baseline path from the prompt, or a simple one.
    # Baseline:
    # [-0.1, 0.1, 15.0], [-0.2, 0.2, 20.0], ...
    # Let's extract x,y from baseline for 15, 30, 45, 60, 75
    # 15: -0.1, 0.1
    # 30: interpolate between 20 and 34? No, let's just use a simple guess.
    x0 = np.array([-0.2, -0.3, -0.3, -0.4, -0.5])
    y0 = np.array([0.2, 0.3, 0.3, 0.2, 0.2])
    params0 = np.concatenate([x0, y0])
    
    # Optimize
    # We want to maximize area, so minimize -area
    res = minimize(lambda p: -compute_area(p), params0, method='Nelder-Mead', 
                   options={'maxiter': 500, 'xatol': 1e-3, 'fatol': 1e-5})
    
    best_params = res.x
    x = best_params[0::2]
    y = best_params[1::2]
    angles = np.array([15, 30, 45, 60, 75])
    
    path = np.column_stack((x, y, angles))
    return path
