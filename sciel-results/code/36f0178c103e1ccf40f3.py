"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_sum_r = 0.0
    best_config = None
    
    def evaluate_and_polish(x, y, r, W, H, max_iter=500, tol=1e-5):
        """
        Polishes a configuration by maximizing the minimum gap to walls and other circles.
        We assume the relative positions (scaled) are fixed or close, and we adjust 
        positions and W,H to maximize the scale factor (radii).
        
        Actually, a simpler and more robust approach for a fixed topology:
        1. Fix W, H.
        2. Find the maximum uniform scaling factor s such that circles with radii r_i*s 
           fit in W, H without overlapping.
        3. This is a convex feasibility problem (or close to it).
        4. We can use a simple iterative projection method or bisection on s.
        
        However, the current seed allows different radii. But for 21 circles in a small 
        rectangle, equal radii or near-equal radii are often optimal or near-optimal.
        Let's assume equal radii for the initial structural search, then allow slight 
        variations if needed.
        
        Strategy for this function:
        Given initial x, y (relative positions in [0,1]^2) and W, H:
        1. Compute the maximum possible radius R for equal circles at these positions.
           R is limited by:
           - Walls: R <= min(x_i, W-x_i, y_i, H-y_i)
           - Overlaps: 2R <= dist(i, j) for all i<j
           
        To maximize sum(r), if we assume equal radii R, we want to maximize R.
        sum(r) = N * R.
        
        We will search for the best aspect ratio W/(2-W) and the best grid layout.
        """
        pass

    def get_max_radius_for_positions(x, y, W, H):
        """
        Given positions x, y in [0, W] x [0, H], find the maximum radius R 
        such that circles of radius R at these positions are disjoint and inside.
        """
        n = len(x)
        # Wall constraints
        min_wall = np.min(np.minimum(np.minimum(x, W - x), np.minimum(y, H - y)))
        
        # Overlap constraints
        # dist(i,j) >= 2R => R <= dist(i,j)/2
        min_overlap = np.inf
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < min_overlap:
                    min_overlap = d
        
        max_r = min(min_wall, min_overlap / 2.0)
        return max_r

    def relax_positions(x, y, W, H, r, steps=100):
        """
        Simple force-directed relaxation to push circles apart and towards center.
        Used to refine positions for a given W, H and target radius r.
        """
        for _ in range(steps):
            for i in range(N):
                fx = 0.0
                fy = 0.0
                # Wall forces (soft)
                if x[i] < r[i]: fx += (r[i] - x[i])
                if W - x[i] < r[i]: fx -= (r[i] - (W - x[i]))
                if y[i] < r[i]: fy += (r[i] - y[i])
                if H - y[i] < r[i]: fy -= (r[i] - (H - y[i]))
                
                for j in range(N):
                    if i == j: continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < 1e-6:
                        dx = 1e-6
                        d = 1e-6
                    min_d = r[i] + r[j]
                    if d < min_d:
                        # Push apart
                        force = (min_d - d)
                        fx += force * (dx / d)
                        fy += force * (dy / d)
                
                x[i] += 0.01 * fx
                y[i] += 0.01 * fy
                # Clip to bounds
                x[i] = np.clip(x[i], r[i], W - r[i])
                y[i] = np.clip(y[i], r[i], H - r[i])
        
        return x, y

    def generate_grid_layout(rows, cols, W, H):
        """
        Generate a grid layout.
        """
        x = np.zeros(N)
        y = np.zeros(N)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N:
                    break
                # Center of cell
                x[idx] = (j + 0.5) * W / cols
                y[idx] = (i + 0.5) * H / rows
                idx += 1
        
        return x, y

    def generate_hex_layout(W, H):
        """
        Generate a hexagonal-like layout.
        """
        x = np.zeros(N)
        y = np.zeros(N)
        
        # Approximate hex grid
        # 5 rows of 4, 3, 4, 3, 4? Total 18. Need 21.
        # Maybe 4 rows: 5, 4, 5, 4, 3? 
        # Let's try a standard hex packing pattern.
        # For N=21, a 4x5 grid is 20. Add 1?
        # Let's just use a distorted grid or a specific hex count.
        # 5 rows: 4, 5, 4, 5, 3 = 21?
        rows_counts = [4, 5, 4, 5, 3]
        
        idx = 0
        for i, count in enumerate(rows_counts):
            for j in range(count):
                if idx >= N:
                    break
                # Horizontal spacing
                x[idx] = (j + 0.5) * W / count
                # Vertical spacing
                y[idx] = (i + 0.5) * H / len(rows_counts)
                
                # Shift every other row by half a cell width for hex effect
                if i % 2 == 1:
                    x[idx] += W / (2 * count)
                
                idx += 1
        
        return x, y

    def optimize_aspect_ratio(x0, y0, W_start, H_start, r_start):
        """
        Optimize W (and thus H=2-W) to maximize the minimum radius.
        We scale positions to fit the new W, H.
        """
        def score(W):
            H = 2.0 - W
            if H <= 0.1 or W <= 0.1:
                return -1.0
            
            # Scale positions
            x = x0 * W
            y = y0 * H
            
            # Relax positions slightly to optimize for this aspect ratio
            # Use a small number of steps to keep it fast
            x_rel, y_rel = relax_positions(x, y, W, H, np.full(N, r_start), steps=10)
            
            # Calculate max radius
            r_max = get_max_radius_for_positions(x_rel, y_rel, W, H)
            return N * r_max

        # Golden section search
        a = 0.4
        b = 1.6
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = score(c)
        fd = score(d)
        
        for _ in range(20):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = score(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = score(c)
        
        W_opt = (a + b) / 2
        H_opt = 2.0 - W_opt
        return W_opt, H_opt

    # Main loop
    np.random.seed(42)
    
    # Define candidate topologies
    topologies = [
        ("grid_4x5", 4, 5),
        ("grid_5x4", 5, 4),
        ("grid_3x7", 3, 7),
        ("grid_7x3", 7, 3),
        ("hex", None, None), # Special case
    ]
    
    for name, rows, cols in topologies:
        if time.time() > deadline - 10:
            break
            
        # Initial guess for aspect ratio
        # For grid rows x cols, aspect ratio W/H should be roughly cols/rows
        if rows and cols:
            W_init = 1.0 / (1.0 + rows/cols) # W/(W+H) = W/2 -> W/H = 1/(1+W/H)... 
            # W+H=2. W/H = cols/rows => W = H * cols/rows.
            # H(cols/rows + 1) = 2 => H = 2 / (cols/rows + 1)
            # W = 2 * (cols/rows) / (cols/rows + 1)
            W_init = 2.0 * (cols/rows) / (cols/rows + 1.0)
            H_init = 2.0 - W_init
        else:
            W_init = 1.0
            H_init = 1.0
            
        # Generate initial layout
        if rows:
            x0, y0 = generate_grid_layout(rows, cols, W_init, H_init)
        else:
            x0, y0 = generate_hex_layout(W_init, H_init)
        
        # Normalize x0, y0 to [0,1] for aspect ratio optimization
        x0_norm = x0 / W_init
        y0_norm = y0 / H_init
        
        # Initial radius estimate
        r_init = get_max_radius_for_positions(x0, y0, W_init, H_init)
        
        # Optimize aspect ratio
        W_opt, H_opt = optimize_aspect_ratio(x0_norm, y0_norm, W_init, H_init, r_init)
        
        # Scale positions back
        x = x0_norm * W_opt
        y = y0_norm * H_opt
        
        # Detailed relaxation with the optimized aspect ratio
        # Start with the estimated max radius
        r_est = get_max_radius_for_positions(x, y, W_opt, H_opt)
        
        # Run a more detailed relaxation to see if we can improve positions
        # We want to maximize the minimum radius.
        # Let's run a few iterations of "grow and relax"
        current_r = r_est * 0.9
        for _ in range(50):
            # Grow
            current_r *= 1.01
            # Relax
            x, y = relax_positions(x, y, W_opt, H_opt, np.full(N, current_r), steps=5)
            # Check if valid
            r_check = get_max_radius_for_positions(x, y, W_opt, H_opt)
            if r_check < current_r:
                # Shrink back
                current_r = r_check
                break
            if time.time() > deadline - 5:
                break
        
        # Final check
        r_final = get_max_radius_for_positions(x, y, W_opt, H_opt)
        
        # If we found a valid configuration, check if it's better
        # We assume equal radii for this structural search.
        total_r = N * r_final
        
        if total_r > best_sum_r:
            best_sum_r = total_r
            # Create the output array
            # x, y are centers. r_final is the radius.
            config = np.zeros((N, 3))
            config[:, 0] = x
            config[:, 1] = y
            config[:, 2] = r_final
            best_config = config

    # Fallback if nothing found (should not happen)
    if best_config is None:
        # Simple grid fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_config = np.array(cells[:N])

    return best_config

# EVOLVE-BLOCK-END
