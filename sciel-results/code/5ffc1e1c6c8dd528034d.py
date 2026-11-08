import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    # New Approach: Geometric Hexagonal Initialization + Nelder-Mead Optimization
    # This avoids the slow convergence of SA by starting from a structurally dense packing.
    
    def compute_energy(state, w_penalty=100.0, overlap_penalty=100.0):
        w = state[0]
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return 1e10
        
        x = state[1::3]
        y = state[2::3]
        r = state[3::3]
        
        # Objective: Minimize -sum(r) + penalties
        obj = -np.sum(r)
        
        # Boundary Penalties
        bound_pen = 0.0
        for i in range(N):
            if r[i] > w / 2.0 or r[i] > h / 2.0:
                bound_pen += (r[i] - min(w, h)/2.0)**2 * w_penalty
            else:
                if x[i] < r[i]:
                    bound_pen += (r[i] - x[i])**2 * w_penalty
                elif x[i] > w - r[i]:
                    bound_pen += (x[i] - (w - r[i]))**2 * w_penalty
                if y[i] < r[i]:
                    bound_pen += (r[i] - y[i])**2 * w_penalty
                elif y[i] > h - r[i]:
                    bound_pen += (y[i] - (h - r[i]))**2 * w_penalty
        
        # Overlap Penalties
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_pen += (min_dist - dist)**2 * overlap_penalty
        
        return obj + bound_pen + overlap_pen

    def is_valid(state, tol=1e-5):
        w = state[0]
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return False
        
        x = state[1::3]
        y = state[2::3]
        r = state[3::3]
        
        if np.any(r <= 0):
            return False
            
        for i in range(N):
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False
            if r[i] > w/2 + tol or r[i] > h/2 + tol:
                return False
                
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def create_hex_seed(w):
        """
        Creates a hexagonal packing of N circles in a w x (2-w) rectangle.
        We try to fit as many rows as needed.
        """
        h = 2.0 - w
        state = np.zeros(1 + 3*N)
        state[0] = w
        
        # Estimate radius based on area
        # Area = w*h. N circles. r ~ sqrt(w*h/N)
        r_est = np.sqrt(w * h / N) * 0.8
        
        # Hexagonal packing parameters
        # Horizontal distance: 2r
        # Vertical distance: sqrt(3)*r
        # Offset for odd rows: r
        
        # We need to arrange 21 circles.
        # Let's try to fit them into rows.
        # If w/h is large, we have fewer, wider rows.
        # If w/h is small, we have more, narrower rows.
        
        # Let's determine number of rows
        # Approximate number of circles per row: w / (2*r_est)
        # Total rows: N / (w / (2*r_est)) = 2*N*r_est / w
        # Let's just iterate over possible row counts
        
        best_local_state = None
        best_local_energy = 1e10
        
        # Try different row counts
        min_rows = 3
        max_rows = 7
        for n_rows in range(min_rows, max_rows + 1):
            # Distribute N circles into n_rows
            # Base count
            base = N // n_rows
            rem = N % n_rows
            
            # Calculate required radius to fit
            # Width constraint: (max_circles_in_row) * 2r <= w  => r <= w / (2 * max_circles)
            # Height constraint: (n_rows - 1) * sqrt(3) * r + 2r <= h => r <= h / (2 + (n_rows-1)*sqrt(3))
            
            max_circles = base + (1 if rem > 0 else 0) # Actually max is base + 1
            # More precise: if rem>0, some rows have base+1, some base.
            # The tightest width constraint is on the rows with max circles.
            
            r_width = w / (2.0 * (base + 1))
            r_height = h / (2.0 + (n_rows - 1) * np.sqrt(3))
            r = min(r_width, r_height) * 0.95 # Safety factor
            
            if r <= 0:
                continue
            
            # Construct positions
            curr_idx = 0
            local_state = np.zeros(1 + 3*N)
            local_state[0] = w
            
            valid = True
            for row in range(n_rows):
                # How many circles in this row?
                if row < rem:
                    n_in_row = base + 1
                else:
                    n_in_row = base
                    
                if n_in_row == 0:
                    continue
                    
                # Offset for this row
                if row % 2 == 1:
                    offset = r
                else:
                    offset = 0
                    
                # Check if it fits
                # Width needed: (n_in_row - 1) * 2r + 2r = n_in_row * 2r
                # If offset is r, the first circle is at r, last at r + (n-1)*2r.
                # Total span: (n-1)*2r + 2r = 2rn.
                # But with offset, the span is still 2rn?
                # Row 0: r, 3r, 5r... (n-1)*2r + r
                # Row 1: 2r, 4r...
                # Max x = r + (n-1)*2r = (2n-1)r.
                # Min x = r.
                # So we need (2n-1)r <= w.
                
                if (2 * n_in_row - 1) * r > w:
                    # Scale down r for this specific configuration if it doesn't fit
                    # This breaks the uniformity, but let's just skip this row count if it's bad
                    # Or adjust r
                    pass 
                
                y_pos = r + row * np.sqrt(3) * r
                
                for col in range(n_in_row):
                    x_pos = r + offset + col * 2 * r
                    
                    if curr_idx >= N:
                        break
                    if x_pos + r > w or y_pos + r > h:
                        valid = False
                        break
                    
                    local_state[1 + 3*curr_idx] = x_pos
                    local_state[2 + 3*curr_idx] = y_pos
                    local_state[3 + 3*curr_idx] = r
                    curr_idx += 1
                
                if not valid:
                    break
            
            if valid and curr_idx == N:
                e = compute_energy(local_state)
                if e < best_local_energy:
                    best_local_energy = e
                    best_local_state = local_state.copy()
        
        if best_local_state is None:
            # Fallback to grid
            return create_grid_seed(w)
            
        return best_local_state

    def create_grid_seed(w):
        h = 2.0 - w
        state = np.zeros(1 + 3*N)
        state[0] = w
        side = int(np.ceil(np.sqrt(N)))
        # Adjust side to fit N
        # Find grid dimensions
        best_side = 4
        for s in range(3, 8):
            if s * (s+1) >= N or s*s >= N:
                best_side = s
                break
        
        # Simple grid placement
        rows = int(np.ceil(N / best_side))
        cols = best_side
        
        r_est = min(w / (2*cols), h / (2*rows)) * 0.9
        
        for i in range(N):
            col = i % cols
            row = i // cols
            xi = (col + 0.5) * (w / cols)
            yi = (row + 0.5) * (h / rows)
            ri = r_est
            
            state[1 + 3*i] = xi
            state[2 + 3*i] = yi
            state[3 + 3*i] = ri
            
        return state

    best_state = None
    best_score = 0.0
    
    start_time = time.time()
    time_limit = 110.0
    
    # Try a range of aspect ratios
    # Square 1x1 is w=1.0
    # Elongated: w=1.5, 1.2, 0.8, 0.5
    
    aspect_ratios = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.9, 0.8, 0.7, 0.6, 0.5]
    
    for w_init in aspect_ratios:
        if time.time() - start_time > time_limit - 5:
            break
            
        # 1. Generate Geometric Seed
        state = create_hex_seed(w_init)
        
        # 2. Optimize with Nelder-Mead
        # Nelder-Mead is good for this size (64 variables) if the landscape is reasonably smooth
        # We use a penalty method, so it's non-smooth at boundaries, but NM handles it okay.
        
        def obj_func(vars):
            return compute_energy(vars)
            
        # Initial simplex size
        # We need to set initial_simplex for Nelder-Mead or let it auto-generate.
        # Auto-generation can be bad if scale is wrong.
        # Let's provide an initial simplex manually for better control?
        # Or just use default and hope.
        
        try:
            res = minimize(obj_func, state, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            if is_valid(res.x):
                score = np.sum(res.x[3::3])
                if score > best_score:
                    best_score = score
                    best_state = res.x.copy()
                    
        except Exception:
            pass
            
        # 3. Try a second optimization pass from the result if it's valid, with smaller steps?
        # Or just accept.
        
    if best_state is None:
        # Fallback: Grid in 1x1
        w = 1.0
        h = 1.0
        side = 5
        result = []
        for i in range(N):
            col = i % side
            row = i // side
            xi = (col + 0.5) / side
            yi = (row + 0.5) / side
            ri = 0.5 / side - 1e-4
            result.append([xi, yi, ri])
        return np.array(result)
        
    # Extract result
    w = best_state[0]
    x = best_state[1::3]
    y = best_state[2::3]
    r = best_state[3::3]
    
    result = []
    for i in range(N):
        result.append([x[i], y[i], r[i]])
        
    return np.array(result)
