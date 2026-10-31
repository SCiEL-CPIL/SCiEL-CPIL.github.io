"""Circle packing in perimeter-4 rectangle: Simulated Annealing with Hard Constraints."""
import numpy as np
import time

N = 21
_IU = np.triu_indices(N, k=1)

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin

    best_value = 0.0
    best_result = None
    
    # Precompute indices for vectorized checks
    i_idx = _IU[0]
    j_idx = _IU[1]

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.01 or h <= 0.01:
            return False, 0.0
        if np.any(r < 0):
            return False, 0.0
        # Wall checks
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
        # Pairwise checks
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        d2 = dx * dx + dy * dy
        r_sum = r[i_idx] + r[j_idx]
        if np.any(d2 < r_sum * r_sum - tol):
            return False, 0.0
        return True, float(np.sum(r))

    def get_max_radius(x, y, r, w, h, i):
        """Calculate max possible radius for circle i given current positions of others."""
        # Wall constraints
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        
        # Neighbor constraints
        # Distance to all others minus their radii
        dx = x[i] - x
        dy = y[i] - y
        d = np.sqrt(dx * dx + dy * dy)
        d[i] = np.inf # Ignore self
        r_allowed = d - r
        r_neighbor = np.min(r_allowed)
        
        return max(0.0, min(r_wall, r_neighbor))

    def expand_all(x, y, r, w, h, max_rounds=10):
        """Iteratively expand radii to local maximum."""
        r = r.copy()
        for _ in range(max_rounds):
            improved = False
            for i in range(N):
                r_max = get_max_radius(x, y, r, w, h, i)
                if r_max > r[i] + 1e-10:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
        return r

    def generate_initial_guesses():
        guesses = []
        # Square-ish aspect ratios
        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.65]
        side = int(np.ceil(np.sqrt(N)))
        
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3: continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.4
                x[i] = cx
                y[i] = cy
                r[i] = r0
            guesses.append((x, y, r, w0, h0))
            
        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(5):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                r0 = rng.uniform(0.03, 0.08)
                x[i] = rng.uniform(r0, w0 - r0)
                y[i] = rng.uniform(r0, h0 - r0)
                r[i] = r0
            guesses.append((x, y, r, w0, h0))
            
        return guesses

    def simulated_annealing(x0, y0, r0, w0, h0, time_budget=10):
        """
        SA with hard constraints. 
        Moves: 
        1. Move a circle's center.
        2. Change aspect ratio (w).
        3. Expand a circle's radius (if valid).
        Acceptance: 
        - If valid: Accept if improves sum(r). 
        - If invalid: Reject (Hard Constraint).
        - Special 'Kick': If stuck, randomize a few circles to escape local minima.
        """
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        w = w0
        h = 2.0 - w
        
        # Ensure initial state is valid by shrinking if necessary
        # Simple repair: shrink all radii until valid
        while True:
            valid, val = check_validity(x, y, r, w, h)
            if valid: break
            r *= 0.9
            if np.min(r) < 1e-6: return None
            
        best_local_val = val
        best_local_state = (x.copy(), y.copy(), r.copy(), w, h)
        
        T = 0.05 # Initial "temperature" for step size
        end_time = time.time() + time_budget
        stuck_count = 0
        max_stuck = 50
        
        while time.time() < end_time:
            # Decay temperature (step size)
            T *= 0.9995
            
            # Choose move type
            move_type = np.random.randint(0, 3)
            i = np.random.randint(0, N)
            
            new_x = x.copy()
            new_y = y.copy()
            new_r = r.copy()
            new_w = w
            new_h = h
            
            if move_type == 0: # Move circle
                step = T * 0.1
                new_x[i] += np.random.normal(0, step)
                new_y[i] += np.random.normal(0, step)
                # Clip to bounds (approximate, validity check will catch rest)
                new_x[i] = np.clip(new_x[i], 0, new_w)
                new_y[i] = np.clip(new_y[i], 0, new_h)
            elif move_type == 1: # Change Aspect Ratio
                step = T * 0.05
                new_w += np.random.normal(0, step)
                new_w = np.clip(new_w, 0.5, 1.5)
                new_h = 2.0 - new_w
                # Scale positions to fit new box? Or just let validity fail?
                # Scaling is better for continuity
                scale_x = new_w / w
                scale_y = new_h / h
                new_x = x * scale_x
                new_y = y * scale_y
                new_r = r * np.minimum(scale_x, scale_y) # Conservative radius scaling
            else: # Expand radius
                # Try to expand circle i to its max possible
                # First, check if we can expand it slightly
                r_max = get_max_radius(new_x, new_y, new_r, new_w, new_h, i)
                if r_max > new_r[i]:
                    new_r[i] = r_max
                else:
                    # Try a small random increase
                    new_r[i] += np.random.uniform(-0.005, 0.01)
                    if new_r[i] < 0: new_r[i] = 0

            valid, new_val = check_validity(new_x, new_y, new_r, new_w, new_h)
            
            if valid:
                if new_val > best_local_val:
                    best_local_val = new_val
                    best_local_state = (new_x.copy(), new_y.copy(), new_r.copy(), new_w, new_h)
                    x, y, r, w, h = new_x, new_y, new_r, new_w, new_h
                    stuck_count = 0
                else:
                    # Accept worse valid moves with probability exp((new_val - best_local_val)/T_val)
                    # Since we are maximizing, delta is negative.
                    # We use a simple acceptance criterion for valid moves that don't improve
                    # to allow wandering.
                    if np.random.rand() < np.exp((new_val - best_local_val) / (T * 2.0)):
                        x, y, r, w, h = new_x, new_y, new_r, new_w, new_h
                        stuck_count += 1
                    else:
                        stuck_count += 1
            else:
                # Invalid move: Reject
                stuck_count += 1
                # Occasionally kick if stuck
                if stuck_count > max_stuck:
                    # Randomize 3 circles
                    for k in range(3):
                        idx = np.random.randint(0, N)
                        x[idx] = np.random.uniform(0.1, w - 0.1)
                        y[idx] = np.random.uniform(0.1, h - 0.1)
                        r[idx] = 0.05 # Reset radius to allow repacking
                    # Re-expand
                    r = expand_all(x, y, r, w, h, max_rounds=5)
                    valid, val = check_validity(x, y, r, w, h)
                    if valid:
                        x, y, r, w, h = x, y, r, w, h
                        stuck_count = 0
                    else:
                        # If still invalid after kick, shrink back to last valid
                        x, y, r, w, h = best_local_state
                        stuck_count = 0
                        
        return best_local_state

    # Main Loop
    guesses = generate_initial_guesses()
    
    for i, (x0, y0, r0, w0, h0) in enumerate(guesses):
        if time.time() > deadline - 15:
            break
            
        # Initial expansion
        r_exp = expand_all(x0, y0, r0, w0, h0, max_rounds=10)
        
        # SA Optimization
        remaining = deadline - (time.time() - start_time)
        time_budget = min(10, remaining - 5)
        if time_budget < 2:
            continue
            
        result = simulated_annealing(x0, y0, r_exp, w0, h0, time_budget)
        
        if result is not None:
            x, y, r, w, h = result
            # Final expansion
            r = expand_all(x, y, r, w, h, max_rounds=20)
            valid, val = check_validity(x, y, r, w, h)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)
                
    # Final Polish on Best Result
    if best_result is not None and time.time() < deadline - 5:
        x, y, r, w, h = best_result
        # One last aggressive expansion
        r = expand_all(x, y, r, w, h, max_rounds=50)
        valid, val = check_validity(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            r0 = 0.5 / side - 0.005
            cells.append((cx, cy, r0))
        return np.array(cells)

    x, y, r, w, h = best_result
    return np.column_stack((x, y, r))
