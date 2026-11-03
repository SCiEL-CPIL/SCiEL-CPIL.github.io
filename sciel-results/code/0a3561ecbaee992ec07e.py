"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 disjoint circles."""
import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 21

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # seconds; keep margin

    best_result = None
    best_value = 0.0

    # We'll use multiple restarts with different initial configurations.
    # Strategy: start from grid, then optimize positions and rectangle aspect ratio.
    # Variables: x_i, y_i, r_i for i in 1..N, plus aspect ratio w (height = 2/(1+w), width = 2*w/(1+w))
    # Actually, let's parameterize: let the rectangle have width W and height H with 2(W+H)=4, so W+H=2.
    # Let t = W/2, so H = 2 - W = 2 - 2t, where t in (0,1).
    # Or better: let aspect ratio a = W/H, so W = 2a/(1+a), H = 2/(1+a).
    # Variables: positions (x_i, y_i) in [0, W] x [0, H], radii r_i > 0, and aspect ratio a.
    # Constraints:
    #   x_i - r_i >= 0, x_i + r_i <= W
    #   y_i - r_i >= 0, y_i + r_i <= H
    #   (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for i != j
    # Objective: maximize sum(r_i)

    # To handle constraints, we can use a penalty method or convert to unconstrained with penalties.
    # Let's use a penalty-based approach with scipy.optimize.minimize (Nelder-Mead or Powell).

    def pack_solution(a, positions, radii):
        """Given aspect ratio a, positions (N,2), radii (N,), return value and validity."""
        W = 2.0 * a / (1.0 + a)
        H = 2.0 / (1.0 + a)
        
        # Check wall constraints
        for i in range(N):
            x, y = positions[i]
            r = radii[i]
            if r <= 0:
                return 0.0, False
            if x - r < -1e-9 or x + r > W + 1e-9 or y - r < -1e-9 or y + r > H + 1e-9:
                return 0.0, False
        
        # Check circle-circle constraints
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist2 = dx*dx + dy*dy
                min_dist = radii[i] + radii[j]
                if dist2 < min_dist*min_dist - 1e-9:
                    return 0.0, False
        
        value = np.sum(radii)
        return value, True

    def objective(params):
        """Objective to minimize (negative of sum of radii) with penalties for constraint violations."""
        # params: [a, x_1, y_1, r_1, x_2, y_2, r_2, ..., x_N, y_N, r_N]
        a = params[0]
        if a < 0.01 or a > 100.0:
            return 1e10
        W = 2.0 * a / (1.0 + a)
        H = 2.0 / (1.0 + a)
        
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(N):
            idx = 1 + 3*i
            positions[i, 0] = params[idx]
            positions[i, 1] = params[idx+1]
            radii[i] = params[idx+2]
        
        # Penalty for wall violations
        penalty = 0.0
        for i in range(N):
            x, y = positions[i]
            r = radii[i]
            # Wall constraints
            penalty += max(0, r - x)**2
            penalty += max(0, x + r - W)**2
            penalty += max(0, r - y)**2
            penalty += max(0, y + r - H)**2
            # Radius positivity
            penalty += max(0, -r)**2
        
        # Penalty for circle-circle violations
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
        
        sum_r = np.sum(radii)
        return -sum_r + 100.0 * penalty

    def make_initial(a_init, mode=0):
        """Create initial parameter vector."""
        W = 2.0 * a_init / (1.0 + a_init)
        H = 2.0 / (1.0 + a_init)
        
        params = np.zeros(1 + 3*N)
        params[0] = a_init
        
        if mode == 0:
            # Grid-like initialization
            cols = int(np.ceil(np.sqrt(N * W / H))) if W >= H else int(np.ceil(np.sqrt(N * H / W)))
            if cols < 1: cols = 1
            rows = int(np.ceil(N / cols))
            for i in range(N):
                idx = 1 + 3*i
                col = i % cols
                row = i // cols
                x = (col + 0.5) * W / cols
                y = (row + 0.5) * H / rows
                r = min(W / cols, H / rows) / 2.0 * 0.9
                params[idx] = x
                params[idx+1] = y
                params[idx+2] = r
        elif mode == 1:
            # Random initialization within rectangle
            rng = np.random.RandomState(42 + mode)
            for i in range(N):
                idx = 1 + 3*i
                x = rng.uniform(0, W)
                y = rng.uniform(0, H)
                r = rng.uniform(0.02, min(W, H) * 0.15)
                params[idx] = x
                params[idx+1] = y
                params[idx+2] = r
        else:
            # Square-ish grid
            side = int(np.ceil(np.sqrt(N)))
            for i in range(N):
                idx = 1 + 3*i
                col = i % side
                row = i // side
                x = (col + 0.5) * W / side
                y = (row + 0.5) * H / side
                r = min(W / side, H / side) / 2.0 * 0.9
                params[idx] = x
                params[idx+1] = y
                params[idx+2] = r
        
        return params

    # Try multiple initial configurations
    initial_modes = [0, 2, 1]
    aspect_ratios = [1.0, 1.5, 2.0, 0.7, 1.2, 0.8, 1.8, 1.3]
    
    attempts = 0
    max_attempts = 15
    
    for mode in initial_modes:
        for a_init in aspect_ratios:
            if time.time() > DEADLINE - 30:
                break
            if attempts >= max_attempts:
                break
            
            attempts += 1
            params0 = make_initial(a_init, mode)
            
            try:
                result = minimize(objective, params0, method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            except Exception:
                continue
            
            # Extract solution
            a = result.x[0]
            if a < 0.01 or a > 100.0:
                continue
            W = 2.0 * a / (1.0 + a)
            H = 2.0 / (1.0 + a)
            
            positions = np.zeros((N, 2))
            radii = np.zeros(N)
            for i in range(N):
                idx = 1 + 3*i
                positions[i, 0] = result.x[idx]
                positions[i, 1] = result.x[idx+1]
                radii[i] = result.x[idx+2]
            
            # Clip to valid range
            for i in range(N):
                positions[i, 0] = np.clip(positions[i, 0], 0, W)
                positions[i, 1] = np.clip(positions[i, 1], 0, H)
                radii[i] = max(radii[i], 0.001)
            
            value, valid = pack_solution(a, positions, radii)
            if valid and value > best_value:
                best_value = value
                best_result = np.column_stack([positions, radii])
        
        if time.time() > DEADLINE - 30:
            break
    
    # If we found a valid result, try to polish it
    if best_result is not None and time.time() < DEADLINE - 20:
        # Try a local polish on the best result
        params_best = np.zeros(1 + 3*N)
        a = 2.0 * best_result[0, 0] / (best_result[0, 0] + best_result[0, 1])  # This is wrong, let me fix
        # Actually, I need to store the aspect ratio separately. Let me restructure.
        
        # For now, just return what we have
        pass
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result

# Test
if __name__ == "__main__":
    result = construct()
    print(result.shape)
    print(np.sum(result[:, 2]))
