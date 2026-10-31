"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Pack 21 circles in a perimeter-4 rectangle, maximizing sum of radii.
    Uses Simulated Annealing with a penalty-based objective function.
    """
    
    # Constants
    PERIMETER = 4.0
    MAX_ITER = 15000
    T_START = 0.5
    T_END = 1e-6
    ALPHA = (T_END / T_START) ** (1.0 / MAX_ITER)
    
    def evaluate(params):
        """
        params: [x1, y1, r1, ..., x21, y21, r21, W]
        Returns (value, penalty)
        """
        W = params[-1]
        H = 2.0 - W
        
        if W <= 0 or H <= 0:
            return -1e9, 1e9
            
        # Extract circles
        circles = params[:-1].reshape(N, 3)
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Check radii positivity
        if np.any(rs <= 0):
            return -1e9, 1e9
            
        # Boundary Violations
        viol_x_left = np.maximum(0, rs - xs)
        viol_x_right = np.maximum(0, xs - (W - rs))
        viol_y_bottom = np.maximum(0, rs - ys)
        viol_y_top = np.maximum(0, ys - (H - rs))
        
        boundary_pen = np.sum(viol_x_left**2 + viol_x_right**2 + viol_y_bottom**2 + viol_y_top**2)
        
        # Overlap Violations
        iu, ju = np.triu_indices(N, k=1)
        dx = xs[iu] - xs[ju]
        dy = ys[iu] - ys[ju]
        d_ij = np.sqrt(dx**2 + dy**2)
        min_d_ij = rs[iu] + rs[ju]
        
        diff = min_d_ij - d_ij
        pos_diff = np.maximum(0, diff)
        overlap_pen = np.sum(pos_diff**2)
        
        total_pen = boundary_pen + 10.0 * overlap_pen
        
        value = np.sum(rs)
        
        return value, total_pen

    def energy(params):
        """
        Energy to minimize.
        E = -Sum(r) + Lambda * Penalty
        """
        val, pen = evaluate(params)
        return -val + 100.0 * pen

    best_params = None
    best_energy = float('inf')
    
    for restart in range(5):
        np.random.seed(123 + restart)
        
        # Initialize positions in a grid
        cols = 5
        rows = 5
        W = np.random.uniform(0.9, 1.1)
        H = 2.0 - W
        
        params = np.zeros(N * 3 + 1)
        idx = 0
        for i in range(N):
            col = i % cols
            row = i // cols
            x = (col + 0.5) / cols * W + np.random.uniform(-0.02, 0.02)
            y = (row + 0.5) / rows * H + np.random.uniform(-0.02, 0.02)
            r = 0.15
            x = np.clip(x, r, W - r)
            y = np.clip(y, r, H - r)
            
            params[idx] = x
            params[idx+1] = y
            params[idx+2] = r
            idx += 3
        params[-1] = W
        
        current_energy = energy(params)
        current_params = params.copy()
        
        for i in range(MAX_ITER):
            T = T_START * (ALPHA ** i)
            
            new_params = current_params.copy()
            
            step_size = 0.05 * (T / T_START + 0.1)
            
            mask = np.random.rand(len(new_params)) < 0.5
            noise = np.random.normal(0, step_size, size=len(new_params))
            new_params += noise * mask
            
            new_params[-1] = np.clip(new_params[-1], 0.1, 1.9)
            
            r_indices = np.arange(2, len(new_params)-1, 3)
            new_params[r_indices] = np.maximum(new_params[r_indices], 0.01)
            
            new_energy = energy(new_params)
            
            delta_e = new_energy - current_energy
            if delta_e < 0:
                accept = True
            else:
                if T > 1e-10:
                    prob = np.exp(-delta_e / T)
                    accept = np.random.rand() < prob
                else:
                    accept = False
            
            if accept:
                current_energy = new_energy
                current_params = new_params
                
                val, pen = evaluate(current_params)
                if pen < 1e-4:
                    if best_energy is None or -val < best_energy:
                        best_energy = -val
                        best_params = current_params.copy()
        
        if best_params is None:
             val, pen = evaluate(current_params)
             if pen < 1e-4:
                 best_params = current_params.copy()
                 best_energy = -val

    if best_params is None:
        side = 5
        cells = []
        for i in range(21):
            c = i % side
            r = i // side
            x = (c + 0.5) / side
            y = (r + 0.5) / side
            rad = 0.5 / side - 0.01
            cells.append([x, y, rad])
        return np.array(cells)
    
    circles = best_params[:-1].reshape(N, 3)
    
    W = best_params[-1]
    H = 2.0 - W
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    valid = True
    for i in range(N):
        if xs[i] < rs[i] or xs[i] > W - rs[i] or ys[i] < rs[i] or ys[i] > H - rs[i]:
            valid = False
            break
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                if d < rs[i] + rs[j]:
                    valid = False
                    break
            if not valid: break
            
    if not valid:
        circles[:, 2] *= 0.9
        
    return circles
# EVOLVE-BLOCK-END
