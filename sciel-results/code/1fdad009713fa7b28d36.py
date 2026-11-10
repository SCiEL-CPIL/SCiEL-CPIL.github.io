import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    x_grid, y_grid = np.meshgrid(xs, ys)
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    n_points = len(x_flat)
    
    # Time management
    start_time = time.time()
    time_limit = 575.0  # Safety margin before 630s kill
    
    def _make_densified(path):
        """Build the full densified path: start + path + end, with linear interpolation."""
        full = [(0.0, 0.0, 0.0)]
        for p in path:
            full.append((float(p[0]), float(p[1]), float(p[2])))
        full.append((-2.0, 0.0, 90.0))
        dens = []
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(m):
                t = (k + 1) / m
                x = p1[0] + (p2[0] - p1[0]) * t
                y = p1[1] + (p2[1] - p1[1]) * t
                a = p1[2] + (p2[2] - p1[2]) * t
                dens.append((x, y, a))
        dens.append(full[-1])
        return dens
    
    def _placement_mask(cx, cy, ca_deg):
        """Compute mask for a single placement."""
        ca = np.deg2rad(ca_deg)
        cos_a = np.cos(ca)
        sin_a = np.sin(ca)
        
        dx = x_flat - cx
        dy = y_flat - cy
        
        u = dx * cos_a + dy * sin_a
        v = -dx * sin_a + dy * cos_a
        
        in_arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        
        return in_arm1 | in_arm2
    
    def evaluate_path(path):
        """path: list of (x, y, a). Returns area score."""
        dens = _make_densified(path)
        mask = np.ones(n_points, dtype=bool)
        
        for (cx, cy, ca_deg) in dens:
            m = _placement_mask(cx, cy, ca_deg)
            mask &= m
            
            if not np.any(mask):
                return 0.0
        
        return np.sum(mask) * 0.0001
    
    def evaluate_path_fast(path):
        """Faster evaluation using precomputed arrays."""
        dens = _make_densified(path)
        n_dens = len(dens)
        
        # Precompute all placement parameters
        cxs = np.array([d[0] for d in dens])
        cys = np.array([d[1] for d in dens])
        cas = np.array([d[2] for d in dens])
        
        cas_rad = np.deg2rad(cas)
        cos_as = np.cos(cas_rad)
        sin_as = np.sin(cas_rad)
        
        # For each grid point, check if it's inside ALL placements
        # Process in chunks to manage memory
        chunk_size = 5000
        n_chunks = (n_points + chunk_size - 1) // chunk_size
        
        min_mask = np.ones(n_points, dtype=bool)
        
        for chunk_start in range(0, n_points, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_points)
            xc = x_flat[chunk_start:chunk_end]
            yc = y_flat[chunk_start:chunk_end]
            
            # For each placement, compute mask for this chunk
            chunk_mask = np.ones(chunk_end - chunk_start, dtype=bool)
            
            for i in range(n_dens):
                dx = xc - cxs[i]
                dy = yc - cys[i]
                
                u = dx * cos_as[i] + dy * sin_as[i]
                v = -dx * sin_as[i] + dy * cos_as[i]
                
                in_arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
                in_arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
                
                chunk_mask &= (in_arm1 | in_arm2)
                
                if not np.any(chunk_mask):
                    return 0.0
            
            min_mask[chunk_start:chunk_end] = chunk_mask
        
        return np.sum(min_mask) * 0.0001
    
    # Initial path: hand-tuned starting point
    initial_path = [
        (-0.1, 0.1, 15.0),
        (-0.2, 0.2, 20.0),
        (-0.3, 0.3, 34.0),
        (-0.3, 0.3, 40.0),
        (-0.4, 0.3, 50.0),
        (-0.5, 0.2, 55.0),
        (-0.6, 0.2, 75.0),
        (-0.7, 0.3, 86.0),
    ]
    
    best_score = evaluate_path_fast(initial_path)
    best_path = initial_path[:]
    current_path = initial_path[:]
    
    # Local search with adaptive step sizes and simulated annealing
    np.random.seed(42)
    epsilon_xy = 0.15
    epsilon_a = 3.0
    no_improve = 0
    max_no_improve = 25
    temperature = 0.001  # For simulated annealing acceptance
    
    iteration = 0
    while time.time() - start_time < time_limit:
        # Pick random control point
        idx = np.random.randint(len(current_path))
        p = current_path[idx]
        
        # Perturb all three coordinates
        new_x = p[0] + np.random.normal(0, epsilon_xy)
        new_y = p[1] + np.random.normal(0, epsilon_xy)
        new_a = p[2] + np.random.normal(0, epsilon_a)
        
        # Constrain angle to [0, 90]
        new_a = np.clip(new_a, 0.0, 90.0)
        
        new_p = (new_x, new_y, new_a)
        test_path = current_path[:]
        test_path[idx] = new_p
        
        score = evaluate_path_fast(test_path)
        
        # Simulated annealing: accept worse solutions with probability exp(-delta/T)
        if score > best_score:
            best_score = score
            best_path = test_path
            current_path = test_path
            no_improve = 0
            temperature = max(0.0001, temperature * 0.99)
        else:
            delta = score - current_score if 'current_score' in dir() else score - best_score
            if delta < 0:
                accept_prob = np.exp(delta / temperature)
                if np.random.random() < accept_prob:
                    current_path = test_path
                    no_improve += 1
                else:
                    no_improve += 1
            else:
                current_path = test_path
                no_improve += 1
            
            if no_improve >= max_no_improve:
                # Shrink step sizes
                epsilon_xy *= 0.85
                epsilon_a *= 0.85
                no_improve = 0
                # If step sizes get too small, try a restart from best
                if epsilon_xy < 0.01 and epsilon_a < 0.5:
                    current_path = best_path[:]
                    epsilon_xy = 0.3
                    epsilon_a = 5.0
                    temperature = 0.001
        
        # Update current_score for next iteration
        current_score = score
        
        iteration += 1
        if iteration % 100 == 0:
            # Occasionally try adding or removing a control point
            if np.random.random() < 0.1 and len(current_path) < 15:
                # Add a new control point
                insert_idx = np.random.randint(len(current_path) + 1)
                if insert_idx == 0:
                    new_point = (current_path[0][0] + np.random.normal(0, 0.1),
                                  current_path[0][1] + np.random.normal(0, 0.1),
                                  max(0, current_path[0][2] - np.random.uniform(0, 10)))
                elif insert_idx == len(current_path):
                    new_point = (current_path[-1][0] + np.random.normal(0, 0.1),
                                  current_path[-1][1] + np.random.normal(0, 0.1),
                                  min(90, current_path[-1][2] + np.random.uniform(0, 10)))
                else:
                    p1 = current_path[insert_idx - 1]
                    p2 = current_path[insert_idx]
                    new_point = ((p1[0] + p2[0])/2 + np.random.normal(0, 0.1),
                                  (p1[1] + p2[1])/2 + np.random.normal(0, 0.1),
                                  (p1[2] + p2[2])/2 + np.random.normal(0, 2))
                    new_point = (new_point[0], new_point[1], np.clip(new_point[2], 0, 90))
                
                test_path = current_path[:]
                test_path.insert(insert_idx, new_point)
                
                test_score = evaluate_path_fast(test_path)
                if test_score >= score * 0.95:  # Accept if not too bad
                    current_path = test_path
                    if test_score > best_score:
                        best_score = test_score
                        best_path = test_path[:]
            
            elif np.random.random() < 0.1 and len(current_path) > 4:
                # Remove a control point
                remove_idx = np.random.randint(len(current_path))
                test_path = current_path[:]
                test_path.pop(remove_idx)
                
                test_score = evaluate_path_fast(test_path)
                if test_score >= score * 0.95:
                    current_path = test_path
                    if test_score > best_score:
                        best_score = test_score
                        best_path = test_path[:]
    
    return np.array(best_path)
