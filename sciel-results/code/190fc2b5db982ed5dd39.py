"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    def remaining():
        return deadline - time.time()
    
    def hex_initial_config(n):
        """Generate a hexagonal packing configuration for n circles."""
        # Try different grid arrangements and pick the best
        best_config = None
        best_score = -1
        
        # Hexagonal arrangement: rows with alternating offsets
        # For n circles, try arrangements like 5+5+5+5+4, 6+5+5+5+5, etc.
        for rows in range(3, 10):
            # Distribute n circles across rows
            base = n // rows
            extra = n % rows
            counts = [base + (1 if i < extra else 0) for i in range(rows)]
            
            # Build hex packing
            circles = []
            # Row height for hex packing
            # If radius r, row spacing = r * sqrt(3), horizontal spacing = 2r
            # But we need to fit in unit square, so we'll normalize later
            # Use unit radius for now, then scale
            
            r = 1.0  # unit radius for layout
            row_h = r * np.sqrt(3)
            col_w = 2 * r
            
            for i, count in enumerate(counts):
                y = i * row_h
                x_offset = (i % 2) * r  # hex offset
                for j in range(count):
                    x = x_offset + j * col_w
                    circles.append([x, y, r])
            
            if len(circles) != n:
                continue
            
            # Center and scale to fit in unit square
            xs = [c[0] for c in circles]
            ys = [c[1] for c in circles]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Add radius margin
            pad_x = r
            pad_y = r
            total_w = (max_x - min_x) + 2 * pad_x
            total_h = (max_y - min_y) + 2 * pad_y
            
            scale = min(1.0 / total_w, 1.0 / total_h)
            
            new_circles = []
            for c in circles:
                nx = (c[0] - min_x + pad_x) * scale
                ny = (c[1] - min_y + pad_y) * scale
                nr = c[2] * scale
                new_circles.append([nx, ny, nr])
            
            # Check validity
            valid = True
            for c in new_circles:
                if c[0] - c[2] < -1e-9 or c[0] + c[2] > 1 + 1e-9:
                    valid = False
                    break
                if c[1] - c[2] < -1e-9 or c[1] + c[2] > 1 + 1e-9:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check non-overlap
            for i in range(len(new_circles)):
                for j in range(i+1, len(new_circles)):
                    ci, cj = new_circles[i], new_circles[j]
                    dist = np.sqrt((ci[0]-cj[0])**2 + (ci[1]-cj[1])**2)
                    if dist < ci[2] + cj[2] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                score = sum(c[2] for c in new_circles)
                if score > best_score:
                    best_score = score
                    best_config = new_circles
        
        if best_config is None:
            # Fallback: grid
            side = int(np.ceil(np.sqrt(n)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                     for i in range(side) for j in range(side)]
            best_config = cells[:n]
        
        return np.array(best_config)
    
    def optimize(config, max_iters=5000, lr_init=0.01, penalty_init=100.0):
        """Gradient-based optimization with penalty method."""
        config = config.copy()
        n = len(config)
        
        # Variables: x, y, r for each circle
        # We'll optimize positions and radii
        
        penalty = penalty_init
        lr = lr_init
        
        best_config = config.copy()
        best_score = -np.inf
        best_valid_score = -np.inf
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Compute current score (sum of radii)
            current_score = np.sum(config[:, 2])
            
            # Check validity
            valid = True
            for i in range(n):
                x, y, r = config[i]
                if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                    valid = False
                    break
            
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        dx = config[i,0] - config[j,0]
                        dy = config[i,1] - config[j,1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < config[i,2] + config[j,2] - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                if current_score > best_valid_score:
                    best_valid_score = current_score
                    best_config = config.copy()
            
            # Compute gradient
            grad = np.zeros_like(config)
            
            # Objective gradient: maximize sum of radii
            # d/dr_i = 1 for each circle
            grad[:, 2] = 1.0
            
            # Penalty for boundary violations
            for i in range(n):
                x, y, r = config[i]
                # Left boundary
                if x - r < 0:
                    grad[i, 0] -= penalty
                    grad[i, 2] -= penalty
                # Right boundary
                if x + r > 1:
                    grad[i, 0] += penalty
                    grad[i, 2] += penalty
                # Bottom boundary
                if y - r < 0:
                    grad[i, 1] -= penalty
                    grad[i, 2] -= penalty
                # Top boundary
                if y + r > 1:
                    grad[i, 1] += penalty
                    grad[i, 2] += penalty
            
            # Penalty for overlap
            for i in range(n):
                for j in range(i+1, n):
                    dx = config[i,0] - config[j,0]
                    dy = config[i,1] - config[j,1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = config[i,2] + config[j,2]
                    
                    if dist < min_dist and dist > 1e-12:
                        # Overlap penalty
                        overlap = min_dist - dist
                        # Gradient of -overlap w.r.t. positions
                        # d(dist)/dx_i = dx/dist, d(min_dist)/dx_i = 0
                        # So gradient of (min_dist - dist) w.r.t. x_i = -dx/dist
                        # We want to push apart, so add penalty * (dx/dist) to x_i
                        factor = penalty * overlap / dist
                        grad[i, 0] += factor * dx
                        grad[i, 1] += factor * dy
                        grad[j, 0] -= factor * dx
                        grad[j, 1] -= factor * dy
                        
                        # Gradient w.r.t. radii: d(min_dist)/dr_i = 1
                        # So gradient of (min_dist - dist) w.r.t. r_i = 1
                        grad[i, 2] -= penalty * overlap
                        grad[j, 2] -= penalty * overlap
            
            # Adaptive learning rate
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad = grad / grad_norm
            
            # Update
            config += lr * grad
            
            # Clamp radii to positive
            config[:, 2] = np.maximum(config[:, 2], 1e-6)
            
            # Adaptive penalty: increase if many violations
            violations = 0
            for i in range(n):
                x, y, r = config[i]
                if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                    violations += 1
            for i in range(n):
                for j in range(i+1, n):
                    dx = config[i,0] - config[j,0]
                    dy = config[i,1] - config[j,1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < config[i,2] + config[j,2]:
                        violations += 1
            
            if violations > 5:
                penalty *= 1.2
            elif violations < 2:
                penalty = max(penalty * 0.95, 10.0)
            
            # Decay learning rate
            lr = lr_init * (0.999 ** it)
            lr = max(lr, 1e-6)
        
        return best_config if best_valid_score > -np.inf else config
    
    # Generate initial configuration
    config = hex_initial_config(N)
    
    # Run optimization
    best_result = config.copy()
    best_score = -np.inf
    
    def get_score(cfg):
        """Get valid score or -inf if invalid."""
        n = len(cfg)
        valid = True
        for i in range(n):
            x, y, r = cfg[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                return -np.inf
        for i in range(n):
            for j in range(i+1, n):
                dx = cfg[i,0] - cfg[j,0]
                dy = cfg[i,1] - cfg[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < cfg[i,2] + cfg[j,2] - 1e-9:
                    return -np.inf
        return np.sum(cfg[:, 2])
    
    # Multiple optimization runs with different initial conditions
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        if restart == 0:
            cfg = config.copy()
        else:
            # Perturb the best result
            cfg = best_result.copy()
            noise_scale = 0.01 / (restart + 1)
            cfg[:, :2] += np.random.randn(N, 2) * noise_scale
            # Keep radii mostly the same, small perturbation
            cfg[:, 2] += np.random.randn(N) * noise_scale * 0.5
            cfg[:, 2] = np.maximum(cfg[:, 2], 1e-4)
        
        # Optimize
        result = optimize(cfg, max_iters=3000, lr_init=0.005, penalty_init=50.0)
        
        score = get_score(result)
        if score > best_score:
            best_score = score
            best_result = result.copy()
    
    # Final local refinement
    if time.time() < deadline - 5.0:
        # Do a final fine-tuning
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            result = optimize(best_result, max_iters=2000, lr_init=0.001, penalty_init=200.0)
            score = get_score(result)
            if score > best_score:
                best_score = score
                best_result = result.copy()
    
    # Ensure validity of final result
    final_score = get_score(best_result)
    if final_score == -np.inf:
        # Fallback to initial config if optimization failed
        return config
    
    return best_result

# EVOLVE-BLOCK-END
