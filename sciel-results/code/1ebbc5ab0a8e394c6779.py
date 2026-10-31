import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial configuration: Random points in the square, small radii
    # Using a fixed seed for reproducibility of the search path if needed, 
    # but random restarts are usually better for escaping local optima.
    # However, for a single run within time limit, a good initialization is key.
    # Let's try a few random restarts if time permits, or just one strong run.
    
    best_score = -np.inf
    best_config = None
    
    # Helper to validate and score
    def score_config(config):
        # config: (N, 3) array [x, y, r]
        x = config[:, 0]
        y = config[:, 1]
        r = config[:, 2]
        
        # Check boundaries
        if np.any(r < 0):
            return -np.inf
        if np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r):
            return -np.inf
            
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j] - 1e-9:
                    return -np.inf
        return np.sum(r)

    # Gradient ascent step
    def gradient_step(config, lr, penalty_weight):
        x = config[:, 0].copy()
        y = config[:, 1].copy()
        r = config[:, 2].copy()
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        grad_r = np.zeros(N)
        
        # Objective gradient: d(sum(r))/dr_i = 1
        grad_r += 1.0
        
        # Penalty gradients
        # 1. Boundary penalties
        # Penalty: sum( max(0, r - x)^2 + max(0, r - (1-x))^2 + max(0, r - y)^2 + max(0, r - (1-y))^2 )
        # For circle i:
        # P_x_left = max(0, r_i - x_i)^2
        # dP/dx_i = 2 * max(0, r_i - x_i) * (-1)
        # dP/dr_i = 2 * max(0, r_i - x_i) * (1)
        
        v_xl = np.maximum(0, r - x)
        v_xr = np.maximum(0, r - (1 - x))
        v_yl = np.maximum(0, r - y)
        v_yr = np.maximum(0, r - (1 - y))
        
        grad_x -= 2 * v_xl
        grad_x += 2 * v_xr
        grad_y -= 2 * v_yl
        grad_y += 2 * v_yr
        
        grad_r += 2 * (v_xl + v_xr + v_yl + v_yr)
        
        # 2. Overlap penalties
        # Penalty: sum_{i<j} max(0, r_i + r_j - dist_ij)^2
        # Let delta_ij = r_i + r_j - dist_ij
        # If delta_ij > 0:
        # dP/dr_i = 2 * delta_ij
        # dP/dx_i = 2 * delta_ij * (-(x_i - x_j) / dist_ij)
        # dP/dy_i = 2 * delta_ij * (-(y_i - y_j) / dist_ij)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-8:
                    # Handle singularity
                    dist = 1e-8
                    dx = 1e-8
                    dy = 0
                    
                delta = r[i] + r[j] - dist
                if delta > 0:
                    # Gradient contribution to i
                    # dP/dx_i = 2 * delta * (-dx / dist)
                    # dP/dy_i = 2 * delta * (-dy / dist)
                    # dP/dr_i = 2 * delta
                    
                    factor = 2 * delta
                    grad_x[i] -= factor * (dx / dist)
                    grad_y[i] -= factor * (dy / dist)
                    grad_r[i] += factor
                    
                    # Gradient contribution to j
                    # dP/dx_j = 2 * delta * (dx / dist)  (since d(dist)/dx_j = -dx/dist, and sign flips)
                    # Wait, dist = sqrt((xi-xj)^2...). d(dist)/dx_j = -(xi-xj)/dist = -dx/dist.
                    # Penalty term is (ri+rj-dist)^2. dP/ddist = -2(ri+rj-dist).
                    # dP/dx_j = dP/ddist * d(dist)/dx_j = -2*delta * (-dx/dist) = 2*delta*dx/dist.
                    # So gradient for j is opposite to i in x,y?
                    # Let's re-verify.
                    # P = (ri+rj-d)^2.
                    # dP/dxi = 2(ri+rj-d) * (-1) * d(xi-xj)/dxi / d = -2(ri+rj-d) * (1) / d = -2 delta / d.
                    # Wait, d/dxi (xi-xj) = 1. So d(dist)/dxi = (xi-xj)/dist = dx/dist.
                    # So dP/dxi = 2(ri+rj-d) * (-1) * (dx/dist) = -2 delta dx/dist.
                    # My previous derivation: grad_x[i] -= 2 * delta * (dx / dist). This matches.
                    
                    # For j: d(dist)/dxj = -dx/dist.
                    # dP/dxj = 2(ri+rj-d) * (-1) * (-dx/dist) = 2 delta dx/dist.
                    grad_x[j] += factor * (dx / dist)
                    grad_y[j] += factor * (dy / dist)
                    grad_r[j] += factor
        
        # Apply penalty weight
        grad_x *= penalty_weight
        grad_y *= penalty_weight
        grad_r *= penalty_weight
        
        # Update
        x += lr * grad_x
        y += lr * grad_y
        r += lr * grad_r
        
        # Project back to feasible region? 
        # Simple projection: ensure r >= 0 and inside square roughly.
        # If we project, we might break the gradient flow, but it helps stability.
        r = np.maximum(r, 0.001)
        
        # Soft projection to square to prevent wild excursions
        # x must be in [r, 1-r]. If not, clamp.
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        # Note: Clamping x to r changes the effective gradient, but prevents invalid states.
        
        return np.column_stack((x, y, r))

    # Main loop
    num_restarts = 5
    steps_per_restart = 2000
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize
        # Try to initialize with a somewhat spread out pattern to avoid immediate overlaps
        # Random uniform in [0.1, 0.9] with small radii
        x = np.random.uniform(0.2, 0.8, N)
        y = np.random.uniform(0.2, 0.8, N)
        r = np.full(N, 0.05)
        config = np.column_stack((x, y, r))
        
        # Check initial validity, if not, adjust
        # If initial config is invalid, the penalty will push it out.
        
        current_score = -np.inf
        best_config_in_restart = None
        best_score_in_restart = -np.inf
        
        # Annealing schedule
        # Start with high penalty weight to satisfy constraints, then lower it to maximize radii
        # Or start low and increase? 
        # Usually, start with valid config (small r) and increase r.
        # If we start with small r, constraints are satisfied. 
        # We want to increase r. 
        # If we use penalty method: Maximize sum(r) - lambda * penalty.
        # If lambda is high, it stays valid but small.
        # If lambda is low, it grows but might overlap.
        # Strategy: Start with lambda high, gradually decrease lambda? No, that allows violation.
        # Better: Start with valid config. Increase r via gradient. 
        # The penalty acts as a barrier. 
        # Let's keep lambda constant and moderate, or increase it if violations appear.
        
        # Let's try: Start with lambda=100. Run steps. 
        # If we get stuck, we can restart.
        
        lr = 0.01
        penalty_weight = 50.0
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5.0:
                break
                
            config = gradient_step(config, lr, penalty_weight)
            
            # Check validity and score
            s = score_config(config)
            if s > best_score_in_restart:
                best_score_in_restart = s
                best_config_in_restart = config.copy()
                
            # Adaptive learning rate? 
            # If score is improving, keep lr. If oscillating, reduce lr.
            # Simple decay
            if step % 100 == 0:
                lr *= 0.95
                # Increase penalty weight slightly to enforce constraints more strictly as we grow?
                # Or decrease to allow more growth?
                # If we are valid, penalty is 0. Gradient is just 1 for r.
                # So it will grow until it hits a constraint.
                # Then penalty kicks in.
                # So we don't need to change penalty weight much.
                
        if best_score_in_restart > best_score:
            best_score = best_score_in_restart
            best_config = best_config_in_restart
            
    # Final polish: try to push radii up slightly with very small steps and high penalty
    if best_config is not None and time.time() < deadline - 2.0:
        config = best_config.copy()
        # High penalty to ensure validity
        penalty_weight = 500.0
        lr = 0.001
        for step in range(500):
            if time.time() > deadline - 1.0:
                break
            config = gradient_step(config, lr, penalty_weight)
            s = score_config(config)
            if s > best_score:
                best_score = s
                best_config = config.copy()
                
    if best_config is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
# EVOLVE-BLOCK-END
