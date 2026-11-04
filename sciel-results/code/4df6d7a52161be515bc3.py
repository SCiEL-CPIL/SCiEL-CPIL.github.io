"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Finds a packing of 50 spheres in a unit cube [-0.5, 0.5]^3.
    Uses Gradient Ascent on a log-sum-of-distances objective to maximize
    the minimum distance (approximated by sum of logs).
    Multiple random restarts are used to escape local optima.
    """
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer

    def compute_grad_and_loss(X, r_target=0.13):
        """
        Computes the negative gradient of the log-sum objective.
        Objective: Maximize sum_{i<j} log(d_ij) + sum_{i,k} log(wall_dist_ik)
        We minimize the negative of this.
        """
        # Initialize gradients
        grads = np.zeros_like(X)
        
        # Pairwise distances
        # d_ij = ||X_i - X_j||
        # grad_i of log(d_ij) = (X_i - X_j) / d_ij^2
        # grad_j of log(d_ij) = (X_j - X_i) / d_ij^2
        
        for i in range(N):
            for j in range(i + 1, N):
                diff = X[i] - X[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    dist_sq = 1e-12
                dist = np.sqrt(dist_sq)
                # We want to maximize log(dist), so gradient is diff/dist^2
                # But if dist is very small, this explodes. 
                # We can clamp the force if dist < 2*r_target to avoid instability, 
                # or just let it be large to push them apart strongly.
                force = diff / dist_sq
                grads[i] += force
                grads[j] -= force

        # Wall distances
        # wall_dist = 0.5 - |x_k|
        # grad of log(wall_dist) wrt x_k = -sign(x_k) / (0.5 - |x_k|)
        for i in range(N):
            for k in range(3):
                x_k = X[i, k]
                wall_dist = 0.5 - abs(x_k)
                if wall_dist < 1e-6:
                    wall_dist = 1e-6
                # Gradient of log(wall_dist) is -sign(x_k) / wall_dist
                # We want to maximize, so we add this to the gradient direction
                # Wait, if x_k > 0, wall_dist = 0.5 - x_k. d/dx_k log(0.5-x_k) = -1/(0.5-x_k).
                # This is negative, meaning increasing x_k decreases the log.
                # So the gradient points towards the center (negative x_k direction if x_k>0).
                # Correct.
                grad_val = -np.sign(x_k) / wall_dist
                # If x_k is 0, sign is 0, but limit is ambiguous. 
                # However, log(0.5) is max at 0. So grad should be 0?
                # Actually, d/dx log(0.5-|x|) is -sign(x)/(0.5-|x|). At x=0, it's discontinuous but effectively 0 from both sides if we consider the max.
                # Let's just use the formula. If x_k == 0, force is 0.
                if x_k == 0:
                    grad_val = 0
                grads[i, k] += grad_val

        return grads

    def run_optimization(seed, steps=5000):
        """
        Runs gradient ascent with Adam-like adaptation.
        """
        rng = np.random.default_rng(seed)
        
        # Initialize
        # Start with a jittered grid or random
        X = rng.uniform(-0.4, 0.4, size=(N, 3))
        
        # Adam parameters
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        lr = 0.01
        
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        
        best_score = 0.0
        best_X = X.copy()
        
        for step in range(steps):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            # Compute gradient (direction of increase)
            grad = compute_grad_and_loss(X)
            
            # Adam update
            t = step + 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            X_new = X + lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Clip to cube to ensure validity
            X_new = np.clip(X_new, -0.5, 0.5)
            
            # Evaluate score
            # Calculate min distance
            min_dist = 0.5
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(X_new[i] - X_new[j]) / 2.0
                    if d < min_dist:
                        min_dist = d
            for i in range(N):
                for k in range(3):
                    d_wall = 0.5 - abs(X_new[i, k])
                    if d_wall < min_dist:
                        min_dist = d_wall
            
            if min_dist > best_score:
                best_score = min_dist
                best_X = X_new.copy()
            
            X = X_new
            
        return best_X, best_score

    best_config = None
    best_global_score = 0.0
    
    # Multiple restarts
    num_restarts = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = num_restarts
        # Run optimization
        config, score = run_optimization(seed, steps=2000)
        
        if score > best_global_score:
            best_global_score = score
            best_config = config
            
        num_restarts += 1
        
    if best_config is None:
        # Fallback: Grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_config = grid[:N]
        
    return best_config

# EVOLVE-BLOCK-END
