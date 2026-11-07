"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def objective(h):
    """
    Compute the evaluation value:
    value = max(np.correlate(h, 1 - h, mode="full")) / len(h) * 2
    """
    # np.correlate is O(N^2) for small N, which is fine for N=100.
    # We can optimize by pre-computing 1-h if h is static, but h changes.
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0

def construct():
    """
    Uses Gradient-Based Optimization (L-BFGS-B) to minimize the maximum correlation.
    Strategy:
    1. The objective function f(h) = max_k (corr_k) is convex? 
       Actually, max of linear functions is convex. 
       Each correlation term is linear in h. 
       So the objective is convex. 
       Wait, let's verify.
       corr_k = sum_{i} h_i * (1 - h_{i+k}) = sum h_i - sum h_i h_{i+k}.
       This is NOT linear. It's quadratic.
       sum h_i h_{i+k} is a quadratic form.
       So the objective is max of concave functions (since -quadratic is convex? No, -x^2 is concave).
       Let's check the Hessian.
       f_k(h) = sum h_i - sum h_i h_{i+k}.
       Gradient: grad f_k = 1 - (h_{i+k} + h_{i-k}) roughly.
       Hessian: - (shift + shift^T). 
       The matrix for sum h_i h_{i+k} is a symmetric circulant-like matrix.
       If the matrix A_k (where (A_k)_{ij} = 1 if j-i=k) is positive semi-definite, then -sum h_i h_{i+k} is convex.
       However, circulant matrices with entries 1 on the k-th diagonal are not necessarily PSD.
       Actually, let's just use a robust non-smooth optimizer or smooth approximation.
       
       Alternatively, since N=100 is small, we can use a multi-start local search with a very good initial guess derived from spectral analysis or random projections, followed by fine-tuning with Nelder-Mead or L-BFGS-B (if we smooth the max).
       
       Let's try a different angle:
       The objective is max_k C_k(h).
       We can minimize this by using a "soft max" approximation or just direct minimization with a method that handles non-smoothness (like Nelder-Mead or Powell).
       
       However, a more powerful strategy for "minimize max of several functions" is to introduce an auxiliary variable t and minimize t subject to C_k(h) <= t for all k.
       This is a convex optimization problem IF C_k(h) are convex.
       Let's check convexity of C_k(h) = sum_i h_i (1 - h_{i+k}).
       Let's test with N=2, k=1.
       h=[x, y]. C_1 = x(1-y) + y(1-x) = x + y - 2xy.
       Hessian of x+y-2xy is [[0, -2], [-2, 0]]. Eigenvalues 2, -2. Not convex. It's indefinite.
       So the problem is non-convex.
       
       Therefore, gradient descent on the raw max might get stuck.
       
       Let's try a different deterministic approach:
       Spectral/Structural initialization.
       The term sum h_i h_{i+k} is related to the autocorrelation.
       We want to minimize max_k (sum h_i - sum h_i h_{i+k}).
       Since sum h_i = 50 is fixed, we want to MAXIMIZE min_k (sum h_i h_{i+k}).
       Let Q_k = sum h_i h_{i+k}. We want to maximize the minimum of these quadratic forms.
       
       This looks like a max-min problem.
       
       Let's use a multi-start strategy with Nelder-Mead (derivative-free, robust to non-smoothness) on a set of diverse initial points, but make it faster by using a good heuristic for initial points.
       
       Heuristic: Random Gaussian projected to sum=50, clipped.
       Or: Alternating patterns.
       
       Actually, the current program uses SA. A "MATERIALLY DIFFERENT" strategy is using a **Genetic Algorithm** or **Particle Swarm Optimization** (PSO) or **Deterministic Annealing**.
       
       Let's go with **Particle Swarm Optimization (PSO)**. It is a swarm intelligence method, distinct from single-agent SA or gradient methods. It maintains a population of particles, each moving through the solution space based on its own history and the group's history. It often finds better global minima than SA for non-convex problems with constraints, especially with proper velocity clamping and boundary handling.
       
       Implementation details:
       1. Population of M particles (e.g., 20-50).
       2. Each particle has a position h and velocity v.
       3. Update: v = w*v + c1*r1*(pbest - h) + c2*r2*(gbest - h).
       4. h = h + v.
       5. Constraint handling: Project h to sum=50 and clip to [0,1]. If sum constraint is violated after clipping, use a repair step (distribute remaining mass to particles with room).
       6. Run for a fixed number of iterations or time limit.
       
       This is significantly different from SA (single trajectory, temperature-based acceptance) and Gradient methods (local, smoothness-dependent).
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Objective function for PSO (returns value to minimize)
    def obj_func(h):
        if not np.all(np.isfinite(h)):
            return 1e9
        # Ensure validity
        if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
            return 1e9
        if abs(np.sum(h) - N/2.0) > 1e-6:
            # Soft penalty for sum constraint violation during search
            return 1e9 + (np.sum(h) - N/2.0)**2
        return objective(h)
    
    # Repair function to enforce constraints exactly
    def repair(h):
        h = np.clip(h, 0.0, 1.0)
        diff = N/2.0 - np.sum(h)
        if abs(diff) > 1e-9:
            # Distribute diff to elements that have room
            # If diff > 0, we need to increase some elements
            # If diff < 0, we need to decrease some elements
            if diff > 0:
                # Find indices where h < 1
                mask = h < 1.0
                if np.sum(mask) > 0:
                    # Distribute proportionally to available space? Or evenly?
                    # Evenly is simpler but might violate bounds.
                    # Let's just add diff/num_masked to those, then re-clip.
                    # This might not sum exactly.
                    # Better: iterative water-filling
                    for _ in range(50):
                        h = np.clip(h, 0.0, 1.0)
                        current_sum = np.sum(h)
                        rem = N/2.0 - current_sum
                        if abs(rem) < 1e-9:
                            break
                        if rem > 0:
                            mask = h < 1.0
                        else:
                            mask = h > 0.0
                        if np.sum(mask) == 0:
                            break
                        h[mask] += rem / np.sum(mask)
            else:
                mask = h > 0.0
                if np.sum(mask) > 0:
                    for _ in range(50):
                        h = np.clip(h, 0.0, 1.0)
                        current_sum = np.sum(h)
                        rem = N/2.0 - current_sum
                        if abs(rem) < 1e-9:
                            break
                        if rem > 0:
                            mask = h < 1.0
                        else:
                            mask = h > 0.0
                        if np.sum(mask) == 0:
                            break
                        h[mask] += rem / np.sum(mask)
        return h

    # PSO Parameters
    swarm_size = 30
    max_iters = 2000
    w = 0.7  # Inertia
    c1 = 1.5 # Cognitive
    c2 = 1.5 # Social
    
    # Initialize Swarm
    # Diverse initializations
    positions = np.zeros((swarm_size, N))
    velocities = np.zeros((swarm_size, N))
    
    # 1. Random uniform
    for i in range(swarm_size // 3):
        h = np.random.uniform(0, 1, N)
        h = repair(h)
        positions[i] = h
        velocities[i] = np.random.uniform(-0.05, 0.05, N)
        
    # 2. Alternating
    for i in range(swarm_size // 3, 2 * swarm_size // 3):
        h = np.zeros(N)
        h[1::2] = 1.0
        # Add small noise
        h += np.random.uniform(-0.01, 0.01, N)
        h = repair(h)
        positions[i] = h
        velocities[i] = np.random.uniform(-0.05, 0.05, N)
        
    # 3. Smooth random (low frequency)
    for i in range(2 * swarm_size // 3, swarm_size):
        # Generate random low-freq signal
        freqs = np.random.randint(1, 5, 3)
        h = np.zeros(N)
        for f in freqs:
            h += np.sin(np.linspace(0, f * 2 * np.pi, N)) * np.random.uniform(0.1, 0.5)
        h = (h - np.min(h)) / (np.max(h) - np.min(h) + 1e-9) # Normalize to [0,1]
        h = repair(h)
        positions[i] = h
        velocities[i] = np.random.uniform(-0.05, 0.05, N)
        
    # Initialize bests
    pbests = positions.copy()
    pbest_vals = np.array([obj_func(p) for p in positions])
    gbest = positions[np.argmin(pbest_vals)]
    gbest_val = np.min(pbest_vals)
    
    best_h = gbest.copy()
    best_val = gbest_val
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        for i in range(swarm_size):
            # Update velocity
            r1 = np.random.random(N)
            r2 = np.random.random(N)
            velocities[i] = w * velocities[i] + c1 * r1 * (pbests[i] - positions[i]) + c2 * r2 * (gbest - positions[i])
            
            # Clip velocity to prevent explosion
            velocities[i] = np.clip(velocities[i], -0.1, 0.1)
            
            # Update position
            positions[i] = positions[i] + velocities[i]
            
            # Repair
            positions[i] = repair(positions[i])
            
            # Evaluate
            val = obj_func(positions[i])
            
            # Update personal best
            if val < pbest_vals[i]:
                pbest_vals[i] = val
                pbests[i] = positions[i].copy()
                
                # Update global best
                if val < gbest_val:
                    gbest_val = val
                    gbest = positions[i].copy()
                    best_h = gbest.copy()
                    best_val = gbest_val
                    
    # Final check
    if not np.all(np.isfinite(best_h)) or np.abs(np.sum(best_h) - N/2.0) > 1e-9 * N:
        return np.full(N, 0.5)
        
    return best_h

# EVOLVE-BLOCK-END
