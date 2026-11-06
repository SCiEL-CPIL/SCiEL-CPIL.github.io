import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 24
    # Precompute constants
    k = np.arange(n)
    
    # Fast max norm calculator
    def calc_max_norm(c):
        # c is a numpy array of ints
        # Use a fixed grid for speed, then refine the top peak
        m = 4096
        thetas = np.linspace(0, 2*np.pi, m, endpoint=False)
        # We can precompute thetas? No, it's small.
        # But we can precompute the powers for the grid?
        # The grid is fixed.
        # powers_grid shape (n, m)
        # exp(i * k * theta)
        # This is independent of c!
        # Let's precompute it once outside the function?
        # No, construct() is called once.
        
        # Precomputation inside construct is better.
        pass 

    # Let's restructure to precompute the grid powers.
    m = 4096
    thetas = np.linspace(0, 2*np.pi, m, endpoint=False)
    # powers_grid[j, i] = exp(i * k_j * theta_i)
    # k_j = j
    # theta_i = i * 2*pi / m
    # powers_grid = exp(1j * np.outer(k, thetas))
    powers_grid = np.exp(1j * np.outer(k, thetas))
    
    def get_max_norm(c):
        # c: numpy array
        vals = np.dot(c, powers_grid)
        max_val = np.max(np.abs(vals))
        idx = np.argmax(np.abs(vals))
        
        # Refine
        # Find neighbors
        # Simple refinement: sample 20 points around thetas[idx]
        delta = 2 * np.pi / m
        # Ensure we don't go out of bounds, though wrap-around is handled by periodicity
        # Just sample in [thetas[idx]-delta, thetas[idx]+delta]
        fine_thetas = np.linspace(thetas[idx] - delta, thetas[idx] + delta, 50)
        # We need powers for these fine thetas
        # This is expensive to compute every time.
        # Alternative: Just use the grid max. 
        # The grid resolution is 1/4096. The error is small.
        # For N=24, the polynomial is band-limited.
        # Let's check if grid max is sufficient.
        # The task says "accuracy about 1e-9". 
        # Grid max might be off by ~1e-4.
        # We need refinement.
        
        # To make refinement fast, we can precompute a small set of "refinement vectors"?
        # No, the peak location changes.
        
        # Let's do a lightweight refinement:
        # Use the derivative?
        # Or just do a small golden section search on the function |P(e^{i theta})|.
        # We need to evaluate |P(e^{i theta})| for arbitrary theta.
        # def eval_p(theta):
        #     z = np.exp(1j * theta)
        #     # Compute sum c_k z^k
        #     # Horner's method?
        #     # Or just dot product with powers?
        #     # powers = z ** k
        #     # This is O(N) = 24 ops.
        #     # Very fast.
        
        def eval_p(theta):
            z = np.exp(1j * theta)
            # Horner
            res = 0.0
            for ck in c[::-1]:
                res = res * z + ck
            return np.abs(res)
        
        # Golden section search around thetas[idx]
        # Range [thetas[idx] - 2*delta, thetas[idx] + 2*delta]
        # 20 iterations of golden section
        gr = 0.6180339887498949
        a = thetas[idx] - 2*delta
        b = thetas[idx] + 2*delta
        c1 = b - gr * (b - a)
        c2 = a + gr * (b - a)
        f1 = eval_p(c1)
        f2 = eval_p(c2)
        
        for _ in range(20):
            if f1 < f2:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - gr * (b - a)
                f1 = eval_p(c1)
            else:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + gr * (b - a)
                f2 = eval_p(c2)
                
        max_refined = max(f1, f2)
        # Also check the grid max itself?
        # The grid point might be the max.
        f_grid = eval_p(thetas[idx])
        return max(max_refined, f_grid)

    # Initial best
    best_c = np.random.choice([-1, 1], size=n)
    best_score = get_max_norm(best_c)
    
    # Simulated Annealing / Local Search
    # We will run multiple restarts.
    # Each restart: Start from random or best, run SA, polish.
    
    restarts = 0
    while time.time() < deadline:
        # Choose start
        if random.random() < 0.5:
            c = np.random.choice([-1, 1], size=n)
        else:
            c = best_c.copy()
            # Perturb
            for _ in range(5):
                idx = random.randint(0, n-1)
                c[idx] *= -1
        
        current_score = get_max_norm(c)
        
        # SA Parameters
        T = 1.0
        T_min = 1e-6
        alpha = 0.99
        # Number of steps per restart
        # We can adapt this based on time remaining.
        # Let's do fixed steps, e.g., 5000
        steps = 5000
        
        for i in range(steps):
            if time.time() > deadline - 5.0:
                break
            
            # Pick random flip
            idx = random.randint(0, n-1)
            c[idx] *= -1
            new_score = get_max_norm(c)
            
            delta_e = new_score - current_score
            
            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_c = c.copy()
            
            T *= alpha
            
            if T < T_min:
                break
        
        # Polish: Steepest Descent
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for j in range(n):
                c[j] *= -1
                new_score = get_max_norm(c)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_c = c.copy()
                    improved = True
                else:
                    c[j] *= -1 # revert
                    # Early exit if we found an improvement? 
                    # No, we should check all flips to find the best one?
                    # Steepest descent usually means pick the best flip.
                    # But "First Improvement" is faster.
                    # Let's stick to First Improvement for speed, or just one pass.
                    # Actually, if we revert, we might miss a better flip.
                    # Let's just do one pass of "if any flip improves, do it and restart pass".
        
        restarts += 1

    return best_c.tolist()
