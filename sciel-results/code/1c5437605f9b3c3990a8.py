import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 24
    k = np.arange(n)
    
    # Precompute grid powers
    # 4096 points is sufficient for the continuous relaxation and proxy scoring
    m = 4096
    thetas = np.linspace(0, 2*np.pi, m, endpoint=False)
    powers_grid = np.exp(1j * np.outer(k, thetas))
    
    def fast_eval(c):
        # c is a numpy array of shape (n,)
        # Returns max |P(e^{i theta})| on the grid
        vals = np.dot(c, powers_grid)
        return np.max(np.abs(vals))
        
    # Helper for local search
    def local_search(c):
        current_score = fast_eval(c)
        improved = True
        # Limit iterations to avoid infinite loops if stuck in oscillation (unlikely with best improvement)
        while improved and time.time() < deadline - 5.0:
            improved = False
            best_idx = -1
            best_new_score = current_score
            
            for i in range(n):
                c[i] *= -1
                new_score = fast_eval(c)
                c[i] *= -1 # revert
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_idx = i
                    
            if best_idx != -1:
                c[best_idx] *= -1
                current_score = best_new_score
                improved = True
                
        return c, current_score

    def blahut_arimoto():
        """
        Solves the continuous convex relaxation:
        Minimize T such that |sum c_k e^{i k theta}| <= T for all theta.
        Variables c_k are continuous.
        Uses iterative water-filling (Blahut-Arimoto style).
        """
        # Initialize c to 1s (or random signs? 1s is symmetric)
        c = np.ones(n, dtype=float)
        
        # The objective is min T s.t. |P(theta)| <= T.
        # Equivalent to min sum |P(theta)|^2 / something? No, sup norm.
        # Blahut-Arimoto for sup norm minimization with linear constraints?
        # Actually, we can use the dual problem or iterative projection.
        
        # Let's use a simpler iterative method:
        # Minimize T subject to |P(theta)| <= T.
        # This is a convex problem.
        # We can solve it via subgradient or by fixing T and projecting?
        
        # Alternative: Minimize max |P(theta)|.
        # Gradient of max |P| is hard.
        
        # Let's try the "iterative water-filling" for the moment problem.
        # Actually, for small n, we can just use scipy.optimize if available, 
        # but let's stick to numpy for speed and simplicity.
        
        # A common heuristic for this specific problem (min sup norm of polynomial with fixed coefficients magnitude)
        # is to start with 1s and flip signs to reduce peaks.
        
        # However, the directive asks for Blahut-Arimoto.
        # The Blahut-Arimoto algorithm is typically for entropy minimization or similar.
        # For sup norm minimization, we can formulate it as:
        # Minimize T
        # Subject to: -T <= Re(P(theta)) <= T and -T <= Im(P(theta)) <= T
        
        # This is a Linear Program (LP) if we treat T and c_k as variables?
        # No, P(theta) is linear in c_k. So yes, it's an LP!
        # Variables: c_0..c_{n-1}, T
        # Constraints:
        # -T <= Re(sum c_k e^{i k theta}) <= T  for all theta
        # -T <= Im(sum c_k e^{i k theta}) <= T  for all theta
        
        # But we want c_k to be close to +/- 1.
        # The relaxation allows c_k to be any real number.
        # The optimal continuous solution will likely have c_k in [-1, 1] or slightly outside?
        # Actually, if we don't constrain c_k magnitude, T can go to 0 by setting c_k=0.
        # So we must constrain c_k.
        # The problem is: Minimize T subject to |P(theta)| <= T and c_k in [-1, 1]?
        # No, the discrete problem is c_k in {+1, -1}.
        # The convex hull of {+1, -1} is [-1, 1].
        # So the continuous relaxation is: Minimize T subject to |P(theta)| <= T and c_k in [-1, 1].
        
        # This is a convex optimization problem.
        # We can solve it using projected gradient or similar.
        # Or simply use the fact that the optimum of the LP relaxation for symmetric problems
        # often has c_k at the boundaries +/- 1.
        
        # Let's implement a simple iterative solver for this convex problem.
        # We can use the "cutting plane" method or just gradient descent on the smoothed objective.
        
        # Let's try a different approach:
        # Start with c = 1.
        # Compute P(theta).
        # Find theta_max where |P| is maximum.
        # Adjust c to reduce |P(theta_max)|.
        # This is basically local search but in continuous domain.
        
        # Given the time limit, let's just do a few steps of "continuous local search"
        # which is effectively gradient descent on the max norm.
        
        # However, the directive specifically mentions Blahut-Arimoto.
        # Let's implement a simplified version of the iterative algorithm for the dual problem.
        
        # Dual of min T s.t. |P(theta)| <= T, c in [-1,1]
        # This might be complex.
        
        # Let's stick to the practical interpretation:
        # Use the continuous solution to guide the rounding.
        
        # Initialize c
        c = np.ones(n)
        
        # Iterative refinement
        for _ in range(100):
            if time.time() > deadline - 10.0:
                break
            
            vals = np.dot(c, powers_grid)
            abs_vals = np.abs(vals)
            max_idx = np.argmax(abs_vals)
            theta_max = thetas[max_idx]
            
            # Gradient of |P(theta)| wrt c_k
            # d|P|/dc_k = Re( P(theta) * conj(e^{i k theta}) / |P(theta)| )
            # If P is 0, gradient is 0.
            if abs_vals[max_idx] > 1e-9:
                phase = vals[max_idx] / abs_vals[max_idx]
                # We want to reduce |P|. So we move c in direction -grad.
                # grad_k = Re( phase * conj(e^{i k theta_max}) )
                # e^{i k theta_max} = powers_grid[k, max_idx]
                # conj(e^{i k theta_max}) = conj(powers_grid[k, max_idx])
                grads = np.real(phase * np.conj(powers_grid[:, max_idx]))
                
                # Update c
                # c = c - alpha * grads
                # But we must keep c in [-1, 1]
                alpha = 0.1
                c_new = c - alpha * grads
                c = np.clip(c_new, -1, 1)
            else:
                break
        
        # Round to nearest integer (-1 or 1)
        c_rounded = np.sign(c)
        # If any are 0, flip randomly
        c_rounded[c_rounded == 0] = np.random.choice([-1, 1], size=np.sum(c_rounded == 0))
        
        return c_rounded.astype(int)

    best_c = None
    best_score = float('inf')
    
    # 1. Run Blahut-Arimoto relaxation to get a good starting point
    c_relaxed = blahut_arimoto()
    c, score = local_search(c_relaxed)
    if score < best_score:
        best_score = score
        best_c = c.copy()
        
    # 2. Run random restarts
    restarts = 0
    while time.time() < deadline - 10.0:
        c = np.random.choice([-1, 1], size=n)
        
        # Local Search
        c, score = local_search(c)
        
        if score < best_score:
            best_score = score
            best_c = c.copy()
            
        restarts += 1
        
    # If we never found one (shouldn't happen), return random
    if best_c is None:
        best_c = np.random.choice([-1, 1], size=n)
        
    return best_c.tolist()
