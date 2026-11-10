"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option using smooth policy optimization.
    
    Strategy:
    1. Represent the exercise decision at each step j as a smooth function of the state S_j:
       p_j(S) = sigmoid( w_j * (S - B_j) )
       where B_j is the effective boundary and w_j controls the steepness.
       We exercise with probability 1 - p_j(S) (i.e., when S is low).
       Actually, let's define the probability of *not* exercising (continuing) as:
       q_j(S) = sigmoid( w_j * (B_j - S) )
       If S < B_j, B_j - S > 0, sigmoid is high -> high probability of continuing? No, that's wrong.
       
       Let's define the probability of EXERCISING as:
       P(exercise | S, j) = sigmoid( w_j * (B_j - S) )
       If S is very small, B_j - S is large positive -> sigmoid ~ 1 -> Exercise.
       If S is very large, B_j - S is large negative -> sigmoid ~ 0 -> Don't exercise.
       This makes sense.
       
    2. The objective is the expected discounted payoff.
       V = E[ sum_{j=1}^{12} D_j * I(exercise_j) * (K - S_j)^+ ]
       
       Since the decisions are coupled (exercising at j precludes decisions at j+1...),
       the gradient is complex. However, we can use the "score function" estimator (REINFORCE)
       or simply use the fact that for a fixed set of paths, we can compute the value for a
       deterministic policy (threshold) and use finite differences or subgradient methods.
       
       BUT, a better structural change: Use the "Smooth Policy" directly in the expectation
       via Monte Carlo rollouts? No, that's expensive.
       
       Alternative Structural Change: 
       Use the training paths to compute the **exact** optimal exercise decision for each path
       assuming we know the continuation value? No, that's circular.
       
       Let's stick to the **Smooth Threshold Policy** but optimize it using a **vectorized
       coordinate ascent** that is much faster than the random search, or simply use a 
       **Grid Search** over a 2D manifold of boundaries (e.g., log-linear boundaries) with 
       vectorized evaluation.
       
       Actually, the most robust "different" approach that often works well for Bermudans 
       with limited time is **Longstaff-Schwarz (LSM)** but done correctly with 
       **Polynomial Regression** and **Out-of-Sample** validation, or simply 
       **Backward Induction on the Training Paths** (Monte Carlo Tree/DP).
       
       Let's try **Monte Carlo Backward Induction (DP)** on the training paths.
       This is fundamentally different from optimizing a parametric boundary.
       
       Algorithm:
       1. Sort paths? No, DP on paths is tricky because states don't form a grid.
       
       Let's go back to **Optimization** but make it **Vectorized and Deterministic**.
       Instead of random search, use **Coordinate Ascent with Vectorized Evaluation**.
       The previous code was slow because `compute_objective` was called 5000*5 times 
       and was a Python loop over 12 steps.
       
       We can vectorize the objective calculation for a single set of boundaries.
       But even better: Let's use a **Global Optimization** approach with a 
       **Differentiable Surrogate**?
       
       Let's try a **Hybrid**: 
       1. Initialize boundaries using a heuristic (e.g., American Put approximation).
       2. Use **Nelder-Mead** or **Powell** method from SciPy? 
          SciPy is available. `scipy.optimize.minimize` with method='Nelder-Mead' 
          or 'Powell' is robust for non-smooth, low-dimensional (12D) problems.
       
       This is a structural change: Replacing custom random search with a 
       robust, library-backed global/local optimizer (Nelder-Mead) that 
       handles non-smoothness better than random perturbations and requires 
       fewer function evaluations to converge in 12D.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    
    if training_paths is None or len(training_paths) == 0:
        class FallbackPolicy:
            def exercise(self, current_state, time):
                return current_state < K
        return FallbackPolicy()

    paths = np.asarray(training_paths, dtype=float)
    n_paths, n_cols = paths.shape
    
    if n_cols >= 13:
        start_idx = 1
        end_idx = 12
    else:
        start_idx = 0
        end_idx = 11
        
    S_ex = paths[:, start_idx:end_idx+1]  # Shape (n_paths, 12)
    
    dt = T / n_exercises
    disc = np.exp(-r * dt)
    discount_factors = np.array([disc ** (n_exercises - 1 - j) for j in range(n_exercises)])
    
    intrinsic = np.maximum(K - S_ex, 0.0)  # Shape (n_paths, 12)
    
    # Initial guess for boundaries
    # Heuristic: American Put boundary is below K.
    # Early steps: lower boundary. Late steps: higher boundary (close to K).
    # B_j = K * (0.6 + 0.4 * j / (n_exercises - 1))
    boundaries_init = K * (0.6 + 0.4 * np.arange(n_exercises) / (n_exercises - 1))
    boundaries_init[-1] = K
    
    # Vectorized Objective Function
    # Input: bounds (12,)
    # Output: -Value (to minimize)
    
    def compute_value_vec(bounds):
        """
        Computes the expected discounted payoff for a given set of boundaries.
        Uses a loop over time steps but vectorized over paths.
        This is fast enough for Nelder-Mead.
        """
        total_value = 0.0
        active = np.ones(n_paths, dtype=bool)
        
        for j in range(n_exercises):
            if not np.any(active):
                break
            
            # Exercise condition: S_j < bounds[j]
            # Only check active paths
            s_j = S_ex[:, j]
            intrinsic_j = intrinsic[:, j]
            
            # Mask: active AND s_j < bounds[j]
            ex_mask = active & (s_j < bounds[j])
            
            if np.any(ex_mask):
                total_value += discount_factors[j] * np.sum(intrinsic_j[ex_mask])
                active[ex_mask] = False
        
        return total_value / n_paths
    
    def objective_func(bounds):
        """
        Objective for minimization: -Value
        Enforces monotonicity and bounds.
        """
        # Project bounds to be monotonic and within [0, K]
        # B_0 <= B_1 <= ... <= B_11 = K
        # We optimize B_0 ... B_10, B_11 is fixed to K.
        # To handle monotonicity in the optimizer, we can parameterize 
        # using deltas: B_j = B_{j-1} + delta_j, where delta_j >= 0.
        # But Nelder-Mead works on unconstrained variables.
        # Simple approach: Clip and sort? No, sorting changes the meaning.
        # We will enforce monotonicity by "forward fill" of max.
        
        b = np.asarray(bounds).copy()
        
        # Enforce lower bound 0
        b = np.maximum(b, 0.0)
        
        # Enforce monotonicity: B_j >= B_{j-1}
        for j in range(1, n_exercises):
            if b[j] < b[j-1]:
                b[j] = b[j-1]
                
        # Enforce upper bound K for all, and fix last to K
        b = np.minimum(b, K)
        b[-1] = K
        
        val = compute_value_vec(b)
        return -val
    
    # Use SciPy Nelder-Mead
    try:
        from scipy.optimize import minimize
        
        # Initial guess
        x0 = boundaries_init.copy()
        
        # Options
        options = {
            'xatol': 0.01,
            'fatol': 0.001,
            'maxiter': 200,
            'maxfev': 500, # Limit function evaluations to stay within time
            'disp': False
        }
        
        result = minimize(objective_func, x0, method='Nelder-Mead', options=options)
        
        best_bounds = result.x
        best_val = -result.fun
        
        # Check time
        if time.time() > deadline - 5.0:
            pass # Use whatever we have
            
        # Post-process best bounds
        best_bounds = np.maximum(best_bounds, 0.0)
        for j in range(1, n_exercises):
            if best_bounds[j] < best_bounds[j-1]:
                best_bounds[j] = best_bounds[j-1]
        best_bounds = np.minimum(best_bounds, K)
        best_bounds[-1] = K
        
    except Exception:
        # Fallback to initial heuristic if optimization fails or times out badly
        best_bounds = boundaries_init
        
    class ExercisePolicy:
        def __init__(self, boundaries, n_exercises):
            self.boundaries = boundaries
            self.n_exercises = n_exercises
            
        def exercise(self, current_state, time):
            """
            current_state: array of stock prices.
            time: step index 0..11.
            """
            if time < 0 or time >= self.n_exercises:
                return np.zeros(len(current_state), dtype=bool)
            
            B = self.boundaries[time]
            return current_state < B
    
    return ExercisePolicy(best_bounds, n_exercises)
# EVOLVE-BLOCK-END
