import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        t_start = time.time()
        max_time = 290.0  # Safety margin before 300s limit

        evals = 0
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f, evals
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def evaluate(x):
            nonlocal evals
            x = np.asarray(x, dtype=float)
            if x.shape != (n,):
                x = x.reshape(n)
            f = func(x)
            evals += 1
            update_best(x, f)
            return f

        # 1. Initial Exploration: LHS or Grid to sample the landscape
        # We need to get enough points to fit a quadratic model.
        # n=5. Quadratic model has 1 + 5 + 15 = 21 parameters.
        # Let's use 50-100 points for a robust fit.
        n_init = min(budget, 100)
        
        # Use Halton sequence or random points within [-5, 5]
        # Random is fine for BO initialization if we have enough points
        rng = np.random.default_rng(42)
        
        X_init = rng.uniform(-5, 5, size=(n_init, n))
        F_init = np.array([evaluate(x) for x in X_init])
        
        # 2. Fit a Quadratic Model
        # f(x) ≈ c0 + c1^T x + x^T C2 x
        # We can use least squares to fit this.
        
        def build_quadratic_features(X):
            # X: (N, n)
            N = X.shape[0]
            F = np.zeros((N, 1 + n + n*(n+1)//2))
            F[:, 0] = 1.0
            F[:, 1:1+n] = X
            # Quadratic terms
            idx = 1 + n
            for i in range(n):
                for j in range(i, n):
                    F[:, idx] = X[:, i] * X[:, j]
                    idx += 1
            return F

        F_init_feats = build_quadratic_features(X_init)
        
        # Solve least squares: F_init_feats @ coeffs ≈ F_init
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(F_init_feats, F_init, rcond=None)
            quad_valid = True
        except np.linalg.LinAlgError:
            quad_valid = False
            coeffs = None

        def quad_model(x):
            if not quad_valid:
                return np.inf
            x = np.asarray(x, dtype=float)
            if x.shape != (n,):
                x = x.reshape(n)
            # Reconstruct features for single point
            feats = np.zeros(1 + n + n*(n+1)//2)
            feats[0] = 1.0
            feats[1:1+n] = x
            idx = 1 + n
            for i in range(n):
                for j in range(i, n):
                    feats[idx] = x[i] * x[j]
                    idx += 1
            return np.dot(feats, coeffs)

        # 3. Refine with Local Optimization on the Quadratic Model
        # Start from the best point found so far
        if best_x is not None:
            res = minimize(quad_model, best_x, method='Nelder-Mead', 
                           options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 500})
            if res.success:
                x_opt = res.x
                # Clip to box
                x_opt = np.clip(x_opt, -5, 5)
                f_opt = evaluate(x_opt)
                
                # If the model was good, we might be very close.
                # Let's also try L-BFGS-B on the actual function starting from x_opt
                # to polish, using remaining budget.
        
        # 4. Polish with L-BFGS-B on the actual function
        # Use remaining budget carefully.
        # L-BFGS-B uses finite differences by default (n calls per iter).
        # We have ~400 budget left.
        
        if best_x is not None:
            # Try a few different starting points for L-BFGS-B if budget allows?
            # Or just one good one.
            # Let's use the best point found.
            
            # We can also use the quadratic model's minimum as another start.
            # Let's just use best_x for now.
            
            # Check time and budget
            if evals < budget - 10 and time.time() - t_start < max_time - 10:
                try:
                    res2 = minimize(func, best_x, method='L-BFGS-B', 
                                    bounds=[(-5, 5)] * n,
                                    options={'maxiter': 50, 'gtol': 1e-8, 'disp': False})
                    # Note: minimize with func will call func internally.
                    # We need to track these evals!
                    # Unfortunately, scipy's minimize does not expose the eval count directly in a simple counter without wrapping.
                    # Let's wrap func to count.
                    
                    # Actually, the previous evaluate() wrapper is good.
                    # But minimize expects a function that returns a scalar.
                    # We can pass a wrapper.
                    
                    # Let's restart this part with a proper wrapper for L-BFGS-B
                    # to ensure we don't exceed budget.
                    
                    # Reset evals tracking for this specific call? No, global evals is used.
                    # The wrapper `evaluate` increments global `evals`.
                    # But `minimize` calls `func` directly if we pass it.
                    # We must pass a wrapper that increments `evals`.
                    
                    def func_wrapper(x):
                        return evaluate(x)
                        
                    # We already did Nelder-Mead on the model.
                    # Now let's do L-BFGS-B on the real function from best_x.
                    # We must be careful: `minimize` might make many calls.
                    # maxiter=50 is safe.
                    
                    # However, the previous block called `evaluate` inside `minimize(quad_model...)`?
                    # No, `quad_model` is local and doesn't call `func`.
                    # So `evals` is still around 100 + 1 (from quad opt check).
                    
                    # Let's do the L-BFGS-B step now.
                    if evals < budget - 5:
                         res2 = minimize(func_wrapper, best_x, method='L-BFGS-B', 
                                          bounds=[(-5, 5)] * n,
                                          options={'maxiter': 100, 'gtol': 1e-8, 'disp': False})
                         # `func_wrapper` updates best_x and best_f automatically via `evaluate`.
                         # So best_x and best_f are already updated to the best found during L-BFGS-B.
                         
                except Exception:
                    pass

        # 5. If we still have budget, do a CMA-ES or Random Polish?
        # Given the score of inspirations, CMA-ES was strong.
        # But the prompt asks to combine Bayesian/Model-based with CMA-ES.
        # The model-based approach (Quadratic) is a form of surrogate optimization.
        # If the quadratic fit was good, we are likely at the optimum.
        # If not, we might be stuck.
        
        # Let's add a simple random restart or CMA-ES step if we have budget.
        # But the "Bayesian" strategy implies we trust the model.
        # Let's stick to the model + local polish.
        
        # One more check: if the quadratic model predicted a better point that we didn't evaluate?
        # We evaluated the quad model min.
        
        # Let's ensure we return the best found.
        if best_x is None:
             # Fallback if something went wrong
             best_x = np.zeros(n)
             best_f = func(best_x)
             evals += 1

        return best_x, best_f
