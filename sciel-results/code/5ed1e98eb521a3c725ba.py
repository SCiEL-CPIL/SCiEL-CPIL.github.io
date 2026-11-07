import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        eval_count = [0]
        all_X = []
        all_F = []
        
        def count_eval(x):
            eval_count[0] += 1
            if eval_count[0] > self.budget:
                raise Exception("Budget exceeded")
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            return f

        # 1. Initialization: Latin Hypercube Sampling for better coverage
        n_init = 40
        rng = np.random.default_rng(0)
        
        # LHS in 5D
        n_per_dim = n_init
        X_init = []
        for d in range(self.dim):
            samples = (np.arange(n_per_dim) + rng.random(n_per_dim)) / n_per_dim
            X_init.append(samples)
        
        X_init = np.array(X_init).T
        X_init = lo + (hi - lo) * X_init
        
        for x in X_init:
            if eval_count[0] >= self.budget:
                break
            count_eval(x)
            
        # Current non-dominated set
        # We'll maintain a list of solutions.
        
        def is_dominated(f, archive):
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def get_non_dominated(F):
            F = np.array(F)
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i != j and mask[j]:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            mask[i] = False
                            break
            return mask

        # Extract initial non-dominated set
        F_init = np.array(all_F)
        mask = get_non_dominated(F_init)
        nd_X = np.array(all_X)[mask]
        nd_F = np.array(all_F)[mask]
        
        # If no non-dominated points (shouldn't happen), fallback to all
        if len(nd_X) == 0:
            nd_X = np.array(all_X)
            nd_F = np.array(all_F)

        # Strategy:
        # We have ~160 evals left.
        # We will use Adaptive Tchebycheff Scalarization.
        # We generate a set of weight vectors.
        # For each weight, we run L-BFGS-B from the best starting point in the current archive 
        # (or center/random if archive is empty).
        
        # Generate weight vectors on a grid for 3 objectives.
        # Grid size m=8 gives (8+1)*(8+2)/2 = 45 weights.
        # 160 / 45 ~ 3.5 evals per weight. Too low for L-BFGS-B to converge well.
        # However, L-BFGS-B is very efficient. Let's try smaller grid or fewer weights?
        # Or use the "key weights" approach from the original but with better starting points.
        
        # Let's use a hybrid:
        # 1. A few key weights (vertices, centroid, edges) with more budget.
        # 2. Remaining budget for random weights or local search on non-dominated points.
        
        key_weights = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1/3.0, 1/3.0, 1/3.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
            np.array([0.7, 0.15, 0.15]),
            np.array([0.15, 0.7, 0.15]),
            np.array([0.15, 0.15, 0.7]),
        ]
        
        # Add some random weights to cover other parts of the front
        for _ in range(5):
            w = rng.random(self.n_obj)
            w /= w.sum()
            key_weights.append(w)
            
        # Shuffle to mix up
        rng.shuffle(key_weights)
        
        # We need to estimate the ideal point (z*) and nadir point (z^nad) for Tchebycheff.
        # z*_i = min(F_i)
        # z^nad_i = max(F_i)
        # Since we don't know the true nadir, we use the current max as an approximation,
        # or just use Weighted Sum if Tchebycheff is tricky to scale.
        # Weighted Sum is robust but only finds convex hull.
        # Tchebycheff: min max_i w_i * (f_i - z*_i) / (z^nad_i - z*_i)
        # If range is 0, handle division by zero.
        
        z_ideal = np.min(nd_F, axis=0)
        z_nadir = np.max(nd_F, axis=0)
        range_f = z_nadir - z_ideal
        range_f[range_f == 0] = 1.0 # Avoid div by zero
        
        for w in key_weights:
            if eval_count[0] >= self.budget - 5:
                break
            if time.time() > deadline - 10:
                break
                
            # Define Tchebycheff objective
            def tchebycheff_obj(x):
                f = count_eval(x)
                # Normalize
                f_norm = (f - z_ideal) / range_f
                return np.max(w * f_norm)
            
            # Choose starting point
            # Best point for this weight in current archive?
            # Calculate tchebycheff value for all current points
            # We don't have F for all points in a vectorized way easily inside this loop without extra evals.
            # We can just use the point in nd_F that minimizes the weighted sum as a proxy for start.
            
            if len(nd_F) > 0:
                # Weighted sum proxy for start
                ws = nd_F @ w
                best_idx = np.argmin(ws)
                x0 = nd_X[best_idx]
            else:
                x0 = (lo + hi) / 2
                
            # Run L-BFGS-B
            try:
                # maxiter=5, maxls=5. 
                # L-BFGS-B uses finite differences. With dim=5, each gradient step costs ~6 evals.
                # So 5 iterations is ~30 evals.
                # If we have 15 weights, 15*30 = 450 > 160.
                # We need to limit evals per run.
                # Let's set maxfun explicitly.
                remaining = self.budget - eval_count[0]
                # Allow up to 15 evals per run if budget is tight, more if we have lots left?
                # But we have fixed list of weights.
                # Let's just let it run but catch the exception.
                # To be safe, let's limit maxiter based on remaining budget.
                # If remaining > 100, allow 10 iter. Else 5.
                max_iter = 10 if remaining > 100 else 5
                
                res = minimize(tchebycheff_obj, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': max_iter, 'maxls': 5, 'ftol': 1e-8})
                
                # Update ideal/nadir if we found better points
                # res.x is the result. We need to get its F.
                # It was evaluated inside tchebycheff_obj, so it's in all_F.
                # The last evaluation was res.x? Not necessarily, L-BFGS-B might evaluate other points.
                # But the final point is res.x.
                # We should ensure res.x is in our archive.
                # Let's explicitly evaluate res.x to be sure it's recorded and to get f_res.
                # Wait, count_eval raises exception if budget exceeded.
                # If minimize finished, it didn't raise. So we have budget for 1 more eval.
                
                # Check if we have budget for one more eval to record the final point explicitly
                # (though it should already be there if it was the last eval)
                if eval_count[0] < self.budget:
                    f_res = count_eval(res.x)
                    # Update ideal
                    z_ideal = np.minimum(z_ideal, f_res)
                    # Update nadir? Only if it's worse? No, nadir should be max.
                    z_nadir = np.maximum(z_nadir, f_res)
                    range_f = z_nadir - z_ideal
                    range_f[range_f == 0] = 1.0
                    
                    # Update non-dominated set
                    # Check if f_res is dominated
                    dominated = False
                    for af in nd_F:
                        if np.all(af <= f_res) and np.any(af < f_res):
                            dominated = True
                            break
                    
                    if not dominated:
                        # Remove any points dominated by f_res
                        new_nd_F = []
                        new_nd_X = []
                        for i, af in enumerate(nd_F):
                            if not (np.all(f_res <= af) and np.any(f_res < af)):
                                new_nd_F.append(af)
                                new_nd_X.append(nd_X[i])
                        
                        nd_F = np.array(new_nd_F)
                        nd_X = np.array(new_nd_X)
                        
                        # Add f_res
                        if len(nd_F) == 0:
                            nd_F = f_res.reshape(1, -1)
                            nd_X = res.x.reshape(1, -1)
                        else:
                            nd_F = np.vstack([nd_F, f_res])
                            nd_X = np.vstack([nd_X, res.x])
                            
            except Exception:
                # Budget exceeded or other error
                # We might have partially evaluated.
                # Just break the loop.
                break

        return all_X
