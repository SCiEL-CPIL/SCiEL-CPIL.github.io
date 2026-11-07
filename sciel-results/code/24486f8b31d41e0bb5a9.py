"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Quadratic surrogate model optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        # History for surrogate model
        X = [x.copy()]
        F = [fx]
        
        # Initial exploration: sample a few points to get a sense of the landscape
        # The bent cigar is narrow in one direction and wide in others.
        # We need points that span the space to fit a paraboloid.
        # Let's do a few random steps first to populate history.
        
        sigma = 2.0
        for i in range(min(10, self.budget - 1)):
            if time.time() > deadline: break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            X.append(y)
            F.append(fy)
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()
            
            # Keep x as the current "center" for local search? 
            # Or just use the best point as center?
            # Let's use the best point as the center for the surrogate to focus on the basin.
            # But we also need spread.
            # Let's keep a history of the last 25 points.
            if len(X) > 25:
                X.pop(0)
                F.pop(0)
                
            # For the first few steps, just do random walk to get spread
            x = y

        # Now use surrogate model
        # We want to fit f(x) ~ (x - c)^T A (x - c) + b
        # Or simpler: least squares quadratic fit in 5D.
        # Features: 1, x_1..x_5, x_1^2..x_5^2, x_1*x_2...
        # Total features: 1 + 5 + 5 + 10 = 21.
        # We have 25 points. This is overdetermined, good.
        
        for i in range(10, self.budget):
            if time.time() > deadline: break
            
            # Build design matrix for quadratic fit
            # Using only the last 25 points
            n_pts = len(X)
            if n_pts < 15: # Need enough points for a decent fit
                # Fallback to random step
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                X.append(y)
                F.append(fy)
                if len(X) > 25:
                    X.pop(0)
                    F.pop(0)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                x = y
                continue
            
            X_arr = np.array(X)
            F_arr = np.array(F)
            
            # Construct features
            # 1, x, x^2, x_i x_j
            ones = np.ones((n_pts, 1))
            x_lin = X_arr
            x_sq = X_arr ** 2
            # Interactions
            x_int = []
            for i in range(self.dim):
                for j in range(i+1, self.dim):
                    x_int.append(X_arr[:, i] * X_arr[:, j])
            if x_int:
                x_int = np.column_stack(x_int)
            else:
                x_int = np.zeros((n_pts, 0))
                
            Phi = np.hstack([ones, x_lin, x_sq, x_int])
            
            # Solve least squares
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(Phi, F_arr, rcond=None)
            except:
                coeffs = None
            
            if coeffs is not None:
                # Predict minimizer of the quadratic
                # f(x) = c0 + c_lin^T x + c_sq^T (x^2) + c_int^T (x_i x_j)
                # This is equivalent to x^T H x + g^T x + c0
                # H is symmetric.
                # Let's extract H and g.
                # Phi columns:
                # 0: 1
                # 1..5: x_1..x_5
                # 6..10: x_1^2..x_5^2
                # 11..20: x_1x_2, x_1x_3, ...
                
                c0 = coeffs[0]
                c_lin = coeffs[1:6]
                c_sq = coeffs[6:11]
                c_int = coeffs[11:21]
                
                # Build Hessian H (5x5) and gradient vector g (5)
                H = np.zeros((self.dim, self.dim))
                g = c_lin.copy()
                
                for i in range(self.dim):
                    H[i, i] = 2 * c_sq[i]
                    for j in range(i+1, self.dim):
                        # Find index in c_int
                        # Indices: (0,1), (0,2), ..., (0,4), (1,2), ...
                        # Let's just iterate
                        pass
                
                # Simpler: reconstruct H and g from coeffs
                # col 1+i is x_i -> g[i] += c_lin[i]
                # col 6+i is x_i^2 -> H[i,i] += 2*c_sq[i]
                # col 11+k is x_i x_j -> H[i,j] += c_int[k], H[j,i] += c_int[k]
                
                # Let's map interactions properly
                int_idx = 0
                for i in range(self.dim):
                    for j in range(i+1, self.dim):
                        val = c_int[int_idx]
                        H[i, j] += val
                        H[j, i] += val
                        int_idx += 1
                        
                # The quadratic form is x^T H x + g^T x + c0
                # Minimizer x* = -0.5 * H^{-1} g
                # Check if H is positive definite (or at least invertible)
                try:
                    # Add small regularization to ensure stability
                    H_reg = H + 1e-6 * np.eye(self.dim)
                    x_pred = -0.5 * np.linalg.solve(H_reg, g)
                except:
                    x_pred = None
                    
                if x_pred is not None:
                    # Clip to box
                    x_pred = np.clip(x_pred, -5, 5)
                    
                    # If prediction is very close to current best, take a small random step
                    # Otherwise, step towards prediction
                    dist = np.linalg.norm(x_pred - best_x)
                    if dist < 0.1:
                        # Local search
                        y = np.clip(best_x + 0.1 * rng.standard_normal(self.dim), -5, 5)
                    else:
                        # Step towards prediction, but maybe not all the way to avoid overshooting
                        # Or just go there. Surrogate is local, so it should be okay.
                        # Let's take a step of 0.5 towards it, or just evaluate it.
                        # Evaluating it directly is more aggressive.
                        y = x_pred
                        
                    fy = func(y)
                    X.append(y)
                    F.append(fy)
                    if len(X) > 25:
                        X.pop(0)
                        F.pop(0)
                        
                    if fy < best_fx:
                        best_fx = fy
                        best_x = y.copy()
                        
                    # Update current center for next surrogate fit
                    # Should we move x to y? Yes, to keep the model local to the progress.
                    x = y
            else:
                # Fallback
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                X.append(y)
                F.append(fy)
                if len(X) > 25:
                    X.pop(0)
                    F.pop(0)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                x = y

        return best_x, best_fx
# EVOLVE-BLOCK-END
