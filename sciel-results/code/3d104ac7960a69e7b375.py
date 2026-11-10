# EVOLVE-BLOCK-START
import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Formulate as a quadratic program w^T M w with constraints w >= 0, sum(w) = 1.
    Use scipy.optimize.minimize (L-BFGS-B) to maximize P.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin
    
    # We need to find w (length n) to maximize P = sum_{k,l} w_k w_l * (sum_{i,j: s_i+s_j < 2*s_l-s_k} w_i w_j)
    # This is a quartic form in w, but we can write it as w^T A w where A is a 4-index tensor contracted.
    # Actually, P = sum_{k,l,i,j} w_k w_l w_i w_j * I(s_i + s_j < 2*s_l - s_k)
    # This is a homogeneous quartic polynomial. It is NOT a quadratic form w^T M w.
    # Wait, the prompt says "expressing the target probability P as a quadratic form w^T M w".
    # Let's re-read carefully.
    # "P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)"
    # This is indeed a 4th degree polynomial.
    # However, if we fix the support points, the objective is a 4th order polynomial.
    # Minimizing/maximizing a 4th order polynomial is generally non-convex.
    # But maybe the prompt implies a specific structure or approximation?
    # Or maybe I should just use a general non-linear optimizer on the 4th order polynomial?
    # The prompt says: "Formulate the optimization as a convex quadratic program... expressing P as a quadratic form w^T M w".
    # This is mathematically impossible for a general 4th order term unless there's a trick.
    # Let's look at the structure again.
    # X1, X2, X3, X4 are i.i.d.
    # P[X1 + X2 + X3 < 2 X4]
    # Let S = X1 + X2 + X3. We want P[S < 2 X4].
    # This is not a quadratic form in w.
    # Perhaps the "quadratic form" hint is misleading or refers to an approximation?
    # Or maybe I should just use `scipy.optimize.minimize` with the exact 4th order objective?
    # The directive says: "Use scipy.optimize.minimize with a BFGS or L-BFGS-B solver to maximize this co..." (cut off)
    # Let's assume the strategy is to use L-BFGS-B on the exact objective function, treating it as a general non-linear optimization.
    # The "quadratic program" part might be a misinterpretation in the prompt generation or refers to a local approximation.
    # I will implement the exact objective and use L-BFGS-B.
    
    # Grid size
    # The evaluator builds n^2 pairwise sums. n=64 is 4096 sums, very fast.
    # n=128 is 16384 sums, still fast.
    # Let's try n=64 first for optimization speed.
    n = 64
    
    # Support points
    # We need to choose support points. The prompt says "fixed grid".
    # Let's use a grid from 0 to some max_val.
    # What is a good max_val?
    # If support is too small, P might be limited.
    # If support is too large, the optimizer might struggle.
    # Let's try a grid from 0 to 10, or 0 to 100.
    # Actually, scaling the support doesn't change P if we scale the weights? No, the inequality is scale-invariant if we scale all support points.
    # s_i + s_j < 2 s_l - s_k. If we multiply all s by c, the inequality becomes c(s_i+s_j) < c(2s_l-s_k), which is the same.
    # So the scale of the support doesn't matter, only the relative positions.
    # We can fix the support to be, say, 0 to 1, or 0 to n-1.
    # Let's use support = np.linspace(0, 1, n).
    # But wait, if we use a fixed grid, we are restricting the support.
    # Can we optimize the support points as well?
    # The prompt says "parameterizing the discrete measure's probability mass function directly on a fine, fixed grid".
    # So support is fixed. Only weights are optimized.
    
    support = np.linspace(0.0, 1.0, n, dtype=np.float64)
    
    # Precompute the indicator matrix for the inequality
    # Inequality: s_i + s_j < 2*s_l - s_k
    # This depends on k, l, i, j.
    # We can precompute a 4D array, but that's n^4 which is 64^4 = 16,777,216. That's manageable.
    # But we can do better.
    # For each pair (k, l), we need the sum of w_i w_j over (i, j) with s_i + s_j < 2*s_l - s_k.
    # Let T_{kl} = sum_{i,j: s_i+s_j < 2*s_l-s_k} w_i w_j.
    # Then P = sum_{k,l} w_k w_l T_{kl}.
    # This is a quadratic form in w if T_{kl} were constant, but T_{kl} depends on w.
    # So P = sum_{k,l,i,j} w_k w_l w_i w_j I_{kl,ij}.
    
    # Let's precompute the 4D tensor I_{kl,ij} or just compute the objective on the fly.
    # On the fly:
    # For a given w, compute all pairwise sums s_i + s_j.
    # Then for each k, l, we need sum_{i,j: s_i+s_j < 2*s_l-s_k} w_i w_j.
    # This can be vectorized.
    
    def compute_P(w):
        """
        Compute P[X1 + X2 + X3 < 2*X4] for a given weight vector w.
        w is assumed to be normalized (sum to 1).
        """
        # w is 1D array of length n
        # Pairwise sums: S[i, j] = s_i + s_j
        S = support[:, None] + support[None, :]
        # W[i, j] = w_i * w_j
        W = w[:, None] * w[None, :]
        
        # For each k, l, we need sum of W[i, j] where S[i, j] < 2*s_l - s_k
        # Let's create a 4D mask or use broadcasting.
        # Threshold for (k, l) is T_{kl} = 2*s_l - s_k
        # T is an n x n matrix.
        T = 2.0 * support[None, :] - support[:, None]  # T[k, l] = 2*s_l - s_k
        
        # We need for each (k, l), sum of W[i, j] where S[i, j] < T[k, l]
        # This is a 4D operation.
        # Let's reshape S and W to be compatible.
        # S is n x n. W is n x n.
        # We want M[k, l] = sum_{i, j} W[i, j] * (S[i, j] < T[k, l])
        # This can be done by broadcasting S and T.
        # S[None, None, :, :] has shape (1, 1, n, n)
        # T[:, :, None, None] has shape (n, n, 1, 1)
        # Mask = S[None, None, :, :] < T[:, :, None, None]  # shape (n, n, n, n)
        # This is 64^4 = 16.7M booleans, which is fine.
        
        mask = S[None, None, :, :] < T[:, :, None, None]
        # M[k, l] = sum_{i, j} W[i, j] * mask[k, l, i, j]
        M = np.einsum('ij,klij->kl', W, mask)
        
        # P = sum_{k, l} w_k w_l M[k, l]
        P = np.einsum('k,l,kl->', w, w, M)
        return P
    
    # Initial guess: uniform weights
    w0 = np.ones(n) / n
    
    # Objective function for minimization (we want to maximize P)
    def objective(w):
        # Ensure w is non-negative and sums to 1
        # L-BFGS-B handles bounds, but we need to handle the sum constraint.
        # We can use Lagrange multipliers or just project back.
        # A common trick is to optimize over n-1 variables and set the last one to 1 - sum(others).
        # But L-BFGS-B doesn't handle equality constraints well.
        # Alternatively, we can use a penalty method or just rely on the fact that the optimum
        # of a convex function (if it were convex) under linear constraints lies on the boundary.
        # But P is not convex.
        # Let's try a different approach: use `scipy.optimize.minimize` with method='SLSQP' which handles
        # equality constraints, or use a transformation.
        # Transformation: w_i = exp(v_i) / sum(exp(v_j)). This ensures w_i > 0 and sum(w_i) = 1.
        # Then we optimize over v.
        
        # Let's stick with L-BFGS-B and use a penalty for the sum constraint if it drifts.
        # Or better, use the softmax parameterization.
        # w_i = exp(v_i) / Z
        # This is smooth and ensures w_i > 0, sum(w_i) = 1.
        # The gradient will be well-defined.
        
        # However, the prompt specifically mentions "convex quadratic program" and "L-BFGS-B".
        # L-BFGS-B is for bound-constrained problems.
        # If we use the softmax parameterization, there are no bounds on v, so we can use BFGS or L-BFGS-B (with no bounds).
        
        # Let's implement the softmax version.
        # v -> w = softmax(v)
        # P(w) -> maximize
        # Objective for minimizer: -P(softmax(v))
        
        # But wait, the prompt says "parameterizing the discrete measure's probability mass function directly...".
        # This suggests w is the variable.
        # If we use w directly, we need to handle sum(w)=1 and w>=0.
        # L-BFGS-B can handle w>=0 with bounds.
        # For sum(w)=1, we can fix w_n = 1 - sum_{i=1}^{n-1} w_i.
        # Then we optimize over w_1...w_{n-1} with bounds 0 <= w_i <= 1.
        # This is a valid bound-constrained problem.
        
        # Let's try this approach.
        # Optimize over w[0:n-1]. w[n-1] = 1 - sum(w[0:n-1]).
        # Constraint: 0 <= w[n-1] <= 1 => 0 <= 1 - sum(w[0:n-1]) <= 1 => 0 <= sum(w[0:n-1]) <= 1.
        # Also 0 <= w[i] <= 1 for i < n-1.
        
        def objective_w(w_partial):
            # w_partial is length n-1
            w_full = np.empty(n)
            w_full[:n-1] = w_partial
            w_full[n-1] = 1.0 - np.sum(w_partial)
            
            # Check if w_full is valid (non-negative)
            if np.any(w_full < -1e-9):
                return 1e6  # Penalty
            
            # Normalize just in case (should be 1 already)
            w_norm = w_full / np.sum(w_full)
            
            return -compute_P(w_norm)
        
        # Initial guess
        w_partial0 = np.ones(n-1) / n
        
        # Bounds for w_partial
        # w_i >= 0
        # w_i <= 1 (since w_full sums to 1 and is non-negative)
        # Also, sum(w_partial) <= 1 => each w_i <= 1 is not sufficient, but L-BFGS-B doesn't handle sum constraints.
        # We need to ensure 1 - sum(w_partial) >= 0.
        # This is a linear constraint, not a bound.
        # L-BFGS-B does NOT support linear inequality constraints like sum(x) <= 1.
        # So this approach is flawed for L-BFGS-B.
        
        # Let's go back to the softmax parameterization.
        # Optimize over v (length n).
        # w = softmax(v)
        # No constraints on v.
        
        def objective_v(v):
            # Compute softmax
            v_max = np.max(v)
            exp_v = np.exp(v - v_max)
            w = exp_v / np.sum(exp_v)
            
            return -compute_P(w)
        
        # Initial guess for v
        # Uniform w corresponds to v = 0 (or any constant)
        v0 = np.zeros(n)
        
        # Use L-BFGS-B
        # Since there are no bounds, we can use BFGS or L-BFGS-B with no bounds.
        
        best_v = v0
        best_score = -objective_v(v0)
        
        # Perform optimization
        try:
            from scipy.optimize import minimize
            
            result = minimize(
                objective_v,
                v0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            if result.success or result.fun < -best_score:
                best_v = result.x
                best_score = -result.fun
                
        except Exception as e:
            print(f"Optimization failed: {e}")
            
        # Convert best_v back to weights
        v_max = np.max(best_v)
        exp_v = np.exp(best_v - v_max)
        weights = exp_v / np.sum(exp_v)
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            # Return best found so far
            pass
        
        # Validate
        if np.any(weights < -1e-9) or np.sum(weights) <= 1e-9:
            # Fallback to uniform
            weights = np.ones(n) / n
            support = np.linspace(0.0, 1.0, n, dtype=np.float64)
        
        return support, weights

# EVOLVE-BLOCK-END
