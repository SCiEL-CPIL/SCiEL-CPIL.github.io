import numpy as np
import time
import scipy.optimize as opt

N = 600
TIME_LIMIT = 145  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def construct():
    start_time = time.time()
    
    # We formulate this as a convex optimization problem.
    # Minimize t
    # Subject to:
    #   sum(f) = 1  (normalization, since the score is scale-invariant)
    #   f >= 0
    #   conv(f, f)[k] <= t for all k
    
    # Variables: f (length N), t (scalar)
    # Total variables: N + 1
    
    # Objective: t
    c = np.zeros(N + 1)
    c[-1] = 1.0
    
    # Constraints:
    # 1. Equality: sum(f) = 1
    #    A_eq * x = b_eq
    A_eq = np.zeros((1, N + 1))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])
    
    # 2. Inequalities: conv(f, f)[k] <= t
    #    This is quadratic. scipy.optimize.linprog cannot handle quadratic constraints.
    #    However, the problem is convex. We can use scipy.optimize.minimize with SLSQP or trust-constr.
    #    SLSQP can handle quadratic constraints if we provide the Hessian or if it's simple enough?
    #    Actually, SLSQP in scipy supports inequality constraints g(x) <= 0.
    #    g_k(x) = conv_k(f) - t <= 0.
    
    # Let's try SLSQP. It might be slow for N=600 with 1200 constraints.
    # Alternatively, we can use a bisection on t.
    # For a fixed t, check if there exists f >= 0, sum(f)=1 such that conv(f,f) <= t.
    # This is a feasibility problem.
    # Since conv(f,f) is convex, the set {f | conv(f,f) <= t} is convex.
    # Intersection with f>=0, sum(f)=1 is convex.
    # We can solve this as a convex feasibility problem.
    # Or, simpler: use `scipy.optimize.minimize` to minimize sum(f^2) subject to constraints? No.
    
    # Let's stick to the direct minimization with SLSQP first, but it might be too slow.
    # A better approach: Bisection on t.
    # Lower bound for t: 0 (theoretically, but practically > 0)
    # Upper bound for t: compute_score(np.ones(N)/N)
    
    f_flat = np.ones(N) / N
    t_high = compute_score(f_flat)
    t_low = 0.0
    
    # We will bisect on t.
    # For a given t, we want to find if there is f such that:
    # 1. sum(f) = 1
    # 2. f >= 0
    # 3. conv(f, f) <= t
    
    # This can be formulated as a quadratic program (QP) or solved via interior point methods.
    # scipy.optimize.minimize can solve this.
    # Minimize 0 subject to constraints.
    # Or minimize ||f - f0||^2 for some f0 to help convergence?
    
    # Let's use `scipy.optimize.minimize` with method 'SLSQP' for the feasibility check.
    # Actually, a more robust way for feasibility of convex constraints is to minimize a quadratic penalty or just use a solver.
    # Let's try to minimize the violation.
    
    # A simpler and potentially faster approach for the feasibility check:
    # The condition conv(f, f) <= t is equivalent to ||f||_2^2 <= t (for the 0-th lag) and other lags.
    # Wait, conv(f,f)[0] = sum(f^2). So sum(f^2) <= t is a necessary condition.
    # If t < 1/N, no solution exists because by Cauchy-Schwarz, (sum f)^2 <= N sum f^2 => 1 <= N sum f^2 => sum f^2 >= 1/N.
    # So t_low = 1/N is a good lower bound.
    
    t_low = 1.0 / N
    
    # We will perform bisection.
    # For a given t, solve:
    # Minimize 0
    # Subject to:
    #   sum(f) = 1
    #   f >= 0
    #   conv(f, f)[k] - t <= 0 for all k
    
    # This is a convex feasibility problem.
    # We can use `scipy.optimize.minimize` to minimize the maximum constraint violation?
    # Or just try to find a point.
    
    # Let's use a different strategy:
    # The optimal solution often has a specific structure.
    # But let's try the bisection with SLSQP feasibility.
    
    def is_feasible(t_target):
        # Minimize 0 subject to constraints
        # Initial guess: uniform
        x0 = np.ones(N) / N
        
        # Constraints for SLSQP
        # Inequality constraints: g(x) <= 0
        # We need to define the function and its gradient for speed.
        
        # The number of constraints is 2N-1.
        # This might be too many for SLSQP to be fast.
        
        # Alternative: Use a penalty method or just check the optimal value of a related QP.
        # Actually, let's use `scipy.optimize.linprog` if we relax the quadratic constraints? No.
        
        # Let's try a different approach:
        # Minimize t subject to constraints using `scipy.optimize.minimize` with `trust-constr` which supports quadratic constraints?
        # No, `trust-constr` supports linear and general nonlinear.
        
        # Let's go back to the direct optimization of t with SLSQP, but limit the iterations and use a good initial guess.
        # Or, better yet, use the fact that the problem is convex and use a specialized solver if available?
        # Only standard scipy is available.
        
        # Let's try to solve the original problem directly with SLSQP, but carefully.
        # Variables: f (N), t (1).
        # Objective: t
        # Constraints:
        #   sum(f) - 1 = 0
        #   -f_i <= 0
        #   conv_k(f) - t <= 0
        
        # This is 1 equality, N inequalities, 2N-1 inequalities.
        # Total ~ 3N constraints.
        # SLSQP might struggle.
        
        # Let's try a bisection approach where the feasibility check is done by minimizing the maximum constraint violation.
        # Minimize max(0, max_k(conv_k(f) - t), max_i(-f_i), sum(f)-1, 1-sum(f))
        # This is a convex problem.
        
        def obj_viol(x):
            f = x[:N]
            t_val = x[N]
            conv = np.convolve(f, f, mode='full')
            viol_conv = np.max(conv - t_val)
            viol_nonneg = np.max(-f)
            viol_sum = np.abs(np.sum(f) - 1.0)
            return np.max([0.0, viol_conv, viol_nonneg, viol_sum])
        
        # This objective is not differentiable, but we can use Nelder-Mead? No, too slow.
        # We can smooth it with LSE.
        
        beta = 100.0
        def obj_viol_smooth(x):
            f = x[:N]
            t_val = x[N]
            conv = np.convolve(f, f, mode='full')
            c1 = np.max(conv - t_val)
            c2 = np.max(-f)
            c3 = np.abs(np.sum(f) - 1.0)
            # LSE of [c1, c2, c3, 0]
            arr = np.array([c1, c2, c3, 0.0])
            m = np.max(arr)
            if m < 1e-15:
                return 0.0
            return m + np.log(np.sum(np.exp(beta * (arr - m)))) / beta
        
        def grad_viol_smooth(x):
            f = x[:N]
            t_val = x[N]
            conv = np.convolve(f, f, mode='full')
            c1 = np.max(conv - t_val)
            c2 = np.max(-f)
            c3 = np.abs(np.sum(f) - 1.0)
            arr = np.array([c1, c2, c3, 0.0])
            m = np.max(arr)
            if m < 1e-15:
                return np.zeros(N+1)
            
            exp_arr = np.exp(beta * (arr - m))
            sum_exp = np.sum(exp_arr)
            p = exp_arr / sum_exp
            
            # Gradient of LSE is sum p_i * grad(arr_i)
            grad = np.zeros(N+1)
            
            # grad c1: max(conv - t). Subgradient is one-hot at max index, or averaged.
            # For smooth max, it's the weighted average of subgradients.
            # Subgradient of conv_k - t w.r.t f is 2 * f shifted.
            # Subgradient of conv_k - t w.r.t t is -1.
            
            # This gradient computation is complex. Let's use numerical gradient or just use a different method.
            
            # Let's try a simpler approach:
            # Use `scipy.optimize.minimize` with method='SLSQP' on the original problem, but with a tighter bound and fewer iterations.
            # Or, use the fact that we just need a good upper bound.
            
            # Let's try a different strategy:
            # Randomized search + local refinement.
            # 1. Generate random f >= 0, sum=1.
            # 2. Compute score.
            # 3. Use `scipy.optimize.minimize` (L-BFGS-B) to minimize the smooth max (LSE) of conv/s^2.
            # This is what the seed program did, but maybe with better initialization and more careful tuning.
            
            # The directive says: "Make a STRUCTURAL change".
            # The seed program uses FISTA. L-BFGS-B is a different method (quasi-Newton).
            # Also, the seed program had a complex gradient. L-BFGS-B can use numerical gradients if we don't provide them, but that's slow.
            # We should provide the gradient.
            
            # Let's implement the LSE objective and gradient carefully.
            return None # Placeholder
            
        # Since the gradient is hard, let's use a different structural change:
        # **Grid Search on the Number of Non-Zero Elements (Sparsity)**
        # Hypothesis: The optimal solution might be sparse or have a specific support.
        # We can try to optimize over subsets of indices.
        # But 600 choose k is huge.
        
        # Let's try a **Multi-Start L-BFGS-B** approach with a robust smooth objective.
        
        return False # Placeholder for is_feasible
        
    # Let's implement the L-BFGS-B approach with multiple starts.
    
    def objective_and_grad(f, beta):
        f = np.maximum(f, 0.0)
        s = np.sum(f)
        if s < 1e-15:
            return 1e10, np.zeros_like(f)
        
        conv = np.convolve(f, f, mode='full')
        z = conv / (s * s)
        
        z_max = np.max(z)
        if z_max < 1e-15:
            return 0.0, np.zeros_like(f)
        
        exp_z = np.exp(beta * (z - z_max))
        sum_exp = np.sum(exp_z)
        obj = z_max + np.log(sum_exp) / beta
        
        p = exp_z / sum_exp
        
        # Gradient computation as in seed program
        c1 = np.sum(p * conv)
        
        S = np.zeros(N)
        for j in range(N):
            i_start = max(0, j)
            i_end = min(2*N-1, N+j)
            if i_end > i_start:
                S[j] = np.dot(p[i_start:i_end], f[i_start-j:i_end-j])
                
        grad = (2.0 / (s**3)) * (s * S - c1)
        
        return obj, grad
    
    best_f = None
    best_score = np.inf
    
    # Initializations
    inits = []
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Cosine
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f_cos = np.cos(np.pi * x / 0.25) ** 2
    f_cos = np.maximum(f_cos, 0.0)
    inits.append(f_cos / np.sum(f_cos))
    
    # 3. Random smooth
    rng = np.random.RandomState(42)
    for _ in range(5):
        f_rand = rng.rand(N)
        kernel = np.ones(20) / 20
        f_rand = np.convolve(f_rand, kernel, mode='same')
        f_rand = np.maximum(f_rand, 0.0)
        inits.append(f_rand / np.sum(f_rand))
        
    # 4. Sparse random
    for _ in range(5):
        f_sparse = np.zeros(N)
        indices = rng.choice(N, size=100, replace=False)
        f_sparse[indices] = rng.rand(100)
        f_sparse = np.maximum(f_sparse, 0.0)
        inits.append(f_sparse / np.sum(f_sparse))
        
    for f_init in inits:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        try:
            # Use L-BFGS-B
            res = opt.minimize(
                objective_and_grad,
                f_init,
                method='L-BFGS-B',
                jac=True,
                args=(50.0,),
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            f_opt = np.maximum(res.x, 0.0)
            score = compute_score(f_opt)
            
            if score < best_score:
                best_score = score
                best_f = f_opt.copy()
                
        except Exception as e:
            pass
            
    if best_f is None:
        best_f = np.ones(N) / N
        best_score = compute_score(best_f)
        
    # Final polish
    if time.time() - start_time < TIME_LIMIT - 5:
        try:
            res = opt.minimize(
                objective_and_grad,
                best_f,
                method='L-BFGS-B',
                jac=True,
                args=(100.0,),
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            f_opt = np.maximum(res.x, 0.0)
            score = compute_score(f_opt)
            if score < best_score:
                best_f = f_opt
                best_score = score
        except Exception:
            pass
            
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
