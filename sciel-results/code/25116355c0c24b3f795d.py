import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 24
    # Initial guess: Rudin-Shapiro
    # p, q = [1], [1]
    # while len(p) < N:
    #     p, q = p + q, p + [-c for c in q]
    # x_init = np.array(p[:N], dtype=float)
    
    # Better initial guess: Random or RS
    # Let's use RS as it's a good local minimum
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    x_init = np.array(p[:N], dtype=float)
    
    # Evaluation setup
    # Sample points on unit circle
    K = 4096
    theta = np.linspace(0, 2*np.pi, K, endpoint=False)
    z = np.exp(1j * theta)
    # M[j, k] = z[j]^k
    # M = z[:, None] ** np.arange(N)
    # More efficient:
    M = np.empty((K, N), dtype=complex)
    M[:, 0] = 1
    for k in range(1, N):
        M[:, k] = M[:, k-1] * z
    
    def eval_norm(x):
        """Compute max |P(z)| for coefficients x."""
        v = M @ x
        return np.max(np.abs(v))
    
    # Local Search Function
    def local_search(x):
        x = x.copy()
        current_norm = eval_norm(x)
        improved = True
        while improved:
            improved = False
            for k in range(N):
                # Try flipping bit k
                x_new = x.copy()
                x_new[k] *= -1
                new_norm = eval_norm(x_new)
                if new_norm < current_norm:
                    x = x_new
                    current_norm = new_norm
                    improved = True
        return x, current_norm

    best_x = x_init.copy()
    best_norm = eval_norm(best_x)
    
    # 1. Local search from RS
    best_x, best_norm = local_search(best_x)
    
    # 2. Continuous Relaxation + Polish
    # If we have time, try to optimize a smooth surrogate
    if time.time() < deadline - 20:
        try:
            from scipy.optimize import minimize
            
            # Smooth surrogate: log-sum-exp of |P(z)|^2
            # f(x) = (1/beta) * log( sum( exp(beta * |M @ x|^2) ) )
            # We want to minimize f(x).
            # beta controls the sharpness. Start with moderate beta.
            
            def smooth_objective(x):
                v = M @ x
                mag2 = np.abs(v)**2
                # To avoid overflow, subtract max
                max_mag2 = np.max(mag2)
                # If max_mag2 is very large, beta * max_mag2 might overflow
                # Scale down
                beta = 100.0
                scaled = beta * (mag2 - max_mag2)
                # log-sum-exp trick
                log_sum = max_mag2 + (1.0/beta) * np.log(np.sum(np.exp(scaled)))
                return log_sum
            
            def smooth_gradient(x):
                v = M @ x
                mag2 = np.abs(v)**2
                max_mag2 = np.max(mag2)
                beta = 100.0
                scaled = beta * (mag2 - max_mag2)
                weights = np.exp(scaled) / np.sum(np.exp(scaled))
                # Gradient of log_sum w.r.t x
                # d/dx (1/beta log sum exp(beta |M x|^2))
                # = 1/beta * sum ( exp(beta |v_j|^2) / sum exp(...) * d/dx (beta |v_j|^2) )
                # d/dx (beta |v_j|^2) = 2 beta Re( M_j^* v_j )
                # So grad = sum( weights_j * 2 Re( M_j^* v_j ) )
                # Note: M_j is row j. M_j^* is conjugate.
                # v_j = sum M_jk x_k
                # Re( conj(M_j) * v )
                # This is a matrix vector product
                # grad = 2 * (M.conj().T @ (weights * v)).real
                grad = 2.0 * (M.conj().T @ (weights * v)).real
                return grad

            # Multiple restarts with different beta or initial points
            for restart in range(5):
                if time.time() > deadline - 5:
                    break
                
                # Perturb best_x or use random
                if restart == 0:
                    x0 = best_x + 0.1 * np.random.randn(N)
                else:
                    x0 = np.random.choice([-1, 1], size=N).astype(float)
                
                # Clip to [-1, 1] for L-BFGS-B
                x0 = np.clip(x0, -1.0, 1.0)
                
                res = minimize(
                    smooth_objective, 
                    x0, 
                    jac=smooth_gradient, 
                    method='L-BFGS-B', 
                    bounds=[(-1.0, 1.0)] * N,
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                
                x_opt = res.x
                # Project to +/- 1
                x_proj = np.where(x_opt >= 0, 1.0, -1.0)
                
                # Polish with local search
                x_polished, norm_polished = local_search(x_proj)
                
                if norm_polished < best_norm:
                    best_norm = norm_polished
                    best_x = x_polished
                    
        except Exception as e:
            # Fallback to pure local search if scipy fails
            pass

    # 3. Final Random Restarts Local Search if time permits
    if time.time() < deadline - 10:
        for i in range(50):
            if time.time() > deadline - 2:
                break
            # Random start
            x_rand = np.random.choice([-1, 1], size=N).astype(float)
            x_local, norm_local = local_search(x_rand)
            if norm_local < best_norm:
                best_norm = norm_local
                best_x = x_local

    # Return integer coefficients
    return [int(x) for x in best_x]

if __name__ == "__main__":
    # Self test
    coeffs = construct()
    print(coeffs)
