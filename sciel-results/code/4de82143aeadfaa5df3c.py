"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_chebyt

def construct():
    """
    Find step heights h of length N=400 that minimize:
        value = abs(2 * N * max(convolve(h,h)) / sum(h)^2)
    
    Strategy:
    1. Parametrize h as a linear combination of Chebyshev polynomials (low-dim).
    2. Use differential evolution for global search over coefficients.
    3. Polish with L-BFGS-B using a soft-max surrogate.
    """
    deadline = time.time() + 145  # 145s safety margin (120s limit + buffer)
    
    # Number of Chebyshev basis functions to use
    n_basis = 12
    
    # Map Chebyshev polynomials to our domain [-1/4, 1/4]
    # x in [-1, 1] -> t = x/2 - 1/2 maps to [-1/4, 1/4]
    # Actually, let's use t directly in [-1/4, 1/4] and scale Chebyshev
    # Chebyshev T_k(x) for x in [-1,1]. We want x = 2*t where t in [-1/4,1/4] -> x in [-1/2, 1/2]
    # Let's just use x = 4*t, so t in [-1/4,1/4] -> x in [-1,1]
    t = np.linspace(-0.25, 0.25, N)
    x = 4 * t  # now x in [-1, 1]
    
    # Precompute Chebyshev basis
    basis = np.zeros((N, n_basis))
    for k in range(n_basis):
        basis[:, k] = eval_chebyt(k, x)
    
    # The DC component (k=0) is constant 1, so sum(h) = N * c_0 + ...
    # We need sum(h) != 0. Let's fix c_0 = 1 to ensure non-zero sum (or at least make it likely)
    # Actually, better to let it vary but constrain. Or just set c_0=1 and optimize others.
    # If c_0=1, sum(h) = N + sum_{k>=1} c_k * sum(basis[:,k]). Since sum of Chebyshev over symmetric interval is ~0 for k>=1,
    # sum(h) will be close to N. This is safe.
    
    # So we optimize c_1, ..., c_{n_basis-1}, with c_0 fixed at 1.
    n_vars = n_basis - 1
    
    def coeffs_to_h(coeffs):
        """Convert coefficient vector (c_1,...,c_{n-1}) to h, with c_0=1."""
        c = np.zeros(n_basis)
        c[0] = 1.0
        c[1:] = coeffs
        h = basis @ c
        return h
    
    def value_from_h(h):
        """Compute the task value."""
        s = np.sum(h)
        if s == 0:
            return np.inf
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)  # signed max
        return abs(2 * N * max_conv / (s * s))
    
    def objective(coeffs):
        """Objective for optimization: value_from_h(coeffs_to_h(coeffs))."""
        h = coeffs_to_h(coeffs)
        return value_from_h(h)
    
    # Soft-max surrogate for smoother optimization
    def soft_max_surrogate(coeffs, beta=500):
        """
        Use log-sum-exp approximation to max(conv) for smoother gradients.
        max(z) ≈ (1/beta) * log(sum(exp(beta*z)))
        """
        h = coeffs_to_h(coeffs)
        s = np.sum(h)
        if s == 0:
            return np.inf
        conv = np.convolve(h, h, mode='full')
        # Soft max
        max_val = (1.0 / beta) * np.log(np.sum(np.exp(beta * conv)))
        return abs(2 * N * max_val / (s * s))
    
    # Step 1: Differential Evolution for global search
    # Bounds for coefficients: let's try [-5, 5] initially
    bounds = [(-5.0, 5.0)] * n_vars
    
    best_result = None
    best_value = np.inf
    
    # Run DE with limited time
    de_start = time.time()
    try:
        de_result = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            polish=False,  # We'll do our own polish
            init='sobol'
        )
        if de_result.success or de_result.fun < best_value:
            best_value = de_result.fun
            best_result = de_result.x
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 10:
        if best_result is not None:
            return coeffs_to_h(best_result)
        return np.ones(N)
    
    # Step 2: Polish with L-BFGS-B using soft-max surrogate
    if best_result is not None:
        # Try multiple beta values for soft-max
        for beta in [100, 500, 2000]:
            if time.time() > deadline - 5:
                break
            try:
                polish_result = minimize(
                    soft_max_surrogate,
                    best_result,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if polish_result.fun < best_value:
                    # Verify with hard max
                    hard_val = objective(polish_result.x)
                    if hard_val < best_value:
                        best_value = hard_val
                        best_result = polish_result.x
            except Exception:
                pass
    
    # Step 3: Also try a few random restarts if time permits
    if time.time() < deadline - 20:
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            # Random initial point
            x0 = np.random.uniform(-5, 5, n_vars)
            try:
                result = minimize(
                    soft_max_surrogate,
                    x0,
                    args=(500,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                hard_val = objective(result.x)
                if hard_val < best_value:
                    best_value = hard_val
                    best_result = result.x
            except Exception:
                pass
    
    if best_result is not None:
        return coeffs_to_h(best_result)
    return np.ones(N)

# EVOLVE-BLOCK-END
