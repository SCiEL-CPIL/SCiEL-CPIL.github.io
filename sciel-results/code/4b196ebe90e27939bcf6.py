"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _c3_upper_bound(h_initial):
    """
    Optimizes the signed step function to minimize the upper bound on C3.
    Metric: max_k |conv(h, h)_k| / (sum_i |h_i|)^2
    Uses multi-start L-BFGS-B with tanh parameterization.
    """
    
    def _objective(x, T):
        h = np.tanh(x)
        conv = np.convolve(h, h, mode='full')
        eps = 1e-12
        abs_conv = np.sqrt(conv**2 + eps)
        
        # Stable soft-max
        max_abs = np.max(abs_conv)
        # If max_abs is very small, return a large penalty to avoid division by zero issues later
        if max_abs < 1e-9:
            return 1e6
            
        log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
        log_max = T * log_sum_exp
        
        sum_abs_h = np.sum(np.abs(h))
        if sum_abs_h < 1e-6:
            return 1e6
            
        value = log_max / (sum_abs_h**2)
        return value

    def _exact_value(h):
        conv = np.convolve(h, h, mode='full')
        sum_abs_h = np.sum(np.abs(h))
        if sum_abs_h < 1e-6:
            return float('inf')
        return np.max(np.abs(conv)) / (sum_abs_h**2)

    best_val = float('inf')
    best_h = None
    
    # Define several initializations in the tanh-space (x)
    # We want h = tanh(x) to cover different regions of [-1, 1]
    # 1. All positive (Seed-like)
    x0_1 = np.full(N, 5.0)
    # 2. All negative
    x0_2 = np.full(N, -5.0)
    # 3. Alternating signs (High frequency)
    x0_3 = np.array([5.0 if i % 2 == 0 else -5.0 for i in range(N)])
    # 4. Random noise (Broken symmetry)
    x0_4 = np.random.randn(N) * 2.0
    # 5. Sine wave (Mid frequency)
    t = np.linspace(0, 2*np.pi, N)
    x0_5 = 5.0 * np.sin(t)
    
    inits = [x0_1, x0_2, x0_3, x0_4, x0_5]
    
    # Temperature for soft-max
    # Lower T is more accurate to max, but steeper gradients.
    # Start with T=5.0, maybe refine if needed, but 5.0 is a good balance.
    T = 5.0
    
    for x0 in inits:
        # Add tiny noise to break exact symmetries
        x0 = x0 + np.random.randn(N) * 1e-4
        
        try:
            res = minimize(
                lambda x: _objective(x, T),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            h_opt = np.tanh(res.x)
            val = _exact_value(h_opt)
            
            if val < best_val:
                best_val = val
                best_h = h_opt
                
        except Exception:
            continue
            
    # Fallback to seed if all optimizations failed
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h


def construct():
    return _c3_upper_bound(np.full(N, 5.0))
# EVOLVE-BLOCK-END
