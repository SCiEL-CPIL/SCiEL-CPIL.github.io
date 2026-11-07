"""Convex optimization approach for Erdos minimum overlap using trust-constr."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds


def construct():
    n = N
    
    # Deterministic initialization: start with 0.5 and add a small sinusoidal perturbation
    # to break symmetry and help the optimizer find a non-trivial local minimum.
    # The seed is 0.5, but we can start from a valid perturbation.
    i = np.arange(n)
    h_init = 0.5 + 0.1 * np.sin(2 * np.pi * i / n)
    # Project to sum n/2
    h_init = h_init - np.mean(h_init) + 0.5
    h_init = np.clip(h_init, 0.0, 1.0)
    # Re-adjust sum precisely
    diff = (n / 2.0 - np.sum(h_init)) / n
    h_init = np.clip(h_init + diff, 0.0, 1.0)
    
    # Calculate initial t
    lags_init = np.correlate(h_init, 1 - h_init, mode="full")
    t_init = np.max(lags_init) + 0.01
    
    x0 = np.concatenate([np.array([t_init]), h_init])
    
    # Bounds: t >= 0, 0 <= h_i <= 1
    bounds = Bounds(lb=[0.0] + [0.0] * n, ub=[np.inf] + [1.0] * n)
    
    # Linear constraint: sum(h) = n/2
    A_sum = np.zeros((1, n + 1))
    A_sum[0, 1:] = 1.0
    b_sum = np.array([n / 2.0])
    linear_constraint = LinearConstraint(A_sum, b_sum, b_sum)
    
    # Nonlinear constraints: t - C_k >= 0 for all lags k
    # There are 2n - 1 lags in 'full' mode.
    num_lags = 2 * n - 1
    
    def corr_func(x):
        t = x[0]
        h = x[1:]
        lags = np.correlate(h, 1 - h, mode="full")
        return t - lags
    
    def corr_jac(x):
        t = x[0]
        h = x[1:]
        jac = np.zeros((num_lags, n + 1))
        
        # Derivative of t - C_k w.r.t t is 1
        jac[:, 0] = 1.0
        
        # Derivative w.r.t h_j
        # C_k = sum_i h_i (1 - h_{i+k})
        # d C_k / d h_j = 1 - h_{j+k} - h_{j-k} (with bounds checks)
        # Index mapping: np.correlate(h, v, 'full') index l corresponds to lag l - (n-1)
        # Let lag = l - (n - 1).
        # C_lag = sum_i h_i (1 - h_{i+lag})
        # Valid i: 0 <= i < n and 0 <= i+lag < n.
        
        for l in range(num_lags):
            lag = l - (n - 1)
            # We need to compute grad for each j
            # d/dh_j [ sum_i h_i - h_i h_{i+lag} ]
            # = 1 (from sum h_i) 
            #   - (if j in valid i range) h_{j+lag} 
            #   - (if j+lag in valid i range, i.e., j = i+lag) h_{j-lag}
            
            # Let's compute this efficiently
            # Term 1: 1 for all j
            jac[l, 1:] = 1.0
            
            # Term 2: - h_{j+lag} if 0 <= j+lag < n
            if lag >= 0:
                # j+lag < n => j < n - lag
                # j >= 0
                if n - lag > 0:
                    jac[l, 1:(n - lag)] -= h[lag:n]
            else:
                # j+lag >= 0 => j >= -lag
                # j < n
                if -lag < n:
                    jac[l, (n - lag):] -= h[:n + lag] # h[0 : n+lag]
            
            # Term 3: - h_{j-lag} if 0 <= j-lag < n
            # Note: this term comes from - h_i h_{i+lag} where j = i+lag => i = j-lag.
            # So we subtract h_{j-lag} if j-lag is a valid index for i.
            # 0 <= j-lag < n => lag <= j < n + lag.
            
            if lag >= 0:
                # j >= lag, j < n
                if lag < n:
                    jac[l, (1 + lag):] -= h[:n - lag]
            else:
                # j < n + lag (which is < n since lag < 0)
                # j >= lag (always true for j>=0 if lag < 0? No, j-lag >= 0 => j >= -lag)
                # Wait, condition is 0 <= j-lag < n.
                # If lag < 0, let L = -lag > 0.
                # 0 <= j + L < n => -L <= j < n - L.
                # Since j >= 0, we have 0 <= j < n - L.
                # So j in [0, n-L-1].
                # Indices in jac are 1..n. So columns 1 .. (n-L).
                # h_{j-lag} = h_{j+L}.
                # For j in 0..n-L-1, h_{j+L} is h[L .. n-1].
                if -lag < n:
                    jac[l, 1:(n + lag)] -= h[-lag:] # h[-lag : n]
            
        return jac

    nonlinear_constraint = NonlinearConstraint(corr_func, 0.0, np.inf, jac=corr_jac)
    
    try:
        result = minimize(
            lambda x: x[0],
            x0,
            method='trust-constr',
            bounds=bounds,
            constraints=[linear_constraint, nonlinear_constraint],
            options={'maxiter': 500, 'gtol': 1e-9, 'xtol': 1e-9}
        )
        
        if result.success:
            h = result.x[1:]
            # Post-processing to ensure strict validity
            h = np.clip(h, 0.0, 1.0)
            current_sum = np.sum(h)
            if abs(current_sum - n / 2.0) > 1e-9 * n:
                diff = (n / 2.0 - current_sum) / n
                h = h + diff
                h = np.clip(h, 0.0, 1.0)
            
            # Final check
            if not np.all(np.isfinite(h)):
                return np.full(n, 0.5)
            if np.min(h) < -1e-9 or np.max(h) > 1 + 1e-9:
                return np.full(n, 0.5)
            if abs(np.sum(h) - n / 2.0) > 1e-9 * n:
                return np.full(n, 0.5)
                
            return h
        else:
            # Fallback: use the initialization if optimization fails
            h = h_init
            h = np.clip(h, 0.0, 1.0)
            current_sum = np.sum(h)
            diff = (n / 2.0 - current_sum) / n
            h = np.clip(h + diff, 0.0, 1.0)
            return h
    except Exception:
        return np.full(n, 0.5)
# EVOLVE-BLOCK-END
