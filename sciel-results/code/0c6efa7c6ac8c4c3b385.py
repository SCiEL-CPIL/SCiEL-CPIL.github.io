import numpy as np
from scipy import signal
from scipy.optimize import minimize

N = 600

def compute_exact_score(x):
    """
    Computes the ratio of the maximum non-zero lag autocorrelation
    to the zero-lag autocorrelation.
    """
    # Ensure non-negative
    if np.any(x < 0):
        return np.inf
    
    # Autocorrelation
    # np.correlate(x, x, mode='full') returns length 2N-1
    # Index N-1 is lag 0
    c = np.correlate(x, x, mode='full')
    c0 = c[N-1]
    
    if c0 == 0:
        return np.inf
        
    # Non-zero lags are indices 0 to N-2 and N to 2N-2
    # We want the max of these
    # Note: c is symmetric, so c[k] == c[2N-2-k]
    # We can just take the first half (lags 0 to N-1)
    # Lag 0 is at N-1.
    # Lags 1 to N-1 are at N-2 down to 0?
    # Let's check indices.
    # np.correlate([1,2,3], [1,2,3], 'full')
    # [3, 8, 14, 8, 3]
    # N=3. N-1=2. c[2]=14 (lag 0).
    # Lags:
    # k=0: sum x_i x_i = 14
    # k=1: x_0 x_1 + x_1 x_2 = 2+6=8. Index 1? c[1]=8.
    # k=2: x_0 x_2 = 3. Index 0? c[0]=3.
    # So lags 1..N-1 are at indices N-2 .. 0.
    # And N .. 2N-2 are the negative lags.
    # Since x is real, c is symmetric.
    # Max non-zero lag is max(c[0:N-1])
    
    max_side = np.max(c[:N-1])
    
    return max_side / c0

def construct():
    # Generate candidate windows
    candidates = []
    
    # Standard windows
    candidates.append(signal.windows.hann(N))
    candidates.append(signal.windows.hamming(N))
    candidates.append(signal.windows.blackman(N))
    candidates.append(signal.windows.blackmanharris(N))
    
    # Kaiser windows with different beta values
    for beta in [4, 6, 8, 10, 12, 15]:
        candidates.append(signal.windows.kaiser(N, beta))
        
    # Find the best initial candidate
    best_w = None
    best_score = np.inf
    
    for w in candidates:
        # Ensure non-negative (windows are, but just in case of numerical issues)
        w = np.maximum(w, 0)
        s = compute_exact_score(w)
        if s < best_score:
            best_score = s
            best_w = w
            
    # Local optimization
    # Objective: minimize smooth approximation of max_side / c0
    # f(x) = (1/beta) log(sum(exp(beta * c_k))) / c0
    # This is differentiable.
    
    beta_smooth = 50.0
    
    def objective(x):
        c = np.correlate(x, x, mode='full')
        c0 = c[N-1]
        if c0 <= 1e-10:
            return 1e10
            
        # Non-zero lags
        sides = c[:N-1]
        
        # Smooth max
        # Subtract max for numerical stability
        max_val = np.max(sides)
        if max_val > 700:
            return 1e10
            
        log_sum_exp = max_val + np.log(np.sum(np.exp(sides - max_val)))
        
        # We want to minimize log_sum_exp / c0
        return log_sum_exp / c0

    # Bounds: x_i >= 0
    bounds = [(0, None)] * N
    
    # Start optimization from best_w
    # Use L-BFGS-B
    try:
        res = minimize(
            objective, 
            best_w, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 100, 'ftol': 1e-9}
        )
        
        # Check if the optimized result is better using the EXACT score
        if res.success:
            exact_score = compute_exact_score(res.x)
            if exact_score < best_score:
                best_w = res.x
                best_score = exact_score
    except Exception:
        # If optimization fails, stick with the best window
        pass

    return list(best_w)
