import numpy as np

N = 100

def construct():
    h = np.full(N, 0.5)
    target_sum = N / 2.0
    
    lambda_val = 5.0
    learning_rate = 0.1
    steps = 200
    
    for step in range(steps):
        g = 1.0 - h
        c = np.correlate(h, g, mode="full")
        
        # Softmax gradient
        c_max = np.max(c)
        c_shifted = c - c_max
        weights = np.exp(lambda_val * c_shifted)
        sum_weights = np.sum(weights)
        grad_c = weights / sum_weights
        
        grad_h = np.zeros(N)
        
        # Precompute valid ranges for lags?
        # Loop over k (lag)
        for k in range(2 * N - 1):
            w = grad_c[k]
            if w < 1e-12:
                continue
            lag = k - (N - 1)
            
            # Vectorized update for grad_h
            # Term 1: I(0 <= j+lag < N) * (1 - h[j+lag] - h[j]*I(lag==0))
            # Term 2: -I(0 <= j-lag < N) * h[j-lag]
            
            # We can create masks
            # j is 0..N-1
            
            # Term 1 mask: 0 <= j+lag < N
            # j >= -lag, j < N-lag
            start1 = max(0, -lag)
            end1 = min(N, N - lag)
            
            if start1 < end1:
                # Indices j in [start1, end1)
                # h[j+lag] is valid
                # If lag == 0, subtract h[j]
                if lag == 0:
                    grad_h[start1:end1] += w * (1.0 - h[start1:end1] - h[start1:end1])
                    # Wait, 1 - h[j] - h[j] = 1 - 2h[j]?
                    # Formula: (1 - h[j+lag]) - h[j] * I(lag=0)
                    # If lag=0: (1 - h[j]) - h[j] = 1 - 2h[j]
                    # Yes.
                else:
                    grad_h[start1:end1] += w * (1.0 - h[start1:end1 + lag])
            
            # Term 2 mask: 0 <= j-lag < N
            # j >= lag, j < N+lag
            start2 = max(0, lag)
            end2 = min(N, N + lag)
            
            if start2 < end2:
                # Indices j in [start2, end2)
                # i = j - lag is valid
                # Subtract h[j-lag]
                grad_h[start2:end2] -= w * h[start2:end2 - lag]
        
        # Update
        h_new = h - learning_rate * grad_h
        
        # Project
        h_new = np.clip(h_new, 0.0, 1.0)
        
        # Bisection for sum
        def proj_sum(lam):
            return np.sum(np.clip(h_new + lam, 0.0, 1.0))
        
        lo = -1.0
        hi = 1.0
        if proj_sum(lo) > target_sum:
            lam = lo
        elif proj_sum(hi) < target_sum:
            lam = hi
        else:
            for _ in range(40):
                mid = (lo + hi) / 2.0
                if proj_sum(mid) < target_sum:
                    lo = mid
                else:
                    hi = mid
            lam = (lo + hi) / 2.0
            
        h = np.clip(h_new + lam, 0.0, 1.0)
        
        # Annealing
        lambda_val *= 1.1
        if lambda_val > 50.0:
            lambda_val = 50.0
        learning_rate *= 0.98

    return h
