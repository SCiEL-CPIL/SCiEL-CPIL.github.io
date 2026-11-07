import numpy as np
import time

N = 100
TARGET_SUM = N / 2.0

def compute_corr(h):
    # h is shape (N,)
    # We need C_k for k in -(N-1) to N-1
    # np.correlate(h, 1-h, mode='full') gives length 2N-1
    # The center is lag 0.
    corr = np.correlate(h, 1 - h, mode='full')
    return corr

def project_to_box_and_sum(h):
    # Project h to [0, 1] and sum = TARGET_SUM
    # First clip to [0, 1]
    h_clipped = np.clip(h, 0.0, 1.0)
    
    # Now adjust sum
    current_sum = np.sum(h_clipped)
    if np.isclose(current_sum, TARGET_SUM, atol=1e-9):
        return h_clipped
    
    # Binary search for lambda
    # We want sum(clip(h_i - lambda, 0, 1)) = TARGET_SUM
    # f(lambda) is monotonic decreasing
    
    lo, hi = -1.0, 2.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_trial = np.clip(h_clipped - mid, 0.0, 1.0)
        s = np.sum(h_trial)
        if s > TARGET_SUM:
            lo = mid
        else:
            hi = mid
            
    h_final = np.clip(h_clipped - (lo + hi) / 2.0, 0.0, 1.0)
    return h_final

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize
    # Start with a random balanced sequence
    np.random.seed(42)
    h = np.random.rand(N)
    h = project_to_box_and_sum(h)
    
    best_h = h.copy()
    best_val = np.max(compute_corr(best_h))
    
    beta = 10.0  # Soft max temperature
    step_size = 0.1
    
    while time.time() < deadline:
        C = compute_corr(h)
        max_C = np.max(C)
        
        if max_C < best_val:
            best_val = max_C
            best_h = h.copy()
            
        # Compute Soft Max Gradient
        # P_k = exp(beta * C_k) / sum(exp(beta * C_k))
        # To avoid overflow, subtract max
        C_max = np.max(C)
        exp_C = np.exp(beta * (C - C_max))
        sum_exp = np.sum(exp_C)
        P = exp_C / sum_exp
        
        # Gradient of C_k wrt h_i
        # C_k = sum_j h_j (1 - h_{j+k})
        # d C_k / d h_i = (1 - h_{i+k}) - h_{i-k}
        # We need sum_k P_k * ( (1 - h_{i+k}) - h_{i-k} )
        
        # Vectorized approach:
        # grad_i = sum_k P_k (1 - h_{i+k}) - sum_k P_k h_{i-k}
        # Note: P_k corresponds to lag k.
        # h_{i+k} is a shifted version of h.
        
        # Let's compute term1_i = sum_k P_k (1 - h_{i+k})
        # This is a convolution of P with (1-h)?
        # Careful with indices.
        # np.correlate(A, B) computes sum A_i B_{i+k}.
        # Here we want sum_k P_k (1 - h_{i+k}).
        # This is correlate(1-h, P) ?
        # Let's check.
        # correlate(h, P) at index i (shifted) is sum h_j P_{j-i}?
        # Standard numpy correlate: c_k = sum_n a_n b_{n+k}.
        # We want G_i = sum_k P_k (1 - h_{i+k}).
        # Let a = 1-h, b = P.
        # c_k = sum_n a_n b_{n+k}.
        # If we set n = i+k, then k = n-i.
        # G_i = sum_n (1-h_n) P_{n-i}.
        # This is correlate(1-h, P) at index corresponding to shift -i?
        # Numpy correlate(a, b, 'full') length is N+M-1.
        # Index i in 'full' corresponds to shift i - (M-1)?
        
        # Simpler: Use FFT for convolution if N is large, but N=100 is small.
        # Direct loop or vectorized shift sum.
        
        grad = np.zeros(N)
        # P is length 2N-1. Center is N-1.
        # P[k] corresponds to lag k - (N-1).
        # Let lag l = k - (N-1).
        # P_l = P[k].
        
        # term1: sum_l P_l (1 - h_{i+l})
        # term2: sum_l P_l h_{i-l}
        
        # We can compute this by summing over l.
        # Since N is small, we can just loop over l? 199 iterations * 100 ops = 20k ops. Very fast.
        
        for l_idx in range(len(P)):
            l = l_idx - (N - 1)
            p_val = P[l_idx]
            if p_val < 1e-6:
                continue
            
            # term1 contribution: p_val * (1 - h shifted by l)
            # h_{i+l}
            if l >= 0:
                # i+l < N => i < N-l
                # h[0:N-l] gets (1 - h[l:N])
                # h[N-l:N] is 0 (out of bounds, h is 0 outside? No, h is defined on [0, N-1])
                # In correlation, h_i is 0 for i<0 or i>=N.
                # So h_{i+l} is 0 if i+l >= N.
                # So for i in [0, N-l-1], h_{i+l} = h[l:N].
                # For i in [N-l, N-1], h_{i+l} = 0.
                # So (1 - h_{i+l}) is 1 for i >= N-l, and 1-h[l:N] for i < N-l.
                
                # Vectorized:
                # Create array of 1-h
                # Shift h by l
                h_shifted = np.zeros(N)
                if l < N:
                    h_shifted[:N-l] = h[l:]
                # else h_shifted is 0
                
                term1_part = p_val * (1.0 - h_shifted)
                
            else:
                # l < 0
                # h_{i+l} = h_{i-|l|}
                # Non-zero for i >= |l|
                h_shifted = np.zeros(N)
                if -l < N:
                    h_shifted[-l:] = h[:-l]
                
                term1_part = p_val * (1.0 - h_shifted)
            
            grad += term1_part
            
            # term2 contribution: - p_val * h_{i-l}
            # h_{i-l}
            if l >= 0:
                # h_{i-l} non-zero for i >= l
                h_shifted2 = np.zeros(N)
                if l < N:
                    h_shifted2[l:] = h[:N-l]
            else:
                # l < 0, i-l = i+|l|
                # non-zero for i+|l| < N => i < N-|l|
                h_shifted2 = np.zeros(N)
                if -l < N:
                    h_shifted2[:N-(-l)] = h[-l:]
            
            grad -= p_val * h_shifted2

        # Update
        h = h - step_size * grad
        h = project_to_box_and_sum(h)
        
        # Adaptive step size?
        # If max_C increased, reduce step?
        
    return best_h
