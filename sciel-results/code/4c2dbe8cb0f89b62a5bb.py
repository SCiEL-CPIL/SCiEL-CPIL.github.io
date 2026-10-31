import numpy as np
import time

N = 600
L = 0.5  # interval length [-1/4, 1/4]

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution.
    
    For a step function with heights h_i on N equal intervals of total length L,
    the autoconvolution at lag 0 is sum(h_i^2 * dx) where dx = L/N.
    The normalization is by max(h) * sum(h_i) * dx.
    
    The ratio is (sum h_i^2) / (max(h) * sum h_i), which is scale invariant.
    """
    h = np.asarray(h)
    s1 = np.sum(h)
    s2 = np.sum(h**2)
    hmax = np.max(h)
    if s1 < 1e-15 or hmax < 1e-15:
        return np.inf
    return s2 / (hmax * s1)

def construct():
    """Construct a nonnegative step function with 600 intervals on [-1/4, 1/4]
    that minimizes the normalized maximum of its autoconvolution."""
    
    deadline = time.time() + 110.0  # safety margin before 120s limit
    best_h = np.ones(N)
    best_val = compute_autoconv_max(best_h)
    
    # Analytical single-spike solution
    # Minimize f(eps) = (1 + (N-1)*eps^2) / (1 + (N-1)*eps)
    # Optimal eps = (sqrt(N) - 1) / (N - 1)
    eps_opt = (np.sqrt(N) - 1) / (N - 1)
    h_spike = np.full(N, eps_opt)
    h_spike[0] = 1.0  # one spike at index 0
    val_spike = compute_autoconv_max(h_spike)
    if val_spike < best_val:
        best_val = val_spike
        best_h = h_spike.copy()
    
    # Try multiple spikes
    for k in [2, 3, 5, 10, 20, 50]:
        if k >= N:
            break
        # Optimize eps for k spikes of height 1, rest eps
        # ratio = (k + (N-k)*eps^2) / (k + (N-k)*eps)
        # Derivative zero: (N-k)*eps^2 + 2k*eps - k = 0
        # eps = (-2k + sqrt(4k^2 + 4k(N-k))) / (2(N-k))
        #     = (-k + sqrt(k^2 + k(N-k))) / (N-k)
        #     = (-k + sqrt(k*N)) / (N-k)
        #     = k*(-1 + sqrt(N/k)) / (N-k)
        #     = k*sqrt(N/k) * (-1/sqrt(k) + 1/sqrt(k)) ... let me redo
        # Actually: (N-k)*eps^2 + 2k*eps - k = 0
        # eps = [-2k + sqrt(4k^2 + 4k(N-k))] / (2(N-k))
        #     = [-k + sqrt(k^2 + k(N-k))] / (N-k)
        #     = [-k + sqrt(kN)] / (N-k)
        #     = k(-1 + sqrt(N/k)) / (N-k)
        # For k=1: (-1 + sqrt(N))/(N-1) which matches our earlier result.
        
        if N > k:
            eps_k = (-k + np.sqrt(k * N)) / (N - k)
            if eps_k > 0 and eps_k < 1:
                h_k = np.full(N, eps_k)
                for i in range(k):
                    h_k[i] = 1.0
                val_k = compute_autoconv_max(h_k)
                if val_k < best_val:
                    best_val = val_k
                    best_h = h_k.copy()
    
    # Local refinement using coordinate descent on the best structure
    # We'll try to optimize the heights directly, but with a smart initialization
    current_h = best_h.copy()
    current_val = best_val
    
    # Coordinate descent: optimize each height one at a time
    # For a fixed set of other heights, the optimal h_i is:
    # minimize (h_i^2 + S2_other) / (max(h_i, hmax_other) * (h_i + S1_other))
    # where S2_other = sum of squares of other heights, S1_other = sum of other heights
    # hmax_other = max of other heights
    
    # This is a 1D optimization for each coordinate.
    # Let's implement a simple coordinate descent.
    
    max_iter = 200
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        improved = False
        S1 = np.sum(current_h)
        S2 = np.sum(current_h**2)
        hmax = np.max(current_h)
        
        for i in range(N):
            if time.time() > deadline - 5:
                break
            
            h_i_old = current_h[i]
            S1_other = S1 - h_i_old
            S2_other = S2 - h_i_old**2
            hmax_other = np.max(np.delete(current_h, i)) if N > 1 else 0
            
            # We want to minimize f(h_i) = (h_i^2 + S2_other) / (max(h_i, hmax_other) * (h_i + S1_other))
            # for h_i >= 0
            
            # Case 1: h_i <= hmax_other
            # f(h_i) = (h_i^2 + S2_other) / (hmax_other * (h_i + S1_other))
            # f'(h_i) = [2*h_i * hmax_other * (h_i + S1_other) - (h_i^2 + S2_other) * hmax_other] / (hmax_other^2 * (h_i + S1_other)^2)
            # = [2*h_i*(h_i + S1_other) - (h_i^2 + S2_other)] / (hmax_other * (h_i + S1_other)^2)
            # = [2h_i^2 + 2h_i*S1_other - h_i^2 - S2_other] / (hmax_other * (h_i + S1_other)^2)
            # = [h_i^2 + 2h_i*S1_other - S2_other] / (hmax_other * (h_i + S1_other)^2)
            # Set to 0: h_i^2 + 2h_i*S1_other - S2_other = 0
            # h_i = [-2*S1_other + sqrt(4*S1_other^2 + 4*S2_other)] / 2
            #     = -S1_other + sqrt(S1_other^2 + S2_other)
            
            h_i_case1 = -S1_other + np.sqrt(S1_other**2 + S2_other)
            
            # Case 2: h_i > hmax_other
            # f(h_i) = (h_i^2 + S2_other) / (h_i * (h_i + S1_other))
            # f'(h_i) = [2*h_i * h_i*(h_i + S1_other) - (h_i^2 + S2_other) * (2*h_i + S1_other)] / [h_i^2 * (h_i + S1_other)^2]
            # This is more complex. Let's just use a simple scan.
            
            # Actually, let's just do a local search around the current value.
            # Try a few candidate values.
            candidates = [h_i_old, h_i_case1, 0.0, 2*h_i_old, h_i_old/2, h_i_old*1.5, h_i_old/1.5]
            candidates = [max(0.0, c) for c in candidates if c is not None and not np.isnan(c)]
            
            best_h_i = h_i_old
            best_f = np.inf
            
            for h_i_cand in candidates:
                new_h = current_h.copy()
                new_h[i] = h_i_cand
                f_val = compute_autoconv_max(new_h)
                if f_val < best_f:
                    best_f = f_val
                    best_h_i = h_i_cand
            
            if best_h_i != h_i_old:
                current_h[i] = best_h_i
                improved = True
                S1 = np.sum(current_h)
                S2 = np.sum(current_h**2)
                hmax = np.max(current_h)
        
        # Update best
        current_val = compute_autoconv_max(current_h)
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        if not improved:
            break
    
    # Final check: ensure nonnegativity
    best_h = np.maximum(best_h, 0.0)
    
    # Verify the value
    final_val = compute_autoconv_max(best_h)
    
    # Return as list of floats
    return list(best_h)
