import numpy as np

N = 600

def construct():
    # Parameterize h_i using a low-order cosine series to ensure smoothness
    # h_i = A_0 + sum_{k=1}^M A_k cos(2*pi*k*i/N)
    # We use M=50 to allow for more flexibility while keeping the system small enough for optimization.
    M = 50
    i = np.arange(N)
    
    # Initial guess: Uniform
    A = np.zeros(M + 1)
    A[0] = 1.0 / N
    
    def coeffs_to_h(A):
        h = np.zeros(N)
        for k in range(M + 1):
            if k == 0:
                h += A[0]
            else:
                h += A[k] * np.cos(2 * np.pi * k * i / N)
        return h

    def compute_convolution(h):
        # Linear convolution
        return np.convolve(h, h, mode='full')

    # Remez-like Iterative Minimax
    # We iteratively identify the peaks and adjust coefficients to lower them.
    # Since the problem is non-linear, we use a sub-gradient step on the max function.
    
    h = coeffs_to_h(A)
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s > 0:
        h = h / s
    else:
        h = np.ones(N) / N
        
    g = compute_convolution(h)
    max_g = np.max(g)
    
    # Iterative adjustment
    # We use a projected gradient descent on the coefficients A
    # Objective: max(g(A))
    # Gradient of max(g) w.r.t A is the gradient of the active components.
    
    for _ in range(200):
        # Compute h and g
        h_raw = coeffs_to_h(A)
        # Enforce non-negativity by projecting? 
        # Better to penalize in the objective or handle in gradient.
        # For simplicity, we use a barrier or just clip and renormalize, 
        # but this breaks differentiability. 
        # Instead, we optimize A such that h > 0 is satisfied.
        
        # Let's use a simple subgradient step on A
        # grad_A = sum_{k in active} (d g_k / d A)
        
        # Calculate gradient of g_k w.r.t A
        # g_k = sum_j h_j h_{k-j}
        # d g_k / d A_m = sum_j (d h_j / d A_m) h_{k-j} + h_j (d h_{k-j} / d A_m)
        # d h_j / d A_0 = 1
        # d h_j / d A_m (m>0) = cos(2*pi*m*j/N)
        
        active_indices = np.where(g == max_g)[0]
        # To avoid too many active points, take the top few
        if len(active_indices) > 10:
            top_vals = np.sort(g)[-10:]
            threshold = top_vals[0]
            active_indices = np.where(g >= threshold)[0]
            
        grad_A = np.zeros(M + 1)
        
        # Pre-compute cosines for gradient
        # grad_h[j, m] = d h_j / d A_m
        # This is N x (M+1). 600 x 51. Manageable.
        
        # We need h values at shifted indices.
        # g_k = sum_j h_j h_{k-j}
        # Note: h is defined on 0..N-1. 
        # In linear convolution, indices are 0..2N-2.
        # h_j is 0 if j < 0 or j >= N.
        
        for k in active_indices:
            # g_k = sum_{j=0}^{N-1} h_j h_{k-j}
            # d g_k / d A_m = sum_j (dh_j/dA_m) h_{k-j} + h_j (dh_{k-j}/dA_m)
            
            # Vectorized computation might be faster, but N=600 is small.
            # We iterate j
            for j in range(N):
                if h[j] < 1e-12: continue # Optimization
                
                # Term 1: (dh_j/dA_m) * h_{k-j}
                # h_{k-j} is non-zero only if 0 <= k-j < N
                idx2 = k - j
                if 0 <= idx2 < N:
                    val2 = h[idx2]
                    # Add to grad_A[m] += dh_j/dA_m * val2
                    grad_A[0] += 1.0 * val2
                    for m in range(1, M + 1):
                        grad_A[m] += np.cos(2 * np.pi * m * j / N) * val2
                    
                # Term 2: h_j * (dh_{k-j}/dA_m)
                # dh_{k-j}/dA_m is non-zero only if 0 <= k-j < N
                if 0 <= idx2 < N:
                    val1 = h[j]
                    grad_A[0] += val1 * 1.0
                    for m in range(1, M + 1):
                        grad_A[m] += val1 * np.cos(2 * np.pi * m * idx2 / N)
        
        # Normalize gradient
        norm_grad = np.linalg.norm(grad_A)
        if norm_grad < 1e-12:
            break
            
        # Step size
        # We want to decrease max_g.
        # A_new = A - alpha * grad_A
        # Determine alpha by line search or fixed small step.
        # Since we are minimizing, we go opposite to gradient.
        alpha = 1e-5 / (norm_grad * max_g) # Heuristic
        
        A_new = A - alpha * grad_A
        
        # Check if h_new is non-negative
        h_new = coeffs_to_h(A_new)
        if np.min(h_new) < 0:
            # Reduce step size
            alpha = alpha / 2
            A_new = A - alpha * grad_A
            h_new = coeffs_to_h(A_new)
            if np.min(h_new) < 0:
                # Fallback: keep A
                continue
                
        A = A_new
        h = h_new
        # Normalize h to sum 1
        s = np.sum(h)
        if s <= 0:
            break
        h = h / s
        g = compute_convolution(h)
        new_max_g = np.max(g)
        
        if new_max_g < max_g:
            max_g = new_max_g
        else:
            # If it increased, reduce alpha and retry? 
            # For simplicity, just break or continue with smaller alpha.
            pass
            
    # Final cleanup
    h = coeffs_to_h(A)
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s > 0:
        h = h / s
    else:
        h = np.ones(N) / N
        
    return list(h)
