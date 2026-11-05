import numpy as np

N = 400

def construct():
    # We want to minimize max(c) where c = conv(h, h).
    # c is the IFFT of |H|^2.
    # Let P = |H|^2.
    # We want to choose P (non-negative, even) to minimize max(IFFT(P)).
    # Constraint: sum(h) = 1.
    # sum(h) = H[0] (if using DFT normalization appropriately).
    # Actually, sum(h) = (1/L) * sum(H) ? No.
    # H = fft(h). H[0] = sum(h).
    # So we need H[0] = 1 (if we normalize h to sum 1).
    # But H is complex. P = |H|^2.
    # P[0] = |H[0]|^2 = 1.
    # So we fix P[0] = 1.
    # We want to minimize max(IFFT(P)).
    
    L = 2 * N
    # Initialize P to be flat (which gives delta c, peak 1/N)
    # Or peaked?
    # Let's start with a random P and optimize.
    
    # Gradient descent on P
    # Objective: max(IFFT(P)).
    # This is non-smooth. Use soft-max or just gradient of the max element.
    
    # Better: Minimize the sum of squares of the positive part?
    # Or just minimize the max by projecting.
    
    # Let's try to make c flat.
    # Target c_k = 1/(2N-1).
    # This requires P to be a delta? No.
    # If c is flat, P is delta.
    # P[0] = 1. P[k] = 0 for k != 0.
    # This gives c_k = 1/L * 1 * exp(...) = 1/L.
    # But c must be real and even.
    # If P is delta at 0, c is constant 1/L.
    # But c = conv(h,h) has support 2N-1.
    # The circular convolution would be constant.
    # The linear convolution is the circular convolution if L >= 2N-1.
    # So if we design h such that its circular autocorrelation is flat,
    # then its linear autocorrelation is also flat (if no wrap-around).
    
    # So the goal is: Find h such that circular autocorrelation is constant.
    # This implies |H|^2 is a delta at 0.
    # But |H|^2 >= 0 and sum |H|^2 is fixed?
    # No, we can scale h.
    # If |H|^2 is delta at 0, then H is non-zero only at 0.
    # Then h is a constant sequence.
    # If h is constant, c is triangle.
    # Contradiction.
    
    # Why?
    # Because the linear convolution of a constant is a triangle.
    # The circular convolution of a constant is a constant (if L=N).
    # But we use L=2N.
    # If h is constant of length N, padded to 2N, it is not constant in frequency.
    
    # Let's just use the gradient descent on the time domain h,
    # but with a better objective: minimize the max of c.
    # Use subgradient: grad = c at the index of the max.
    
    h = np.ones(N)
    h = h / np.sum(h)
    
    best_h = h.copy()
    best_val = np.inf
    
    L = 2 * N
    
    for i in range(1000):
        c = np.convolve(h, h)
        max_idx = np.argmax(c)
        max_val = c[max_idx]
        
        # Value
        val = 2.0 * N * max_val / (np.sum(h)**2)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Gradient of max(c) wrt h
        # d max(c) / dh = c' at max_idx?
        # c_k = sum_j h_j h_{k-j}
        # d c_k / dh_i = 2 h_{k-i}
        # So grad_i = 2 h_{max_idx - i}
        
        # We want to minimize max(c).
        # So we move h in direction -grad.
        
        grad = 2.0 * np.roll(h, max_idx) # h_{max_idx - i} is h rolled by max_idx?
        # h[j] where j = max_idx - i.
        # np.roll(h, max_idx)[i] = h[(i - max_idx) % N]?
        # We need h_{max_idx - i}.
        # If we define g[i] = h[max_idx - i], then g is h reversed and shifted.
        
        # Let's compute grad explicitly for safety.
        # c_k = sum_j h_j h_{k-j}
        # d c_k / dh_i = h_{k-i} + h_{k-i} = 2 h_{k-i}
        # We need h_{max_idx - i} for i in 0..N-1.
        
        # Indices:
        # i=0 -> h[max_idx]
        # i=1 -> h[max_idx - 1]
        # ...
        
        # If max_idx >= N, h[max_idx - i] might be out of bounds?
        # h is defined 0..N-1.
        # If max_idx - i < 0 or >= N, term is 0.
        
        grad = np.zeros(N)
        for i in range(N):
            j = max_idx - i
            if 0 <= j < N:
                grad[i] += 2.0 * h[j]
            # Also the other term?
            # d/dh_i (h_j h_{k-j})
            # If j=i, term h_{k-i}.
            # If k-j=i, term h_j = h_i? No, h_{k-i}.
            # So it's always 2 h_{k-i}.
            # Wait, if k-i = i, i.e., 2i=k, then term is h_i * h_i, deriv 2 h_i.
            # Formula 2 h_{k-i} gives 2 h_i. Correct.
            
        # Normalize grad
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
            
        # Step
        eta = 0.01
        h = h - eta * grad
        
        # Re-normalize sum to 1
        s = np.sum(h)
        if abs(s) < 1e-9:
            h[0] += 1e-9
            s = np.sum(h)
        h = h / s
        
    return best_h
