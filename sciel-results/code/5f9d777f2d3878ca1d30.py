import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initialize h
    # Start with a random vector or a specific shape
    # A windowed function might be a good start
    h = np.random.randn(N)
    
    # Normalize
    sum_h = np.sum(h)
    if abs(sum_h) < 1e-10:
        h[0] += 1.0
        sum_h = np.sum(h)
    h /= sum_h
    
    best_score = 0.0
    best_h = h.copy()
    
    # Precompute indices for sidelobes
    # Full convolution length is 2N-1
    # Main lobe is at index N-1
    main_lag = N - 1
    
    # Gradient Descent
    # Objective: Minimize sum of squared sidelobes
    # h is normalized to sum=1
    
    step_size = 0.01
    momentum = 0.9
    v = np.zeros(N)
    
    # We can also try to minimize the peak directly, but L2 is smoother
    # Let's stick to L2 sidelobe energy
    
    while time.time() < deadline:
        # Compute Autocorrelation
        A = np.convolve(h, h)
        
        # Mask main lobe
        A_sidelobes = A.copy()
        A_sidelobes[main_lag] = 0.0
        
        # Objective: E = sum(A_sidelobes^2)
        # Gradient: dE/dh_j = sum_k 2 A_k * dA_k/dh_j
        # dA_k/dh_j = 2 * h_{k-j} (with appropriate shifting)
        
        # Efficient gradient calculation
        # grad[j] = sum_{k in S} 2 A_k * 2 h_{k-j}
        # This is a correlation of A_sidelobes with h, but weighted?
        # Let's just compute it directly for N=400, it's fast enough
        grad = np.zeros(N)
        
        # Vectorized approach:
        # For each k, contribution is 4 * A_k * shift(h, k)
        # We can use FFT or just loops. N=400, 2N-1 ~ 800 terms.
        # 800 * 400 = 320,000 ops per step. 
        # Python loops might be slow. Let's use vectorized numpy.
        
        # Alternative: 
        # grad = 4 * (A_sidelobes * h) ? No.
        # Let's use the property that autocorrelation gradient is related to correlation.
        # Actually, let's just use a simple finite difference check or a known fast method.
        
        # Faster method:
        # The gradient of the squared autocorrelation energy is:
        # grad = 4 * ifft(fft(A_sidelobes) * conj(fft(h, 2N)))
        # Wait, A = ifft(|H|^2).
        # Let's stick to direct calculation if it's fast enough, or use scipy.
        
        # Let's try a simpler heuristic update:
        # Minimize peak sidelobe directly?
        
        # Let's use the gradient of the L2 norm of sidelobes.
        # grad = 4 * np.convolve(A_sidelobes, h, mode='same')? 
        # Let's verify.
        # A_k = sum_i h_i h_{k-i}
        # d/dh_j (sum_k A_k^2) = sum_k 2 A_k (2 h_{k-j})
        # This is 4 * sum_k A_k h_{k-j}
        # This is 4 * (A * h)_{j} ? 
        # (A * h)_j = sum_k A_k h_{j-k}
        # We have sum_k A_k h_{k-j}.
        # Since h_{k-j} is a shift, this is a cross-correlation.
        # sum_k A_k h_{k-j} = (A * h)_{j} if A is symmetric?
        # A is symmetric (autocorrelation). h is not.
        # Let's just compute the gradient explicitly using vectorized shifts to be safe.
        
        # Actually, for N=400, we can afford a simple loop in numpy if vectorized per step.
        # But 800 iterations of vector ops of size 400 is fine.
        
        # Let's use the FFT method for gradient.
        # H = fft(h, 2*N)
        # A_full = ifft(|H|^2).real
        # A_sidelobes_full = A_full.copy()
        # A_sidelobes_full[main_lag] = 0
        # Grad in frequency domain?
        
        # Let's try a different approach: Coordinate Descent or simple GD with finite diff if needed?
        # No, analytic gradient is better.
        
        # Let's use the relation:
        # grad = 4 * np.convolve(A_sidelobes, h, mode='full')
        # We need to extract the center part of length N.
        
        conv_grad = np.convolve(A_sidelobes, h, mode='full')
        # The gradient for h[0..N-1] corresponds to indices [N-1 .. 2N-2] in the full convolution?
        # Let's check indices.
        # A_k is defined for k in [-(N-1), N-1] effectively, but np.convolve gives [0, 2N-2].
        # np.convolve(h, h) index k corresponds to lag k - (N-1).
        # So A_sidelobes is aligned with lags.
        # The gradient calculation sum_k A_k h_{k-j}
        # Let's just do it with a loop over k for clarity and correctness, N is small.
        
        # Optimization: Precompute shifts?
        # No, let's just use the FFT trick which is robust.
        
        # Gradient of E = sum_k A_k^2 wrt h
        # E = 1/2 sum_k A_k^2 (factor 1/2 cancels with 2 from derivative)
        # dE/dh = 2 * (A * h) ?
        
        # Let's use a verified simple update:
        # Update h based on negative gradient of the objective.
        
        # To avoid complex gradient derivations errors, let's use a numerical gradient check for the first few steps?
        # No, let's trust the formula: grad = 4 * (A_sidelobes * h) [correlation]
        
        # Let's implement the gradient as:
        # grad[j] = 4 * sum_{k} A_k h_{k-j}
        
        # Using numpy:
        # We can construct a matrix of shifts? Too slow.
        # Let's use FFT.
        # A_sidelobes is the autocorrelation.
        # The operation is cross-correlation of A_sidelobes and h.
        
        # Cross-correlation C[j] = sum_k A_k h_{k-j}
        # C = ifft(fft(A_sidelobes) * conj(fft(h)))
        
        # Let's try this.
        
        H = np.fft.fft(h, 2*N)
        A_fft = np.fft.fft(A_sidelobes, 2*N)
        C = np.fft.ifft(A_fft * np.conj(H)).real
        # C corresponds to the correlation.
        # We need to align it with h indices.
        # The correlation C[j] for j in 0..N-1.
        # The output of ifft is length 2N.
        # The valid part is likely the center.
        
        # Let's just use the direct sum for N=400, it's only 400*800 = 320k operations.
        # In Python, a loop of 400 iterations doing vector ops of size 400 is fast.
        
        grad = np.zeros(N)
        # A_sidelobes has length 2N-1
        # We need sum_k A_k h_{k-j}
        # Note: h_{k-j} is 0 if index out of [0, N-1]
        
        # Vectorized loop
        for j in range(N):
            # Indices k such that k-j is in [0, N-1]
            # k in [j, j+N-1]
            # A_sidelobes indices are 0 to 2N-2
            # So we sum A_sidelobes[j : j+N] * h[0 : N] ?
            # Wait.
            # Term is h_{k-j}. Let m = k-j. m in [0, N-1].
            # k = m + j.
            # Sum over m=0..N-1 of A_{m+j} h_m.
            # This is dot product of A_sidelobes[j : j+N] and h.
            # But A_sidelobes length is 2N-1.
            # If j+N > 2N-1, we need to pad.
            
            # Let's pad A_sidelobes to length 2N
            A_pad = np.zeros(2*N)
            A_pad[:2*N-1] = A_sidelobes
            
            # For each j, grad[j] = 4 * dot(A_pad[j : j+N], h)
            
        # This loop is N=400 iterations.
        # 400 dot products of length 400.
        # This is fast in NumPy.
        
        # Let's precompute A_pad once per iteration.
        A_pad = np.zeros(2*N)
        A_pad[:2*N-1] = A_sidelobes
        
        # Use matrix multiplication?
        # grad = 4 * (A_pad @ h) ? No, it's a sliding dot product.
        # This is a convolution!
        # grad = 4 * np.convolve(A_pad, h, mode='valid')?
        # np.convolve(a, v, 'valid') returns length len(a)-len(v)+1.
        # We want length N.
        # len(A_pad) = 2N. len(h) = N.
        # Result length = 2N - N + 1 = N+1.
        # We want N.
        # Let's check alignment.
        
        # Let's just use the loop, it's safe.
        
        for j in range(N):
            grad[j] = 4.0 * np.dot(A_pad[j:j+N], h)
            
        # Normalize gradient
        # Apply momentum
        v = momentum * v - step_size * grad
        h = h + v
        
        # Renormalize h
        sum_h = np.sum(h)
        if abs(sum_h) < 1e-10:
            h[0] += 1.0
            sum_h = np.sum(h)
        h /= sum_h
        
        # Evaluate score
        A = np.convolve(h, h)
        M = np.max(A)
        S = np.sum(h)
        V = abs(2 * N * M / (S**2))
        score = 2.0 / V
        
        if score > best_score:
            best_score = score
            best_h = h.copy()
            
        # Adaptive step size?
        # If score decreases, reduce step size?
        
    return best_h
