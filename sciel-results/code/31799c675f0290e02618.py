import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    # We work in the frequency domain.
    # Let H be the FFT of h.
    # The autoconvolution c is ifft(|H|^2) (up to scaling and indexing).
    # Actually, np.convolve(h, h) corresponds to the full linear convolution.
    # Let's use FFT to compute the autoconvolution efficiently.
    # Linear convolution length is 2N-1.
    # We can compute it via zero-padding FFT.
    
    def get_autoconv(h):
        # Zero pad to at least 2N-1 for linear convolution
        size = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h, n=size)
        C = np.fft.irfft(H * H, n=size)
        # The linear convolution is the first 2N-1 elements
        return C[:2*N-1]

    def soft_max(x, beta=100.0):
        # Smooth approximation of max(x)
        # Using log-sum-exp for stability
        m = np.max(x)
        if not np.isfinite(m):
            return m
        return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

    def objective_and_grad(h):
        # Objective: minimize soft_max(autoconv(h))
        # We fix sum(h) = 1 to handle scale invariance.
        # Actually, the value is scale invariant. 
        # V = 2*N*max(c)/S^2.
        # If we fix S=1, we minimize max(c).
        
        c = get_autoconv(h)
        obj = soft_max(c, beta=100.0)
        
        # Gradient of soft_max(c) w.r.t h
        # c = conv(h, h)
        # d c_k / d h_i = h_{k-i} (with appropriate bounds)
        # This is effectively another convolution.
        # grad_c = conv(h, h) in a specific way?
        # Let's use the property that grad of sum c_k^2 is 4 * conv(conv(h, h), h).
        # But we have soft_max, which is non-linear.
        # We can compute the gradient numerically or analytically.
        # Analytical:
        # Let L = soft_max(c).
        # dL/dc_k = exp(beta*(c_k - m)) / sum(exp(...))
        # w_k = dL/dc_k
        # dL/dh_i = sum_k w_k * (dc_k/dh_i)
        # dc_k/dh_i = h_{k-i} if valid indices.
        # This is equivalent to conv(w, h) ?
        # Let's check.
        # c_k = sum_j h_j h_{k-j}
        # dc_k/dh_i = h_{k-i} + h_i h_{k-i}? No, derivative of product.
        # If c_k = sum_j h_j h_{k-j}, then d/dh_i c_k = h_{k-i} (term where j=i) + h_i (term where k-j=i => j=k-i).
        # So dc_k/dh_i = h_{k-i} + h_i * delta(k-j-i)? No.
        # c_k = sum_j h_j h_{k-j}.
        # Term with h_i: j=i => h_i h_{k-i}.
        # Term with h_{k-i}: j=k-i => h_{k-i} h_i.
        # So dc_k/dh_i = 2 h_{k-i} (if i != k-i) or h_i^2 derivative is 2h_i.
        # So dc_k/dh_i = h_{k-i} + h_{k-i} = 2 h_{k-i} ?
        # Wait, if we view c as matrix M h where M depends on h? No, c is quadratic.
        # Gradient of scalar f(c(h)) is sum_k (df/dc_k) (dc_k/dh).
        # dc_k/dh = 2 * conv(h, h) ? No.
        # The gradient of the quadratic form h^T A h is (A + A^T)h.
        # Here c = conv(h, h). This is not a simple quadratic form in a fixed matrix sense for the vector h, but it is related to the power spectrum.
        # Actually, it's easier to just use numerical gradients for the soft-max part if analytical is tricky, 
        # or use the fact that grad of sum(c^2) is 4*conv(c, h).
        # For soft max, the weights w_k are localized around the peak.
        # We can approximate the gradient as 2 * conv(w, h) where w are the soft-max weights.
        
        m = np.max(c)
        # Weights for soft max
        exp_terms = np.exp(100.0 * (c - m))
        w = exp_terms / np.sum(exp_terms)
        
        # Gradient of objective w.r.t h
        # d/dh sum_k w_k c_k
        # = sum_k w_k (dc_k/dh)
        # dc_k/dh_i = h_{k-i} + h_{k-i} (from symmetry) ?
        # Let's verify with small example.
        # h=[a,b]. c=[a^2, 2ab, b^2].
        # d c_0 / da = 2a. w_0 * 2a.
        # d c_1 / da = 2b. w_1 * 2b.
        # d c_2 / da = 0.
        # Grad_a = 2 w_0 a + 2 w_1 b.
        # conv(w, h) for w=[w0, w1, w2], h=[a,b] (padded)
        # conv([w0, w1, w2], [a, b, 0]) = [w0 a, w0 b + w1 a, w1 b + w2 a, w2 b]
        # This doesn't look like 2a w0 + 2b w1.
        
        # Let's use a robust numerical gradient check for the first few steps or just use a simpler proxy:
        # Minimize L2 norm of c? No, we need L_inf.
        # Let's use the analytic gradient for the "peak" by treating it as a weighted sum where weights are concentrated on the max.
        # If we just minimize the sum of squares of c, the gradient is 4 * conv(c, h).
        # This is smooth and convex-ish.
        # Let's try minimizing the L2 norm of the autoconvolution first to get a flat baseline, 
        # then switch to soft-max.
        
        # Actually, there is a known trick: 
        # Minimizing the L-infinity norm of the autoconvolution is equivalent to designing a sequence with low PSLR.
        # We can use the "Chirp" or "Barker" like structures as initialization.
        
        # Let's stick to a gradient descent on the time domain using the analytic gradient of the soft-max.
        # We need the correct gradient.
        # c_k = sum_j h_j h_{k-j}
        # Let's compute grad_h = 2 * ifft(fft(w) * fft(h))?
        # No, c is linear convolution.
        # Let's just use the finite difference gradient for the soft-max objective, 
        # it's expensive (N evaluations per step) but N=400 is small.
        # 400 * 400 = 160k ops per step. 1000 steps = 160M ops. Feasible in 150s?
        # Python loop over 400 elements calling np.convolve is slow.
        # We must use analytic gradient.
        
        # Correct analytic gradient:
        # The term h_i appears in c_k for k=i+j.
        # d c_k / d h_i = h_{k-i} + h_{k-i} ? No.
        # c_k = sum_j h_j h_{k-j}.
        # If we vary h_i, we affect terms where j=i or k-j=i.
        # Term j=i: h_i h_{k-i}. Deriv: h_{k-i}.
        # Term k-j=i => j=k-i: h_{k-i} h_i. Deriv: h_{k-i}.
        # So d c_k / d h_i = 2 h_{k-i} (if i != k-i) or h_i^2 deriv is 2h_i.
        # So d c_k / d h_i = h_{k-i} + h_{k-i} is correct?
        # Wait, if i = k-i, i.e., k=2i, then the term is h_i^2. Deriv is 2h_i.
        # The formula 2 h_{k-i} gives 2 h_i. Correct.
        # If i != k-i, we have two distinct terms in the sum: h_i h_{k-i} and h_{k-i} h_i.
        # So derivative is h_{k-i} + h_{k-i} = 2 h_{k-i}.
        # So d c_k / d h_i = 2 h_{k-i} (with h_j = 0 if j out of bounds).
        
        # So grad_obj_i = sum_k w_k * 2 h_{k-i}
        # This is 2 * (w * h)_i where * is cross-correlation?
        # Let's check indices.
        # sum_k w_k h_{k-i}. Let j = k-i => k = j+i.
        # sum_j w_{j+i} h_j.
        # This is the cross-correlation of w and h at lag i.
        # np.correlate(w, h, mode='full')?
        
        # Let's implement this.
        # w is defined on indices 0..2N-2.
        # h is defined on 0..N-1.
        # We need the result for i=0..N-1.
        
        # Using FFT for speed:
        # Correlation can be done via FFT.
        
        return obj, w, c

    # Initialization: Random spectral mask
    # We want H to be "flat" in magnitude to make c flat.
    # If |H| is constant, then c is a Dirac delta (impulse) at 0?
    # No, ifft(|H|^2) where |H| is constant is a Delta function.
    # But we need sum(h) != 0.
    # If h is a random phase sequence with constant amplitude, |H| is not constant.
    # We want to shape |H| such that ifft(|H|^2) is flat.
    # This is the "inverse problem": given a flat target time-domain c, find h.
    # c_target = constant * delta? No, we want c to be as flat as possible, i.e., close to 0 except maybe at 0?
    # But c_0 = sum h_i^2.
    # We want max(c) to be small.
    # This implies we want the energy of h to be spread out in a way that the autoconvolution doesn't peak.
    
    # Strategy:
    # 1. Start with h having random phases (unit modulus).
    # 2. Normalize sum(h) to 1.
    # 3. Iterate:
    #    a. Compute c = conv(h, h).
    #    b. Compute weights w for soft-max of c.
    #    c. Compute gradient g = 2 * correlate(w, h).
    #    d. Update h = h - lr * g.
    #    e. Project h to have sum 1.
    #    f. Optional: Re-phase h to maintain unit modulus? (Non-convex constraint)
    #       If we don't enforce unit modulus, h can collapse.
    #       Let's allow amplitude to vary.
    
    rng = np.random.default_rng(42)
    
    # Initial h: random complex phases, then take real part? 
    # The problem allows real h.
    # Let's start with random real h.
    h = rng.normal(0, 1, N)
    s = np.sum(h)
    if abs(s) < 1e-6:
        h[0] += 1.0
        s = np.sum(h)
    h /= s
    
    best_h = h.copy()
    best_val = compute_value(best_h)
    
    # Gradient Descent
    lr = 0.01
    beta = 50.0 # Softness
    
    for step in range(10000):
        if time.time() > deadline:
            break
            
        c = get_autoconv(h)
        m = np.max(c)
        if not np.isfinite(m):
            break
            
        # Weights for soft max
        # To avoid overflow, subtract max
        exp_terms = np.exp(beta * (c - m))
        w = exp_terms / np.sum(exp_terms)
        
        # Gradient: 2 * correlate(w, h)
        # w has length 2N-1. h has length N.
        # We need the correlation for lags corresponding to h indices.
        # Using FFT:
        size = 1 << (2 * N - 1).bit_length()
        W = np.fft.rfft(w, n=size)
        H = np.fft.rfft(h, n=size)
        # Correlation is ifft(W * conj(H)) ?
        # Let's check: corr(x, y)[k] = sum x[n] y[n-k]
        # FFT of corr is X * conj(Y).
        # Here we want sum_k w_k h_{k-i}.
        # This is (w * h)_i? No, it's cross-correlation.
        # Let's just use np.correlate for simplicity, N=400 is small enough if we do it once per step.
        # np.correlate(w, h, mode='same')? 
        # w length 2N-1, h length N.
        # We want result length N.
        # The full correlation has length 2N-1 + N - 1 = 3N-2.
        # We need the part corresponding to i=0..N-1.
        
        # Let's use the analytic formula directly with FFT to be safe and fast.
        # grad_i = 2 * sum_k w_k h_{k-i}
        # This is the convolution of w with reversed h?
        # Let's stick to np.correlate.
        # np.correlate(w, h, 'full')
        # Index in full corr: 
        # c_full[k] = sum_n w[n] h[n-k] ? No, numpy correlate is:
        # c_k = sum_n x_n y_{n-k}
        # We want sum_k w_k h_{k-i}.
        # Let n=k. sum_n w_n h_{n-i}.
        # This is the correlation of w and h at lag i.
        # np.correlate(w, h, 'full')[i + N - 1] ?
        
        # Let's test with small N.
        # N=2. h=[a,b]. w=[w0, w1, w2].
        # i=0: sum_k w_k h_k. k=0: w0 a. k=1: w1 b. k=2: w2 * 0. Sum = w0 a + w1 b.
        # np.correlate([w0,w1,w2], [a,b], 'full')
        # Result length 4.
        # [w2*0? No, standard definition]
        # Numpy: 
        # c_k = sum_n x_n y_{n-k}
        # x=w, y=h.
        # c_0 = w_2 h_{-2} (0) ... 
        # It's easier to just compute it via matrix multiplication or explicit loop if N is small, 
        # but 10000 steps * 400 is 4M ops. Explicit loop in Python is slow.
        # Let's use FFT.
        
        # We want g_i = sum_k w_k h_{k-i}
        # This is the cross-correlation.
        # In frequency domain: G = IFFT( W * conj(H) ) ?
        # Let's verify.
        # FFT of corr(x, y) is X * conj(Y).
        # So g = IFFT( W * conj(H) ).
        
        G = np.fft.irfft(W * np.conj(H), n=size)
        # The correlation is centered. 
        # The "full" correlation length is 3N-2.
        # The FFT result gives circular correlation.
        # We need to extract the linear part.
        # The linear correlation of w (len Lw=2N-1) and h (len Lh=N) has length Lw+Lh-1 = 3N-2.
        # The indices i=0..N-1 correspond to the "middle" part of the correlation?
        # Actually, g_i = sum_k w_k h_{k-i}.
        # If we define w' = w and h' = h.
        # The term h_{k-i} is non-zero when 0 <= k-i < N => i <= k < i+N.
        # So for a fixed i, we sum w_k over k in [i, i+N-1].
        
        # Let's just use np.correlate. It's optimized in C.
        # Time per call: ~10us? 10000 calls = 100ms. Very fast.
        
        g_full = np.correlate(w, h, mode='full')
        # We need g_i for i=0..N-1.
        # The full correlation index k corresponds to lag k - (len(h)-1) ?
        # Numpy correlate:
        # c_k = sum_n x_n y_{n-k}
        # We want sum_k w_k h_{k-i}.
        # Let's map indices.
        # In numpy correlate(x, y, 'full'), the result c has length len(x)+len(y)-1.
        # c[m] = sum_n x[n] y[n-m] ?? No.
        # Documentation: c_k = sum_n x_n y_{n-k}
        # So c_k is the correlation at lag k.
        # We want sum_k w_k h_{k-i}.
        # This looks like the correlation of w and h, but with indices flipped?
        # Let's just compute it explicitly for the first step to verify, then use the vectorized version.
        
        # Actually, there is a simpler relation.
        # grad_i = 2 * (w * h)_i where * is convolution?
        # Let's check the derivative again.
        # c_k = sum_j h_j h_{k-j}
        # d c_k / d h_i = h_{k-i} + h_{k-i} = 2 h_{k-i}
        # grad_i = sum_k w_k 2 h_{k-i}
        # Let j = k-i => k = j+i.
        # grad_i = 2 sum_j w_{j+i} h_j
        # This is the cross-correlation of w and h at lag i.
        # np.correlate(w, h, 'full') gives c_k = sum_n w_n h_{n-k}.
        # We want sum_j w_{j+i} h_j.
        # Let n = j+i => j = n-i.
        # sum_n w_n h_{n-i}.
        # This is exactly c_i from np.correlate(w, h, 'full')?
        # Yes, if the lag is defined as k=i.
        # But we need to be careful with the centering.
        # np.correlate(w, h, 'full') returns an array of length len(w)+len(h)-1.
        # The index corresponding to lag 0 is at len(h)-1.
        # So lag i is at index (len(h)-1) + i.
        
        offset = N - 1
        g = 2.0 * g_full[offset : offset + N]
        
        # Update h
        h_new = h - lr * g
        
        # Normalize sum to 1
        s_new = np.sum(h_new)
        if abs(s_new) < 1e-12:
            h_new = h
        else:
            h_new /= s_new
            
        val_new = compute_value(h_new)
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            
        # Adaptive learning rate
        if val_new < best_val:
            lr *= 1.01
        else:
            lr *= 0.95
            
        h = h_new
        
    return best_h

# Run construct to ensure it works
# h = construct()
# print(compute_value(h))
