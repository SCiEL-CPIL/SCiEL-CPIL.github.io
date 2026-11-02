import numpy as np

N = 400

def construct():
    # Initialize with random signs
    x = np.random.choice([-1, 1], size=N)
    
    # We want to minimize the peak sidelobe or L1 norm of sidelobes.
    # Let's minimize the sum of absolute values of the autoconvolution for lags 1..N-1.
    # This encourages a delta-like autoconvolution.
    
    num_iters = 500
    learning_rate = 0.01
    
    for _ in range(num_iters):
        # Compute autoconvolution
        # Efficient way: use FFT
        X = np.fft.fft(x, n=2*N)
        AC = np.fft.ifft(X * np.conj(X)).real[:N]
        
        # Mask out lag 0
        mask = np.ones(N, dtype=bool)
        mask[0] = False
        sidelobes = AC[mask]
        
        # Objective: L1 norm of sidelobes
        # Gradient of |AC[k]| is sign(AC[k]) * dAC[k]/dx
        # dAC[k]/dx_i = x_{i+k} + x_{i-k}
        
        # Compute gradient
        grad = np.zeros(N)
        # For each lag k, contribution to grad[i] is sign(AC[k]) * (x[i+k] + x[i-k])
        # This is a convolution-like operation.
        # grad = sum_{k=1}^{N-1} sign(AC[k]) * (x shifted by k + x shifted by -k)
        
        # Efficient computation using FFT?
        # Let S[k] = sign(AC[k]) for k>0, S[0]=0.
        # grad[i] = sum_{k=1}^{N-1} S[k] x[i+k] + sum_{k=1}^{N-1} S[k] x[i-k]
        # This is correlation of x with S.
        
        S = np.zeros(N)
        S[1:] = np.sign(AC[1:])
        
        # grad = corr(x, S) + corr(x, S_rev) ?
        # Let's just loop for simplicity if N is small, but N=400 is small enough for vectorized ops?
        # Actually, N=400, N^2 = 160,000 ops per iter. 500 iters = 80M ops. Might be slow in Python loop.
        # Use FFT for correlation.
        
        # corr(x, S) is ifft(fft(x) * conj(fft(S)))
        # We need sum of corr(x, S) and corr(x, S) reversed?
        # x[i+k] is shift right.
        
        # Let's use the property that grad is the gradient of the L1 norm of the autoconvolution sidelobes.
        # We can approximate the gradient of the max or L1.
        
        # Simpler: Gradient of sum |AC[k]|
        # d/dx_i sum_{k} |AC[k]| = sum_{k} sign(AC[k]) * (x[i+k] + x[i-k])
        
        # Compute this sum efficiently.
        # Term 1: sum_{k} sign(AC[k]) x[i+k]
        # This is a circular correlation of x with sign(AC).
        
        S_full = np.zeros(2*N)
        S_full[1:N] = np.sign(AC[1:N])
        S_full[N+1:2*N] = np.sign(AC[1:N])[::-1] # For the x[i-k] part?
        
        # Actually, let's just use a simple projection method or steepest descent with a few steps.
        # Given time constraints, I will use a simpler heuristic: 
        # Flip the sign of x_i if it reduces the objective.
        
        # Objective: sum |AC[k]| for k>0
        obj = np.sum(np.abs(AC[1:]))
        
        # Try flipping each bit? Too slow (400 * 500 * 400) = 80M. Maybe okay in C, but Python?
        # Let's try gradient descent on the L1 norm using FFT-based gradient.
        
        # grad[i] = sum_{k=1}^{N-1} sign(AC[k]) * (x[i+k] + x[i-k])
        # This is the convolution of x with a kernel K where K[k] = sign(AC[k]) + sign(AC[-k])
        # Since AC is real and symmetric, sign(AC[k]) = sign(AC[-k]).
        # So K[k] = 2 * sign(AC[k]) for k != 0, K[0]=0.
        
        K = np.zeros(N)
        K[1:] = 2.0 * np.sign(AC[1:])
        # We need circular correlation.
        # grad = ifft(fft(x) * conj(fft(K)))
        
        grad = np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(K))).real
        
        # Update
        x_new = x - learning_rate * grad
        
        # Project to signs? The problem says "signed step functions", usually implies +/- 1 or real values?
        # "sequence of signed step heights". Could be real.
        # But "signed autocorrelation" often implies binary or PAM-2.
        # The seed program uses `np.random.randn` (real).
        # Let's keep it real but normalize?
        # If we keep it real, we can just do gradient descent.
        
        x = x_new
        
    return x
