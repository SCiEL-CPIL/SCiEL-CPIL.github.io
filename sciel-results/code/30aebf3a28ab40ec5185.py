import numpy as np

N = 600

def construct():
    # Parameters for the magnitude spectrum A
    # A has length N//2 + 1 (rfft size)
    n_fft = N // 2 + 1
    A = np.zeros(n_fft, dtype=np.float64)
    
    # Initialize A: Start with a broad peak in frequency to create a spread-out time signal?
    # Or start with DC to get a flat signal?
    # If A is DC only, u is constant, h is constant, v is delta.
    # Score: max(v) / (sum h)^2.
    # v[0] = sum h^2. sum h = N h.
    # Score = (N h^2) / (N^2 h^2) = 1/N.
    # If we spread energy in frequency, u becomes oscillatory.
    # h = softplus(u).
    # If u oscillates, some h are small, some large.
    # sum h might decrease?
    # v = irfft(A^2). If A is spread, A^2 is spread. v becomes smoother.
    # max(v) might decrease relative to sum(h)^2?
    
    # Let's initialize with a random positive spectrum
    A = np.abs(np.random.randn(n_fft)) * 0.1
    A[0] = 1.0  # Ensure some DC component
    
    # Adam optimizer
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m = np.zeros_like(A)
    v = np.zeros_like(A)
    t = 0
    
    best_A = A.copy()
    best_score = np.inf
    
    # Precompute basis vectors for gradients? 
    # rfft(delta_k) is just a complex exponential sequence.
    # We can compute it on the fly or precompute.
    # For k=0, rfft(delta_0) is all 1s.
    # For k>0, it's cos/sin.
    
    for it in range(2000):
        u = np.fft.irfft(A, n=N)
        h = np.logaddexp(0, u)
        
        # Compute v = irfft(A^2)
        # A^2 is real, so we can treat it as the magnitude of a zero-phase spectrum
        v = np.fft.irfft(A * A, n=N)
        
        Q = np.sum(h)
        if Q < 1e-10:
            continue
            
        M = np.max(v)
        score = M / (Q * Q)
        
        if score < best_score:
            best_score = score
            best_A = A.copy()
            
        # Gradient calculation
        # dQ/dA
        sigma_u = 1 / (1 + np.exp(-np.clip(u, -50, 50)))
        grad_Q = np.fft.rfft(sigma_u, n=N).real  # Real part because A is real
        
        # dM/dA
        k_star = np.argmax(v)
        
        # Vector w such that w_m = exp(i * 2*pi * k_star * m / N)
        # This is the m-th component of the rfft of a delta at k_star
        # rfft(delta_k) corresponds to the DFT of the delta
        # DFT of delta at k is exp(i * 2*pi * m * k / N)
        
        # We need Re(2 * A * w)
        # w is complex.
        
        # Efficient computation:
        # w_m = cos(2*pi*m*k/N) + i*sin(2*pi*m*k/N)
        # A is real.
        # A * w is complex.
        # We need the real part: A * cos(2*pi*m*k/N)
        
        # So grad_M = 2 * A * cos(2*pi*m*k_star/N)
        
        m_idx = np.arange(n_fft)
        angle = 2 * np.pi * m_idx * k_star / N
        cos_term = np.cos(angle)
        
        grad_M = 2 * A * cos_term
        
        # Total gradient
        grad_score = (Q * grad_M - M * 2 * grad_Q) / (Q * Q)
        
        # Adam update
        t += 1
        m = beta1 * m + (1 - beta1) * grad_score
        v = beta2 * v + (1 - beta2) * (grad_score * grad_score)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        A -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Enforce non-negativity of A
        A = np.maximum(A, 0)
        
    # Reconstruct h from best_A
    u = np.fft.irfft(best_A, n=N)
    h = np.logaddexp(0, u)
    
    return list(h)
