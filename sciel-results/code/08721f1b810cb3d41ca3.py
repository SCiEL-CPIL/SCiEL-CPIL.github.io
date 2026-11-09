import numpy as np

N = 600

def construct():
    # Number of frequency bins for rfft
    n_fft = N // 2 + 1
    
    # Parameterization: Log-amplitudes and phases
    # We initialize with a DC-only spectrum to start at the uniform distribution
    # which is the global minimum for non-negative h.
    
    # Log amplitudes
    # H[0] = 1, H[k>0] = 0 implies log_A[0] = 0, log_A[k>0] = -inf (or very small)
    # Using a large negative number to approximate 0
    log_A = np.full(n_fft, -10.0)
    log_A[0] = 0.0
    
    # Phases
    # Fixed at 0 for simplicity, or random. 0 is symmetric.
    phases = np.zeros(n_fft)
    
    # Adam optimizer parameters
    lr = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    # State
    m_logA = np.zeros_like(log_A)
    v_logA = np.zeros_like(log_A)
    m_ph = np.zeros_like(phases)
    v_ph = np.zeros_like(phases)
    t = 0
    
    best_h = None
    best_score = np.inf
    
    # Number of iterations
    # Since we start at the optimum, few iterations are needed.
    # But we run some to ensure stability or if initialization isn't perfect.
    for it in range(500):
        # Reconstruct H
        A = np.exp(log_A)
        H = A * np.exp(1j * phases)
        H[0] = np.real(H[0]) # H[0] must be real
        
        # Time domain
        z = np.fft.irfft(H, n=N)
        h = np.logaddexp(0, z) # softplus
        
        # Autoconvolution
        # Use FFT for efficiency
        size = 2 * N
        H_h = np.fft.rfft(h, n=size)
        C = np.fft.irfft(H_h * H_h.conj(), n=size)
        c = C[:2*N-1]
        
        Q = np.sum(h)
        if Q < 1e-10:
            continue
            
        M = np.max(c)
        score = M / (Q * Q)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
        # Gradient calculation
        k_star = np.argmax(c)
        k_lag = k_star - (N - 1)
        
        grad_M = np.zeros(N)
        if k_lag >= 0:
            k = k_lag
            if k < N:
                grad_M[k:] += h[:N-k]
                grad_M[:N-k] += h[k:]
        else:
            k = -k_lag
            if k < N:
                grad_M[:N-k] += h[k:]
                grad_M[k:] += h[:N-k]
                
        dQ_dh = np.ones(N)
        grad_score_h = (Q * grad_M - M * 2 * dQ_dh) / (Q * Q)
        
        # Gradient w.r.t z
        sigmoid_z = 1 / (1 + np.exp(-np.clip(z, -50, 50)))
        grad_z = grad_score_h * sigmoid_z
        
        # Gradient w.r.t H
        grad_H = np.fft.rfft(grad_z, n=N)
        
        # Gradient w.r.t log_A and phases
        # H = exp(log_A + i*phases)
        # dH/dlog_A = H
        # dH/dphases = i*H
        
        # J is real.
        # grad_log_A = Re( grad_H * conj(dH/dlog_A) )? 
        # Actually, using Wirtinger derivatives or just chain rule for real parameters:
        # dJ/dlog_A = Re( dJ/dH * dH/dlog_A* ) + ...
        # Simpler: 
        # grad_log_A = 2 * Re( grad_H * H.conj() )? 
        # Let's verify.
        # If H is real, grad_H is real. H is real.
        # dJ/dA = grad_H. dA/dlog_A = A.
        # dJ/dlog_A = grad_H * A.
        # With complex:
        # grad_log_A = Re( grad_H * H.conj() ) * 2? 
        # Let's check: H = A e^{i phi}.
        # dH = dA e^{i phi} + i A e^{i phi} dphi.
        # dJ = Re( grad_H dH )? No, J is real.
        # dJ = 2 Re( grad_H dH )?
        # If we use the standard complex gradient trick:
        # grad_A = Re( grad_H * e^{-i phi} ) = Re( grad_H * H.conj() / A )
        # dJ/dlog_A = dJ/dA * A = Re( grad_H * H.conj() )
        
        grad_log_A = np.real(grad_H * H.conj())
        
        # grad_phases
        # dJ/dphi = Re( grad_H * i H )?
        # dH/dphi = i H.
        # dJ = Re( grad_H i H )?
        # grad_phases = np.real(grad_H * 1j * H)
        
        grad_phases = np.real(grad_H * 1j * H)
        
        # Update
        t += 1
        
        m_logA = beta1 * m_logA + (1 - beta1) * grad_log_A
        v_logA = beta2 * v_logA + (1 - beta2) * (grad_log_A * grad_log_A)
        m_logA_hat = m_logA / (1 - beta1**t)
        v_logA_hat = v_logA / (1 - beta2**t)
        
        m_ph = beta1 * m_ph + (1 - beta1) * grad_phases
        v_ph = beta2 * v_ph + (1 - beta2) * (grad_phases * grad_phases)
        m_ph_hat = m_ph / (1 - beta1**t)
        v_ph_hat = v_ph / (1 - beta2**t)
        
        log_A -= lr * m_logA_hat / (np.sqrt(v_logA_hat) + eps)
        phases -= lr * m_ph_hat / (np.sqrt(v_ph_hat) + eps)
        
        # Keep H[0] real constraint? 
        # H[0] = A[0] e^{i phi[0]}.
        # If we update phi[0], H[0] becomes complex.
        # irfft expects H[0] to be real.
        # So we should fix phi[0] = 0.
        phases[0] = 0.0
        # Also, if we optimize log_A[0], it's fine.

    if best_h is None:
        return [1.0] * N
    return list(best_h)
