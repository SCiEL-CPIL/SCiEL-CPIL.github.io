"""Optimized signed step function via Gumbel-Softmax relaxation and efficient local search."""
import numpy as np
import time

N = 400
DT = 0.5 / N  # Interval width: 0.5 / 400


def _compute_autoconv(signed_heights):
    """Compute the autoconvolution via FFT and return as a 1D array of length N (lags 0..N-1)."""
    n = 1
    while n < 2 * N:
        n <<= 1
    F = np.fft.rfft(signed_heights, n)
    R = np.fft.irfft(F * np.conj(F), n)
    return R[:N] * DT


def _score_binary(signs):
    """
    Compute the score for a binary sequence.
    The task mentions 'upper bound on C3'. 
    We minimize the maximum absolute value of the sidelobes.
    """
    R = _compute_autoconv(signs)
    sidelobes = R[1:]
    return np.max(np.abs(sidelobes))


def _soft_max_loss_and_grad(R_sidelobes, tau=0.05):
    """
    Compute the soft-max loss and its gradient w.r.t R_sidelobes.
    L = tau * log(sum(exp(R^2 / tau)))
    dL/dR_k = (2 * R_k * exp(R_k^2 / tau)) / sum(exp(R_j^2 / tau))
    """
    R_sq = R_sidelobes ** 2
    m = np.max(R_sq)
    if m < 1e-8:
        return 0.0, np.zeros_like(R_sidelobes)
        
    exp_terms = np.exp((R_sq - m) / tau)
    sum_exp = np.sum(exp_terms)
    
    # Loss value (for logging/monitoring, not used for gradient directly in this form)
    loss = m + tau * np.log(sum_exp)
    
    # Gradient
    grad_R = (2.0 * R_sidelobes * exp_terms) / sum_exp
    
    return loss, grad_R


def _grad_s_from_grad_R(grad_R, s):
    """
    Compute gradient w.r.t s given gradient w.r.t R (sidelobes).
    g_i = sum_{k=1}^{N-1} grad_R[k-1] * (s[i-k] + s[i+k])
    This is 2 * correlation(grad_R, s) (linear, with zero padding).
    """
    n = 1
    while n < 2 * N:
        n <<= 1
        
    # Pad grad_R to length N? 
    # The operator is: for each i, sum over k where i-k and i+k are valid.
    # This is equivalent to circular correlation if we wrap around? No, it's linear.
    # However, autoconvolution via FFT with zero padding computes linear convolution.
    # The adjoint of linear autoconvolution (zero-padded) is linear correlation.
    
    # Let's construct the full gradient vector for R (length N, with R[0] ignored)
    full_grad_R = np.zeros(N)
    full_grad_R[1:] = grad_R
    
    # We can compute this via FFT:
    # corr = ifft(fft(w) * conj(fft(s)))
    W = np.fft.rfft(full_grad_R, n)
    S = np.fft.rfft(s, n)
    # Correlation is ifft(fft(w) * conj(fft(s)))
    G = np.fft.irfft(W * np.conj(S), n)
    
    # The result G[i] = sum_k w_k s_{i-k} (circular).
    # Since we zero-padded to n >= 2N, the circular correlation is equivalent to linear correlation
    # for indices i in [0, N-1] provided we handle the wrap-around correctly?
    # Actually, linear correlation of length N and N-1 has length 2N-2.
    # The 'same' mode output length is N.
    # The FFT method with n >= 2N computes circular correlation of the zero-padded signals.
    # This is equivalent to linear correlation for the valid range.
    
    # So grad_s = 2 * G[:N]
    return 2.0 * G[:N]


def _local_search_fast(signs, max_time=10.0):
    start_time = time.time()
    current_signs = signs.copy()
    R = _compute_autoconv(current_signs)
    current_score = np.max(np.abs(R[1:]))
    
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        best_gain = 0
        best_flip = -1
        best_new_R = None
        best_new_score = current_score
        
        for i in range(N):
            s_i = current_signs[i]
            
            # Calculate Delta R for flipping i
            dR = np.zeros(N)
            
            # Term 1: s[i-k] for k=1..i
            # dR[1:i+1] += -2 * s_i * s[i-1::-1] * DT
            if i > 0:
                dR[1:i+1] += -2.0 * s_i * current_signs[i-1::-1] * DT
                
            # Term 2: s[i+k] for k=1..N-1-i
            # dR[1:N-i] += -2 * s_i * s[i+1:] * DT
            if i < N - 1:
                dR[1:N-i] += -2.0 * s_i * current_signs[i+1:] * DT
                
            new_R = R + dR
            new_score = np.max(np.abs(new_R[1:]))
            
            if new_score < best_new_score - 1e-10:
                best_new_score = new_score
                best_new_R = new_R
                best_flip = i
                best_gain = current_score - new_score
                
        if best_flip != -1:
            current_signs[best_flip] *= -1.0
            R = best_new_R
            current_score = best_new_score
            improved = True
                
    return current_signs

def construct():
    start_time = time.time()
    timeout = 110.0
    
    np.random.seed(42)
    
    # Phase 1: Gradient Descent on Logits with Gumbel-Softmax
    logits = np.zeros(N)
    lr = 0.5
    temperature = 1.0
    tau = 0.05  # Soft-max temperature
    
    best_signs = np.ones(N)
    best_score = _score_binary(best_signs)
    
    # Adam optimizer state
    m = np.zeros(N)
    v = np.zeros(N)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    for step in range(500):
        if time.time() - start_time > timeout - 20:
            break
            
        # 1. Sample Gumbel noise
        u = np.random.rand(N)
        u = np.clip(u, 1e-8, 1 - 1e-8)
        g = -np.log(-np.log(u))
        
        # 2. Compute soft sigmoid
        z = (logits + g) / temperature
        y = 1.0 / (1.0 + np.exp(-z))
        
        # 3. Map to signs: s = 2y - 1
        s_soft = 2.0 * y - 1.0
        # Hard binarize for evaluation
        s_hard = np.sign(s_soft)
        s_hard[s_hard == 0] = 1.0
        
        # 4. Compute Autoconvolution R
        R = _compute_autoconv(s_hard)
        sidelobes = R[1:]
        
        # 5. Compute Soft-Max Loss Gradient w.r.t R
        loss, grad_R = _soft_max_loss_and_grad(sidelobes, tau)
        
        # 6. Compute Gradient w.r.t s_hard
        grad_s = _grad_s_from_grad_R(grad_R, s_hard)
        
        # 7. Backprop through Gumbel-Sigmoid (Straight-Through Estimator)
        # dL/dz = grad_s * 2 * y * (1-y)
        # dL/dlogits = dL/dz / temperature
        dy_dz = y * (1.0 - y)
        grad_logits = (grad_s * 2.0 * dy_dz) / temperature
        
        # 8. Adam Update
        t = step + 1
        m = beta1 * m + (1 - beta1) * grad_logits
        v = beta2 * v + (1 - beta2) * (grad_logits ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        logits -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Evaluate hard signs
        score = _score_binary(s_hard)
        
        if score < best_score:
            best_score = score
            best_signs = s_hard.copy()
            
        # Cool temperature
        temperature *= 0.98
        tau *= 0.99
        
    # Phase 2: Local Search
    current_signs = _local_search_fast(best_signs, max_time=timeout - (time.time() - start_time) - 5)
    current_score = _score_binary(current_signs)
    
    if current_score < best_score:
        best_score = current_score
        best_signs = current_signs.copy()
        
    return best_signs
