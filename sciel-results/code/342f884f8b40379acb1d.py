"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def linear_autoconv(h):
        """Compute linear autoconvolution using FFT."""
        n = len(h)
        size = 2 * n
        H = np.fft.rfft(h, n=size)
        conv = np.fft.irfft(H * H, n=size)[:2*n-1]
        return conv
    
    def score_from_h(h):
        """Compute normalized maximum of autoconvolution."""
        conv = linear_autoconv(h)
        s = np.sum(h**2)
        if s < 1e-12:
            return 1e10
        return np.max(conv) / s
    
    def soft_max(x, beta=10.0):
        """Compute smooth approximation to max using log-sum-exp."""
        x_max = np.max(x)
        return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta
    
    def soft_score_from_h(h, beta=10.0):
        """Compute smooth normalized maximum of autoconvolution."""
        conv = linear_autoconv(h)
        s = np.sum(h**2)
        if s < 1e-12:
            return 1e10, np.zeros(N)
        score = soft_max(conv, beta) / s
        # Gradient with respect to h
        # d(score)/d(h_i) = (1/s) * d(soft_max(conv))/d(h_i) - (2*h_i/s^2) * soft_max(conv)
        # d(soft_max(conv))/d(h_i) = (1/beta) * sum_j exp(beta*(conv_j - max(conv))) * d(conv_j)/d(h_i) / sum_k exp(beta*(conv_k - max(conv)))
        conv_max = np.max(conv)
        exp_terms = np.exp(beta * (conv - conv_max))
        exp_sum = np.sum(exp_terms)
        
        # d(conv_j)/d(h_i) = h_{j-i} + h_{j+i} (for valid indices)
        # This is complex, so use finite differences for gradient
        grad = np.zeros(N)
        eps = 1e-7
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            conv_plus = linear_autoconv(h_plus)
            soft_plus = soft_max(conv_plus, beta)
            s_plus = np.sum(h_plus**2)
            score_plus = soft_plus / s_plus
            
            h_minus = h.copy()
            h_minus[i] -= eps
            conv_minus = linear_autoconv(h_minus)
            soft_minus = soft_max(conv_minus, beta)
            s_minus = np.sum(h_minus**2)
            score_minus = soft_minus / s_minus
            
            grad[i] = (score_plus - score_minus) / (2 * eps)
        
        return score, grad
    
    def h_from_magnitudes(mags):
        """Reconstruct h from DFT magnitudes with zero phase, ensure nonnegativity."""
        H = np.zeros(N, dtype=complex)
        H[0] = mags[0]
        for k in range(1, N//2 + 1):
            if k < N:
                H[k] = mags[k]
                H[N-k] = mags[k]
        h = np.fft.ifft(H).real
        h = np.maximum(h, 0.0)
        return h
    
    def mags_from_h(h):
        """Extract DFT magnitudes from h."""
        H = np.fft.fft(h)
        mags = np.abs(H)
        return mags
    
    # Initialize with all ones
    h_init = np.ones(N)
    mags_init = mags_from_h(h_init)
    
    best_h = h_init.copy()
    best_score = score_from_h(best_h)
    
    # Try some simple initializations
    if time.time() < deadline - 10:
        freqs = np.fft.fftfreq(N)
        lowpass_mags = mags_init * np.exp(-10 * freqs**2)
        h_lp = h_from_magnitudes(lowpass_mags)
        s_lp = score_from_h(h_lp)
        if s_lp < best_score:
            best_score = s_lp
            best_h = h_lp.copy()
    
    if time.time() < deadline - 10:
        highpass_mags = mags_init * (1 - np.exp(-10 * freqs**2))
        h_hp = h_from_magnitudes(highpass_mags)
        s_hp = score_from_h(h_hp)
        if s_hp < best_score:
            best_score = s_hp
            best_h = h_hp.copy()
    
    # Gradient descent on h directly with soft-max surrogate
    if time.time() < deadline - 5:
        h_current = best_h.copy()
        lr = 0.01
        beta = 10.0
        prev_score = best_score
        
        for it in range(100):
            if time.time() > deadline - 2:
                break
            
            score, grad = soft_score_from_h(h_current, beta)
            
            if score < best_score:
                best_score = score
                best_h = h_current.copy()
            
            # Gradient step with line search
            h_new = h_current - lr * grad
            h_new = np.maximum(h_new, 0.0)
            
            new_score = score_from_h(h_new)
            if new_score < score:
                h_current = h_new
            else:
                lr *= 0.5
            
            if score > prev_score:
                lr *= 0.5
            prev_score = score
        
        # Increase beta for sharper max approximation
        if time.time() < deadline - 5:
            beta = 20.0
            h_current = best_h.copy()
            lr = 0.005
            prev_score = best_score
            
            for it in range(50):
                if time.time() > deadline - 2:
                    break
                
                score, grad = soft_score_from_h(h_current, beta)
                
                if score < best_score:
                    best_score = score
                    best_h = h_current.copy()
                
                h_new = h_current - lr * grad
                h_new = np.maximum(h_new, 0.0)
                
                new_score = score_from_h(h_new)
                if new_score < score:
                    h_current = h_new
                else:
                    lr *= 0.5
                
                if score > prev_score:
                    lr *= 0.5
                prev_score = score
    
    # Final local refinement in time domain
    if time.time() < deadline - 3:
        for smooth_iter in range(20):
            if time.time() > deadline - 2:
                break
            h_smooth = np.convolve(best_h, np.ones(5)/5, mode='same')
            h_smooth = np.maximum(h_smooth, 0.0)
            s_smooth = score_from_h(h_smooth)
            if s_smooth < best_score:
                best_score = s_smooth
                best_h = h_smooth.copy()
            else:
                break
    
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
