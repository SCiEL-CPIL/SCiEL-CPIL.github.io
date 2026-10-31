import numpy as np
from scipy.optimize import minimize
import time

N = 600
# Interval is [-0.25, 0.25], so dx = 0.5 / 600
dx = 0.5 / N
# Grid points for the function
x = np.linspace(-0.25, 0.25, N, endpoint=False) + dx/2

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initial guess: Box function
    h = np.ones(N)
    
    # Function to compute the metric (Peak Sidelobe Ratio)
    def metric(h_vec):
        # h_vec is the height array
        # Compute autoconvolution
        # Using FFT for speed
        n_fft = 2 * N
        H = np.fft.fft(h_vec, n=n_fft)
        R = np.fft.ifft(H * np.conj(H)).real
        # R is the circular autoconvolution. 
        # Since support is small, the linear autoconvolution is embedded in R.
        # The peak is at index 0.
        # We want to minimize the max value of R excluding the peak.
        # Note: R is periodic. The peak at 0 might wrap around.
        # We should look at the central part of R.
        # The linear autoconvolution of length N has support 2N-1.
        # In the FFT result of size 2N, the linear autoconv is at indices 0 to N-1 and N to 2N-1 (wrapped).
        # Specifically, R[0] is the peak.
        # R[1..N-1] and R[N..2N-1] contain the sidelobes.
        
        # To be safe, let's compute linear autoconvolution directly if N is small enough, 
        # but N=600, so FFT is better.
        # The linear autoconvolution c[k] = sum h[i] h[i+k].
        # We can get this from the FFT.
        
        # Let's extract the linear autoconvolution from the circular one.
        # The circular autoconvolution R_circ[k] = sum_j h[j] h[(j+k) % N].
        # This is not the same as linear.
        # However, if we zero-pad to 2N, the circular autoconvolution of the zero-padded signal 
        # IS the linear autoconvolution.
        
        h_padded = np.zeros(2 * N)
        h_padded[:N] = h_vec
        H_pad = np.fft.fft(h_padded)
        R_lin = np.fft.ifft(H_pad * np.conj(H_pad)).real
        
        # R_lin has length 2N.
        # The peak is at index 0 (and potentially wrapped at the end, but linear is centered? 
        # No, linear autoconv of length N is length 2N-1.
        # The peak is at index 0 in the "unwrapped" sense?
        # np.fft.ifft of |FFT|^2 gives the circular autoconv.
        # For zero-padded signal, the circular autoconv of length 2N is the linear autoconv 
        # IF the signal length is <= N. Here length is N, padded to 2N.
        # So R_lin[0] is the peak.
        # R_lin[1:N] and R_lin[N:2N-1] are the sidelobes?
        # Actually, the linear autoconvolution is symmetric.
        # R_lin[0] is max.
        
        peak = R_lin[0]
        if peak < 1e-9:
            return 1.0
        
        # Find max sidelobe
        # The sidelobes are R_lin[1:] and R_lin[:0:-1]?
        # R_lin is the sequence c[k] for k=0 to 2N-1.
        # c[0] is peak.
        # c[1] to c[N-1] are positive lags.
        # c[N] to c[2N-1] are negative lags (wrapped).
        # Due to symmetry, c[k] = c[-k].
        # So we just need to check c[1:N].
        
        sidelobes = R_lin[1:N]
        max_side = np.max(sidelobes)
        
        # We want to minimize max_side / peak
        return max_side / peak

    current_h = h.copy()
    best_score = metric(current_h)
    
    # Optimization loop
    # Use coordinate descent on the heights, or gradient descent.
    # Given the "cosine series" hint, we might optimize coefficients of a cosine expansion.
    # But let's stick to direct height optimization with constraints h >= 0.
    
    # Simple gradient descent with line search
    # Or use scipy minimize with bounds.
    
    # Since we have a time limit, we will do a few steps of optimization.
    
    # Let's try a simpler approach: perturb the box function with low-frequency cosines.
    # h(t) = 1 + sum a_k cos(2 pi k t)
    # We need h(t) >= 0.
    
    # Initialize coefficients to 0
    num_modes = 10
    coeffs = np.zeros(num_modes)
    
    def get_h_from_coeffs(c):
        t = x
        h_new = np.ones(N)
        for k in range(1, num_modes + 1):
            h_new += c[k-1] * np.cos(2 * np.pi * k * t)
        # Enforce non-negativity
        h_new = np.maximum(h_new, 0)
        return h_new
    
    def objective(c):
        h_new = get_h_from_coeffs(c)
        return metric(h_new)
    
    # Optimize coeffs
    # Use Nelder-Mead or similar
    try:
        res = minimize(objective, coeffs, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.success:
            best_h = get_h_from_coeffs(res.x)
            best_score = metric(best_h)
            current_h = best_h
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5.0:
        return current_h.tolist()
        
    # Refine with coordinate descent on heights
    # This might be slow. Let's try a few iterations.
    step_size = 0.1
    for _ in range(50):
        if time.time() > deadline - 2.0:
            break
        improved = False
        for i in range(N):
            old_val = current_h[i]
            # Try increasing
            current_h[i] = old_val + step_size
            score = metric(current_h)
            if score < best_score:
                best_score = score
                improved = True
            else:
                current_h[i] = old_val
            # Try decreasing
            if old_val > step_size:
                current_h[i] = old_val - step_size
                score = metric(current_h)
                if score < best_score:
                    best_score = score
                    improved = True
                else:
                    current_h[i] = old_val
        if not improved:
            step_size *= 0.5
            if step_size < 1e-3:
                break
                
    return current_h.tolist()
