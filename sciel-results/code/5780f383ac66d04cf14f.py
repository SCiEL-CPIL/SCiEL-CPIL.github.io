import numpy as np
from scipy.optimize import minimize

N = 400

def construct():
    # Initial guess: uniform
    h0 = np.ones(N) / N
    
    # We want to minimize max(h*h) subject to sum(h)=1
    # h*h is computed via FFT
    
    def objective(h):
        # h is flattened
        h = h.reshape(N)
        # Compute autoconvolution
        # Use FFT for speed
        H = np.fft.fft(h, n=2*N)
        R = np.fft.ifft(H * H).real
        # R is circular convolution? No, ifft of H*H is circular conv of length 2N.
        # We need linear convolution.
        # Linear conv of length 2N-1.
        # FFT of length 2N is sufficient to avoid aliasing for linear conv of two length N vectors?
        # Length of linear conv is 2N-1. Next power of 2 is 512. 2N=800.
        # So FFT length 1024 or 2048 is safe.
        # Let's use 2048 to be safe.
        H = np.fft.fft(h, n=2048)
        R = np.fft.ifft(H * H).real
        # The linear convolution is in the first 2N-1 entries?
        # Actually, ifft gives circular conv.
        # If we zero pad to 2N-1, it's linear.
        # With n=2048, the linear conv is in R[0:2N-1].
        # The rest is zeros (or wrap-around if not padded enough, but 2048 > 799).
        return np.max(R[:2*N-1])
        
    # This is non-differentiable at the max.
    # We can use a smooth approximation or subgradient.
    # Or use scipy minimize with a smooth objective like sum(exp(alpha * R)) / alpha.
    
    alpha = 100.0
    
    def smooth_obj(h):
        h = h.reshape(N)
        H = np.fft.fft(h, n=2048)
        R = np.fft.ifft(H * H).real[:2*N-1]
        # Soft max
        return np.log(np.sum(np.exp(alpha * R))) / alpha
        
    # Constraints
    cons = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}
    ]
    
    # No bounds if we allow negative values.
    # If we want non-negative: bounds = [(0, None)] * N
    
    # Let's try without bounds first.
    
    res = minimize(smooth_obj, h0, method='SLSQP', constraints=cons, options={'maxiter': 100, 'ftol': 1e-9})
    
    h_opt = res.x.reshape(N)
    
    # Calculate the exact value
    H = np.fft.fft(h_opt, n=2048)
    R = np.fft.ifft(H * H).real[:2*N-1]
    max_R = np.max(R)
    S = np.sum(h_opt)
    val = abs(2 * N * max_R / (S * S))
    
    return h_opt

# EVOLVE-BLOCK-END
