import numpy as np
from scipy.signal import fftconvolve

N = 400
w = 1.0 / 200.0  # interval width

def construct():
    # 1. Define the grid
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    
    # 2. Construct a "Wavelet-like" basis matrix B (N x N)
    # We use a set of Gaussian functions with varying scales and positions
    # This approximates a discrete wavelet transform structure
    B = np.zeros((N, N))
    scales = [10, 5, 2, 1] # Widths in grid units
    positions_per_scale = N // len(scales)
    
    idx = 0
    for scale in scales:
        sigma = scale * w
        for p in range(positions_per_scale):
            # Center position
            center = -0.25 + (p + 0.5) * (0.5 / positions_per_scale)
            # Gaussian
            g = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            B[:, idx] = g
            idx += 1
            
    # Normalize basis vectors to unit L2 norm to stabilize optimization
    norms = np.linalg.norm(B, axis=0, keepdims=True)
    norms[norms == 0] = 1
    B = B / norms

    # 3. Initialize coefficients
    np.random.seed(42)
    c = np.random.randn(N) * 0.1
    
    # 4. Helper to compute score
    def get_score(coeffs):
        v = B @ coeffs
        s = np.sign(v)
        s[s == 0] = 1
        # Compute autoconvolution L-inf norm
        # Using FFT as in seed
        n_fft = 2 * N
        S = np.fft.rfft(s, n_fft)
        r = np.fft.irfft(S * np.conj(S), n_fft)[:N]
        return np.max(np.abs(r)) * w

    # 5. Simulated Annealing
    best_c = c.copy()
    best_score = get_score(c)
    current_score = best_score
    T = 1.0
    T_min = 1e-6
    cooling = 0.999
    
    for i in range(20000):
        # Perturb coefficients
        noise = np.random.randn(N) * 0.1
        c_new = c + noise
        
        score_new = get_score(c_new)
        
        delta = score_new - current_score
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            c = c_new
            current_score = score_new
            if current_score < best_score:
                best_score = current_score
                best_c = c.copy()
                
        T *= cooling
        if T < T_min:
            break

    # 6. Reconstruct best signal
    v_best = B @ best_c
    s_final = np.sign(v_best)
    s_final[s_final == 0] = 1
    
    return s_final
