import numpy as np

N = 600

def _objective(h):
    """Compute the normalized maximum of the autoconvolution."""
    norm = np.linalg.norm(h)
    if norm < 1e-12:
        return 1e6
    h_norm = h / norm
    # Use FFT for faster convolution
    # Length of conv is 2*N - 1. We need at least 2*N for FFT to avoid circular convolution effects.
    fft_len = 2 * N
    f_h = np.fft.fft(h_norm, n=fft_len)
    ac = np.fft.ifft(f_h * f_h).real
    return np.max(ac)

def construct():
    np.random.seed(42)
    
    half_n = N // 2
    
    # Initial guess: Symmetric Gaussian
    x = np.linspace(-0.25, 0.25, N)
    sigma = 0.05
    h_init = np.exp(-x**2 / (2 * sigma**2))
    h_init = np.maximum(h_init, 0)
    
    current_h = h_init.copy()
    current_score = _objective(current_h)
    
    best_h = current_h.copy()
    best_score = current_score
    
    # Phase 1: Coarse Search
    T_start = 0.5
    T_end = 1e-4
    alpha = 0.995
    steps = 50000
    
    for i in range(steps):
        T = T_start * (alpha ** i)
        
        h_new = current_h.copy()
        
        # Perturb a random subset of the first half
        num_perturb = max(1, int(half_n * 0.1))
        indices = np.random.choice(half_n, num_perturb, replace=False)
        
        noise_scale = T
        noise = np.random.normal(0, noise_scale, num_perturb)
        
        h_new[indices] += noise
        
        # Enforce symmetry
        h_new[half_n:] = h_new[:half_n][::-1]
        
        h_new = np.clip(h_new, 0, None)
        
        new_score = _objective(h_new)
        
        if new_score < current_score:
            current_h = h_new
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        else:
            delta = new_score - current_score
            prob = np.exp(-delta / T) if delta > 0 else 1.0
            
            if np.random.random() < prob:
                current_h = h_new
                current_score = new_score
        
        if T < T_end:
            break
            
    # Phase 2: Fine Polish
    T_start = 1e-4
    T_end = 1e-7
    alpha = 0.99
    steps = 20000
    
    current_h = best_h.copy()
    current_score = best_score
    
    for i in range(steps):
        T = T_start * (alpha ** i)
        
        h_new = current_h.copy()
        
        # Smaller perturbations
        num_perturb = max(1, int(half_n * 0.02))
        indices = np.random.choice(half_n, num_perturb, replace=False)
        
        noise_scale = T
        noise = np.random.normal(0, noise_scale, num_perturb)
        
        h_new[indices] += noise
        
        # Enforce symmetry
        h_new[half_n:] = h_new[:half_n][::-1]
        
        h_new = np.clip(h_new, 0, None)
        
        new_score = _objective(h_new)
        
        if new_score < current_score:
            current_h = h_new
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        else:
            delta = new_score - current_score
            prob = np.exp(-delta / T) if delta > 0 else 1.0
            
            if np.random.random() < prob:
                current_h = h_new
                current_score = new_score
        
        if T < T_end:
            break
            
    return list(best_h)
