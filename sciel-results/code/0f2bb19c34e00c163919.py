import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    
    def compute_ac(x):
        """Compute autoconvolution using FFT."""
        X = np.fft.fft(x, n=fft_size)
        ac = np.fft.ifft(X * np.conj(X))
        # ac is circular convolution, we need linear
        # The linear autoconvolution is embedded in the circular one
        # For real input, ac is real
        ac = np.real(ac)
        # The first N elements are the positive lags, last N-1 are negative
        # Actually, np.correlate(x, x, 'full') gives lags from -(N-1) to N-1
        # FFT gives circular correlation, need to unwrap
        ac_linear = np.concatenate([ac[:N], ac[fft_size-N+1:fft_size]])
        return ac_linear
    
    def compute_score(x):
        S = np.sum(x)
        if abs(S) < 1e-10:
            return np.inf
        ac = compute_ac(x)
        max_ac = np.max(np.abs(ac))
        return max_ac / (S * S)
    
    best_score = np.inf
    best_x = np.ones(N)
    
    # First, evaluate the baseline
    baseline_score = compute_score(np.ones(N))
    best_score = baseline_score
    best_x = np.ones(N).copy()
    
    np.random.seed(42)
    
    # Local search with random restarts
    for restart in range(100):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        x = np.random.randn(N)
        if abs(np.sum(x)) < 1e-6:
            x[0] += 1.0
        
        current_score = compute_score(x)
        
        # Local search
        for it in range(100):
            if time.time() > deadline - 5:
                break
            
            i = np.random.randint(N)
            improved = False
            
            for delta in [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0]:
                x_trial = x.copy()
                x_trial[i] += delta
                s = compute_score(x_trial)
                if s < current_score:
                    x = x_trial
                    current_score = s
                    improved = True
                    break
            
            if not improved:
                # Occasional large jump
                if np.random.random() < 0.1:
                    x[i] += np.random.randn()
            
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
        
        if time.time() > deadline - 15:
            break
    
    return best_x
