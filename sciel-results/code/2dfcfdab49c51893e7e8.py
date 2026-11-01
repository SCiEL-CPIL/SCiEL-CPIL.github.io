import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a signed step function of length N=400 to minimize the peak 
    autocorrelation sidelobe using efficient delta-based local search.
    """
    np.random.seed(42)
    
    # Initialize randomly
    f = np.random.choice([-1, 1], N)
    
    def compute_acf(f):
        """Compute full autocorrelation via FFT."""
        F = np.fft.fft(f, n=2*N)
        acf = np.fft.ifft(F * np.conj(F)).real
        return acf
    
    def get_sidelobe_array(acf):
        """Get array of sidelobe values (excluding lag 0)."""
        pos_lags = acf[1:N]
        neg_lags = acf[N+1:2*N]
        return np.concatenate([pos_lags, neg_lags])
    
    acf = compute_acf(f)
    sidelobes = get_sidelobe_array(acf)
    current_peak = np.max(np.abs(sidelobes))
    
    # Precompute neighbor_sum[i][k-1] = f[i+k] + f[i-k] for k=1..N-1
    neighbor_sum = np.zeros((N, N-1))
    for i in range(N):
        for k in range(1, N):
            s = 0.0
            if i + k < N:
                s += f[i + k]
            if i - k >= 0:
                s += f[i - k]
            neighbor_sum[i, k-1] = s
    
    # Local search
    for iteration in range(10000):
        # Try flipping each bit in a random subset
        num_candidates = min(200, N)
        candidates = np.random.choice(N, size=num_candidates, replace=False)
        
        best_i = -1
        best_new_peak = current_peak
        best_delta = None
        
        for i in candidates:
            # Delta for flipping f[i]: delta[k] = -2 * f[i] * neighbor_sum[i, k]
            delta = -2.0 * f[i] * neighbor_sum[i]
            
            new_sidelobes = sidelobes + delta
            new_peak = np.max(np.abs(new_sidelobes))
            
            if new_peak < best_new_peak:
                best_new_peak = new_peak
                best_i = i
                best_delta = delta
        
        if best_i >= 0 and best_new_peak < current_peak:
            old_f_i = f[best_i]
            f[best_i] = -old_f_i
            
            # Update sidelobes
            sidelobes = sidelobes + best_delta
            
            # Update neighbor_sum for all positions j that interact with best_i
            for j in range(N):
                k = abs(best_i - j)
                if 1 <= k < N:
                    neighbor_sum[j, k-1] += -2.0 * old_f_i
            
            current_peak = best_new_peak
        else:
            # No improvement; try a random flip and recompute
            i = np.random.randint(N)
            old_f_i = f[i]
            f[i] = -old_f_i
            
            # Recompute ACF from scratch
            acf = compute_acf(f)
            sidelobes = get_sidelobe_array(acf)
            current_peak = np.max(np.abs(sidelobes))
            
            # Recompute neighbor_sum
            for j in range(N):
                for k in range(1, N):
                    s = 0.0
                    if j + k < N:
                        s += f[j + k]
                    if j - k >= 0:
                        s += f[j - k]
                    neighbor_sum[j, k-1] = s
    
    return f
# EVOLVE-BLOCK-END
