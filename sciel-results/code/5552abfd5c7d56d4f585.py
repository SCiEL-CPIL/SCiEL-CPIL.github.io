"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import time
import numpy as np

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Kill at 150s, use 145s as safety margin
    
    def compute_autoconv(f):
        """Compute linear autoconvolution truncated to length N."""
        n_fft = 2 * N
        f_ext = np.zeros(n_fft)
        f_ext[:N] = f
        F = np.fft.rfft(f_ext)
        power = F * np.conj(F)
        autocorr = np.fft.irfft(power, n=n_fft)[:N]
        return autocorr

    def energy_from_autoconv(g):
        """Compute sum of squares of non-zero lags."""
        return np.sum(g[1:N] ** 2)
    
    def compute_neighbor_sums(h):
        """Precompute for each position i, the sum of h[j] over all j such that j-i or i-j is in [0,N-1].
        This is used to quickly compute the change in autoconvolution when flipping h[i]."""
        # For flipping h[i], the change in g_k is delta * (h_{k-i} + h_{i-k}) + delta^2 * delta_{k,2i}
        # We need sum over k in [1,N-1] of (g_k + delta*(h_{k-i}+h_{i-k}) + delta^2*delta_{k,2i})^2 - g_k^2
        # = 2*delta * sum_k g_k*(h_{k-i}+h_{i-k}) + delta^2 * sum_k (h_{k-i}+h_{i-k})^2 + 2*delta^2 * sum_k g_k*delta_{k,2i} + delta^4 * delta_{2i in [1,N-1]}
        # Let's precompute:
        # S1[i] = sum_{k=1}^{N-1} g_k * h_{k-i} (where valid)
        # S2[i] = sum_{k=1}^{N-1} g_k * h_{i-k} (where valid)  
        # S3[i] = sum_{k=1}^{N-1} (h_{k-i} + h_{i-k})^2 (where valid)
        # S4[i] = g_{2i} if 1 <= 2i <= N-1 else 0
        # Then delta_e = 2*delta*(S1[i]+S2[i]) + delta^2*S3[i] + 2*delta^2*S4[i] + delta^4*(1 if 1<=2i<=N-1 else 0)
        # But S1, S2 depend on g which changes... so this approach doesn't work directly.
        # Let's just do the direct computation but optimize it.
        return None

    def fast_delta_energy(h, g, i):
        """Compute change in energy when flipping h[i]. Returns (delta_e, k_start_e, k_end_e, delta_g)."""
        old_val = h[i]
        delta = -2.0 * old_val
        
        k_start = max(1, i - N + 1)
        k_end = min(N - 1, i + N - 1)
        
        if k_start > k_end:
            return 0.0, 0, 0, None
        
        ks = np.arange(k_start, k_end + 1)
        
        # f_{k-i}
        idx1 = ks - i
        mask1 = (idx1 >= 0) & (idx1 < N)
        f_shifted1 = np.zeros(len(ks))
        f_shifted1[mask1] = h[idx1[mask1]]
        
        # f_{i-k}
        idx2 = i - ks
        mask2 = (idx2 >= 0) & (idx2 < N)
        f_shifted2 = np.zeros(len(ks))
        f_shifted2[mask2] = h[idx2[mask2]]
        
        delta_g = delta * (f_shifted1 + f_shifted2)
        
        # Add delta^2 * delta_{k, 2i}
        if 2 * i >= k_start and 2 * i <= k_end:
            idx_2i = 2 * i - k_start
            delta_g[idx_2i] += delta * delta
        
        g_old = g[k_start:k_end + 1]
        g_new = g_old + delta_g
        
        delta_e = np.sum(g_new ** 2 - g_old ** 2)
        
        return delta_e, k_start, k_end, delta_g

    best_result = None
    best_score = float('inf')
    
    rng = np.random.RandomState(42)
    
    # Strategy: Simulated Annealing with optimized evaluation
    num_restarts = 30
    max_sa_steps = 80000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with random sequence
        h = rng.choice([-1.0, 1.0], size=N)
        g = compute_autoconv(h)
        current_energy = energy_from_autoconv(g)
        
        if current_energy < best_score:
            best_score = current_energy
            best_result = h.copy()
            
        # Simulated Annealing with adaptive cooling
        T = 50.0
        T_min = 0.01
        cooling_rate = 0.99995
        
        for step in range(max_sa_steps):
            if time.time() > deadline:
                break
                
            i = rng.randint(N)
            
            delta_e, k_start_e, k_end_e, delta_g = fast_delta_energy(h, g, i)
            
            # Acceptance
            if delta_e < 0 or (T > T_min and rng.rand() < np.exp(-delta_e / T)):
                # Accept
                h[i] = -h[i]
                if delta_g is not None and k_start_e <= k_end_e:
                    g[k_start_e:k_end_e + 1] += delta_g
                    current_energy += delta_e
                    
                    if current_energy < best_score:
                        best_score = current_energy
                        best_result = h.copy()
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
    
    # Final polish: local search on best result
    if best_result is not None and time.time() < deadline - 10:
        h = best_result.copy()
        g = compute_autoconv(h)
        current_energy = energy_from_autoconv(g)
        
        # Local search: try all single flips, accept if improves
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                delta_e, k_start_e, k_end_e, delta_g = fast_delta_energy(h, g, i)
                if delta_e < 0:
                    h[i] = -h[i]
                    g[k_start_e:k_end_e + 1] += delta_g
                    current_energy += delta_e
                    improved = True
                    if current_energy < best_score:
                        best_score = current_energy
                        best_result = h.copy()
    
    if best_result is None:
        best_result = np.ones(N)
        
    return best_result
# EVOLVE-BLOCK-END
