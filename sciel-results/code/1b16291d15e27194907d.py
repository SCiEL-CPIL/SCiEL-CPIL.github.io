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

    best_result = None
    best_score = float('inf')
    
    rng = np.random.RandomState(42)
    
    # Strategy: Simulated Annealing with O(N) flip cost evaluation
    num_restarts = 50
    max_sa_steps = 50000
    
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
            
        # Simulated Annealing
        T = 100.0
        T_min = 0.1
        cooling_rate = 0.9999
        
        for step in range(max_sa_steps):
            if time.time() > deadline:
                break
                
            i = rng.randint(N)
            # Flip h[i]
            old_val = h[i]
            new_val = -old_val
            delta = new_val - old_val  # -2 * old_val
            
            # Calculate change in autoconvolution g
            # g_k = sum_j f_j f_{k-j}
            # g'_k = g_k + delta * (f_{k-i} + f_{i-k}) + delta^2 * delta_{k, 2i}
            
            # Indices k where f_{k-i} is non-zero: 0 <= k-i < N => i <= k < i+N
            # Indices k where f_{i-k} is non-zero: 0 <= i-k < N => i-N < k <= i
            # Combined range: [max(0, i-N+1), min(N-1, i+N-1)]
            
            k_start = max(0, i - N + 1)
            k_end = min(N - 1, i + N - 1)
            
            # We only care about k in [1, N-1] for energy
            k_start_e = max(1, k_start)
            k_end_e = min(N - 1, k_end)
            
            if k_start_e > k_end_e:
                delta_e = 0.0
            else:
                # Compute delta_g for k in [k_start_e, k_end_e]
                ks = np.arange(k_start_e, k_end_e + 1)
                
                # f_{k-i}
                idx1 = ks - i
                # Only valid if 0 <= idx1 < N
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
                # 2i must be in [k_start_e, k_end_e] and 2i < N?
                # The term is delta^2 if k == 2i.
                if 2 * i >= k_start_e and 2 * i <= k_end_e:
                    # Find index in ks
                    idx_2i = 2 * i - k_start_e
                    delta_g[idx_2i] += delta * delta
                
                g_old = g[k_start_e:k_end_e + 1]
                g_new = g_old + delta_g
                
                # Delta energy = sum(g_new^2 - g_old^2)
                delta_e = np.sum(g_new ** 2 - g_old ** 2)
            
            # Acceptance
            if delta_e < 0 or (T > T_min and rng.rand() < np.exp(-delta_e / T)):
                # Accept
                h[i] = new_val
                # Update g
                if k_start_e <= k_end_e:
                    # Recalculate g for the affected range
                    # We already have delta_g, so g[k] += delta_g
                    # But we need to be careful: we computed delta_g based on old h.
                    # So g_new = g_old + delta_g is correct.
                    g[k_start_e:k_end_e + 1] = g_old + delta_g
                    current_energy += delta_e
                    
                    if current_energy < best_score:
                        best_score = current_energy
                        best_result = h.copy()
            else:
                # Reject: h is unchanged, g is unchanged
                pass
            
            T *= cooling_rate
            if T < T_min:
                T = T_min

    if best_result is None:
        best_result = np.ones(N)
        
    return best_result
# EVOLVE-BLOCK-END
