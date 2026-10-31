"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import time
import numpy as np

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Kill at 150s, use 145s as safety margin
    
    n_fft = 2 * N
    # Precompute twiddle factors for incremental FFT updates
    # When h[i] changes by delta, F[k] changes by delta * exp(-2*pi*i*k/N)
    k_indices = np.arange(N // 2 + 1)
    twiddle = np.exp(-2j * np.pi * k_indices[None, :] * np.arange(N)[:, None] / N)
    # twiddle[i, k] = exp(-2*pi*i*k/N) for k in [0, N//2]
    
    def compute_F(heights):
        """Compute the rfft of the zero-padded signal."""
        f_ext = np.zeros(n_fft)
        f_ext[:N] = heights
        return np.fft.rfft(f_ext)
    
    def compute_energy_from_F(F):
        """Compute the autoconvolution energy from the FFT."""
        power = F * np.conj(F)
        autocorr = np.fft.irfft(power, n=n_fft)[:N]
        return np.sum(autocorr[1:N] ** 2)
    
    def flip_update(F, i, old_val):
        """
        Update F after flipping h[i] from old_val to -old_val.
        The change in h[i] is -2*old_val.
        Delta F[k] = -2*old_val * exp(-2*pi*i*k/N)
        """
        delta = -2.0 * old_val
        F_new = F.copy()
        F_new += delta * twiddle[i, :]
        return F_new
    
    def autoconv_energy(heights):
        """Compute the L2 norm of the autoconvolution (as a proxy for C3)."""
        F = compute_F(heights)
        return compute_energy_from_F(F)
    
    best_result = None
    best_score = float('inf')
    
    rng = np.random.RandomState(42)
    
    # Precompute some structured initializations
    structured_inits = []
    
    # Alternating pattern
    h_alt = np.ones(N) * ((np.arange(N) % 2) * 4 - 1)
    structured_inits.append(h_alt)
    
    # Block pattern: +1 for first half, -1 for second
    h_block = np.ones(N)
    h_block[N//2:] = -1.0
    structured_inits.append(h_block)
    
    # Quarter blocks
    h_quarter = np.ones(N)
    h_quarter[N//4:3*N//4] = -1.0
    structured_inits.append(h_quarter)
    
    # Try each structured init first
    for h_init in structured_inits:
        if time.time() > deadline - 5:
            break
        current_energy = autoconv_energy(h_init)
        if current_energy < best_score:
            best_score = current_energy
            best_result = h_init.copy()
    
    # Use much more restarts since each step is now ~50x faster
    num_restarts = 200
    max_sa_steps = 15000
    max_hc_steps = 50000
    
    for r in range(num_restarts):
        if time.time() > deadline - 3:
            break
        
        # Alternate between random and perturbed structured starts
        if r % 3 == 0 and r < len(structured_inits) * 3:
            base = structured_inits[r // 3 % len(structured_inits)]
            h = base.copy()
            # Perturb 10% of positions
            perturb_idx = rng.choice(N, size=N//10, replace=False)
            h[perturb_idx] *= -1
        else:
            h = rng.choice([-1.0, 1.0], size=N)
        
        F = compute_F(h)
        current_energy = compute_energy_from_F(F)
        
        if current_energy < best_score:
            best_score = current_energy
            best_result = h.copy()
        
        # Simulated Annealing with adaptive temperature
        T = 500.0
        T_min = 0.1
        cooling_rate = 0.99995
        
        for step in range(max_sa_steps):
            if time.time() > deadline - 3:
                break
                
            i = rng.randint(N)
            old_val = h[i]
            F = flip_update(F, i, old_val)
            h[i] = -old_val
            new_energy = compute_energy_from_F(F)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or (T > 0 and rng.rand() < np.exp(-delta_e / T)):
                current_energy = new_energy
                if current_energy < best_score:
                    best_score = current_energy
                    best_result = h.copy()
            else:
                # Revert
                F = flip_update(F, i, h[i])
                h[i] = old_val
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # Hill Climbing Polish (Greedy) - much more aggressive with fast eval
        improved = True
        hc_count = 0
        while improved and hc_count < max_hc_steps and time.time() < deadline - 3:
            improved = False
            # Try multiple random flips per iteration to find improvements faster
            for _ in range(10):
                if time.time() > deadline - 3:
                    break
                i = rng.randint(N)
                old_val = h[i]
                F = flip_update(F, i, old_val)
                h[i] = -old_val
                new_energy = compute_energy_from_F(F)
                if new_energy < current_energy:
                    current_energy = new_energy
                    improved = True
                    hc_count += 1
                    if current_energy < best_score:
                        best_score = current_energy
                        best_result = h.copy()
                else:
                    # Revert
                    F = flip_update(F, i, h[i])
                    h[i] = old_val
                    hc_count += 1
    
    # Final exhaustive single-flip local search on the best result found
    if best_result is not None and time.time() < deadline - 1:
        h = best_result.copy()
        F = compute_F(h)
        current_energy = compute_energy_from_F(F)
        
        # Exhaustive: try every single flip, keep improvements
        improved_any = True
        passes = 0
        max_passes = 20
        while improved_any and passes < max_passes and time.time() < deadline - 1:
            improved_any = False
            passes += 1
            for i in range(N):
                if time.time() > deadline - 1:
                    break
                old_val = h[i]
                F = flip_update(F, i, old_val)
                h[i] = -old_val
                new_energy = compute_energy_from_F(F)
                if new_energy < current_energy - 1e-12:
                    current_energy = new_energy
                    improved_any = True
                    if current_energy < best_score:
                        best_score = current_energy
                        best_result = h.copy()
                else:
                    # Revert
                    F = flip_update(F, i, h[i])
                    h[i] = old_val
    
    if best_result is None:
        best_result = np.ones(N)
    
    return best_result
# EVOLVE-BLOCK-END
