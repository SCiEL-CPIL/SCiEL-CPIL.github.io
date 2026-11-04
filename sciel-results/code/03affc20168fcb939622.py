"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconvolution_max_abs(f):
    """Compute max |f*f| over all lags using FFT."""
    n = len(f)
    conv = np.fft.ifft(np.fft.fft(f, 2*n) ** 2).real
    return np.max(np.abs(conv))


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s limit
    
    best_f = None
    best_val = np.inf
    
    # Structural change: Use quasi-periodic complex exponential structure.
    # Represent f as real part of sum of few complex exponentials with 
    # irrational frequencies. This creates structured patterns that may
    # have lower autocorrelation peaks than random or sparse patterns.
    
    # The idea: f[k] = sum_j a_j * cos(2*pi*omega_j*k/N + phi_j)
    # where omega_j are fixed irrational-like frequencies, and we optimize
    # amplitudes a_j and phases phi_j.
    
    # Use a small number of harmonics (M) to keep optimization tractable
    # while creating structured, non-random patterns.
    
    M_range = [3, 5, 7, 10]  # number of harmonics to try
    num_starts = 8  # random initializations per M
    
    np.random.seed(42)
    
    for M in M_range:
        if time.time() > deadline - 30:
            break
        
        for start in range(num_starts):
            if time.time() > deadline - 15:
                break
            
            # Generate quasi-periodic frequencies (irrational-like)
            # Use golden ratio conjugate and related values for good distribution
            phi_golden = (1.0 - np.sqrt(5.0)) / 2.0  # ~ -0.618
            freqs = np.array([
                (phi_golden * (j + 1)) % 1.0 for j in range(M)
            ])
            # Ensure frequencies are spread out
            freqs = np.sort(freqs)
            # Add small perturbation to avoid exact periodicity
            freqs = freqs + np.random.uniform(-0.01, 0.01, size=M)
            freqs = np.abs(freqs) % 1.0
            
            # Initialize amplitudes and phases randomly
            a = np.random.uniform(0.5, 2.0, size=M)
            phi = np.random.uniform(0, 2*np.pi, size=M)
            
            # Build f from harmonic sum
            k = np.arange(N)
            f = np.zeros(N)
            for j in range(M):
                f += a[j] * np.cos(2*np.pi*freqs[j]*k + phi[j])
            
            # Normalize to unit amplitude (scale so max |f| = 1)
            max_abs = np.max(np.abs(f))
            if max_abs > 0:
                f = f / max_abs
            
            current_score = _autoconvolution_max_abs(f)
            
            if current_score < best_val:
                best_val = current_score
                best_f = f.copy()
            
            # Local search: perturb amplitudes and phases
            improved = True
            max_iter = 50
            iter_count = 0
            while improved and iter_count < max_iter and time.time() < deadline - 15:
                improved = False
                iter_count += 1
                
                # Try perturbing each amplitude
                for j in range(M):
                    if time.time() > deadline - 15:
                        break
                    
                    old_a = a[j]
                    a[j] += np.random.uniform(-0.2, 0.2)
                    a[j] = max(0.1, a[j])  # keep positive
                    
                    # Rebuild f
                    f_new = np.zeros(N)
                    for jj in range(M):
                        f_new += a[jj] * np.cos(2*np.pi*freqs[jj]*k + phi[jj])
                    max_abs_new = np.max(np.abs(f_new))
                    if max_abs_new > 0:
                        f_new = f_new / max_abs_new
                    
                    new_score = _autoconvolution_max_abs(f_new)
                    if new_score < current_score - 1e-10:
                        current_score = new_score
                        improved = True
                        f = f_new
                        if new_score < best_val:
                            best_val = new_score
                            best_f = f.copy()
                    else:
                        a[j] = old_a
                
                # Try perturbing each phase
                for j in range(M):
                    if time.time() > deadline - 15:
                        break
                    
                    old_phi = phi[j]
                    phi[j] += np.random.uniform(-0.3, 0.3)
                    phi[j] = phi[j] % (2*np.pi)
                    
                    # Rebuild f
                    f_new = np.zeros(N)
                    for jj in range(M):
                        f_new += a[jj] * np.cos(2*np.pi*freqs[jj]*k + phi[jj])
                    max_abs_new = np.max(np.abs(f_new))
                    if max_abs_new > 0:
                        f_new = f_new / max_abs_new
                    
                    new_score = _autoconvolution_max_abs(f_new)
                    if new_score < current_score - 1e-10:
                        current_score = new_score
                        improved = True
                        f = f_new
                        if new_score < best_val:
                            best_val = new_score
                            best_f = f.copy()
                    else:
                        phi[j] = old_phi
    
    # Final polish: try small perturbations on best solution
    if best_f is not None and time.time() < deadline - 5:
        # Try flipping signs of small subsets
        active = np.nonzero(np.abs(best_f) > 0.01)[0]
        if len(active) > 0:
            # Try flipping random subsets of size 1-3
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                
                subset_size = np.random.randint(1, 4)
                subset = np.random.choice(active, size=min(subset_size, len(active)), replace=False)
                
                f_test = best_f.copy()
                for idx in subset:
                    f_test[idx] = -f_test[idx]
                
                s = _autoconvolution_max_abs(f_test)
                if s < best_val - 1e-10:
                    best_val = s
                    best_f = f_test
    
    if best_f is None:
        return np.ones(N)
    
    return best_f
# EVOLVE-BLOCK-END
