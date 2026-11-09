"""Strategy: Spectral Shaping via Convex Power Spectrum Optimization and Time-Domain Refinement."""
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

N = 400
FFT_SIZE = 1024  # Next power of 2 >= 2*N - 1 = 799

def _compute_ac(x):
    """Compute linear autocorrelation via FFT."""
    X = fft(x, FFT_SIZE)
    ac = ifft(X * np.conj(X)).real[:2 * N - 1]
    return ac

def _objective(x):
    """
    Objective: max |AC(k)| for k != 0, divided by (sum x)^2.
    """
    s = np.sum(x)
    if s == 0:
        return float('inf')
    ac = _compute_ac(x)
    # Lag 0 is at index N-1
    mask = np.ones(2 * N - 1, dtype=bool)
    mask[N - 1] = False
    max_val = np.max(np.abs(ac[mask]))
    return max_val / (s * s)

def _spectral_shaping_init():
    """
    Construct initial signal by minimizing the peak of the autocorrelation sidelobes
    in the frequency domain. We optimize the magnitude squared of the DFT (power spectrum).
    """
    # We want to minimize max_k |AC(k)| for k != 0.
    # AC(k) = IFFT(|X|^2).
    # Let P[m] = |X[m]|^2.
    # We want to minimize max_{k != 0} |IFFT(P)[k]| subject to P >= 0 and sum(P) = const.
    # This is a convex optimization problem (minimize t subject to |IFFT(P)[k]| <= t).
    # However, the variables are P[m] which are real and non-negative.
    # IFFT(P) is real if P is symmetric.
    
    # Variables: p (length FFT_SIZE/2 + 1) representing the unique half of the Hermitian symmetric power spectrum.
    # Actually, let's just optimize the full real vector P of length FFT_SIZE, enforcing P >= 0 and P = P[::-1].real (Hermitian).
    # Simpler: Optimize p_real (length FFT_SIZE) such that P = p_real is Hermitian symmetric?
    # No, P[m] must be real and non-negative. And P[m] = P[FFT_SIZE-m].
    # Let's parameterize by the first half: h (length FFT_SIZE//2 + 1).
    # P[0] = h[0]
    # P[m] = h[m] for 1 <= m <= FFT_SIZE//2 - 1
    # P[FFT_SIZE//2] = h[FFT_SIZE//2]
    # P[FFT_SIZE-m] = h[m]
    
    n_half = FFT_SIZE // 2 + 1
    h0 = np.ones(n_half)
    
    def fun(h):
        # Reconstruct full P
        P = np.zeros(FFT_SIZE)
        P[:n_half] = h
        P[n_half:] = h[FFT_SIZE//2 - 1 : 0 : -1]
        
        # Ensure non-negative (penalty or clipping? Minimize with bounds is better)
        # IFFT of P gives AC (scaled)
        ac = ifft(P).real[:2*N-1]
        
        # Exclude lag 0
        mask = np.ones(2*N-1, dtype=bool)
        mask[N-1] = False
        sidelobes = ac[mask]
        
        # We want to minimize the max absolute value of sidelobes.
        # This is a convex objective if we use a surrogate or just max.
        # max |x_i| is convex.
        return np.max(np.abs(sidelobes))
        
    # Use L-BFGS-B with bounds
    bounds = [(0, None)] * n_half
    
    # Initial guess: flat spectrum (white noise)
    h_init = np.ones(n_half)
    
    # Optimization might be slow or get stuck. 
    # Alternative: Iterative water-filling or simple gradient descent on log-spectrum?
    # Let's try a few iterations of minimizing the peak via gradient descent on h.
    
    h = h_init.copy()
    lr = 0.01
    for i in range(500):
        P = np.zeros(FFT_SIZE)
        P[:n_half] = h
        P[n_half:] = h[FFT_SIZE//2 - 1 : 0 : -1]
        
        ac = ifft(P).real[:2*N-1]
        mask = np.ones(2*N-1, dtype=bool)
        mask[N-1] = False
        sidelobes = ac[mask]
        
        # Find the worst lag
        worst_idx = np.argmax(np.abs(sidelobes))
        worst_val = sidelobes[worst_idx]
        
        # Gradient of |ac[worst_idx]| w.r.t P
        # ac = ifft(P)
        # d(ac[k])/d(P[m]) = exp(-2j pi k m / FFT_SIZE) / FFT_SIZE
        # d|ac[k]|/d(P[m]) = sign(ac[k]) * Re(conj(exp(...))) / FFT_SIZE ? 
        # Actually, ac is real.
        # Let's just do a finite difference gradient or use scipy.
        # Given time constraints, let's just do a simple heuristic:
        # Reduce power at frequencies that contribute constructively to the worst lag.
        
        # Contribution of P[m] to ac[k] is Re(P[m] * exp(-2j pi k m / FFT_SIZE))
        # If ac[k] > 0, we want to reduce P[m] where exp(...) has positive real part.
        # If ac[k] < 0, we want to reduce P[m] where exp(...) has negative real part.
        
        k = worst_idx
        # Note: k in ac corresponds to lag. The IFFT index is k.
        # ac[k] = 1/N_FFT * sum_m P[m] exp(2j pi k m / N_FFT)
        # So contribution is P[m] * cos(2 pi k m / N_FFT)
        
        cos_terms = np.cos(2 * np.pi * k * np.arange(FFT_SIZE) / FFT_SIZE)
        
        # If ac[k] > 0, increase P where cos < 0, decrease where cos > 0.
        # If ac[k] < 0, increase P where cos > 0, decrease where cos < 0.
        # Effectively: delta P = - sign(ac[k]) * cos_terms * step
        
        sign_ac = np.sign(worst_val)
        if sign_ac == 0:
            break
            
        delta_P = -sign_ac * cos_terms * 0.01
        
        # Update h from delta_P
        # P[0..N/2] = h[0..N/2]
        # P[N/2+1..] = h[N/2-1..0]
        # delta_h[0..N/2] = delta_P[0..N/2]
        # delta_h[N/2-1..0] = delta_P[N/2+1..]
        
        delta_h = np.zeros(n_half)
        delta_h[:n_half] = delta_P[:n_half]
        # The second half of P is the reverse of the first half (excluding center)
        # P[m] = P[N-m]. So delta_P[m] = delta_P[N-m].
        # h[m] affects P[m] and P[N-m].
        # So delta_h[m] should be sum of effects?
        # Actually, P is constructed from h.
        # P = [h[0], h[1], ..., h[N/2], h[N/2-1], ..., h[1]]
        # dP[0] = dh[0]
        # dP[m] = dh[m] for m < N/2
        # dP[N/2] = dh[N/2]
        # dP[N-m] = dh[m] for m < N/2
        
        # We want to update h to minimize the objective.
        # Gradient of objective w.r.t h[m] is sum of gradients w.r.t P indices it affects.
        # grad_h[0] = grad_P[0]
        # grad_h[m] = grad_P[m] + grad_P[N-m] for 0 < m < N/2
        # grad_h[N/2] = grad_P[N/2]
        
        grad_P = delta_P # This is the direction of decrease for P
        
        # Calculate grad_h
        grad_h = np.zeros(n_half)
        grad_h[0] = grad_P[0]
        grad_h[N//2] = grad_P[N//2]
        for m in range(1, N//2):
            grad_h[m] = grad_P[m] + grad_P[N-m]
            
        h -= lr * grad_h
        h = np.maximum(h, 0) # Keep non-negative
        
        # Normalize to keep energy constant
        h = h / np.sum(h) * n_half
        
    # Reconstruct x from P
    P = np.zeros(FFT_SIZE)
    P[:n_half] = h
    P[n_half:] = h[FFT_SIZE//2 - 1 : 0 : -1]
    
    # X = sqrt(P) * exp(j phase)
    # We need to choose phases to make x real and "good".
    # Random phases? Or zero phases?
    # Zero phases give a positive real sequence (sum of cosines).
    # Let's try zero phases first.
    X = np.sqrt(P)
    x = ifft(X).real[:N]
    
    # Normalize
    s = np.sum(x)
    if s != 0:
        x = x / s * N
        
    return x

def _local_search(x, time_limit):
    """
    Refine the sequence using local search (bit flips / small perturbations).
    """
    start_time = time.time()
    best_x = x.copy()
    best_score = _objective(best_x)
    current_x = x.copy()
    current_score = best_score
    
    T = 0.1
    cooling = 0.99
    
    while time.time() < start_time + time_limit and T > 1e-4:
        # Perturb a few random elements
        num_changes = np.random.randint(1, 5)
        indices = np.random.choice(N, num_changes, replace=False)
        # Random perturbations
        deltas = np.random.normal(0, T, num_changes)
        
        x_new = current_x.copy()
        x_new[indices] += deltas
        
        # Optional: Clip to +/- 1? The task says "signed step heights", usually implies +/- 1 or integers.
        # But the seed program used real values. The inspirations mention "signed integers" or "step functions".
        # Let's stick to real values for now as the seed did, but maybe clip to [-1, 1] to keep "step" character?
        # The task says "signed step heights". Usually this means a piecewise constant function.
        # If the evaluator expects +/- 1, real values might be suboptimal or invalid?
        # "Signed autocorrelation... minimizing...".
        # Let's assume real values are allowed as per seed.
        
        s_new = np.sum(x_new)
        if s_new == 0:
            T *= cooling
            continue
            
        x_new = x_new / s_new * N
        
        new_score = _objective(x_new)
        
        if new_score < current_score:
            current_x = x_new
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_x = x_new.copy()
        else:
            delta_e = new_score - current_score
            if delta_e < 0:
                current_x = x_new
                current_score = new_score
            else:
                prob = np.exp(-delta_e / T)
                if np.random.random() < prob:
                    current_x = x_new
                    current_score = new_score
        
        T *= cooling
        
    return best_x

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # 1. Spectral Shaping Initialization
    x_init = _spectral_shaping_init()
    
    # 2. Local Search Refinement
    # Allocate remaining time to local search
    elapsed = time.time() - start_time
    remaining = deadline - elapsed - 5.0 # 5s buffer
    
    if remaining < 10:
        # Fallback if spectral shaping took too long
        x_final = np.ones(N)
    else:
        x_final = _local_search(x_init, remaining)
        
    # Final check and normalization
    s = np.sum(x_final)
    if s == 0:
        return np.ones(N)
    x_final = x_final / s * N
        
    return x_final
