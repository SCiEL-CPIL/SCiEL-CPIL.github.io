import numpy as np
import time
from scipy.optimize import minimize
import scipy.fft as fft

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

def compute_value(h):
    """Exact evaluation of the objective function."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return float('inf')
    if not np.all(np.isfinite(h)):
        return float('inf')
    z = np.convolve(h, h)
    return abs(2 * N * np.max(z) / (s * s))

def spectral_objective(P):
    """
    Differentiable objective for the power spectrum P.
    Minimizes the peak of the circular autocorrelation r[k] = ifft(P)[k].
    Uses log-sum-exp for smooth max approximation.
    
    P is the power spectrum (real, non-negative).
    We optimize over the independent half-spectrum to ensure real h.
    """
    if not np.all(np.isfinite(P)):
        return float('inf')
    
    # Reconstruct full spectrum from half-spectrum P (length N/2 + 1)
    # P_full[0] = P[0]
    # P_full[k] = P[k] for k=1..N/2
    # P_full[N-k] = P[k]
    
    N_half = N // 2
    P_full = np.zeros(N)
    P_full[0] = P[0]
    if N_half > 0:
        P_full[1:N_half+1] = P[1:N_half+1]
        P_full[N-N_half:N-1] = P[1:N_half+1]
    
    # Ensure non-negativity (optimization might drift slightly negative)
    P_full = np.maximum(P_full, 0)
    
    # Compute circular autocorrelation
    # r[k] = ifft(P_full)[k]
    # Note: ifft of power spectrum gives circular autocorrelation
    r = fft.ifft(P_full)
    r = r.real
    
    # The peak of circular autocorrelation is r[0] (if P is PSD of h, r[0] = sum|h|^2).
    # Wait, the objective is max(z) where z is linear autocorrelation.
    # For large N, circular approx linear if h is band-limited or zero-padded effectively.
    # However, r[0] is always the maximum for PSD? No, sidelobes can be negative or positive.
    # We want to minimize the max value of r.
    
    # Smooth max of r
    tau = 0.5 # Temperature for LSE
    r_scaled = r / tau
    r_max_scaled = np.max(r_scaled)
    exp_terms = np.exp(r_scaled - r_max_scaled)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        log_sum_exp = r_max_scaled
    else:
        log_sum_exp = r_max_scaled + np.log(sum_exp)
        
    max_r_smooth = log_sum_exp * tau
    
    # We also want to ensure the DC component (sum h) is not too small to avoid division by zero issues in final scaling
    # sum(h) = N * h_hat[0]. h_hat[0] = sqrt(P[0]) * phase.
    # Actually, we just return the peak. The scaling is handled by the fact that objective is scale invariant.
    # But if P[0] is 0, sum(h) is 0. We should penalize small P[0].
    
    # Add a penalty if P[0] is too small
    penalty = 10.0 / (P[0] + 1e-6)
    
    return max_r_smooth + penalty

def time_domain_objective(h):
    """
    Objective for the final polish in time domain.
    Uses exact max but with a very smooth surrogate if needed, or just L-BFGS-B on exact?
    L-BFGS-B needs differentiability. The exact max is not differentiable at ties.
    We use a very sharp LSE for the final polish.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    z = np.convolve(h, h)
    # Sharp LSE
    tau = 0.01
    z_scaled = z / tau
    z_max_scaled = np.max(z_scaled)
    exp_terms = np.exp(z_scaled - z_max_scaled)
    sum_exp = np.sum(exp_terms)
    if sum_exp == 0:
        log_sum_exp = z_max_scaled
    else:
        log_sum_exp = z_max_scaled + np.log(sum_exp)
    max_z_smooth = log_sum_exp * tau
    
    return abs(2 * N * max_z_smooth / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - SAFETY_MARGIN)
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Phase 1: Spectral Optimization
    # We optimize the half-power spectrum P.
    # Initial guess: Flat spectrum (white noise) -> P[k] = constant.
    # Or a "Golay-like" spectrum.
    # Let's start with a flat spectrum.
    P_init = np.ones(N // 2 + 1)
    
    # Boundary constraints: P >= 0.
    # L-BFGS-B supports bounds.
    bounds = [(0, None)] * (N // 2 + 1)
    
    # To improve the search, we can try multiple initializations in spectral space.
    # 1. Flat
    # 2. Low-pass (Gaussian decay)
    # 3. High-pass
    
    candidates_P = [P_init.copy()]
    
    # Low-pass
    freqs = np.linspace(0, 1, N // 2 + 1)
    P_lp = np.exp(-5 * freqs**2)
    candidates_P.append(P_lp)
    
    # High-pass
    P_hp = 1.0 - np.exp(-5 * freqs**2)
    # Ensure P[0] is not zero for HP, add small epsilon
    P_hp[0] += 1e-3
    candidates_P.append(P_hp)
    
    # Random
    rng = np.random.default_rng(42)
    for _ in range(3):
        P_rand = rng.uniform(0.1, 1.0, N // 2 + 1)
        candidates_P.append(P_rand)
        
    for P_cand in candidates_P:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                spectral_objective, 
                P_cand, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            P_opt = res.x
            # Reconstruct h
            N_half = N // 2
            P_full = np.zeros(N)
            P_full[0] = P_opt[0]
            P_full[1:N_half+1] = P_opt[1:N_half+1]
            P_full[N-N_half:N-1] = P_opt[1:N_half+1]
            
            # h is ifft(sqrt(P_full)) * phase.
            # Since we only have power, we need to choose phases.
            # To minimize peak autocorrelation, phases should be random or specific?
            # Actually, if we minimize the peak of ifft(P), the phases of h don't matter for the *circular* autocorrelation peak?
            # r[k] = ifft(P)[k]. The value r[k] depends only on P, not on the phases of h, 
            # because P = |H|^2. The autocorrelation is determined by the power spectrum.
            # So any phase choice yields the same circular autocorrelation.
            # We can choose random phases to get a specific h.
            
            H = np.sqrt(P_full) * np.exp(1j * rng.uniform(0, 2*np.pi, N))
            # Ensure H[0] is real and positive to fix global phase?
            # H[0] = sum(h). If we want sum(h) > 0, set H[0] real positive.
            H[0] = np.sqrt(P_full[0])
            # Ensure Hermitian symmetry for real h
            # H[k] = conj(H[N-k])
            # My construction above might not preserve this perfectly if I assign random phases independently.
            # Let's construct H properly.
            
            H_half = np.sqrt(P_opt)
            phases = rng.uniform(0, 2*np.pi, len(H_half))
            H_full = np.zeros(N, dtype=complex)
            H_full[0] = H_half[0]
            for k in range(1, N_half + 1):
                H_full[k] = H_half[k] * np.exp(1j * phases[k])
                H_full[N-k] = np.conj(H_full[k])
            
            h_candidate = fft.ifft(H_full).real
            
            val = compute_value(h_candidate)
            if val < best_val:
                best_val = val
                best_h = h_candidate.copy()
                
        except Exception as e:
            pass

    # Phase 2: Time Domain Polish
    # The spectral optimization minimizes circular autocorrelation peak.
    # The exact objective uses linear autocorrelation.
    # For N=400, if the signal is "compact" or if we zero-pad effectively, they are close.
    # But we can polish in time domain using a smooth surrogate of the linear autocorrelation max.
    
    if time.time() < deadline - 5 and best_h is not None:
        try:
            # Use L-BFGS-B on h directly with a sharp LSE surrogate
            res_final = minimize(
                time_domain_objective, 
                best_h, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            h_final = res_final.x
            if np.all(np.isfinite(h_final)):
                val_final = compute_value(h_final)
                if val_final < best_val:
                    best_val = val_final
                    best_h = h_final.copy()
                    
        except Exception:
            pass

    # Ensure validity
    if best_h is None or not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print(len(h))
    print(np.sum(h))
    z = np.convolve(h, h)
    print(abs(2 * N * np.max(z) / np.sum(h)**2))
