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
    """
    if not np.all(np.isfinite(P)):
        return float('inf')
    
    # Ensure P is non-negative and symmetric (for real h)
    # P[k] corresponds to freq k. For real h, P[k] = P[N-k].
    # We optimize over independent frequencies k=0..N/2.
    # But here P is passed as the full spectrum or half? 
    # Let's assume P is the full vector of length N, but we enforce symmetry in the reconstruction or just optimize full vector with symmetry constraints?
    # Better: Optimize over the half-spectrum and reconstruct full.
    
    # Actually, let's define the variable as the half-spectrum coefficients c[0..N/2].
    # P[0] = c[0]
    # P[N/2] = c[N/2] (if N even)
    # P[k] = P[N-k] = c[k] for 1 <= k < N/2
    
    # The input `P` to this function will be the half-spectrum vector of length N//2 + 1.
    half_len = N // 2 + 1
    if len(P) != half_len:
        return float('inf')
        
    # Reconstruct full power spectrum
    P_full = np.zeros(N)
    P_full[0] = P[0]
    if N % 2 == 0:
        P_full[N//2] = P[N//2]
    for k in range(1, N//2):
        P_full[k] = P[k]
        P_full[N-k] = P[k]
        
    # Compute circular autocorrelation
    # r = ifft(P_full)
    # Note: ifft includes 1/N factor. 
    # r[k] = (1/N) * sum P[j] exp(2pi i j k / N)
    r = fft.ifft(P_full).real
    
    # We want to minimize max(r).
    # Scale invariance: we can fix the DC term (sum h) to 1, which means P[0] = N (since sum h = N * r[0] ? No.
    # sum(h) = N * h_hat[0] * N? No.
    # DFT: H[k] = sum h[n] e^{-i...}. h[n] = (1/N) sum H[k] e^{i...}
    # sum(h) = N * H[0] * (1/N) ? No.
    # sum(h) = H[0].
    # P[0] = |H[0]|^2 = (sum h)^2.
    # If we normalize sum(h)=1, then P[0]=1.
    # The objective value is 2*N*max(z)/(sum h)^2.
    # z is linear conv. For large N, linear conv peak approx circular conv peak if wrapped parts are small.
    # Let's just minimize max(r) subject to P[0]=1.
    
    if P[0] < 1e-9:
        return float('inf')
        
    # Normalize r so that r[0] = 1 (which corresponds to sum h = 1)
    # r[0] = (1/N) P[0]. If P[0]=1, r[0]=1/N.
    # We want max(r) / r[0]^2 ?
    # Let's stick to the raw value formula adapted for circular approx.
    # value ~ 2 * N * max(r) / (sum h)^2
    # If sum h = 1, value ~ 2 * N * max(r).
    # But r[0] = 1/N. So max(r) is at least 1/N.
    # Let's compute the value directly from r.
    
    # To make it smooth, use log-sum-exp on r.
    # We want to minimize max(r).
    # Since r[0] is fixed by P[0], and we want P[0]=1, let's enforce P[0]=1 in the variable space or penalty.
    # Let's just use P as free variables and add a penalty for P[0] != 1.
    # Or better, fix P[0]=1 and optimize the rest.
    
    # Let's assume the optimizer handles P[0] freely, but we add a strong penalty if P[0] deviates from 1.
    # Actually, it's easier to fix P[0]=1 and optimize P[1..N/2].
    # But the function signature takes P. Let's just compute the value.
    
    # Scale r to have r[0]=1 for the ratio?
    # value = 2 * N * max(r) / (N * r[0])^2 ? No.
    # sum h = H[0]. P[0] = H[0]^2.
    # r[0] = P[0]/N.
    # value = 2 * N * max(r) / P[0].
    
    r_max = np.max(r)
    # Smooth max
    tau = 0.1
    r_scaled = r / tau
    r_max_scaled = np.max(r_scaled)
    exp_terms = np.exp(r_scaled - r_max_scaled)
    sum_exp = np.sum(exp_terms)
    log_sum_exp = r_max_scaled + np.log(sum_exp) if sum_exp > 0 else r_max_scaled
    max_r_smooth = log_sum_exp * tau
    
    val = 2 * N * max_r_smooth / P[0]
    
    # Penalty for P[0] not being 1 (to keep scale fixed)
    penalty = 100.0 * (P[0] - 1.0)**2
    # Penalty for negative P (should be handled by bounds, but just in case)
    neg_penalty = 100.0 * np.sum(np.maximum(0, -P))**2
    
    return val + penalty + neg_penalty

def spectral_to_time(P_half, rng, iters=1000):
    """
    Convert half power spectrum to time-domain signal h using spectral phase retrieval.
    P_half[0..N/2]
    """
    half_len = N // 2 + 1
    if len(P_half) != half_len:
        return None
        
    # Reconstruct full P
    P_full = np.zeros(N)
    P_full[0] = P_half[0]
    if N % 2 == 0:
        P_full[N//2] = P_half[N//2]
    for k in range(1, N//2):
        P_full[k] = P_half[k]
        P_full[N-k] = P_half[k]
        
    # Initialize H with random phase
    H = np.sqrt(np.maximum(0, P_full)) * np.exp(1j * rng.uniform(0, 2*np.pi, N))
    
    h = np.zeros(N)
    for _ in range(iters):
        # Time domain
        h = fft.ifft(H).real
        # Project to real (done)
        # Frequency domain: update magnitude to sqrt(P_full)
        H_new = fft.fft(h)
        # Preserve phase of H_new, magnitude of target
        phase = np.angle(H_new)
        mag_target = np.sqrt(np.maximum(0, P_full))
        H = mag_target * np.exp(1j * phase)
        
        # Update h
        h = fft.ifft(H).real
        
    # Final check
    if not np.all(np.isfinite(h)):
        return None
        
    # Normalize sum(h) to 1? No, the value formula handles scaling.
    # But we optimized P[0]=1, so sum(h) should be close to 1.
    return h

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - SAFETY_MARGIN)
    
    best_h = None
    best_val = float('inf')
    
    # 1. Initial Guesses
    # Uniform (Seed)
    h_seed = np.ones(N)
    best_h = h_seed.copy()
    best_val = compute_value(h_seed)
    
    # 2. Spectral Optimization
    # Variable: half spectrum P[0..N/2]
    half_len = N // 2 + 1
    
    # Initialize P
    # Start with flat spectrum (white noise) -> P[k] = 1
    # P[0] = 1 (sum h = 1)
    P_init = np.ones(half_len)
    P_init[0] = 1.0
    
    # Bounds: P >= 0
    bounds = [(0, None)] * half_len
    
    # Objective wrapper
    def obj(P):
        return spectral_objective(P)
    
    # Gradient? L-BFGS-B can approximate, but providing gradient is better.
    # Let's try L-BFGS-B first with numerical gradient.
    
    try:
        res = minimize(
            obj, 
            P_init, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        P_opt = res.x
        h_opt = spectral_to_time(P_opt, np.random.default_rng(42), iters=500)
        if h_opt is not None:
            val_opt = compute_value(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
    except Exception as e:
        print(f"Optimization error: {e}")
    
    # 3. Refinement with different initial phases or restarts
    # Try a few random restarts in the spectral phase retrieval if time permits
    if best_h is not None and time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            # Slight perturbation of P
            P_pert = P_opt * (1 + 0.1 * np.random.default_rng(restart).normal(0, 1, half_len))
            P_pert = np.maximum(0, P_pert)
            P_pert[0] = 1.0 # Keep DC fixed
            
            # Re-optimize briefly
            try:
                res_r = minimize(
                    obj, 
                    P_pert, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                P_r = res_r.x
                h_r = spectral_to_time(P_r, np.random.default_rng(100+restart), iters=300)
                if h_r is not None:
                    val_r = compute_value(h_r)
                    if val_r < best_val:
                        best_val = val_r
                        best_h = h_r.copy()
            except Exception:
                pass
    
    # 4. Final Polish on Time Domain?
    # The spectral method might get stuck. A quick local search on h might help.
    if best_h is not None and time.time() < deadline - 5:
        # Simple coordinate descent or small L-BFGS on h
        # But h is 400 dim. L-BFGS might be slow.
        # Let's try a few random perturbations and check value.
        rng = np.random.default_rng(123)
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            h_pert = best_h + 0.01 * rng.normal(0, 1, N)
            val_pert = compute_value(h_pert)
            if val_pert < best_val:
                best_val = val_pert
                best_h = h_pert.copy()

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
