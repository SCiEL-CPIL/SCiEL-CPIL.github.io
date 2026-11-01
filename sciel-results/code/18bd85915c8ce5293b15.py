"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

DEADLINE = 120.0
START_TIME = time.time()

def elapsed():
    return time.time() - START_TIME

def check_deadline(margin=5.0):
    return elapsed() < DEADLINE - margin

def compute_autocorr_peak_sidelobe(x):
    """Compute peak sidelobe of autocorrelation for a step function.
    x: array of N values (continuous or ±1)
    Returns the max |autocorr(k)| for k != 0, normalized."""
    # Zero-pad for linear autocorrelation
    n = len(x)
    fft_size = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, n=fft_size)
    ac = np.fft.irfft(X * np.conj(X), n=fft_size)[:n]
    # Normalize by DC component
    dc = ac[0]
    if abs(dc) < 1e-12:
        return 1.0
    sidelobes = np.abs(ac[1:])
    peak = np.max(sidelobes) / abs(dc)
    return peak

def objective_and_grad(x):
    """Smooth surrogate for peak sidelobe using log-sum-exp."""
    n = len(x)
    fft_size = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, n=fft_size)
    ac = np.fft.irfft(X * np.conj(X), n=fft_size)[:n]
    dc = ac[0]
    if abs(dc) < 1e-12:
        return 1.0, np.zeros_like(x)
    sidelobes = np.abs(ac[1:]) / abs(dc)
    # Soft-max with temperature
    tau = 0.1
    m = np.max(sidelobes)
    log_sum = m + np.log(np.sum(np.exp((sidelobes - m) / tau))) * tau
    # Gradient via finite difference approximation is expensive; use analytic
    # For L-BFGS-B, we can use finite differences
    val = log_sum
    return val, None

def refine_discrete(x_cont):
    """Refine continuous solution to discrete ±1 values using local search."""
    x = np.sign(x_cont).copy()
    x[x == 0] = 1.0
    best_val = compute_autocorr_peak_sidelobe(x)
    n = len(x)
    # Local search: try flipping individual bits
    improved = True
    while improved and check_deadline(10.0):
        improved = False
        for i in range(n):
            if not check_deadline(10.0):
                break
            old = x[i]
            x[i] = -old
            new_val = compute_autocorr_peak_sidelobe(x)
            if new_val < best_val - 1e-12:
                best_val = new_val
                improved = True
            else:
                x[i] = old
    return x, best_val

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Phase 1: Continuous relaxation with L-BFGS-B
    # Start with random initialization
    best_x = None
    best_val = float('inf')
    
    n_restarts = 3
    for restart in range(n_restarts):
        if not check_deadline(30.0):
            break
        
        # Random initialization
        np.random.seed(42 + restart)
        x0 = np.random.uniform(-1, 1, N)
        
        # Use L-BFGS-B with finite difference gradient
        try:
            res = minimize(
                lambda x: compute_autocorr_peak_sidelobe(x),
                x0,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 5000}
            )
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
        except Exception:
            pass
    
    if best_x is None:
        # Fallback: use all ones
        return np.ones(N)
    
    # Phase 2: Discrete refinement
    x_disc, disc_val = refine_discrete(best_x)
    
    # Phase 3: Try multiple discrete starts and pick best
    candidates = [x_disc]
    # Also try sign of continuous solution
    x_sign = np.sign(best_x)
    x_sign[x_sign == 0] = 1.0
    candidates.append(x_sign)
    
    # Try a few random ±1 initializations with local search
    for trial in range(2):
        if not check_deadline(15.0):
            break
        np.random.seed(1000 + trial)
        x_rand = np.random.choice([-1.0, 1.0], size=N)
        x_ref, val_ref = refine_discrete(x_rand)
        candidates.append(x_ref)
    
    # Evaluate all candidates and pick best
    best_candidate = None
    best_candidate_val = float('inf')
    for cand in candidates:
        val = compute_autocorr_peak_sidelobe(cand)
        if val < best_candidate_val:
            best_candidate_val = val
            best_candidate = cand
    
    if best_candidate is None:
        return np.ones(N)
    
    return best_candidate
# EVOLVE-BLOCK-END
