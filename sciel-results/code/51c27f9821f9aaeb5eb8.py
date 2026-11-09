"""Erdos minimum overlap: find equal-width step heights h in [0,1], sum N/2,
minimizing the maximum overlap of h with shifted (1-h)."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 100

def _overlap_max(h):
    """Compute the maximum overlap of h with shifted (1-h) over all shifts k=0..N-1.
    O_k = sum(h) - autocorr(k) = N/2 - A_k where A_k = sum h_i * h_{i+k mod N}
    We want to minimize max_k O_k.
    """
    H = np.fft.fft(h)
    A = np.real(np.fft.ifft(H * np.conj(H))) * N
    O = N / 2.0 - A
    return float(np.max(O))

def _objective(params):
    """Objective for optimization: maximize min A_k = minimize max O_k.
    We'll minimize max O_k directly."""
    h = np.clip(params, 0.0, 1.0)
    # Adjust sum to be exactly N/2
    s = h.sum()
    if s < 1e-10:
        h = np.full(N, 0.5)
    else:
        h = h * (N / 2.0) / s
        h = np.clip(h, 0.0, 1.0)
    return _overlap_max(h)

def construct():
    """
    Returns a sequence of step heights h in [0, 1] with sum N/2.
    Uses optimization to find the minimum maximum overlap.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Baseline: constant 0.5
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Try differential evolution for global optimization
    # Use a subset of time for DE
    de_deadline = start_time + 60.0
    
    def de_objective(x):
        if time.time() > de_deadline:
            return 1e10
        h = np.clip(x, 0.0, 1.0)
        s = h.sum()
        if s < 1e-10:
            return 1e10
        h = h * (N / 2.0) / s
        h = np.clip(h, 0.0, 1.0)
        return _overlap_max(h)
    
    try:
        bounds = [(0.0, 1.0)] * N
        result = differential_evolution(
            de_objective, bounds,
            maxiter=50, popsize=15,
            tol=1e-6, mutation=(0.3, 1.0),
            recombination=0.7, seed=42,
            polish=True,
            init='sobol'
        )
        
        if result.fun < best_score:
            h_candidate = np.clip(result.x, 0.0, 1.0)
            s = h_candidate.sum()
            if s > 1e-10:
                h_candidate = h_candidate * (N / 2.0) / s
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
            # Fix sum precisely
            diff = N / 2.0 - h_candidate.sum()
            for j in range(N):
                if h_candidate[j] < 1.0 and diff > 0:
                    adj = min(1.0 - h_candidate[j], diff)
                    h_candidate[j] += adj
                    diff -= adj
                elif h_candidate[j] > 0.0 and diff < 0:
                    adj = min(-diff, h_candidate[j])
                    h_candidate[j] -= adj
                    diff += adj
                if abs(diff) < 1e-12:
                    break
            
            score = _overlap_max(h_candidate)
            if score < best_score:
                best_score = score
                best_h = h_candidate.copy()
    except Exception:
        pass
    
    # Try local refinement with L-BFGS-B
    if time.time() < deadline - 10.0:
        try:
            # Start from current best
            x0 = best_h.copy()
            
            def local_obj(x):
                h = np.clip(x, 0.0, 1.0)
                s = h.sum()
                if s < 1e-10:
                    return 1e10
                h = h * (N / 2.0) / s
                h = np.clip(h, 0.0, 1.0)
                return _overlap_max(h)
            
            result2 = minimize(
                local_obj, x0,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if result2.fun < best_score:
                h_candidate = np.clip(result2.x, 0.0, 1.0)
                s = h_candidate.sum()
                if s > 1e-10:
                    h_candidate = h_candidate * (N / 2.0) / s
                    h_candidate = np.clip(h_candidate, 0.0, 1.0)
                diff = N / 2.0 - h_candidate.sum()
                for j in range(N):
                    if h_candidate[j] < 1.0 and diff > 0:
                        adj = min(1.0 - h_candidate[j], diff)
                        h_candidate[j] += adj
                        diff -= adj
                    elif h_candidate[j] > 0.0 and diff < 0:
                        adj = min(-diff, h_candidate[j])
                        h_candidate[j] -= adj
                        diff += adj
                    if abs(diff) < 1e-12:
                        break
                
                score = _overlap_max(h_candidate)
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Try binary-like constructions
    if time.time() < deadline - 5.0:
        # Try to find good binary sequences
        # Use a greedy or randomized approach
        for trial in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Random binary sequence with 50 ones
            h_bin = np.zeros(N)
            ones_positions = np.random.choice(N, 50, replace=False)
            h_bin[ones_positions] = 1.0
            
            score = _overlap_max(h_bin)
            if score < best_score:
                best_score = score
                best_h = h_bin.copy()
    
    # Final verification and cleanup
    best_h = np.clip(best_h, 0.0, 1.0)
    # Ensure exact sum
    diff = N / 2.0 - best_h.sum()
    if abs(diff) > 1e-10:
        # Distribute correction
        for j in range(N):
            if best_h[j] < 1.0 and diff > 0:
                adj = min(1.0 - best_h[j], diff)
                best_h[j] += adj
                diff -= adj
            elif best_h[j] > 0.0 and diff < 0:
                adj = min(-diff, best_h[j])
                best_h[j] -= adj
                diff += adj
            if abs(diff) < 1e-12:
                break
    
    best_h = np.clip(best_h, 0.0, 1.0)
    
    # Final check
    assert abs(best_h.sum() - N / 2.0) < 1e-6, f"Sum is {best_h.sum()}, expected {N/2.0}"
    assert np.all(best_h >= 0.0 - 1e-10) and np.all(best_h <= 1.0 + 1e-10)
    
    return best_h

# EVOLVE-BLOCK-START
# (kept for compatibility)
# EVOLVE-BLOCK-END
