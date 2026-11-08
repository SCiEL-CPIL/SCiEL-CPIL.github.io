"""Combined approach: Fourier initialization + Projected Nelder-Mead refinement."""
import time
import numpy as np
from scipy.optimize import minimize, minimize_nelder_mead

N = 100

# EVOLVE-BLOCK-START
def _compute_max_overlap(h):
    """Compute max over all circular shifts k of sum_j h[j]*(1-h[(j+k)%N]) * (2/N)."""
    n = len(h)
    dx = 2.0 / n
    H = np.fft.rfft(h, n=2*n)
    autocorr = np.fft.irfft(H * np.conj(H), n=2*n)[:n]
    sum_h = np.sum(h)
    C = np.zeros(n)
    C[0] = autocorr[0]
    if n > 1:
        C[1:] = autocorr[n-1:0:-1]
    overlaps = (sum_h - C) * dx
    return np.max(overlaps)

def _project(h):
    """Project h onto feasible set: [0,1]^N with sum = N/2."""
    h = np.array(h, dtype=float)
    h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            # Need to increase sum. Increase points that are < 1.
            mask = h < 1.0 - 1e-10
            cnt = np.sum(mask)
            if cnt > 0:
                # Distribute increase evenly among non-saturated points
                # But if we add too much, some might exceed 1. 
                # Simple approach: add diff/cnt, then clip, then re-balance.
                h[mask] += diff / cnt
        else:
            # Need to decrease sum. Decrease points that are > 0.
            mask = h > 1e-10
            cnt = np.sum(mask)
            if cnt > 0:
                h[mask] += diff / cnt
        
        h = np.clip(h, 0.0, 1.0)
        
        # Second pass to fix any residual sum error due to clipping
        diff2 = N / 2.0 - np.sum(h)
        if abs(diff2) > 1e-10:
            if diff2 > 0:
                mask = h < 1.0 - 1e-10
            else:
                mask = h > 1e-10
            cnt = np.sum(mask)
            if cnt > 0:
                h[mask] += diff2 / cnt
            h = np.clip(h, 0.0, 1.0)
            
            # If still off, it's a pathological case, force adjust one element
            diff3 = N / 2.0 - np.sum(h)
            if abs(diff3) > 1e-8:
                # Find an element that can absorb the change
                if diff3 > 0:
                    idx = np.argmin(h)
                    h[idx] += diff3
                else:
                    idx = np.argmax(h)
                    h[idx] += diff3
                h = np.clip(h, 0.0, 1.0)
    return h

def _objective_fourier(coeffs):
    """Objective function for optimization: max overlap from Fourier coefficients."""
    x = np.linspace(0, 2, N, endpoint=False)
    h = np.full(N, 0.5)
    for k in range(len(coeffs) // 2):
        a_k = coeffs[2*k]
        b_k = coeffs[2*k+1]
        h += a_k * np.cos(np.pi * k * x) + b_k * np.sin(np.pi * k * x)
    # Project to ensure validity for evaluation
    h = _project(h)
    return _compute_max_overlap(h)

def _objective_direct(h):
    """Objective function for direct optimization: max overlap + penalty for infeasibility."""
    # h is already projected in the Nelder-Mead wrapper if we use it, 
    # but let's assume h is raw and project inside for safety? 
    # Actually, Nelder-Mead will explore outside bounds. 
    # We should project h before computing overlap, but the "distance" 
    # from the ideal point matters for the simplex geometry.
    # Better: Project h, compute overlap.
    h_proj = _project(h)
    return _compute_max_overlap(h_proj)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    x = np.linspace(0, 2, N, endpoint=False)
    
    # Phase 1: Multi-start Fourier Optimization
    # This provides good initial points in the smooth manifold.
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Try some analytic candidates
    candidates = []
    h1 = 0.5 + 0.5 * np.sin(np.pi * x)
    candidates.append(_project(h1))
    h2 = 0.5 + 0.5 * np.cos(np.pi * x)
    candidates.append(_project(h2))
    h3 = (x < 1.0).astype(float)
    candidates.append(_project(h3))
    
    for h_cand in candidates:
        s = _compute_max_overlap(h_cand)
        if s < best_score:
            best_score = s
            best_h = h_cand.copy()
            
    n_modes = 6
    n_coeffs = 2 * n_modes
    bounds = [(-0.5, 0.5)] * n_coeffs
    
    np.random.seed(42)
    
    # Run a few L-BFGS-B starts
    starts = []
    starts.append(np.zeros(n_coeffs))
    for _ in range(5):
        starts.append(np.random.uniform(-0.3, 0.3, n_coeffs))
        
    for x0 in starts:
        if time.time() > deadline - 30: # Leave 30s for direct refinement
            break
        try:
            result = minimize(
                _objective_fourier, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if result.fun < best_score:
                h_new = np.full(N, 0.5)
                for k in range(n_modes):
                    h_new += result.x[2*k] * np.cos(np.pi * k * x)
                    h_new += result.x[2*k+1] * np.sin(np.pi * k * x)
                h_new = _project(h_new)
                actual_score = _compute_max_overlap(h_new)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_new.copy()
        except Exception:
            pass
    
    # Phase 2: Direct Space Refinement with Projected Nelder-Mead
    # We take the best solution from Phase 1 and try to improve it 
    # by allowing non-smooth variations.
    
    # We will run Nelder-Mead on the full 100D vector.
    # To make it faster, we might restrict the "step" size or iterations.
    
    current_h = best_h.copy()
    current_score = best_score
    
    # Initialize simplex for Nelder-Mead
    # Standard Nelder-Mead creates a simplex around x0.
    # We need to ensure the initial simplex is feasible or close.
    # The _project function will handle feasibility in the objective.
    
    # We can also try a few random perturbations of the best_h to start Nelder-Mead
    # from slightly different feasible points to avoid getting stuck.
    
    refinement_starts = [current_h]
    
    # Add a few perturbed versions
    np.random.seed(123)
    for _ in range(3):
        perturbed = current_h + np.random.uniform(-0.05, 0.05, N)
        perturbed = _project(perturbed)
        refinement_starts.append(perturbed)
        
    for h_start in refinement_starts:
        if time.time() > deadline - 5:
            break
            
        try:
            # Nelder-Mead parameters
            # initial_simplex: if None, it creates one based on x0.
            # We let it create one, but we must ensure the objective handles projection.
            
            result = minimize(
                _objective_direct, 
                h_start, 
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
            )
            
            h_new = _project(result.x)
            new_score = _compute_max_overlap(h_new)
            
            if new_score < current_score:
                current_score = new_score
                current_h = h_new
        except Exception:
            pass
    
    # Phase 3: Final Polish with Coordinate Descent
    # A quick coordinate descent can clean up any remaining errors.
    
    h_final = current_h
    score_final = current_score
    
    for _ in range(50): # Limit iterations
        if time.time() > deadline - 1:
            break
        improved = False
        for j in range(N):
            old_val = h_final[j]
            # Try small adjustments
            for delta in [-0.01, 0.01, -0.05, 0.05]:
                h_try = h_final.copy()
                h_try[j] = old_val + delta
                h_try = _project(h_try)
                s = _compute_max_overlap(h_try)
                if s < score_final - 1e-8:
                    h_final = h_try
                    score_final = s
                    improved = True
                    break
        if not improved:
            break
    
    # Final validation
    h_final = _project(h_final)
    if abs(np.sum(h_final) - N/2) > 1e-6:
        h_final = np.full(N, 0.5)
        
    return h_final
# EVOLVE-BLOCK-END
