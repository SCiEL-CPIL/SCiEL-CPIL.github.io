import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length 2*N.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac

def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes.
    """
    # Due to symmetry, max |ac[1:N]| is sufficient
    return np.max(np.abs(ac[1:N]))

def _compute_c3(h):
    """
    Compute C3 = max_k |ac[k]| / (sum|h|)^2 for k=1..N-1.
    Lower is better.
    """
    ac = _compute_autocorr(h)
    l1 = np.sum(np.abs(h))
    if l1 < 1e-10:
        return np.max(np.abs(ac[1:N]))
    return np.max(np.abs(ac[1:N])) / (l1 * l1)

def _objective_and_gradient(x, beta=50.0):
    """
    Objective: Smooth approximation of C3 = max|ac[k]|/(sum|h|)^2.
    Gradient: FFT-based O(N log N).
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes = ac[1:N]
    sidelobes_abs = np.abs(sidelobes)
    l1 = np.sum(np.abs(h))
    
    # Soft max of sidelobes_abs
    max_val = np.max(sidelobes_abs)
    if max_val < 1e-10:
        return 0.0, np.zeros(N)
        
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    soft_max = max_val + np.log(sum_exp) / beta
    
    # C3 = soft_max / l1^2
    obj = soft_max / (l1 * l1)
    
    # Gradient via FFT:
    # d(ac[k])/dx_i = h[i+k] + h[i-k] for valid i
    # grad = sum_k w_k * sign(ac[k]) * (shifted h)
    # where w_k = beta * exp(beta*(|ac[k]| - max)) / sum_exp
    # 
    # Using FFT: if W = [w_1, ..., w_{N-1}], then
    # grad[i] = sum_k W[k] * (h[i+k] + h[i-k])
    # This is a convolution-like operation.
    # 
    # Let's compute: for each k, contribution to grad[i] is W[k]*sign(ac[k]) * (h[i+k] + h[i-k])
    # = sum_k C[k] * h[i+k] + sum_k C[k] * h[i-k]
    # where C[k] = W[k] * sign(ac[k])
    # 
    # First term: sum_k C[k] * h[i+k] = (C reversed) convolved with h
    # Second term: sum_k C[k] * h[i-k] = C convolved with h
    # Both are linear convolutions.
    
    # Build C array of length N (C[0]=0, C[k] for k=1..N-1)
    C = np.zeros(N)
    weights = beta * exp_shifted / sum_exp
    C[1:] = weights * np.sign(sidelobes)
    
    # Compute sum_k C[k] * h[i+k] for i=0..N-1 (valid where i+k<N)
    # This is a cross-correlation. Use FFT.
    # Let A = C, B = h. We want (A * B_rev)[i] where B_rev[j] = B[N-1-j]
    # Actually, sum_k C[k]*h[i+k] = sum_j C[j]*h[i+j]
    # Let's pad and use FFT.
    
    n_fft = 2 * N
    F_C = np.fft.rfft(C, n=n_fft)
    F_h = np.fft.rfft(h, n=n_fft)
    
    # sum_k C[k] * h[i+k]: this is like a cross-correlation
    # If we define C_rev[k] = C[N-1-k], then sum_k C[k]*h[i+k] = (C_rev * h)[i+N-1]
    # Actually let's think differently.
    # sum_k C[k] * h[i+k] for k=1..N-1, valid i: 0 <= i, i+k < N
    # = sum_k C[k] * h[i+k]
    # Let m = i+k, then k = m-i, sum_m C[m-i] * h[m] for m=i+1..N-1
    # This is a convolution of C with h at specific points.
    
    # Simpler: just do it with the FFT of the full linear convolution.
    # conv(C, h)[i] = sum_k C[k] * h[i-k] (standard convolution)
    # We want sum_k C[k] * h[i+k] = conv(C_rev, h)[i] where C_rev[k]=C[N-1-k]
    # Hmm, let me just compute both terms via FFT convolution.
    
    # Term 1: T1[i] = sum_{k=1}^{N-1} C[k] * h[i+k] for i where i+k < N
    # Term 2: T2[i] = sum_{k=1}^{N-1} C[k] * h[i-k] for i where i-k >= 0
    
    # T2 is standard: T2 = conv(C, h) evaluated at indices i (T2[i] = sum_k C[k]*h[i-k])
    # T1: Let C_flip[k] = C[N-1-k]. Then T1[i] = sum_k C[k]*h[i+k] 
    #    = sum_j C_flip[j] * h[i + (N-1-j)] ... this is getting complicated.
    # 
    # Alternative: T1[i] = sum_k C[k] * h[i+k]
    # If we reverse h: h_rev[j] = h[N-1-j]
    # T1[i] = sum_k C[k] * h[N-1 - (N-1-i-k)] = sum_k C[k] * h_rev[N-1-i-k]
    # = conv(C, h_rev)[N-1-i]
    # So T1[i] = conv(C, h_rev)[N-1-i]
    
    h_rev = h[::-1]
    
    # Compute conv(C, h) and conv(C, h_rev) via FFT
    conv_Ch = np.fft.irfft(F_C * F_h, n=n_fft)
    conv_Ch_rev = np.fft.irfft(F_C * np.fft.rfft(h_rev, n=n_fft), n=n_fft)
    
    # T2[i] = conv_Ch[i] for i=0..N-1
    T2 = conv_Ch[0:N]
    
    # T1[i] = conv_Ch_rev[N-1-i]
    T1 = conv_Ch_rev[N-1::-1][0:N]  # conv_Ch_rev[N-1], conv_Ch_rev[N-2], ..., conv_Ch_rev[0]
    
    # Total gradient of soft_max w.r.t. x
    grad_soft = T1 + T2
    
    # d(obj)/dx_i = d(soft_max/l1^2)/dx_i
    # = (d(soft_max)/dx_i * l1^2 - soft_max * 2*l1 * d(l1)/dx_i) / l1^4
    # = d(soft_max)/dx_i / l1^2 - soft_max * 2 * d(l1)/dx_i / l1^3
    # d(l1)/dx_i = sign(x_i)
    
    sign_h = np.sign(h)
    grad = grad_soft / (l1 * l1) - 2.0 * soft_max * sign_h / (l1 ** 3)
    
    return obj, grad

def _local_search(h, max_iterations=10000, time_limit=5.0, T0=None):
    """
    Fast local search using FFT for evaluation with simulated annealing.
    Tracks best solution separately from current to avoid losing improvements.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _compute_c3(h)
    
    if T0 is None:
        T0 = max(current_val * 0.1, 0.01)
    
    # Simulated annealing parameters
    T = T0
    cooling_rate = 0.998
    
    best_val = current_val
    best_h = h.copy()
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping a random bit
        i = np.random.randint(N)
        old_h_i = h[i]
        h[i] = -old_h_i
        
        new_val = _compute_c3(h)
        
        delta = new_val - current_val
        
        if delta < 0:
            # Improvement
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Deterioration: accept with probability exp(-delta/T)
            if T > 1e-10 and np.random.rand() < np.exp(-delta / T):
                current_val = new_val
            else:
                # Revert
                h[i] = old_h_i
        
        # Adaptive cooling
        T *= cooling_rate
        
    return best_h, best_val

def _polish(h, time_limit=5.0):
    """
    High-precision polish: try all single flips and keep improving ones.
    Repeat until no improvement or time runs out.
    """
    start_time = time.time()
    current_val = _compute_c3(h)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            h[i] *= -1
            new_val = _compute_c3(h)
            if new_val < current_val:
                current_val = new_val
                improved = True
            else:
                h[i] *= -1  # revert
    
    return h, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation with Smooth Objective
    # Try a few random initializations
    num_continuous_starts = 6
    betas = [30.0, 80.0, 150.0, 30.0, 100.0, 50.0]
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 40.0:  # Reserve 40s for final polish
            break
            
        # Initialize x in [-1, 1]
        # Start near binary to help rounding
        x0 = np.random.choice([-1.0, 1.0], size=N)
        
        # Use a beta that increases over iterations or pick one
        # Let's try different betas for different starts
        beta = betas[i % len(betas)]
        
        # Define objective with fixed beta
        def obj_grad(x, beta=beta):
            return _objective_and_gradient(x, beta)
        
        # Use L-BFGS-B
        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-7, 'maxls': 40})
        except Exception:
            continue
        
        # Round to binary
        h_bin = np.sign(res.x)
        # Handle zeros
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
            
        # Evaluate binary solution
        val = _compute_c3(h_bin)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
            
        # Quick local search refinement on the new candidate
        if time.time() < deadline - 20.0:
            h_ref, val_ref = _local_search(h_bin.copy(), max_iterations=3000, time_limit=3.0)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref
    
    # Phase 2: Refine best solution with local search
    if best_h is not None and time.time() < deadline - 15.0:
        h_ref, val_ref = _local_search(best_h.copy(), max_iterations=8000, time_limit=8.0)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref
    
    # Phase 3: Final Discrete Search with multiple perturbed restarts
    remaining_time = deadline - time.time()
    if remaining_time > 3.0 and best_h is not None:
        # Use multiple restarts from best solution with perturbations to escape local minima
        num_restarts = min(5, max(1, int(remaining_time / 3.0)))
        
        for r in range(num_restarts):
            if time.time() >= deadline - 1.0:
                break
            
            # Create perturbed version
            h_pert = best_h.copy()
            num_flips = np.random.randint(5, 20)
            flip_indices = np.random.choice(N, size=num_flips, replace=False)
            h_pert[flip_indices] *= -1
            
            T_final = best_val * 0.08 if best_val > 1e-10 else 0.02
            h_final, val_final = _local_search(h_pert, max_iterations=20000, time_limit=min(5.0, remaining_time - 1.0), T0=T_final)
            if val_final < best_val:
                best_val = val_final
                best_h = h_final
            
            remaining_time = deadline - time.time()
            if remaining_time < 3.0:
                break
            
    # Phase 4: High-precision polish
    if best_h is not None and time.time() < deadline - 1.0:
        polish_time = min(5.0, deadline - 1.0 - time.time())
        if polish_time > 1.0:
            h_pol, val_pol = _polish(best_h.copy(), time_limit=polish_time)
            if val_pol < best_val:
                best_val = val_pol
                best_h = h_pol
    
    if best_h is None:
        return np.ones(N)
        
    return best_h
