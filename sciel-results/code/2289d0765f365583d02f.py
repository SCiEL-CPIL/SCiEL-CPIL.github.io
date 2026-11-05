"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT (faster than np.convolve for large N)."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _objective_exact(h):
    """Exact objective using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses Simulated Annealing in the time domain to globally minimize the
    peak-to-mean ratio of the autoconvolution.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    # Initialize with all ones
    best_h = np.ones(N)
    best_val = _eval_value(best_h)
    
    # Phase 1: Quick parametric initialization to provide a good starting point
    # This helps SA converge faster by starting near a good region
    for m in range(1, 11):
        if time.time() > deadline - 100:
            break
        for a in np.linspace(-0.5, 0.5, 11):
            h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # Phase 2: Simulated Annealing
    # We optimize h directly. The objective is scale-invariant (value depends on h/sum(h)),
    # so we can fix the sum to 0 or N to reduce dimensionality? 
    # Actually, the value is invariant to scaling h by a constant.
    # value(h) = value(k*h) for k != 0.
    # So we can constrain sum(h) = N (mean = 1) to reduce one degree of freedom.
    # However, SA works well in full space. Let's just run SA on h.
    
    current_h = best_h.copy()
    current_val = best_val
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.9995 # Cooling factor
    steps_per_temp = 50
    
    # Estimate time per step
    # One step involves: 
    # 1. Generate perturbation
    # 2. Evaluate objective (FFT + max)
    # 3. Accept/Reject
    
    # We need to adapt steps_per_temp based on time budget
    # Let's run a small benchmark to estimate time per step
    start_time = time.time()
    for _ in range(10):
        h_test = current_h + np.random.randn(N) * 0.1
        _ = _objective_exact(h_test)
    elapsed = time.time() - start_time
    time_per_step = elapsed / 10.0
    
    # Total steps we can afford
    time_budget = deadline - time.time() - 5.0 # 5s safety margin
    max_steps = int(time_budget / time_per_step)
    
    # Adjust steps_per_temp if necessary
    # If max_steps is very small, we just do what we can
    if max_steps < 100:
        return best_h

    # We will run SA for max_steps iterations
    # We need to adjust T schedule to fit max_steps
    # T = T_start * alpha^k
    # We want T_end at k = max_steps
    # T_end = T_start * alpha^max_steps
    # alpha = (T_end / T_start)^(1/max_steps)
    
    # But T_end depends on how fast we cool. Let's fix alpha and see where T ends up.
    # Or fix T_end and compute alpha.
    # Let's use a fixed alpha and see if we finish in time.
    # If we run out of time, we return best_h.
    
    # To be safe, let's calculate alpha such that T reaches T_end at max_steps
    if max_steps > 0:
        alpha = (T_end / T_start) ** (1.0 / max_steps)
    else:
        alpha = 0.999
    
    T = T_start
    
    rng = np.random.RandomState(42)
    
    # Perturbation strategy:
    # 1. Random single index change
    # 2. Random block change
    # 3. Gaussian noise
    
    for step in range(max_steps):
        if time.time() > deadline - 5.0:
            break
            
        # Generate perturbation
        # Mix of local and global moves
        move_type = rng.randint(3)
        
        if move_type == 0:
            # Single index random walk
            idx = rng.randint(N)
            delta = rng.randn() * 0.5
            new_h = current_h.copy()
            new_h[idx] += delta
        elif move_type == 1:
            # Block update
            start_idx = rng.randint(N)
            block_len = rng.randint(1, 10)
            end_idx = min(N, start_idx + block_len)
            deltas = rng.randn(end_idx - start_idx) * 0.5
            new_h = current_h.copy()
            new_h[start_idx:end_idx] += deltas
        else:
            # Global small noise
            new_h = current_h + rng.randn(N) * 0.1
        
        # Evaluate
        if not np.all(np.isfinite(new_h)):
            T *= alpha
            continue
            
        new_val = _objective_exact(new_h)
        
        if not np.isfinite(new_val):
            T *= alpha
            continue
            
        # Acceptance criterion
        if new_val < current_val:
            # Better solution
            current_val = new_val
            current_h = new_h
            
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        else:
            # Worse solution, accept with probability exp(-delta / T)
            delta = new_val - current_val
            if delta > 0:
                # Avoid overflow in exp
                if T > 1e-9:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        current_val = new_val
                        current_h = new_h
                # else T is too small, effectively 0 probability
                
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end

    # Phase 3: Local polish with coordinate descent
    # Use remaining time to refine best_h
    if time.time() < deadline - 2.0:
        h = best_h.copy()
        step_size = 0.1
        for iteration in range(20):
            if time.time() > deadline - 1.0:
                break
            improved = False
            # Try small perturbations on random indices
            indices = rng.permutation(N)
            for i in indices:
                if time.time() > deadline - 1.0:
                    break
                old = h[i]
                for delta in [step_size, -step_size]:
                    h[i] = old + delta
                    if not np.all(np.isfinite(h)):
                        h[i] = old
                        continue
                    v = _objective_exact(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
                        improved = True
                    else:
                        h[i] = old
            if not improved:
                step_size *= 0.5
                if step_size < 1e-4:
                    break

    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
