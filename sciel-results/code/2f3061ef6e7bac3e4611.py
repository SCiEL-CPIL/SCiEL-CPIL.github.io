"""Parametric block-pulse optimization via Differential Evolution."""

import numpy as np
import time
from scipy.optimize import differential_evolution

N = 400

def _autoconv(x):
    """Compute full linear autoconvolution of x (length N) -> length 2N-1."""
    n = 2 * N
    X = np.fft.fft(x, n=n)
    C = np.fft.ifft(X * X).real[:2*N-1]
    return C

def _eval_ratio(x):
    """Compute the raw value: max|c_k| / (sum x)^2 for k != 0."""
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    if len(c) > 1:
        max_sidelobe = np.max(np.abs(c[1:]))
    else:
        max_sidelobe = 0.0
    return max_sidelobe / (s * s)

def _blocks_to_signal(params, n_blocks):
    """
    Convert parametric block representation to signal.
    params: array of shape (3 * n_blocks,)
    Each block has (start, width, amplitude).
    start: continuous, clipped to [0, N-width]
    width: continuous, clipped to [1, N]
    amplitude: continuous, clipped to [-1, 1]
    """
    x = np.zeros(N)
    for i in range(n_blocks):
        start = params[3*i]
        width = params[3*i + 1]
        amp = params[3*i + 2]
        
        # Clip parameters to valid ranges
        width = np.clip(width, 1, N)
        start = np.clip(start, 0, N - width)
        
        # Integer indices for the block
        i_start = int(np.floor(start))
        i_end = int(np.floor(start + width))
        
        # Ensure indices are within bounds
        i_start = max(0, min(i_start, N-1))
        i_end = max(i_start, min(i_end, N))
        
        if i_start < i_end:
            x[i_start:i_end] += amp
            
    return x

def _objective(params, n_blocks, deadline):
    """
    Objective function for differential evolution.
    Returns the metric value to be minimized.
    """
    if time.time() > deadline:
        return np.inf
        
    x = _blocks_to_signal(params, n_blocks)
    val = _eval_ratio(x)
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial best: all +1
    best_x = np.ones(N, dtype=np.float64)
    best_val = _eval_ratio(best_x)
    
    # Try a few random block configurations as seeds
    np.random.seed(42)
    rng = np.random.default_rng(0)
    
    # Strategy: Use a small number of blocks (e.g., 5-10)
    # This allows for complex shapes but keeps dimensionality low (15-30 vars)
    
    n_blocks_options = [5, 8, 12]
    
    for n_blocks in n_blocks_options:
        if time.time() > deadline:
            break
            
        # Bounds for differential evolution
        # start: [0, N], width: [1, N], amp: [-1, 1]
        bounds = []
        for _ in range(n_blocks):
            bounds.append((0, N))       # start
            bounds.append((1, N))       # width
            bounds.append((-1, 1))      # amp
            
        # Initialize with random params
        init_params = np.array([
            np.random.uniform(0, N),
            np.random.uniform(1, N),
            np.random.uniform(-1, 1)
        ] * n_blocks)
        
        # Evaluate initial
        x_init = _blocks_to_signal(init_params, n_blocks)
        val_init = _eval_ratio(x_init)
        
        if val_init < best_val:
            best_val = val_init
            best_x = x_init.copy()
            
        # Run Differential Evolution
        # maxiter and popsize tuned for speed within time limit
        # We use 'best1bin' strategy by default in DE
        try:
            res = differential_evolution(
                _objective, 
                bounds, 
                args=(n_blocks, deadline),
                maxiter=50, 
                popsize=15,
                seed=42,
                tol=0.01,
                polish=False, # Disable polishing to save time, we'll do local refine
                init='sobol'
            )
            
            if res.success or res.fun < best_val:
                x_opt = _blocks_to_signal(res.x, n_blocks)
                val_opt = _eval_ratio(x_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_x = x_opt.copy()
        except Exception as e:
            pass
    
    # Local Refinement: 
    # The block representation might leave "gaps" or have suboptimal boundaries.
    # We can now try to refine the signal by allowing small continuous adjustments 
    # or simply rounding if we want discrete, but the task allows "signed step heights".
    # The prompt says "sequence of signed step heights". It doesn't explicitly restrict to +/-1.
    # "Signed step functions" usually implies piecewise constant.
    # Let's try a local gradient step on the continuous signal to further minimize.
    
    if time.time() < deadline - 20:
        # Simple local search on the continuous signal
        # We'll do a few steps of projected gradient descent on the raw signal
        # Objective: minimize max|autoconv[1:]| / sum(x)^2
        # This is non-smooth due to max and abs.
        # Instead, let's just try flipping small segments or adjusting amplitudes slightly.
        
        # Given the block structure, maybe just try to adjust the amplitudes of the blocks 
        # found by DE using a local optimizer?
        # Or simpler: take the best_x, and try to minimize via a few random perturbations of the block params
        # and re-evaluate.
        
        pass # Skipping complex local refine for now to ensure robustness and time safety.
        # The DE result with continuous amplitudes should be significantly better than +/-1 fixed.
        
    return best_x
