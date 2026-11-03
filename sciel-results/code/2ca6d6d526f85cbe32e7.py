import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute max(autoconv) / ||h||_2^2 using FFT on fine grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    rng = np.random.default_rng(42)
    
    # === Phase 1: Block-based optimization ===
    # Represent h as piecewise constant over B blocks
    # Optimize block heights using L-BFGS-B
    
    for n_blocks in [10, 15, 20, 30]:
        if time.time() > deadline - 30.0:
            break
        
        block_size = N // n_blocks
        block_indices = np.arange(n_blocks) * block_size + np.arange(block_size)[:None, :]
        block_indices = block_indices.ravel()
        
        def blocks_to_h(block_heights):
            h = np.zeros(N)
            for b in range(n_blocks):
                start = b * block_size
                end = start + block_size
                h[start:end] = max(0.0, block_heights[b])
            return h
        
        def block_objective(block_heights):
            h = blocks_to_h(block_heights)
            return ratio_from_h(h)
        
        # Multiple random restarts in block space
        for restart in range(5):
            if time.time() > deadline - 10.0:
                break
            
            # Initialize with different strategies
            if restart == 0:
                init = np.ones(n_blocks)
            elif restart == 1:
                init = np.exp(-np.linspace(0, 3, n_blocks))
            elif restart == 2:
                init = 0.5 + 0.5 * np.cos(np.pi * np.linspace(-1, 1, n_blocks))
            elif restart == 3:
                init = rng.uniform(0.1, 1.0, n_blocks)
            else:
                init = np.exp(-2.0 * (np.linspace(-1, 1, n_blocks))**2)
            
            try:
                result = opt.minimize(
                    block_objective,
                    init,
                    method='L-BFGS-B',
                    bounds=[(0.0, 5.0)] * n_blocks,
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if result.fun < best_val:
                    h_opt = blocks_to_h(result.x)
                    update_best(h_opt, result.fun)
            except Exception:
                pass
    
    if time.time() > deadline - 15.0:
        return list(best_h)
    
    # === Phase 2: Local refinement via coordinate descent ===
    for restart in range(5):
        if time.time() > deadline - 5.0:
            break
        
        h_cur = best_h.copy()
        val_cur = best_val
        
        # Add small noise
        noise = rng.normal(0, 0.03, N)
        h_cur = np.clip(h_cur + noise, 0, None)
        val_cur = ratio_from_h(h_cur)
        
        # Coordinate descent with larger steps
        for it in range(500):
            if time.time() > deadline - 2.0:
                break
            
            idx = rng.integers(0, N)
            current_val = h_cur[idx]
            
            # Try a range of candidates
            candidates = np.concatenate([
                [0.0, 1.0],
                current_val * np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.5]),
                current_val + np.array([-0.2, -0.1, 0.1, 0.2])
            ])
            candidates = np.clip(candidates, 0, 5.0)
            
            best_candidate = h_cur[idx]
            best_candidate_val = val_cur
            
            for cand in candidates:
                h_try = h_cur.copy()
                h_try[idx] = cand
                v_try = ratio_from_h(h_try)
                if v_try < best_candidate_val:
                    best_candidate_val = v_try
                    best_candidate = cand
            
            if best_candidate != h_cur[idx]:
                h_cur[idx] = best_candidate
                val_cur = best_candidate_val
        
        if val_cur < best_val:
            update_best(h_cur, val_cur)
    
    # === Phase 3: Try analytical shapes ===
    if time.time() < deadline - 2.0:
        x_norm = np.linspace(-1.0, 1.0, N)
        
        shapes = []
        for width in [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
            shapes.append(np.exp(-width * x_norm**2))
        
        shapes.append(0.5 + 0.5 * np.cos(np.pi * x_norm))
        shapes.append(np.clip(0.42 + 0.5 * np.cos(np.pi * x_norm) + 0.08 * np.cos(2 * np.pi * x_norm), 0, None))
        shapes.append(1.0 - np.abs(x_norm))
        shapes.append(np.ones(N))
        shapes.append(np.exp(-3.0 * x_norm**2) * (1.0 + 0.3 * np.cos(4 * np.pi * x_norm)))
        
        for h_shape in shapes:
            val = ratio_from_h(h_shape)
            update_best(h_shape, val)
    
    return list(best_h)
