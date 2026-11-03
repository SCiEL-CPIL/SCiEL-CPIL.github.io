import numpy as np
import time

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute for fast evaluation
# We use a slightly finer grid for convolution to be accurate, but N=600 is the constraint.
# Actually, the task says "step function with 600 equal intervals".
# The autoconvolution of a step function is a piecewise linear function.
# Its maximum occurs at one of the breakpoints or in the middle of an interval.
# However, computing it exactly via FFT on the step values is fast enough.
# Let's use FFT on the step values directly, but we need to handle the convolution length.
# If h is length N, conv is length 2N-1.
# The ratio is max(conv) / sum(h^2 * W^2)? No, L2 norm of step function is sqrt(sum(h_i^2 * W)).
# Let's stick to the previous program's evaluation which was verified.

FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
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
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    # Strategy: 
    # 1. Initialize with several diverse candidates (not just Gaussian).
    #    - Constant
    #    - Two-sided Gaussian
    #    - "Bump" (compact support)
    #    - Random noise
    # 2. Use a block-based local search.
    #    Divide the 600 intervals into K blocks (e.g., 30 blocks of 20 intervals).
    #    Optimize the height of each block. This reduces dimensionality from 600 to 30,
    #    allowing for much more effective local search (Nelder-Mead or simple gradient-free).
    #    Then, refine within blocks if time permits.
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    # Candidate generators
    def gen_constant():
        return np.ones(N) * 1.0
        
    def gen_gaussian(center, width, height):
        g = np.exp(-0.5 * ((X - center) / width) ** 2)
        return height * g + 0.1
        
    def gen_bump(center, width, height):
        mask = np.abs(X - center) < width
        h = np.zeros(N)
        h[mask] = height
        return h
        
    def gen_random():
        return np.random.rand(N) * 2.0
        
    candidates = [
        gen_constant(),
        gen_gaussian(0, 0.05, 5.0),
        gen_gaussian(0, 0.1, 2.0),
        gen_bump(0, 0.05, 10.0),
        gen_bump(0, 0.1, 5.0),
        gen_random()
    ]
    
    for h in candidates:
        v = ratio_from_h(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Block-based optimization
    # Let's use 24 blocks of 25 intervals each.
    # This allows us to change the "shape" significantly.
    K_BLOCKS = 24
    BLOCK_SIZE = N // K_BLOCKS # 25
    
    def h_to_blocks(h):
        blocks = []
        for i in range(K_BLOCKS):
            start = i * BLOCK_SIZE
            end = (i + 1) * BLOCK_SIZE
            if i == K_BLOCKS - 1:
                end = N
            blocks.append(h[start:end].mean())
        return np.array(blocks)
        
    def blocks_to_h(blocks):
        h = np.zeros(N)
        for i in range(K_BLOCKS):
            start = i * BLOCK_SIZE
            end = (i + 1) * BLOCK_SIZE
            if i == K_BLOCKS - 1:
                end = N
            h[start:end] = blocks[i]
        return h

    current_blocks = h_to_blocks(best_h)
    current_val = best_val
    
    # Simple Nelder-Mead implementation on the block heights
    # We need to handle non-negativity.
    
    def objective(blocks):
        h = blocks_to_h(np.clip(blocks, 0.0, None))
        return ratio_from_h(h)
        
    # Initialize simplex
    dim = K_BLOCKS
    simplex = np.zeros((dim + 1, dim))
    simplex[0] = current_blocks
    for i in range(1, dim + 1):
        simplex[i] = current_blocks.copy()
        simplex[i][i-1] += 0.1 * max(1.0, abs(current_blocks[i-1]))
        
    fvals = np.array([objective(simplex[i]) for i in range(dim + 1)])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    max_iter = 2000
    for it in range(max_iter):
        if time.time() > deadline - 10.0:
            break
            
        # Sort simplex
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        # Check convergence
        if np.max(fvals) - np.min(fvals) < 1e-8:
            break
            
        x_avg = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = x_avg + alpha * (x_avg - simplex[-1])
        f_r = objective(x_r)
        
        if f_r < fvals[0]:
            # Expansion
            x_e = x_avg + gamma * (x_r - x_avg)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction
            if f_r < fvals[-1]:
                x_c = x_avg + rho * (x_r - x_avg)
            else:
                x_c = x_avg - rho * (simplex[-1] - x_avg)
            f_c = objective(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, dim + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
        
        # Update best
        if fvals[0] < best_val:
            best_val = fvals[0]
            best_h = blocks_to_h(np.clip(simplex[0], 0.0, None))

    # If time remains, do fine-grained coordinate descent on the full 600 dims
    # starting from the block-optimized solution.
    if time.time() < deadline - 5.0:
        current_h = best_h.copy()
        current_v = best_val
        rng = np.random.default_rng(42)
        
        # Use a smaller step size for fine tuning
        # We will iterate through coordinates in random order, or random blocks of coordinates.
        # To speed up, we can update multiple coordinates at once if they are adjacent?
        # No, FFT is the bottleneck. 
        # Let's try to reduce the cost of evaluation.
        # Actually, 600 points is small. FFT of size 16384 (next power of 2 for 2*8000) is fast.
        # Let's just do random coordinate updates.
        
        for _ in range(5000):
            if time.time() > deadline - 2.0:
                break
            idx = rng.integers(0, N)
            h_try = current_h.copy()
            # Random perturbation
            h_try[idx] += rng.normal(0, 0.1)
            if h_try[idx] < 0:
                h_try[idx] = 0
            v_try = ratio_from_h(h_try)
            if v_try < current_v:
                current_h = h_try
                current_v = v_try
                if current_v < best_val:
                    best_val = current_v
                    best_h = current_h.copy()
                    
    return list(best_h)
