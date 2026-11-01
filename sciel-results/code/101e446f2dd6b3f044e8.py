import numpy as np
import time

N = 400

def _compute_score(h):
    """Compute the C3 upper bound: peak of autoconvolution / ||h||_1^2"""
    g = np.convolve(h, h, mode='full')
    l1_norm = np.sum(np.abs(h))
    if l1_norm < 1e-9:
        return np.inf
    return np.max(np.abs(g)) / (l1_norm ** 2)

def _optimize_continuous(h_init, deadline):
    """Optimize using smooth surrogate with proper FFT gradient"""
    h = h_init.copy()
    best_h = h.copy()
    best_score = _compute_score(h)
    
    epsilon = 0.5
    lr = 0.05
    max_iters = 3000
    size_fft = 1 << (2*N - 1).bit_length()  # Next power of 2 >= 2N-1
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute autoconvolution via FFT
        H = np.fft.fft(h, n=size_fft)
        G = np.fft.ifft(H * H).real[:2*N-1]
        
        l1_norm = np.sum(np.abs(h))
        if l1_norm < 1e-6:
            h = np.random.choice([-1.0, 1.0], size=N)
            continue
        
        current_score = np.max(np.abs(G)) / (l1_norm ** 2)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
        
        # Soft-max gradient
        abs_g = np.abs(G)
        max_abs_g = np.max(abs_g)
        if max_abs_g < 1e-12:
            break
        
        exp_term = np.exp((abs_g - max_abs_g) / epsilon)
        sum_exp = np.sum(exp_term)
        if sum_exp < 1e-9:
            break
        
        weights = exp_term / sum_exp
        dL_dg = (1.0 / epsilon) * weights * np.sign(G)
        
        # FFT gradient: grad_h = ifft(2 * FFT(dL_dg) * FFT(h))
        DL = np.fft.fft(dL_dg, n=size_fft)
        Grad_H = 2 * DL * H
        grad_h = np.fft.ifft(Grad_H).real[:N]
        
        # Normalize gradient by current score scale
        grad_norm = np.linalg.norm(grad_h)
        if grad_norm > 1e-12:
            h = h - lr * grad_h / grad_norm * max_abs_g
        
        # Soft threshold to encourage sparsity/signed structure
        h[np.abs(h) < 0.1] *= 0.9
        
        # Normalize
        l1_norm = np.sum(np.abs(h))
        if l1_norm > 1e-9:
            h = h / l1_norm
        else:
            h = np.random.choice([-1.0, 1.0], size=N)
        
        # Schedule
        if i < 500:
            epsilon *= 0.995
        else:
            epsilon *= 0.999
        epsilon = max(epsilon, 0.01)
        
        if i > 500:
            lr *= 0.9995
    
    return best_h, best_score

def _discrete_refine(h, deadline):
    """Local search on discrete ±1 signs"""
    h = np.sign(h)
    if np.all(h == 0):
        h = np.random.choice([-1.0, 1.0], size=N)
    
    best_h = h.copy()
    best_score = _compute_score(h)
    
    # Greedy coordinate descent
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            # Try flipping sign
            h_new = h.copy()
            h_new[i] *= -1
            score_new = _compute_score(h_new)
            if score_new < best_score - 1e-10:
                h = h_new
                best_h = h.copy()
                best_score = score_new
                improved = True
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    best_h = None
    best_score = np.inf
    
    # Multi-start with diverse initializations
    n_starts = 8
    starts_per_start = (deadline - start_time) / n_starts
    
    for s in range(n_starts):
        if time.time() > deadline:
            break
        
        sub_deadline = min(deadline - 5.0, start_time + (s+1) * starts_per_start)
        
        # Generate diverse initializations
        if s == 0:
            h_init = np.random.choice([-1.0, 1.0], size=N)
        elif s == 1:
            # Alternating pattern
            h_init = np.ones(N)
            h_init[1::2] = -1.0
        elif s == 2:
            # Block pattern
            h_init = np.ones(N)
            h_init[N//4:3*N//4] = -1.0
        elif s == 3:
            # Sine-based
            x = np.linspace(0, 4*np.pi, N)
            h_init = np.sign(np.sin(x))
            h_init[h_init == 0] = 1.0
        elif s == 4:
            # Random with bias
            h_init = np.random.choice([-1.0, 1.0], size=N)
            h_init[:N//2] *= -1
        elif s == 5:
            # Walsh-Hadamard-like
            h_init = np.ones(N)
            for k in range(1, N, 2):
                h_init[k::2] *= -1
        elif s == 6:
            # Random permutation of blocks
            block_size = 10
            n_blocks = N // block_size
            blocks = np.random.choice([-1.0, 1.0], size=n_blocks)
            h_init = np.repeat(blocks, block_size)[:N]
        else:
            h_init = np.random.choice([-1.0, 1.0], size=N)
        
        # Continuous optimization
        h_opt, score_opt = _optimize_continuous(h_init, sub_deadline)
        
        # Discrete refinement
        h_disc, score_disc = _discrete_refine(h_opt, deadline - 2.0)
        
        if score_disc < best_score:
            best_score = score_disc
            best_h = h_disc.copy()
    
    if best_h is None:
        best_h = np.random.choice([-1.0, 1.0], size=N)
    
    return best_h

# EVOLVE-BLOCK-END
