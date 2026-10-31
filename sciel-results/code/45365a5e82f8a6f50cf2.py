import numpy as np
import time

N = 50
DEADLINE_MARGIN = 5.0  # Return 5s before the 150s kill

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    The ratio is homogeneous of degree 0, so scaling h does not change the value.
    We assume h is non-negative.
    """
    h = np.asarray(h, dtype=np.float64)
    # Enforce non-negativity for safety
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    # Normalize to sum=1 to improve numerical stability and simplify ratio
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    
    # Full convolution
    # Using np.convolve for exactness. N=50 is small.
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    
    if conv_inf < 1e-15:
        return 0.0
        
    return h2_sq / conv_inf

def smooth_max(x, beta=100.0):
    """
    Smooth approximation of max using softmax.
    As beta -> infinity, this approaches the true max.
    """
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    exp_x = np.exp(beta * (x - x_max))
    return x_max + np.log(np.sum(exp_x)) / beta

def spectral_objective(h, beta=100.0):
    """
    Compute -ratio using smooth max for differentiability.
    h is normalized to sum=1 and non-negative.
    """
    h = np.maximum(h, 0.0)
    s = np.sum(h)
    if s < 1e-15:
        return 1e10
    h_norm = h / s
    h2_sq = np.sum(h_norm**2)
    
    # Compute autoconvolution via FFT for efficiency
    n_conv = 2 * N - 1
    H = np.fft.rfft(h_norm, n=n_conv)
    conv = np.fft.irfft(H * H, n=n_conv)
    
    # Use smooth max instead of hard max
    conv_smooth = smooth_max(conv, beta=beta)
    
    if conv_smooth < 1e-15:
        return 1e10
        
    return -h2_sq / conv_smooth

def spectral_projected_gradient_descent(h_init, t0, deadline, max_iter=5000, lr_init=0.01):
    """
    Projected gradient descent with adaptive learning rate in the simplex.
    Uses spectral computation for efficiency.
    """
    h = np.array(h_init, dtype=np.float64)
    h = np.maximum(h, 0.0)
    s = np.sum(h)
    if s < 1e-15:
        return h
    h = h / s
    
    best_h = h.copy()
    best_score = compute_ratio(h)
    
    lr = lr_init
    beta = 50.0  # Start with moderate smoothness
    
    for it in range(max_iter):
        if time.time() > deadline - 2:
            break
        
        # Compute gradient of spectral_objective
        # Use finite differences for simplicity (N=50 is small)
        grad = np.zeros_like(h)
        eps = 1e-7
        f0 = spectral_objective(h, beta=beta)
        
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            # Project back to simplex
            h_plus = np.maximum(h_plus, 0.0)
            s_plus = np.sum(h_plus)
            if s_plus > 1e-15:
                h_plus = h_plus / s_plus
            grad[i] = (spectral_objective(h_plus, beta=beta) - f0) / eps
        
        # Projected gradient step
        h_new = h - lr * grad
        
        # Project to simplex: h >= 0, sum(h) = 1
        h_new = np.maximum(h_new, 0.0)
        s_new = np.sum(h_new)
        if s_new < 1e-15:
            h_new = h.copy()
        else:
            h_new = h_new / s_new
        
        # Adaptive learning rate
        f_new = spectral_objective(h_new, beta=beta)
        f_old = f0
        
        if f_new < f_old:  # Improvement (remember we're minimizing negative)
            h = h_new
            lr = min(lr * 1.05, 0.1)
            # Increase smoothness as we converge
            beta = min(beta * 1.02, 500.0)
        else:
            lr = max(lr * 0.5, 1e-5)
        
        # Update best
        sc = compute_ratio(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
    
    return best_h

def construct():
    t0 = time.time()
    deadline = t0 + 150.0 - DEADLINE_MARGIN
    
    def elapsed():
        return time.time() - t0
        
    def time_left():
        return deadline - time.time()

    # Initial best: single spike
    best_h = np.zeros(N, dtype=np.float64)
    best_h[N // 2] = 1.0
    best_score = compute_ratio(best_h)
    
    candidates = []
    
    # Helper to add candidate
    def add_candidate(h):
        nonlocal best_h, best_score
        h = np.maximum(h, 0.0)
        if np.sum(h) < 1e-15:
            return
        sc = compute_ratio(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        candidates.append((sc, h.copy()))
        
    # --- Phase 1: Generate Structured Candidates ---
    
    # 1. Single spike (already set)
    add_candidate(best_h)
    
    # 2. Two spikes
    for i in range(N):
        for j in range(i+1, N):
            h = np.zeros(N)
            h[i] = 0.5
            h[j] = 0.5
            add_candidate(h)
            if elapsed() > deadline - 5:
                return list(best_h)
                
    # 3. Three spikes
    # Sample a subset of triplets to save time
    # Focus on "spread out" triplets
    for i in range(0, N, 2):
        for j in range(i+2, N, 2):
            for k in range(j+2, N, 2):
                h = np.zeros(N)
                h[i] = 1.0/3
                h[j] = 1.0/3
                h[k] = 1.0/3
                add_candidate(h)
                if elapsed() > deadline - 5:
                    return list(best_h)

    # 4. Sidon-like sets
    # A Sidon set ensures all pair sums are distinct, minimizing collisions in convolution.
    # This tends to lower ||h*h||_inf relative to ||h||_2^2.
    
    def find_sidon_set_greedy(n, start_idx=0, step=1):
        """
        Greedy construction of a Sidon set.
        """
        A = []
        used_sums = set()
        # We want to pick indices from 0 to N-1
        for x in range(start_idx, N, step):
            valid = True
            new_sums = []
            for a in A:
                s = a + x
                if s in used_sums:
                    valid = False
                    break
                new_sums.append(s)
            if 2 * x in used_sums:
                valid = False
            if valid:
                A.append(x)
                used_sums.update(new_sums)
                used_sums.add(2 * x)
        return A

    # Try different steps and starts to find diverse Sidon sets
    for step in [1, 2, 3]:
        for start in range(0, 5, step):
            A = find_sidon_set_greedy(N, start_idx=start, step=step)
            if len(A) >= 2:
                h = np.zeros(N)
                for idx in A:
                    h[idx] = 1.0
                add_candidate(h)
                if elapsed() > deadline - 5:
                    return list(best_h)

    # 5. Random Sparse Sets
    rng = np.random.default_rng(42)
    # Increase number of random restarts and vary sparsity
    for _ in range(80):
        k = rng.integers(2, 20) # Wider range of sparsity
        idx = rng.choice(N, size=k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        add_candidate(h)
        if elapsed() > deadline - 5:
            return list(best_h)

    # 6. Uniform
    add_candidate(np.ones(N))
    
    # 7. Gaussian
    x = np.arange(N)
    for sigma in [2, 5, 10, 20]:
        h = np.exp(-0.5 * ((x - N/2) / sigma)**2)
        add_candidate(h)

    # 8. Structured Sparse Patterns (e.g., arithmetic progressions)
    for start in range(0, N, 5):
        for step in range(2, 12):
            indices = list(range(start, N, step))
            if len(indices) >= 2:
                h = np.zeros(N)
                h[indices] = 1.0
                add_candidate(h)
                if elapsed() > deadline - 5:
                    return list(best_h)

    # 9. Quadratic residue-like patterns
    for base in [2, 3, 5, 7]:
        h = np.zeros(N)
        for i in range(N):
            if (i * i) % base == 0:
                h[i] = 1.0
        if np.sum(h) >= 2:
            add_candidate(h)
        if elapsed() > deadline - 5:
            return list(best_h)

    # 10. Fibonacci-like spacing
    fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34]
    h = np.zeros(N)
    pos = 0
    for f in fibs:
        if pos < N:
            h[pos] = 1.0
            pos += f
    if np.sum(h) >= 2:
        add_candidate(h)
    if elapsed() > deadline - 5:
        return list(best_h)

    # Sort candidates by score, keep top few for refinement
    candidates.sort(key=lambda x: -x[0])
    
    # --- Phase 2: Refinement using Spectral Projected Gradient Descent ---
    
    # We refine the top candidates using smooth spectral optimization.
    # The smooth max allows gradient-based optimization to work better.
    
    top_candidates = candidates[:30]  # Refine more candidates
    
    for sc, h_init in top_candidates:
        if elapsed() > deadline - 3:
            break
            
        # Normalize h_init
        h_norm_init = h_init / np.sum(h_init)
        
        try:
            h_opt = spectral_projected_gradient_descent(h_norm_init, t0, deadline, 
                                                        max_iter=3000, lr_init=0.01)
            
            if elapsed() > deadline - 2:
                break
                
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # --- Phase 3: Final Polish with higher precision ---
    # If we have time, do another pass on the best solution with more iterations
    if elapsed() < deadline - 2:
        h_norm_best = best_h / np.sum(best_h)
        try:
            h_opt = spectral_projected_gradient_descent(h_norm_best, t0, deadline,
                                                        max_iter=5000, lr_init=0.005)
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure non-negative
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
