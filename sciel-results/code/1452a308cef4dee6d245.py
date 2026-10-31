import numpy as np
import time

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    l2_sq = np.sum(h * h)
    hh = np.convolve(h, h, mode='full')
    hh_max = np.max(hh)
    if hh_max < 1e-300:
        return 0.0
    return l2_sq / (l1 * hh_max)

def project_simplex(v):
    """
    Project vector v onto the standard simplex {x >= 0, sum(x) = 1}.
    """
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def gradient_ascent_optimize(init_h, max_iters=2000, eta_init=0.05, eta_decay=0.999, tol=1e-12):
    """
    Perform projected gradient ascent on the log-ratio objective.
    Objective: max log(||h||_2^2) - log(max(h*h)) subject to h in simplex.
    """
    h = init_h.copy()
    h = project_simplex(h)
    
    best_h = h.copy()
    best_score = compute_ratio(h)
    
    eta = eta_init
    
    for it in range(max_iters):
        # Compute current metrics
        l2_sq = np.sum(h * h)
        hh = np.convolve(h, h, mode='full')
        hh_max = np.max(hh)
        
        if l2_sq < 1e-15 or hh_max < 1e-15:
            break
            
        # Current log-objective (for tracking, though we use gradient of ratio directly)
        # We maximize J = log(l2_sq) - log(hh_max)
        # Grad J = 2*h/l2_sq - (1/hh_max) * grad(hh_max)
        
        # Find argmax of hh
        k_star = np.argmax(hh)
        
        # Gradient of hh_max w.r.t h
        # (h*h)_k = sum_j h_j h_{j+k}
        # d/dh_i (h*h)_k = h_{i+k} + h_{i-k} (with bounds)
        
        # We need to compute this for all i.
        # Note: hh is full convolution, length 2N-1.
        # hh[k] corresponds to lag k-N+1 in standard autocorrelation indexing?
        # np.convolve(h, h, 'full') index k corresponds to sum h[i] h[k-i].
        # Let's stick to the definition.
        # d(hh[k_star])/dh[i] = h[i] * h[k_star - i] + h[k_star - i] * h[i] ? No.
        # hh[k] = sum_j h[j] h[k-j].
        # d(hh[k])/dh[i] = h[i] h[k-i] + h[k-i] h[i] ? 
        # If we differentiate sum_j h[j] h[k-j] w.r.t h[i]:
        # Term where j=i: h[i] h[k-i].
        # Term where k-j=i => j=k-i: h[k-i] h[i].
        # So yes, 2 * h[i] * h[k-i] if both indices valid.
        # But wait, h is defined on 0..N-1.
        # If k-i is out of bounds, the term is 0.
        
        grad_hh_max = np.zeros(N)
        # Vectorized calculation
        # For each i in 0..N-1, check if k_star - i is in 0..N-1
        # Actually, it's easier to loop or use np.roll if we were doing circular, but it's linear.
        
        # Efficient way:
        # grad_hh_max[i] = h[i] * h[k_star - i] + h[k_star - i] * h[i] 
        # This is only if k_star - i is in [0, N-1].
        
        # Let's just loop for clarity and safety, N=50 is small.
        for i in range(N):
            j = k_star - i
            if 0 <= j < N:
                grad_hh_max[i] = 2.0 * h[i] * h[j]
            else:
                # Check if the term comes from the other side of the convolution sum?
                # No, the derivative is local.
                # However, note that (h*h)_k is symmetric.
                # The formula d/dh_i (sum_j h_j h_{k-j}) is indeed h_{k-i} + h_{k-i} = 2 h_{k-i} if valid.
                # Wait, if j=k-i is valid, then h[k-i] exists.
                # Is there another term?
                # The sum is over j. The only j that depends on h[i] is j=i and j=k-i.
                # If i = k-i (i.e., 2i=k), then it's the same term, so it's 2 h[i]^2?
                # Let's check: d/dx (x*x) = 2x. Yes.
                # So the formula 2 * h[i] * h[k-i] works even if i=k-i.
                pass
        
        # Gradient of log(l2_sq)
        grad_l2_log = 2.0 * h / l2_sq
        
        # Gradient of log(hh_max)
        grad_hh_max_log = grad_hh_max / hh_max
        
        # Total gradient
        grad = grad_l2_log - grad_hh_max_log
        
        # Gradient Ascent Step
        h_new = h + eta * grad
        
        # Project to simplex
        h_new = project_simplex(h_new)
        
        # Calculate new score
        new_score = compute_ratio(h_new)
        
        if new_score > best_score:
            best_score = new_score
            best_h = h_new.copy()
            
        h = h_new
        eta *= eta_decay
        
        # Convergence check
        # if np.max(np.abs(grad)) < tol: break
        
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = compute_ratio(best_h)
    
    rng = np.random.default_rng(42)
    
    # List of initializers
    initializers = []
    
    # 1. Uniform
    initializers.append(np.ones(N) / N)
    
    # 2. Gaussian with different sigmas
    for sigma in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        x = np.arange(N) - N/2.0
        g = np.exp(-x**2 / (2*sigma**2))
        g = g / np.sum(g)
        initializers.append(g)
        
    # 3. Random dense vectors
    for _ in range(10):
        r = rng.random(N)
        r = r / np.sum(r)
        initializers.append(r)
        
    # 4. Random sparse vectors (for diversity, though gradient might densify)
    for _ in range(5):
        K = rng.integers(5, 20)
        S = rng.choice(N, size=K, replace=False)
        r = np.zeros(N)
        r[S] = 1.0 / K
        initializers.append(r)
        
    # 5. Alternating patterns
    for period in [2, 3, 4, 5, 7]:
        r = np.zeros(N)
        r[::period] = 1.0
        r = r / np.sum(r)
        initializers.append(r)

    # Run gradient ascent for each initializer
    for i, init_h in enumerate(initializers):
        if time.time() > deadline - 5:
            break
            
        # Adjust iterations based on time remaining
        time_left = deadline - time.time()
        # Estimate time per iteration (roughly 0.1ms for N=50)
        # Let's just use a fixed reasonable number and check time
        max_iters = 5000
        
        try:
            h_opt, score_opt = gradient_ascent_optimize(init_h, max_iters=max_iters)
            
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception as e:
            # If one fails, continue with others
            pass

    # If we still have time, do a few more random restarts with smaller steps or different params
    while time.time() < deadline - 5:
        # Generate a new random init
        r = rng.random(N)
        r = r / np.sum(r)
        # Mix with best found so far to stay in basin
        mix = rng.random()
        init_h = (1-mix) * best_h + mix * r
        init_h = project_simplex(init_h)
        
        h_opt, score_opt = gradient_ascent_optimize(init_h, max_iters=2000, eta_init=0.02)
        if score_opt > best_score:
            best_score = score_opt
            best_h = h_opt.copy()

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
