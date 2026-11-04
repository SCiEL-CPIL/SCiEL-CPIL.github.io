"""Model-and-optimize bootstrap: parameterize step heights and optimize the objective."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    
    Strategy:
    1. Generate diverse initial points (random, structured).
    2. Use smooth surrogate (log-sum-exp) + L-BFGS-B gradient optimization.
    3. Final exact evaluation and polish.
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid
    
    # First evaluate the seed
    val_seed, _ = evaluate(best_h)
    best_value = val_seed
    
    rng = np.random.RandomState(42)
    
    def smooth_objective_and_grad(h, beta):
        """
        Smooth surrogate for |2N * max(conv) / sum(h)^2|.
        Uses log-sum-exp for the max, and handles the abs by optimizing
        the squared version or by tracking sign.
        We minimize (2N * soft_max(conv) / sum(h)^2)^2 to avoid abs non-diff,
        but that changes the landscape. Instead, we minimize:
            f(h) = 2N * soft_max(conv) / sum(h)^2
        and take abs at the end. The gradient of f is well-defined.
        """
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)
        
        conv = np.convolve(h, h)
        
        # Log-sum-exp for numerical stability
        max_conv = np.max(conv)
        if max_conv > 0:
            logsumexp = max_conv + np.log(np.sum(np.exp(beta * (conv - max_conv))))
        else:
            logsumexp = max_conv + np.log(np.sum(np.exp(beta * (conv - max_conv))))
        
        soft_max = logsumexp / beta
        
        # f = 2N * soft_max / s^2
        f = 2.0 * N * soft_max / (s * s)
        
        # Gradient of soft_max w.r.t. h:
        # soft_max = (1/beta) * log(sum(exp(beta * conv_i)))
        # d(soft_max)/d(conv_i) = exp(beta * conv_i) / sum(exp(beta * conv_j))
        # d(conv_i)/d(h_k) = h_{i-k} (for valid indices in full conv)
        # 
        # Full conv: conv[i] = sum_j h[j] * h[i-j] for i in 0..2N-2
        # d(conv[i])/d(h[k]) = h[i-k] if 0<=i-k<N, plus h[k] contribution from the other term
        # Actually for autoconvolution: conv[i] = sum_{j=max(0,i-N+1)}^{min(N-1,i)} h[j]*h[i-j]
        # d(conv[i])/d(h[k]) = h[i-k] (if 0<=i-k<N) + h[k] (if 0<=i-k<N, same term counted twice when i-k==k? No)
        # Let me think: conv[i] = sum_j h[j] h[i-j]
        # d(conv[i])/d(h[k]) = h[i-k] + h[k] * (delta if i-2k in range... no)
        # Actually: d/dh[k] [sum_j h[j] h[i-j]] = h[i-k] (from j=k term: d h[j]/d h[k] * h[i-j] = h[i-k])
        #   + sum_j h[j] * d h[i-j]/d h[k] = sum_j h[j] * delta_{i-j,k} = h[i-k] (from j=i-k)
        # So d(conv[i])/d(h[k]) = 2*h[i-k] if 0<=i-k<N, else 0.
        
        # Compute weights for soft_max gradient
        shifted = beta * (conv - max_conv)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp  # shape (2N-1,)
        
        # Gradient of soft_max w.r.t. conv: weights (shape 2N-1)
        # Gradient of f w.r.t. conv: 2N / s^2 * weights
        grad_conv = (2.0 * N / (s * s)) * weights
        
        # Gradient w.r.t. h: sum_i grad_conv[i] * d(conv[i])/d(h[k])
        # d(conv[i])/d(h[k]) = 2 * h[i-k] if 0 <= i-k < N else 0
        # So grad_h[k] = sum_i grad_conv[i] * 2 * h[i-k] (if 0<=i-k<N)
        # = 2 * sum_i grad_conv[i] * h[i-k] where i-k in [0,N-1]
        # Let j = i-k, so i = j+k, j in [0,N-1], i in [k, k+N-1]
        # grad_h[k] = 2 * sum_{j=0}^{N-1} grad_conv[j+k] * h[j]
        
        grad_h = np.zeros(N)
        for k in range(N):
            # i ranges from k to k+N-1
            i_start = k
            i_end = k + N  # exclusive
            if i_end > len(grad_conv):
                i_end = len(grad_conv)
            if i_start < i_end:
                grad_h[k] = 2.0 * np.dot(grad_conv[i_start:i_end], h[:i_end-i_start])
        
        # Gradient of 1/s^2 term: d/dh[k] [2N * soft_max / s^2] 
        # = 2N * [ (d soft_max/dh[k]) * 1/s^2 + soft_max * d(1/s^2)/dh[k] ]
        # d(1/s^2)/dh[k] = -2/s^3
        # So we need to add: 2N * soft_max * (-2/s^3) * 1 = -4N * soft_max / s^3
        # This is added to each component of grad_h.
        
        # Wait, I computed grad_conv as gradient of f w.r.t. conv, but f = 2N*soft_max/s^2
        # The soft_max part gradient is already in grad_conv.
        # The 1/s^2 part: d/dh[k] (1/s^2) = -2/s^3
        # So total grad_h[k] = 2N/s^2 * d(soft_max)/dh[k] + 2N*soft_max*(-2/s^3)
        # = grad_soft_max_part[k] - 4N*soft_max/s^3
        
        correction = -4.0 * N * soft_max / (s ** 3)
        grad_h += correction
        
        return f, grad_h
    
    def smooth_objective_wrapper(h, beta):
        f, g = smooth_objective_and_grad(h, beta)
        return f
    
    def smooth_grad_wrapper(h, beta):
        f, g = smooth_objective_and_grad(h, beta)
        return g
    
    # Generate initial points
    init_points = []
    
    # Seed
    init_points.append(np.ones(N).astype(float))
    
    # Random uniform
    for trial in range(200):
        if time.time() - start_time > DEADLINE:
            break
        h = rng.uniform(-1, 1, N)
        init_points.append(h)
    
    # Bimodal sparse
    for trial in range(200):
        if time.time() - start_time > DEADLINE:
            break
        h = np.zeros(N)
        k = rng.randint(50, 350)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        init_points.append(h)
    
    # Random normal
    for scale in [0.5, 1.0, 2.0]:
        for trial in range(100):
            if time.time() - start_time > DEADLINE:
                break
            h = rng.normal(0, scale, N)
            init_points.append(h)
    
    # Alternating
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    init_points.append(h_alt)
    
    # Cosine
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 10]:
        init_points.append(np.cos(2 * np.pi * freq * t))
    
    # Fourier synthesis
    for n_comp in [3, 5, 7]:
        for trial in range(50):
            if time.time() - start_time > DEADLINE:
                break
            freqs = rng.randint(1, 30, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h += a * np.cos(2 * np.pi * f * t + p)
            init_points.append(h)
    
    # Evaluate all init points and keep top candidates
    evals = []
    for h in init_points:
        if time.time() - start_time > DEADLINE:
            break
        val, valid = evaluate(h)
        if valid:
            evals.append((val, h))
    
    evals.sort(key=lambda x: x[0])
    
    # Take top candidates for optimization
    n_optimize = min(20, len(evals))
    
    # For each top candidate, run L-BFGS with increasing beta
    for beta in [10, 50, 100, 200]:
        if time.time() - start_time > DEADLINE:
            break
        
        for idx in range(n_optimize):
            if time.time() - start_time > DEADLINE:
                break
            
            h0 = evals[idx][1].copy()
            
            try:
                result = minimize(
                    smooth_objective_wrapper,
                    h0,
                    args=(beta,),
                    jac=lambda h, b: smooth_grad_wrapper(h, b),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                if result.success or result.fun < 1e10:
                    h_opt = result.x
                    val, valid = evaluate(h_opt)
                    if valid and val < best_value:
                        best_value = val
                        best_h = h_opt.copy()
                        
            except Exception:
                pass
    
    # Final polish: small perturbations around best_h
    if time.time() - start_time < DEADLINE:
        h_curr = best_h.copy()
        lr = 0.01
        for iteration in range(200):
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            for coord in rng.permutation(N):
                if time.time() - start_time > DEADLINE:
                    break
                h_try = h_curr.copy()
                h_try[coord] += lr * rng.uniform(-1, 1)
                val_try, valid = evaluate(h_try)
                if valid and val_try < best_value:
                    best_value = val_try
                    best_h = h_try.copy()
                    h_curr = h_try
                    improved = True
                    if best_value < 0.5:
                        break
            if best_value < 0.5:
                break
            if not improved:
                lr *= 0.7
    
    return best_h
# EVOLVE-BLOCK-END
