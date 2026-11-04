import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    # Keep a safety margin of 20 seconds
    deadline = start_time + 100.0

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    # Initial best: Linear Ramp
    # h_i = i + 1
    h_best = np.arange(1, N + 1, dtype=np.float64)
    val_best = compute_value(h_best)
    
    # Candidate shapes to test
    candidates = []
    
    # 1. Linear ramp (already best)
    candidates.append(h_best)
    
    # 2. Quadratic ramp
    h_quad = np.arange(1, N + 1, dtype=np.float64) ** 2
    candidates.append(h_quad)
    
    # 3. Square root ramp
    h_sqrt = np.sqrt(np.arange(1, N + 1, dtype=np.float64))
    candidates.append(h_sqrt)
    
    # 4. Exponential ramp
    h_exp = np.exp(np.arange(N, dtype=np.float64) / N)
    candidates.append(h_exp)
    
    # 5. Asymmetric Sine (shifted to be positive and asymmetric)
    # h_i = 1 + sin(pi * i / N) is symmetric.
    # h_i = 1 + sin(pi * (i - N/4) / N) ?
    # Let's try a general power law: h_i = (i + 1)^alpha
    for alpha in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        h_pow = np.arange(1, N + 1, dtype=np.float64) ** alpha
        candidates.append(h_pow)

    # Evaluate all candidates
    for h in candidates:
        val = compute_value(h)
        if val < val_best:
            val_best = val
            h_best = h.copy()

    # Local search: Perturb the exponent of the power law
    # Assume the best shape is roughly a power law or can be perturbed
    # We will do a simple random search around the best candidate found
    rng = np.random.default_rng(42)
    
    # If the best is a power law, we can optimize the exponent
    # Let's just do random perturbations on the current best vector
    # But random perturbations on a ramp might break the structure.
    # Better: Search over a parametric family.
    
    # Family: h_i = (i + 1)^alpha * exp(beta * i / N)
    best_alpha = 1.0
    best_beta = 0.0
    
    # Grid search over alpha and beta
    alphas = np.linspace(0.5, 2.0, 15)
    betas = np.linspace(-2.0, 2.0, 15)
    
    for a in alphas:
        for b in betas:
            h_test = (np.arange(1, N + 1, dtype=np.float64) ** a) * np.exp(b * np.arange(N, dtype=np.float64) / N)
            val = compute_value(h_test)
            if val < val_best:
                val_best = val
                h_best = h_test.copy()
                best_alpha = a
                best_beta = b
                
    # Fine-tune around best_alpha, best_beta
    for _ in range(50):
        if time.time() > deadline:
            break
        a_new = best_alpha + rng.normal(0, 0.05)
        b_new = best_beta + rng.normal(0, 0.05)
        h_test = (np.arange(1, N + 1, dtype=np.float64) ** a_new) * np.exp(b_new * np.arange(N, dtype=np.float64) / N)
        val = compute_value(h_test)
        if val < val_best:
            val_best = val
            h_best = h_test.copy()
            best_alpha = a_new
            best_beta = b_new

    return h_best
