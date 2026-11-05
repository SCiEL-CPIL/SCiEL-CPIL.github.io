import numpy as np
from scipy.optimize import minimize

N = 400

def construct():
    # Seed value
    seed_val = 2.0
    
    # Best result found
    best_h = np.ones(N)
    best_val = seed_val
    
    def get_val(h):
        s = np.sum(h)
        if abs(s) < 1e-6:
            return np.inf
        conv = np.convolve(h, h)
        m = np.max(conv)
        return abs(2 * N * m / (s * s))

    def objective(h):
        # We optimize the raw value directly
        # Powell's method is derivative-free and handles non-smoothness better
        return get_val(h)

    # Generate diverse initializations
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Seed
    inits.append(np.ones(N))
    
    # 2. Random Normal
    h_rand = rng.normal(size=N)
    inits.append(h_rand)
    
    # 3. Random Uniform
    h_uni = rng.uniform(-1, 1, size=N)
    inits.append(h_uni)
    
    # 4. Sine wave
    x = np.arange(N)
    h_sin = np.sin(2 * np.pi * x / N)
    inits.append(h_sin)
    
    # 5. High frequency noise
    h_noise = rng.normal(size=N)
    h_noise = np.diff(h_noise) # High freq
    inits.append(h_noise)
    
    # 6. Block structure
    h_block = np.zeros(N)
    h_block[:N//2] = 1
    h_block[N//2:] = -1
    inits.append(h_block)
    
    # Run Powell's method from each start
    # Powell is slow, so limit iterations and function evaluations
    for x0 in inits:
        try:
            # Scale x0 to avoid extreme values
            norm = np.linalg.norm(x0)
            if norm > 0:
                x0 = x0 / norm * 10.0
            
            res = minimize(objective, x0, method='Powell', 
                           options={'maxiter': 200, 'maxfev': 2000})
            
            val_res = get_val(res.x)
            
            if val_res < best_val:
                best_val = val_res
                best_h = res.x
        except Exception:
            pass
            
    return best_h
