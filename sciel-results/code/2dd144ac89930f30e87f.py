import numpy as np
import scipy.optimize as opt
import time

def construct():
    N = 50
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_score = 0.0
    # Default to uniform, which gives ~1.92
    best_h = np.ones(N) / N
    
    def compute_ratio(h):
        # h is assumed to be non-negative and sum to 1
        l2_sq = np.sum(h**2)
        if l2_sq == 0:
            return 0.0
        c = np.convolve(h, h, mode='full')
        c_max = np.max(c)
        if c_max == 0:
            return 0.0
        return l2_sq / c_max

    def objective(y):
        # y are the log-parameters
        y_max = np.max(y)
        h = np.exp(y - y_max)
        h_sum = np.sum(h)
        if h_sum == 0:
            return 0.0
        h = h / h_sum
        
        l2_sq = np.sum(h**2)
        c = np.convolve(h, h, mode='full')
        
        # Smooth max approximation to help gradient methods
        # Temperature t=20 is a good balance
        t = 20.0
        # log-sum-exp trick
        c_max_smooth = (np.log(np.sum(np.exp(c * t))) / t)
        
        if l2_sq == 0 or c_max_smooth == 0:
            return 0.0
            
        val = l2_sq / c_max_smooth
        if val <= 0:
            return 0.0
        return -np.log(val)

    # Run several random restarts
    n_restarts = 100
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Initialize y
        # Mix of strategies
        if i % 4 == 0:
            # Uniform
            y = np.zeros(N)
        elif i % 4 == 1:
            # Random Gaussian
            y = np.random.randn(N) * 0.5
        elif i % 4 == 2:
            # Sparse random (few peaks)
            y = np.zeros(N)
            idx = np.random.choice(N, size=3, replace=False)
            y[idx] = np.random.rand(3)
        else:
            # "Flat" spectrum initialization:
            # h_k = |P(omega^k)|^2 with random phases
            # This often leads to flat autoconvolution
            coeffs = np.random.randn(N) + 1j * np.random.randn(N)
            # Make it sparse? No, full random is good for flatness
            h_init = np.abs(np.fft.fft(coeffs))**2
            # Normalize h_init to sum 1
            h_sum = np.sum(h_init)
            if h_sum > 0:
                h_init = h_init / h_sum
            # Convert to y (log space)
            # Add small epsilon to avoid log(0)
            y = np.log(h_init + 1e-9)
            
        try:
            # Use L-BFGS-B for efficiency
            res = opt.minimize(objective, y, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success:
                y_opt = res.x
                y_max = np.max(y_opt)
                h_opt = np.exp(y_opt - y_max)
                h_opt = h_opt / np.sum(h_opt)
                
                score = compute_ratio(h_opt)
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            pass

    return [float(x) for x in best_h]
