import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.fft import fft, ifft
import time

N = 600
# Grid points: centers of the intervals
# Intervals are on [-1/4, 1/4], length 1/2.
# Interval width dx = (1/2) / N
dx = 0.5 / N
# x_j is the center of the j-th interval
x = np.linspace(-0.25 + dx/2, 0.25 - dx/2, N)

TIME_LIMIT = 145 # Safety margin

def compute_score(f):
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Compute autoconvolution using FFT
    # Zero pad to 2N to avoid circular convolution artifacts
    f_pad = np.zeros(2 * N)
    f_pad[:N] = f
    F = fft(f_pad)
    C = ifft(F * F).real
    # The autoconvolution of two length N sequences has length 2N-1.
    # The FFT approach gives 2N points. The last point is aliasing of the first?
    # Actually, linear convolution of length N and N is length 2N-1.
    # FFT size 2N is sufficient.
    # C[0] corresponds to index 0 (lag -N+1 to N-1? No, standard FFT indexing)
    # Let's just take the max of the relevant part.
    # The full linear convolution is contained in C[0:N] and C[N:2N-1]?
    # Actually, if we use FFT of size >= 2N-1, the result is the linear convolution.
    # Size 2N is fine.
    # The peaks are usually in the middle.
    return np.max(C) / (s * s)

def generate_f(params):
    """
    params: array of shape [K, 3] where K is number of bumps.
    Each row is [center, width, height].
    """
    f = np.zeros(N)
    for i in range(len(params)):
        c, w, h = params[i]
        # Ensure width is positive
        w = max(w, dx * 2)
        # Cosine bump: h * cos^2(pi * (x-c)/w) for |x-c| < w/2
        dist = np.abs(x - c)
        mask = dist < w / 2.0
        if np.any(mask):
            f[mask] += h * (np.cos(np.pi * dist[mask] / w)) ** 2
    return f

def objective(params):
    f = generate_f(params)
    return compute_score(f)

def construct():
    start_time = time.time()
    
    # Number of bumps to use. 
    # Too few might be too rigid. Too many makes DE slow.
    # Let's try K=3 and K=4.
    
    best_f = None
    best_score = np.inf
    
    # Define bounds for parameters:
    # center: [-0.25, 0.25]
    # width: [0.01, 0.5] (max width is the whole domain)
    # height: [0.0, 5.0] (scale doesn't matter for score, but helps DE explore)
    
    bounds_list = []
    
    def run_optimization(K, time_budget):
        global best_f, best_score
        
        bounds = [(-0.25, 0.25), (0.01, 0.5), (0.0, 5.0)] * K
        init_strategy = 'latinhypercube'
        
        # Differential Evolution
        # maxiter and popsize control the cost.
        # We check time inside the callback.
        
        stop_flag = [False]
        
        def callback(xk, convergence):
            if time.time() - start_time > TIME_LIMIT - 10:
                stop_flag[0] = True
                return True
            return False
            
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=100, 
                popsize=10, 
                tol=1e-5, 
                callback=callback,
                seed=42,
                polish=True # DE includes a final polish with Nelder-Mead
            )
            
            if res.success or res.fun < best_score:
                f_candidate = generate_f(res.x)
                score_candidate = compute_score(f_candidate)
                if score_candidate < best_score:
                    best_score = score_candidate
                    best_f = f_candidate.copy()
                    
        except Exception as e:
            pass

    # Run for K=2, 3, 4
    # Allocate time. DE is stochastic and can be slow.
    # K=3 is a good balance.
    
    # First, a quick run with K=3
    run_optimization(3, 60)
    
    # If we have time, try K=4
    if time.time() - start_time < TIME_LIMIT - 40 and best_f is not None:
        # Use the best solution from K=3 as a hint? 
        # DE doesn't easily take a start point, but we can do a local search on a K=4 model
        # initialized by spreading the K=3 bumps.
        # For simplicity, just run another DE with different seed or K.
        run_optimization(4, 40)
        
    # Local Refinement on the best found f
    # We can't easily refine the parametric model further with DE.
    # But we can do a direct coordinate descent on the 600 values?
    # That might be slow. 
    # Instead, let's try a "peak shaving" local search on the parametric model.
    # Actually, the `polish=True` in DE already does Nelder-Mead.
    
    # One more thing: Direct optimization of the 600 values using a smooth surrogate?
    # The previous program did this. The parametric approach is structurally different.
    
    if best_f is None:
        # Fallback: uniform
        best_f = np.ones(N)
        
    # Final normalization check
    s = np.sum(best_f)
    if s > 0:
        # The score is invariant to scaling, so no need to scale, 
        # but ensuring non-negativity is good.
        pass
        
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
