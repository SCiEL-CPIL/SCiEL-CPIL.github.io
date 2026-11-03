import numpy as np
import time

N = 400
START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _binary_search(x0, n_iters=50000):
    """
    Structural change: Simulated Annealing / Hill Climbing on binary +/-1 signs.
    This explores the discrete space of sign patterns, which is a different
    representation than the continuous amplitude/phase optimization used previously.
    """
    x = x0.copy()
    # Initialize as binary signs based on the current best
    # If x0 is continuous, we can just use its signs. If it's all positive, we start with random signs?
    # Better: Start with random signs or the signs of x0.
    # Let's try random signs first as a fresh start, or keep signs of x0 if they vary.
    # The task says "signed step heights". A pure +/-1 sequence is a valid signed step function.
    # Let's initialize with random +/-1 to escape any local minima of the continuous space.
    rng = np.random.default_rng(42)
    x = rng.choice([-1.0, 1.0], size=N)
    
    # Check if initial random is bad, maybe use x0 signs if x0 has variance
    if np.std(x0) > 1e-6:
        x = np.sign(x0)
        x[x == 0] = 1.0
    else:
        # If x0 is constant, random is better
        x = rng.choice([-1.0, 1.0], size=N)
        
    best_x = x.copy()
    best_val = _objective(best_x)
    
    # Temperature for simulated annealing
    # We want to accept worse solutions occasionally to escape local minima
    # But given the time limit, simple hill climbing with random restarts or
    # very slow cooling might be enough.
    # Let's use a simple hill climbing with random flips.
    # To avoid getting stuck, we can allow some "bad" moves if we have a cooling schedule,
    # but standard hill climbing is faster. Let's try hill climbing first.
    # If it stagnates quickly, we might need SA.
    
    # Optimized objective check: 
    # Instead of full FFT every time, can we do incremental? 
    # No, N=400 is small enough that FFT is fast (~microseconds).
    # 50k iterations * ~50us = 2.5s. This is very fast.
    # We can do many more iterations.
    
    # Let's do 200,000 iterations of hill climbing.
    # Flip one bit at a time.
    
    for it in range(n_iters):
        if _check_deadline():
            break
            
        # Random index to flip
        idx = rng.integers(N)
        
        # Flip
        x[idx] *= -1.0
        
        val = _objective(x)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
        else:
            # Revert
            x[idx] *= -1.0
            
    return best_x, best_val

def construct():
    # Start with a simple initialization
    # Since we are switching to binary search, the initial continuous value
    # only matters if we use its signs.
    # Let's just start with ones, then binary search will handle the rest.
    x0 = np.ones(N)
    
    # Run binary search
    best_x, best_val = _binary_search(x0, n_iters=200000)
    
    # If we have time, try a second round with a different random seed
    # or restart from the best found.
    if not _check_deadline():
        # Restart from best found but with a different random seed for flips?
        # The function uses a fixed rng seed 42.
        # Let's just run it again with a different seed if we modify the function to accept seed.
        # For now, let's just run another instance with a different initial random state.
        # We can modify _binary_search to take a seed.
        pass
        
    return best_x
