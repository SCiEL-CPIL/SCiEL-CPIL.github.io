import numpy as np
from scipy.optimize import minimize
import time

N = 600
DEADLINE = 145.0

def compute_score(f):
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def smooth_max(x, beta):
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    shifted = x - x_max
    return x_max + np.log(np.sum(np.exp(beta * shifted))) / beta

def parametric_to_step(params):
    a, b, c, d, e, f_ = params
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    f = a + b*x**2 + c*x**4 + d*x**6 + e*x**8 + f_*x**10
    return np.maximum(f, 0.0)

def objective(params, beta=30.0):
    f = parametric_to_step(params)
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    f = f / s
    conv = np.convolve(f, f, mode='full')
    return smooth_max(conv, beta)

def local_refinement_symmetric(h, start_time, max_time):
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    s = np.sum(current_h)
    if s < 1e-12:
        return best_h, best_val
    current_h = current_h / s
    
    # Enforce symmetry
    half = N // 2
    current_h[:half] = (current_h[:half] + current_h[half:][::-1]) / 2
    current_h[half:] = current_h[:half][::-1]
    if N % 2 == 1:
        current_h[half] = current_h[half-1]  # or average
    s = np.sum(current_h)
    if s > 1e-12:
        current_h = current_h / s
    current_val = compute_score(current_h)
    best_h = current_h.copy()
    best_val = current_val
    
    rng = np.random.RandomState(42)
    
    while time.time() - start_time < max_time:
        improved = False
        indices = rng.permutation(half)
        for i in indices:
            if time.time() - start_time > max_time:
                break
            for delta in [0.002, 0.005, 0.01, 0.02, -0.002, -0.005, -0.01, -0.02]:
                h_new = current_h.copy()
                h_new[i] = max(0, h_new[i] + delta)
                h_new[N-1-i] = h_new[i]
                s_new = np.sum(h_new)
                if s_new < 1e-12:
                    continue
                h_new = h_new / s_new
                conv_new = np.convolve(h_new, h_new, mode='full')
                val_new = np.max(conv_new)
                if val_new < current_val - 1e-12:
                    current_h = h_new
                    current_val = val_new
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
                    break
        if not improved:
            break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    best_f = np.ones(N) / N
    best_score = compute_score(best_f)
    
    # Parametric optimization
    rng = np.random.RandomState(123)
    best_params = None
    best_obj = np.inf
    
    initial_guesses = [
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, -0.5, 0.0, 0.0, 0.0, 0.0],
        [1.0, -1.5, 0.1, 0.0, 0.0, 0.0],
        [1.0, -0.8, 0.05, 0.0, 0.0, 0.0],
        [1.0, -1.2, 0.08, 0.0, 0.0, 0.0],
    ]
    
    for beta in [20.0, 40.0]:
        if time.time() - start_time > DEADLINE - 40:
            break
        for guess in initial_guesses:
            if time.time() - start_time > DEADLINE - 35:
                break
            x0 = np.array(guess) + rng.uniform(-0.05, 0.05, size=6)
            try:
                result = minimize(objective, x0, args=(beta,), method='Nelder-Mead',
                                  options={'maxiter': 2000, 'adaptive': True})
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
            except:
                pass
        
        for _ in range(5):
            if time.time() - start_time > DEADLINE - 30:
                break
            x0 = [rng.uniform(0.5, 1.5), rng.uniform(-2, 0), rng.uniform(0, 0.2),
                  rng.uniform(-0.05, 0.05), rng.uniform(-0.02, 0.02), rng.uniform(0, 0.01)]
            try:
                result = minimize(objective, x0, args=(beta,), method='Nelder-Mead',
                                  options={'maxiter': 1500, 'adaptive': True})
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
            except:
                pass
    
    if best_params is not None:
        f = parametric_to_step(best_params)
        s = np.sum(f)
        if s > 1e-12:
            f = f / s
            score = compute_score(f)
            if score < best_score:
                best_score = score
                best_f = f
    
    # Local refinement
    if time.time() - start_time < DEADLINE - 5:
        refined_f, refined_score = local_refinement_symmetric(best_f, start_time, DEADLINE - 2)
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]
