import numpy as np
import time

N = 400
_start_time = time.time()
_TIME_LIMIT = 148.0  # Safety margin before 150s kill

def _autoconv_full(f):
    """Full linear autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(f, size)
    C = np.fft.irfft(F * F, size)
    return C[:2 * n - 1]

def _score(f):
    """Compute max(|autoconv(f)|) / (sum(f))^2."""
    s = f.sum()
    if abs(s) < 1e-12:
        return float('inf')
    C = _autoconv_full(f)
    return np.max(np.abs(C)) / (s * s)

def _init_smooth():
    """
    Initialize with a smooth, non-rectangular shape.
    Use a raised cosine window, then smooth it to get continuous values in [0,1].
    """
    t = np.linspace(0, 1, N)
    # Raised cosine
    w = 0.5 * (1 + np.cos(2 * np.pi * t))
    # Scale to have mean ~0.5
    w = w / w.max()
    return w

def _init_binary_tapered():
    """
    Initialize with a tapered binary sequence.
    """
    t = np.linspace(0, 1, N)
    w = 0.5 * (1 + np.cos(2 * np.pi * t))
    mean_w = w.mean()
    f = (w >= mean_w).astype(np.float64)
    if f.sum() == 0:
        f[N // 2] = 1.0
    return f

def _local_search_binary(f, time_budget):
    """
    Binary local search: flip individual bits to reduce the score.
    """
    f = f.copy()
    n = len(f)
    best_score = _score(f)
    best_f = f.copy()
    
    deadline = time.time() + time_budget
    no_improve = 0
    max_no_improve = 20000
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        # Try random flips
        for _ in range(50):
            i = np.random.randint(0, n)
            new_f = f.copy()
            new_f[i] = 1.0 - new_f[i]
            if new_f.sum() == 0:
                continue
            score_new = _score(new_f)
            if score_new < best_score - 1e-12:
                best_score = score_new
                best_f = new_f.copy()
                f = new_f
                improved = True
                no_improve = 0
            else:
                no_improve += 1
        if not improved:
            no_improve += 50
    
    return best_f, best_score

def _local_search_real(f, time_budget, lr=0.01):
    """
    Real-valued local search: adjust individual values to reduce the score.
    """
    f = f.copy()
    n = len(f)
    best_score = _score(f)
    best_f = f.copy()
    
    deadline = time.time() + time_budget
    no_improve = 0
    max_no_improve = 10000
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        # Try random perturbations
        for _ in range(50):
            i = np.random.randint(0, n)
            delta = np.random.uniform(-0.1, 0.1)
            new_f = f.copy()
            new_f[i] = np.clip(new_f[i] + delta, 0, 1)
            if abs(new_f.sum()) < 1e-12:
                continue
            score_new = _score(new_f)
            if score_new < best_score - 1e-12:
                best_score = score_new
                best_f = new_f.copy()
                f = new_f
                improved = True
                no_improve = 0
            else:
                no_improve += 1
        if not improved:
            no_improve += 50
    
    return best_f, best_score

def _gradient_descent_smooth(f, time_budget, lr=0.005):
    """
    Gradient descent on the smooth objective using finite differences.
    """
    f = f.copy()
    n = len(f)
    best_score = _score(f)
    best_f = f.copy()
    
    deadline = time.time() + time_budget
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Compute gradient via finite differences
        grad = np.zeros(n)
        eps = 1e-4
        for i in range(n):
            f_plus = f.copy()
            f_plus[i] = np.clip(f_plus[i] + eps, 0, 1)
            f_minus = f.copy()
            f_minus[i] = np.clip(f_minus[i] - eps, 0, 1)
            
            s_plus = f_plus.sum()
            s_minus = f_minus.sum()
            
            if abs(s_plus) < 1e-12 or abs(s_minus) < 1e-12:
                grad[i] = 0
                continue
            
            C_plus = _autoconv_full(f_plus)
            C_minus = _autoconv_full(f_minus)
            
            score_plus = np.max(np.abs(C_plus)) / (s_plus * s_plus)
            score_minus = np.max(np.abs(C_minus)) / (s_minus * s_minus)
            
            grad[i] = (score_plus - score_minus) / (2 * eps)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            no_improve += 100
            continue
        
        # Update
        new_f = f - lr * grad
        new_f = np.clip(new_f, 0, 1)
        
        if abs(new_f.sum()) < 1e-12:
            no_improve += 1
            continue
        
        score_new = _score(new_f)
        if score_new < best_score - 1e-12:
            best_score = score_new
            best_f = new_f.copy()
            f = new_f
            no_improve = 0
        else:
            no_improve += 1
            # Reduce learning rate
            lr *= 0.95
    
    return best_f, best_score

def _annealing(f, time_budget, T0=1.0, T1=0.01):
    """
    Simulated annealing on binary sequences.
    """
    f = f.copy()
    n = len(f)
    best_score = _score(f)
    best_f = f.copy()
    current_score = best_score
    current_f = f.copy()
    
    deadline = time.time() + time_budget
    no_improve = 0
    max_no_improve = 30000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Temperature schedule
        elapsed = time.time() - (deadline - time_budget)
        progress = min(1.0, elapsed / time_budget)
        T = T0 * (T1 / T0) ** progress
        
        # Random flip
        i = np.random.randint(0, n)
        new_f = current_f.copy()
        new_f[i] = 1.0 - new_f[i]
        if new_f.sum() == 0:
            no_improve += 1
            continue
        
        new_score = _score(new_f)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_f = new_f
            current_score = new_score
            if current_score < best_score - 1e-12:
                best_score = current_score
                best_f = new_f.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
    
    return best_f, best_score

def construct():
    """
    Returns the optimal sequence for the C3-prime task.
    
    The objective is to minimize max(|autoconv(f)|) / (sum(f))^2.
    Uses multiple initialization strategies and optimization methods.
    """
    remaining = _TIME_LIMIT - (time.time() - _start_time)
    if remaining < 5:
        return np.ones(N)
    
    # Try multiple strategies and keep the best
    best_f = None
    best_score = float('inf')
    
    # Strategy 1: Smooth initialization + gradient descent
    if remaining > 30:
        f_init = _init_smooth()
        budget1 = min(remaining - 25, 40.0)
        if budget1 > 5:
            f1, score1 = _gradient_descent_smooth(f_init, budget1)
            if score1 < best_score:
                best_score = score1
                best_f = f1.copy()
        
        remaining = _TIME_LIMIT - (time.time() - _start_time)
    
    # Strategy 2: Binary tapered initialization + binary local search
    if remaining > 20:
        f_init = _init_binary_tapered()
        budget2 = min(remaining - 15, 30.0)
        if budget2 > 5:
            f2, score2 = _local_search_binary(f_init, budget2)
            if score2 < best_score:
                best_score = score2
                best_f = f2.copy()
        
        remaining = _TIME_LIMIT - (time.time() - _start_time)
    
    # Strategy 3: Annealing from best found so far
    if remaining > 15 and best_f is not None:
        budget3 = min(remaining - 10, 25.0)
        if budget3 > 5:
            f3, score3 = _annealing(best_f, budget3)
            if score3 < best_score:
                best_score = score3
                best_f = f3.copy()
        
        remaining = _TIME_LIMIT - (time.time() - _start_time)
    
    # Final polish: binary local search on best
    if remaining > 10 and best_f is not None:
        budget4 = min(remaining - 5, 15.0)
        if budget4 > 3:
            f4, score4 = _local_search_binary(best_f, budget4)
            if score4 < best_score:
                best_score = score4
                best_f = f4.copy()
    
    if best_f is None:
        return np.ones(N)
    
    return best_f

# EVOLVE-BLOCK-END
