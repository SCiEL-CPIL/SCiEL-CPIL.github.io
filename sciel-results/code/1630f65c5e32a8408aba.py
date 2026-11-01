import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin (evaluator kills at 150s)
    
    def time_left():
        return deadline - time.time()
    
    # The metric to minimize: max_k |conv(h, h)[k]| / (sum_i |h_i|)^2
    # For a signed step function with 400 equal intervals, h is a sequence of length N.
    # We want to minimize the upper bound on C3, which is this quantity.
    # Lower is better.
    
    def evaluate(h):
        """Compute the score: max |autoconv| / (sum |h|)^2"""
        conv = np.convolve(h, h, mode='full')
        max_abs_conv = np.max(np.abs(conv))
        sum_abs_h = np.sum(np.abs(h))
        if sum_abs_h == 0:
            return float('inf')
        return max_abs_conv / (sum_abs_h ** 2)
    
    def soft_objective(x, temp=1.0):
        """
        Smooth objective for optimization.
        x is the raw variable, h = tanh(x) gives values in (-1, 1).
        We use a soft-max approximation for the max |conv| term.
        """
        h = np.tanh(x / temp)
        conv = np.convolve(h, h, mode='full')
        abs_conv = np.sqrt(conv**2 + 1e-10)
        
        # Soft max using log-sum-exp
        max_abs = np.max(abs_conv)
        T = temp * 0.5
        shifted = (abs_conv - max_abs) / T
        # Clip to avoid overflow
        shifted = np.clip(shifted, -50, 50)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_abs / T
        soft_max = T * log_sum_exp
        
        sum_abs_h = np.sum(np.sqrt(h**2 + 1e-10))
        value = soft_max / (sum_abs_h ** 2)
        return value
    
    best_h = None
    best_score = float('inf')
    
    # Strategy 1: Start with structured initializations and use L-BFGS-B
    # The key insight: we want h to be a signed step function, so values should
    # be close to +/- 1. We optimize over x where h = tanh(x), so large |x| gives h ~ +/- 1.
    
    initializations = []
    
    # All ones (seed program)
    initializations.append(np.full(N, 5.0))
    
    # Alternating signs
    x_alt = np.zeros(N)
    x_alt[::2] = 5.0
    x_alt[1::2] = -5.0
    initializations.append(x_alt)
    
    # Block patterns: various block sizes
    for block_size in [2, 4, 8, 10, 16, 20, 25, 40, 50, 80, 100]:
        x_block = np.zeros(N)
        for i in range(N):
            block_idx = i // block_size
            if block_idx % 2 == 0:
                x_block[i] = 5.0
            else:
                x_block[i] = -5.0
        initializations.append(x_block)
    
    # Random initializations
    np.random.seed(42)
    for _ in range(10):
        initializations.append(np.random.randn(N) * 3.0)
    
    # Optimize from each initialization
    for idx, x0 in enumerate(initializations):
        if time_left() < 5.0:
            break
        
        try:
            # Use L-BFGS-B with limited iterations to stay within time budget
            max_iter = 200 if idx < 5 else 100
            res = minimize(soft_objective, x0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-5})
            h_candidate = np.tanh(res.x)
            score = evaluate(h_candidate)
            
            if score < best_score:
                best_score = score
                best_h = h_candidate.copy()
                
        except Exception:
            pass
    
    # Discrete refinement: flip individual elements to improve
    if best_h is not None and time_left() > 10.0:
        h = best_h.copy()
        improved = True
        while improved and time_left() > 5.0:
            improved = False
            current_score = evaluate(h)
            
            # Try flipping each element
            best_i = -1
            best_new_score = current_score
            
            for i in range(N):
                h_trial = h.copy()
                h_trial[i] = -h_trial[i]
                new_score = evaluate(h_trial)
                
                if new_score < best_new_score - 1e-10:
                    best_new_score = new_score
                    best_i = i
            
            if best_i != -1:
                h[best_i] = -h[best_i]
                improved = True
                if evaluate(h) < best_score:
                    best_score = evaluate(h)
                    best_h = h.copy()
            else:
                break
    
    # If we still have time, try a few more random restarts with shorter optimization
    if best_h is not None and time_left() > 20.0:
        np.random.seed(123)
        for _ in range(5):
            if time_left() < 5.0:
                break
            x0 = np.random.randn(N) * 2.0
            try:
                res = minimize(soft_objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 50, 'ftol': 1e-5})
                h_candidate = np.tanh(res.x)
                score = evaluate(h_candidate)
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
            except Exception:
                pass
    
    if best_h is not None:
        return best_h
    else:
        # Fallback: all ones
        return np.ones(N)
