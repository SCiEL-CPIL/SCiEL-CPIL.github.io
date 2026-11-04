"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: keep a safety margin before the 150s kill
_DEADLINE_MARGIN = 10.0  # seconds before the 150s limit


def _autoconvolution_max_abs(f):
    """
    Compute max absolute value of autoconvolution (excluding zero-lag self term).
    
    For a step function f with N intervals, the autoconvolution at lag k is:
    (f * f)(k) = sum_j f[j] * f[j+k]
    
    We want max |(f*f)(k)| for k != 0.
    """
    n = len(f)
    # Linear convolution via FFT
    size = 2 * n
    f_pad = np.zeros(size)
    f_pad[:n] = f
    F = np.fft.fft(f_pad, size)
    conv = np.fft.ifft(F * F).real
    # conv[0] is the zero-lag (self) term: sum(f^2)
    # We want max |conv[k]| for k != 0
    if len(conv) > 1:
        return np.max(np.abs(conv[1:]))
    return 0.0


def _compute_score(f):
    """The score is the upper bound on C3 (lower is better)."""
    return _autoconvolution_max_abs(f)


def _smooth_surrogate(f, temperature=0.5):
    """
    Smooth surrogate for the max |conv| objective using log-sum-exp.
    This makes the objective differentiable for gradient-based optimization.
    """
    n = len(f)
    size = 2 * n
    f_pad = np.zeros(size)
    f_pad[:n] = f
    F = np.fft.fft(f_pad, size)
    conv = np.fft.ifft(F * F).real[:n]
    # Exclude lag 0
    conv_nonzero = conv[1:] if len(conv) > 1 else np.array([0.0])
    abs_conv = np.abs(conv_nonzero)
    
    # Log-sum-exp for numerical stability
    max_val = np.max(abs_conv)
    if max_val < 1e-10:
        return max_val
    
    # Soft max: exp((x - max)/T) / sum(exp((x - max)/T)) * max + max
    # Actually, we want a smooth approximation to max
    # Use: T * log(sum(exp(x/T)))
    scaled = abs_conv / temperature
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    smooth_max = temperature * log_sum_exp
    
    return smooth_max


def _gradient_descent_smooth(f, max_iters=1000, learning_rate=0.01, temperature=0.5, deadline=None):
    """
    Gradient descent on the smooth surrogate objective.
    Uses finite differences for gradient computation.
    """
    current_f = f.copy().astype(float)
    current_score = _smooth_surrogate(current_f, temperature)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient via finite differences
        grad = np.zeros_like(current_f)
        eps = 1e-4
        for i in range(N):
            f_plus = current_f.copy()
            f_plus[i] += eps
            f_minus = current_f.copy()
            f_minus[i] -= eps
            
            score_plus = _smooth_surrogate(f_plus, temperature)
            score_minus = _smooth_surrogate(f_minus, temperature)
            
            grad[i] = (score_plus - score_minus) / (2 * eps)
        
        # Gradient step
        current_f -= learning_rate * grad
        
        # Project back to [-1, 1] or snap to {-1, 1}
        # For now, keep continuous and snap at the end
        new_score = _smooth_surrogate(current_f, temperature)
        if new_score < current_score:
            current_score = new_score
        else:
            # Revert if no improvement
            current_f += learning_rate * grad
            break
        
        # Adaptive learning rate
        if it % 100 == 0:
            learning_rate *= 0.9
    
    # Snap to {-1, 1}
    f_snapped = np.sign(current_f)
    f_snapped[f_snapped == 0] = 1.0
    
    return f_snapped, _compute_score(f_snapped)


def _local_search_sign_flip(f, max_rounds=50, deadline=None):
    """
    Local search by flipping individual signs.
    Uses a greedy approach: repeatedly find the best single flip.
    """
    current_f = f.copy()
    current_score = _compute_score(current_f)
    
    for _ in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        
        best_score = current_score
        best_idx = -1
        
        # Try all possible flips
        for i in range(N):
            f_new = current_f.copy()
            f_new[i] *= -1.0
            s = _compute_score(f_new)
            if s < best_score:
                best_score = s
                best_idx = i
        
        if best_idx == -1:
            # No improvement found
            break
        
        current_f[best_idx] *= -1.0
        current_score = best_score
    
    return current_f, current_score


def _local_search_sparse(f, max_rounds=20, deadline=None):
    """
    Local search for sparse representations.
    Tries to turn off zero entries or turn on zero entries.
    """
    current_f = f.copy()
    current_score = _compute_score(current_f)
    
    for _ in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        
        best_score = current_score
        best_idx = -1
        best_action = None  # 'flip', 'turn_on', 'turn_off'
        
        # Try flipping signs of non-zero entries
        for i in range(N):
            if current_f[i] != 0:
                f_new = current_f.copy()
                f_new[i] *= -1.0
                s = _compute_score(f_new)
                if s < best_score:
                    best_score = s
                    best_idx = i
                    best_action = 'flip'
        
        # Try turning on zero entries
        for i in range(N):
            if current_f[i] == 0:
                f_new = current_f.copy()
                f_new[i] = 1.0
                s = _compute_score(f_new)
                if s < best_score:
                    best_score = s
                    best_idx = i
                    best_action = 'turn_on'
        
        # Try turning off non-zero entries
        for i in range(N):
            if current_f[i] != 0:
                f_new = current_f.copy()
                f_new[i] = 0.0
                s = _compute_score(f_new)
                if s < best_score:
                    best_score = s
                    best_idx = i
                    best_action = 'turn_off'
        
        if best_idx == -1:
            # No improvement found
            break
        
        if best_action == 'flip':
            current_f[best_idx] *= -1.0
        elif best_action == 'turn_on':
            current_f[best_idx] = 1.0
        elif best_action == 'turn_off':
            current_f[best_idx] = 0.0
        
        current_score = best_score
    
    return current_f, current_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Strategy: Combine sparse combinatorial design with smooth surrogate optimization
    # 1. Start with sparse random initialization
    # 2. Use smooth surrogate gradient descent to refine
    # 3. Snap to discrete values and do local search
    
    best_f = np.ones(N)
    best_score = _compute_score(best_f)
    
    # Try different sparsity levels
    sparsity_levels = [20, 30, 40, 50, 60, 80, 100, 150, 200]
    
    for K in sparsity_levels:
        if time.time() > deadline - 20:
            break
        
        # Random initialization: pick K random indices
        np.random.seed(42 + K)
        active_indices = np.random.choice(N, size=K, replace=False)
        signs = np.random.choice([-1, 1], size=K)
        
        f = np.zeros(N)
        f[active_indices] = signs
        
        current_score = _compute_score(f)
        
        # Smooth surrogate gradient descent
        if time.time() < deadline - 15:
            f_smooth, score_smooth = _gradient_descent_smooth(
                f, max_iters=500, learning_rate=0.05, temperature=0.5, deadline=deadline - 15
            )
            if score_smooth < current_score:
                f = f_smooth
                current_score = score_smooth
        
        # Local search: flip individual signs
        if time.time() < deadline - 10:
            f, current_score = _local_search_sign_flip(f, max_rounds=30, deadline=deadline - 10)
        
        # Local search: sparse adjustments
        if time.time() < deadline - 5:
            f, current_score = _local_search_sparse(f, max_rounds=10, deadline=deadline - 5)
        
        if current_score < best_score:
            best_score = current_score
            best_f = f.copy()
    
    # Final polish: try all sign flips one more time on best solution
    if time.time() < deadline - 5:
        f, score = _local_search_sign_flip(best_f, max_rounds=20, deadline=deadline - 5)
        if score < best_score:
            best_f = f
            best_score = score
    
    return best_f
# EVOLVE-BLOCK-END
