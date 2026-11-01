"""Rewritten program for third autocorrelation inequality optimization."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 400

# EVOLVE-BLOCK-START

def _autoconvolution(signed_step):
    """
    Compute the autoconvolution of a signed step function with N equal intervals on [-1/4, 1/4].
    
    The autoconvolution at lag k (in units of interval width) is:
      (f * f)(k) = sum_i f(i) * f(i + k)  for k = -(N-1), ..., N-1
    
    We use numpy convolve with 'full' mode.
    """
    return np.convolve(signed_step, signed_step, mode='full')


def _score(signed_step):
    """
    Compute the upper bound on C3 for the given signed step function.
    
    The metric is: max|autoconv| / (sum|h|)^2
    
    Lower is better.
    """
    autoconv = _autoconvolution(signed_step)
    max_abs = np.max(np.abs(autoconv))
    sum_abs = np.sum(np.abs(signed_step))
    if sum_abs == 0:
        return np.inf
    return max_abs / (sum_abs ** 2)


def _generate_block_pattern(block_size, n_blocks=None, sign_pattern=None):
    """
    Generate a signed step function with block structure.
    
    Parameters:
    -----------
    block_size : int
        Size of each block
    n_blocks : int or None
        Number of blocks (default: N // block_size)
    sign_pattern : list of int or None
        Pattern of signs (+1/-1) for each block
    
    Returns:
    --------
    numpy array of length N with values in {-1, 0, 1}
    """
    if n_blocks is None:
        n_blocks = N // block_size
    
    if sign_pattern is None:
        # Default: alternate signs
        sign_pattern = [1 if i % 2 == 0 else -1 for i in range(n_blocks)]
    
    result = np.zeros(N)
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, N)
        result[start:end] = sign_pattern[i % len(sign_pattern)]
    
    return result


def _generate_alternating(period):
    """
    Generate a signed step function with alternating pattern of given period.
    """
    result = np.zeros(N)
    for i in range(N):
        if (i // period) % 2 == 0:
            result[i] = 1
        else:
            result[i] = -1
    return result


def _hill_climb(signed_step, max_iterations=5000, time_budget=None):
    """
    Perform discrete hill-climbing by flipping signs of individual intervals.
    
    Parameters:
    -----------
    signed_step : numpy array
        Initial signed step function
    max_iterations : int
        Maximum number of iterations
    time_budget : float or None
        Time budget in seconds
    
    Returns:
    --------
    numpy array
        Improved signed step function
    """
    current = signed_step.copy()
    current_score = _score(current)
    
    start_time = time.time()
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        improved = False
        best_flip = -1
        best_score = current_score
        
        # Try flipping each position
        for i in range(N):
            old_val = current[i]
            current[i] = -old_val
            new_score = _score(current)
            current[i] = old_val
            
            if new_score < best_score:
                best_score = new_score
                best_flip = i
        
        if best_flip >= 0:
            current[best_flip] = -current[best_flip]
            current_score = best_score
            improved = True
        
        iteration += 1
    
    return current


def _tanh_relaxation(signed_step, max_iter=100, time_budget=None):
    """
    Perform continuous optimization using tanh relaxation.
    
    We optimize over real-valued variables x where signed_step = tanh(x).
    This allows us to use gradient-based methods on a smooth landscape.
    
    Parameters:
    -----------
    signed_step : numpy array
        Initial signed step function (values in {-1, 0, 1})
    max_iter : int
        Maximum number of L-BFGS-B iterations
    time_budget : float or None
        Time budget in seconds
    
    Returns:
    --------
    numpy array
        Improved signed step function (rounded to {-1, 0, 1})
    """
    # Initialize with small values to allow exploration
    x0 = np.zeros(N)
    for i in range(N):
        if signed_step[i] > 0:
            x0[i] = 1.0
        elif signed_step[i] < 0:
            x0[i] = -1.0
    
    start_time = time.time()
    
    def objective(x):
        if time_budget is not None and (time.time() - start_time) > time_budget:
            return np.inf
        f = np.tanh(x)
        return _score(f)
    
    def gradient(x):
        if time_budget is not None and (time.time() - start_time) > time_budget:
            return np.zeros(N)
        f = np.tanh(x)
        autoconv = _autoconvolution(f)
        max_abs = np.max(np.abs(autoconv))
        sum_abs = np.sum(np.abs(f))
        
        if sum_abs == 0:
            return np.zeros(N)
        
        # Gradient of max|autoconv| / (sum|h|)^2
        # This is complex; use numerical gradient as fallback
        eps = 1e-6
        grad = np.zeros(N)
        for i in range(N):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        
        return grad
    
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            jac=None,  # Use numerical gradient
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        x_opt = result.x
    except Exception:
        x_opt = x0
    
    # Round to {-1, 0, 1}
    f_opt = np.tanh(x_opt)
    signed_result = np.sign(f_opt)
    
    return signed_result


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to improve an autoconvolution upper bound.
    
    Returns:
    --------
    numpy array of length 400 with values in {-1, 0, 1}
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_step = None
    best_score = np.inf
    
    # Generate diverse initializations
    initializations = []
    
    # Block patterns with various block sizes
    for block_size in [1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 200]:
        if block_size <= N:
            initializations.append(_generate_block_pattern(block_size))
    
    # Alternating patterns
    for period in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200]:
        if period <= N:
            initializations.append(_generate_alternating(period))
    
    # Random initializations
    np.random.seed(42)
    for _ in range(20):
        initializations.append(np.random.choice([-1, 1], size=N))
    
    # Evaluate each initialization
    for init in initializations:
        if time.time() > deadline - 10:
            break
        score = _score(init)
        if score < best_score:
            best_score = score
            best_step = init.copy()
    
    # Apply hill-climbing to the best initialization
    if best_step is not None and time.time() < deadline - 5:
        improved = _hill_climb(best_step, max_iterations=3000, time_budget=deadline - time.time() - 2)
        improved_score = _score(improved)
        if improved_score < best_score:
            best_score = improved_score
            best_step = improved
    
    # Try tanh relaxation if time permits
    if time.time() < deadline - 10:
        relaxed = _tanh_relaxation(best_step, max_iter=50, time_budget=deadline - time.time() - 3)
        relaxed_score = _score(relaxed)
        if relaxed_score < best_score:
            best_score = relaxed_score
            best_step = relaxed
            # Follow up with hill-climbing
            if time.time() < deadline - 3:
                improved = _hill_climb(best_step, max_iterations=2000, time_budget=deadline - time.time() - 1)
                improved_score = _score(improved)
                if improved_score < best_score:
                    best_score = improved_score
                    best_step = improved
    
    # Final safety check
    if best_step is None:
        best_step = np.ones(N)
    
    return best_step

# EVOLVE-BLOCK-END
