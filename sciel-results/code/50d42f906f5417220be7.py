import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0 # Safety margin before 150s limit

def _compute_ac(x):
    """
    Compute the linear autocorrelation of x.
    x is a 1D array.
    Returns the first N-1 sidelobes (excluding the main peak).
    """
    n_fft = 2 * N
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    # ac[0] is the main peak (sum x^2). Sidelobes are ac[1:N].
    return ac[1:N]

def _soft_max_objective_and_gradient(x, beta=10.0):
    """
    Compute the soft-max objective and its gradient.
    Objective: (1/beta) * log(sum(exp(beta * AC_k^2)))
    This approximates max(AC_k^2).
    """
    ac_sidelobes = _compute_ac(x)
    y = ac_sidelobes ** 2
    
    # Compute weights for soft-max
    # To avoid overflow, subtract max
    max_y = np.max(y)
    exp_y = np.exp(beta * (y - max_y))
    sum_exp = np.sum(exp_y)
    weights = exp_y / sum_exp
    
    # Objective value
    obj = (1.0 / beta) * (np.log(sum_exp) + beta * max_y)
    
    # Gradient
    # grad_y_k = 2 * AC_k * d(AC_k)/dx
    # d(AC_k)/dx_j = x_{j+k} + x_{j-k} (with bounds)
    # We can compute the gradient vector by accumulating contributions from each lag k.
    # grad_x = sum_k weights_k * 2 * AC_k * (shifted_x_k + shifted_x_-k)
    
    grad_x = np.zeros_like(x)
    ac_vals = ac_sidelobes
    for k in range(1, N):
        if weights[k-1] < 1e-10: # Skip negligible contributions
            continue
        # d(AC_k)/dx
        # AC_k = sum_{i=0}^{N-1-k} x_i x_{i+k}
        # d/dx_j AC_k:
        # if j < N-k: term x_j x_{j+k} -> x_{j+k}
        # if j >= k: term x_{j-k} x_j -> x_{j-k}
        
        # Vectorized approach for efficiency if possible, but loop is O(N^2) total
        # Let's just do it simply for clarity and correctness, N=400 is small.
        
        # Contribution to indices 0 to N-k-1 from x_{j+k}
        # Contribution to indices k to N-1 from x_{j-k}
        
        # We can use slicing
        # Part 1: x_{j+k} for j in [0, N-k-1]
        # This adds to grad_x[0:N-k]
        # Part 2: x_{j-k} for j in [k, N-1]
        # This adds to grad_x[k:N]
        
        coeff = 2.0 * ac_vals[k-1] * weights[k-1]
        
        grad_x[0:N-k] += coeff * x[k:N]
        grad_x[k:N] += coeff * x[0:N-k]
        
    return obj, grad_x

def _local_search_binary(h, time_limit=5.0):
    """
    Local search on the binary sequence h (values +/- 1).
    """
    start_time = time.time()
    current_h = h.copy()
    
    # Compute initial metric
    ac = _compute_ac(current_h)
    current_val = np.max(np.abs(ac))
    best_h = current_h.copy()
    best_val = current_val
    
    # Precompute FFT size
    n_fft = 2 * N
    
    while time.time() - start_time < time_limit:
        # Find the lag with the maximum sidelobe
        abs_ac = np.abs(ac)
        max_lag = np.argmax(abs_ac) + 1 # 1-indexed
        
        # Try flipping a bit that contributes significantly to this lag
        # We can try all bits that affect this lag, or just a random one.
        # Let's try the bit that seems to have the largest impact.
        # Impact of flipping h[j] on ac[max_lag]:
        # Delta = -2 * h[j] * (h[j+max_lag] + h[j-max_lag]) (with bounds)
        
        best_delta = 0
        best_j = -1
        
        # Only check bits that are involved in ac[max_lag]
        # j ranges from 0 to N-1
        # Valid j for lag k: j such that j+k < N or j-k >= 0
        # i.e., j < N-k or j >= k
        
        # We can just iterate over all j, it's only 400
        for j in range(N):
            term = 0.0
            if j + max_lag < N:
                term += current_h[j + max_lag]
            if j - max_lag >= 0:
                term += current_h[j - max_lag]
            
            if term == 0:
                continue
                
            delta = -2.0 * current_h[j] * term
            # We want to minimize the max sidelobe.
            # Flipping h[j] changes ac[max_lag] by delta.
            # New ac[max_lag] = ac[max_lag] + delta
            # We want |new_ac| < |old_ac|
            
            old_ac_val = ac[max_lag - 1]
            new_ac_val = old_ac_val + delta
            
            if abs(new_ac_val) < abs(old_ac_val):
                # This flip improves the specific lag.
                # It might worsen other lags, but we check the global max after flipping.
                # For efficiency in local search, we often accept if it improves the worst lag.
                # Let's just pick the first one that improves the worst lag significantly?
                # Or the one that minimizes the new max? That's expensive.
                # Let's just do a simple hill climb: flip if it reduces the global max.
                # But checking global max requires O(N) or O(N log N).
                # Let's just flip a random bit that improves the worst lag, and then re-evaluate.
                
                # To be more aggressive, let's find the j that minimizes the new ac[max_lag] magnitude
                # if we only consider this lag.
                pass

        # Simpler strategy: Random bit flip that reduces the current max sidelobe value.
        # Let's try flipping a random bit and check if it improves the global max.
        j = np.random.randint(N)
        old_h_j = current_h[j]
        current_h[j] = -old_h_j
        
        # Recompute metric
        F = np.fft.rfft(current_h, n=n_fft)
        power = F * np.conj(F)
        ac_new = np.fft.irfft(power, n=n_fft)
        new_val = np.max(np.abs(ac_new[1:N]))
        
        if new_val < current_val:
            current_val = new_val
            ac = ac_new
            if new_val < best_val:
                best_val = new_val
                best_h = current_h.copy()
        else:
            current_h[j] = old_h_j
            # If no improvement, we might get stuck. 
            # We can add a random restart or simulated annealing, but time is tight.
            
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = None
    best_val = float('inf')
    
    # Initial candidates
    # 1. All ones (bad)
    # 2. Random
    # 3. Chirp
    
    candidates = []
    candidates.append(np.random.choice([-1, 1], size=N))
    
    # Chirp
    i = np.arange(N)
    phase = np.pi * i**2 / N
    h_chirp = np.sign(np.cos(phase))
    h_chirp[h_chirp == 0] = 1
    candidates.append(h_chirp)
    
    # Another chirp
    phase2 = -np.pi * i**2 / N
    h_chirp2 = np.sign(np.cos(phase2))
    h_chirp2[h_chirp2 == 0] = 1
    candidates.append(h_chirp2)
    
    # Evaluate initial candidates
    for h in candidates:
        val = _compute_metric(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    # Optimization Loop
    # We will run L-BFGS-B from several starting points.
    # Starting points: the best binary candidate, and some random continuous ones.
    
    start_points = []
    if best_h is not None:
        start_points.append(best_h.astype(float))
        
    for _ in range(5):
        start_points.append(np.random.uniform(-1, 1, size=N))
        
    beta = 10.0
    
    for sp in start_points:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(
                _soft_max_objective_and_gradient,
                sp,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * N,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            x_opt = res.x
            # Round to binary
            h_bin = np.sign(x_opt)
            h_bin[h_bin == 0] = 1
            
            val = _compute_metric(h_bin)
            if val < best_val:
                best_val = val
                best_h = h_bin.copy()
                
        except Exception as e:
            pass
            
    # Local Search Polish
    if best_h is not None and time.time() < deadline - 5.0:
        time_left = deadline - 5.0 - time.time()
        best_h, best_val = _local_search_binary(best_h, time_limit=time_left)
        
    return best_h

def _compute_metric(h):
    """
    Compute the max absolute sidelobe of the autocorrelation.
    h is a 1D array of length N with values +/- 1.
    """
    ac = _compute_ac(h)
    return np.max(np.abs(ac))

