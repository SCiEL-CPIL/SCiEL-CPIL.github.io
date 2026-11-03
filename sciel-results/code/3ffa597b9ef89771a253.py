"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _score_signal(x):
    """Compute the C3' score: max(|autocorr|) / (sum(x)^2).
    Autocorr is circular autocorrelation via FFT.
    """
    if np.all(x == 0):
        return np.inf
    integral = np.sum(x)
    if integral == 0:
        return np.inf
    # Circular autocorrelation
    F = np.fft.fft(x)
    ac = np.fft.ifft(F * np.conj(F)).real
    max_abs = np.max(np.abs(ac))
    return max_abs / (integral * integral)

def _flip_bit(x, i):
    """Flip the sign at index i (multiply by -1)."""
    x[i] = -x[i]
    return x

def _random_signed_sequence(n, rng):
    """Generate a random sequence of +1/-1."""
    return rng.choice([-1, 1], size=n)

def _greedy_local_search(x, max_passes=200, deadline=None, safety_margin=5.0):
    """Greedy local search: flip bits that reduce the score."""
    x = x.copy()
    current_score = _score_signal(x)
    n = len(x)
    
    for _ in range(max_passes):
        if deadline is not None and (time.time() - deadline) > safety_margin:
            break
        improved = False
        # Try random subset of positions for speed
        indices = np.random.permutation(n)
        for i in indices:
            if deadline is not None and (time.time() - deadline) > safety_margin:
                break
            old_val = x[i]
            x[i] = -x[i]
            new_score = _score_signal(x)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                x[i] = old_val
        if not improved:
            break
    return x, current_score

def _simulated_annealing(x, max_iters=10000, deadline=None, safety_margin=5.0):
    """Simulated annealing on bit flips."""
    x = x.copy()
    current_score = _score_signal(x)
    best_x = x.copy()
    best_score = current_score
    n = len(x)
    T = 0.1
    T_min = 0.001
    cooling = 0.999
    
    for it in range(max_iters):
        if deadline is not None and (time.time() - deadline) > safety_margin:
            break
        i = np.random.randint(n)
        x[i] = -x[i]
        new_score = _score_signal(x)
        delta = new_score - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            x[i] = -x[i]
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_x, best_score

def _gradient_descent_complex(n, max_iters=5000, deadline=None, safety_margin=5.0):
    """
    Represent signal as real part of complex exponential sum with random phases.
    x[k] = Re(exp(i * phase[k])) = cos(phase[k])
    Optimize phases to minimize max |autocorr| / (sum x)^2.
    """
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, size=n)
    
    def objective(phases):
        x = np.cos(phases)
        integral = np.sum(x)
        if abs(integral) < 1e-10:
            return 1e10
        F = np.fft.fft(x)
        ac = np.fft.ifft(F * np.conj(F)).real
        max_abs = np.max(np.abs(ac))
        return max_abs / (integral * integral)
    
    best_x = None
    best_score = 1e10
    
    for restart in range(4):
        if deadline is not None and (time.time() - deadline) > safety_margin:
            break
        phases = rng.uniform(0, 2 * np.pi, size=n)
        result = minimize(objective, phases, method='BFGS', 
                         options={'maxiter': max_iters // 4, 'gtol': 1e-6})
        x = np.cos(result.x)
        # Snap to signs for binary-like behavior
        x_signed = np.sign(x)
        if np.all(x_signed == 0):
            x_signed = np.ones(n)
        s = _score_signal(x_signed)
        if s < best_score:
            best_score = s
            best_x = x_signed.copy()
    
    if best_x is None:
        best_x = np.ones(n)
        best_score = _score_signal(best_x)
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s out of 120s limit, 10s safety margin
    
    rng = np.random.default_rng(12345)
    n = N
    
    best_x = None
    best_score = np.inf
    
    # 1. Try some structured sequences
    # All ones
    x0 = np.ones(n)
    s0 = _score_signal(x0)
    best_x = x0.copy()
    best_score = s0
    
    # Alternating
    x1 = np.ones(n)
    x1[1::2] = -1
    s1 = _score_signal(x1)
    if s1 < best_score:
        best_x = x1.copy()
        best_score = s1
    
    # Random initializations
    for _ in range(10):
        if time.time() > deadline - 20:
            break
        xr = _random_signed_sequence(n, rng)
        sr = _score_signal(xr)
        if sr < best_score:
            best_x = xr.copy()
            best_score = sr
    
    # 2. Greedy local search from best
    if time.time() < deadline - 30:
        xg, sg = _greedy_local_search(best_x, max_passes=300, deadline=deadline, safety_margin=10.0)
        if sg < best_score:
            best_x = xg.copy()
            best_score = sg
    
    # 3. Simulated annealing
    if time.time() < deadline - 25:
        xa, sa = _simulated_annealing(best_x, max_iters=8000, deadline=deadline, safety_margin=8.0)
        if sa < best_score:
            best_x = xa.copy()
            best_score = sa
    
    # 4. More greedy passes after SA
    if time.time() < deadline - 15:
        xg2, sg2 = _greedy_local_search(best_x, max_passes=500, deadline=deadline, safety_margin=8.0)
        if sg2 < best_score:
            best_x = xg2.copy()
            best_score = sg2
    
    # 5. Gradient descent on complex phases
    if time.time() < deadline - 10:
        xgd, sgd = _gradient_descent_complex(n, max_iters=4000, deadline=deadline, safety_margin=8.0)
        if sgd < best_score:
            best_x = xgd.copy()
            best_score = sgd
        # Also try greedy from GD result
        if time.time() < deadline - 5:
            xg3, sg3 = _greedy_local_search(xgd, max_passes=200, deadline=deadline, safety_margin=5.0)
            if sg3 < best_score:
                best_x = xg3.copy()
                best_score = sg3
    
    # Final check: ensure we return a valid sequence
    if best_x is None or len(best_x) != n:
        best_x = np.ones(n)
    
    # Ensure it's a numpy array of the right type
    return best_x

# EVOLVE-BLOCK-END
