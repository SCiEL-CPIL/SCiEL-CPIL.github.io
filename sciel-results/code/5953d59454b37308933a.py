"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr(x):
    """Compute circular autocorrelation of x via FFT."""
    n = len(x)
    X = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(X * np.conj(X)).real[:n]
    return acf

def _peak_sidelobe(x):
    """Return the maximum absolute autocorrelation value excluding lag 0."""
    acf = _autocorr(x)
    return np.max(np.abs(acf[1:]))

def _continuous_gradient_descent(x0, n_iter=5000, lr=0.1, deadline=None):
    """
    Optimize continuous vector y to minimize L2 norm of autocorrelation sidelobes.
    Then threshold to get binary seed.
    """
    n = len(x0)
    y = x0.astype(float).copy()
    best_y = y.copy()
    best_cost = np.inf
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute autocorrelation
        Y = np.fft.fft(y, n=2*n)
        acf = np.fft.ifft(Y * np.conj(Y)).real[:n]
        
        # Cost: sum of squared sidelobes
        cost = np.sum(acf[1:]**2)
        
        if cost < best_cost:
            best_cost = cost
            best_y = y.copy()
        
        # Gradient of cost w.r.t. y
        # d/dy_k sum_l acf[l]^2 = 2 * sum_l acf[l] * d(acf[l])/d(y_k)
        # acf[l] = sum_j y[j] * y[(j+l)%n]
        # d(acf[l])/d(y_k) = y[(k+l)%n] + y[(k-l)%n]
        # So grad[k] = 2 * sum_{l=1}^{n-1} acf[l] * (y[(k+l)%n] + y[(k-l)%n])
        
        # This can be computed via convolution
        # grad[k] = 2 * (acf * y)[k] + 2 * (acf * y_rev)[k] where y_rev is reversed
        # Actually: sum_l acf[l] * y[(k+l)%n] = (acf * y)[k] (circular convolution)
        # sum_l acf[l] * y[(k-l)%n] = (acf * y)[k] as well (since acf is symmetric for real signals)
        
        # For real y, acf is symmetric: acf[l] = acf[n-l]
        # So sum_l acf[l] * y[(k+l)%n] + sum_l acf[l] * y[(k-l)%n]
        # = sum_l acf[l] * y[(k+l)%n] + sum_l acf[n-l] * y[(k+l)%n]
        # = sum_l (acf[l] + acf[n-l]) * y[(k+l)%n]
        # = 2 * sum_l acf[l] * y[(k+l)%n] for l != 0, n/2
        # For l=0: acf[0]*y[k], for l=n/2: acf[n/2]*y[(k+n/2)%n]
        
        # Simpler: just compute via FFT
        # grad = 2 * ifft(fft(acf) * conj(fft(y))) ... let me think
        # Actually, circular convolution: (a * b)[k] = sum_l a[l] * b[(k-l)%n]
        # We want sum_l acf[l] * y[(k+l)%n] = sum_l acf[l] * y[(k-(-l))%n]
        # Let m = -l, then sum_m acf[-m] * y[(k-m)%n] = (acf_rev * y)[k] where acf_rev[m] = acf[-m]
        # Since acf is symmetric (acf[m] = acf[-m]), acf_rev = acf
        # So sum_l acf[l] * y[(k+l)%n] = (acf * y)[k]
        # Similarly sum_l acf[l] * y[(k-l)%n] = (acf * y)[k]
        # So grad[k] = 4 * (acf * y)[k] ... wait, that's not right either
        # Let me just do it directly for clarity, or use the FFT approach
        
        # Using FFT: circular convolution a*b = ifft(fft(a)*fft(b))
        A = np.fft.fft(acf)
        Yf = np.fft.fft(y)
        conv = np.fft.ifft(A * Yf).real
        # grad[k] = 2 * (conv[k] + conv[k]) = 4 * conv[k]? 
        # Let me verify: sum_l acf[l]*y[(k+l)%n] is circular cross-correlation, not convolution
        # Cross-correlation: (acf ⋆ y)[k] = sum_l acf[l] * y[(k+l)%n] = ifft(fft(acf)*conj(fft(y)))
        # So grad[k] = 2 * (acf ⋆ y)[k] + 2 * (acf ⋆ y)[k] = 4 * (acf ⋆ y)[k]
        # Wait: sum_l acf[l] * y[(k+l)%n] + sum_l acf[l] * y[(k-l)%n]
        # The second term: let m = -l, sum_m acf[-m] * y[(k+m)%n] = sum_m acf[m] * y[(k+m)%n] (by symmetry)
        # So both terms are equal, giving 2 * (acf ⋆ y)[k]
        # Thus grad = 2 * ifft(fft(acf) * conj(fft(y))).real
        
        cross_corr = np.fft.ifft(np.fft.fft(acf) * np.conj(np.fft.fft(y))).real
        grad = 2.0 * cross_corr
        
        # Exclude lag 0 contribution? The cost only includes lags 1..n-1
        # grad[0] from lag 0 should be 0, but let's just zero it out
        # Actually the gradient computation above includes all lags. Let's recompute excluding lag 0.
        # acf[0] is not in the cost, so its contribution to gradient should be removed.
        # acf[0] * y[k] appears in both sums, contributing 2*acf[0]*y[k] to grad[k]
        # So subtract 2*acf[0]*y from grad
        grad -= 2.0 * acf[0] * y
        
        # Gradient descent step
        y -= lr * grad
        
        # Keep y roughly bounded
        if np.max(np.abs(y)) > 10:
            y /= np.max(np.abs(y)) * 1.1
    
    return best_y

def _binary_from_continuous(y):
    """Threshold continuous vector to binary {-1, 1}."""
    return np.sign(y)
    # If there are zeros, assign randomly
    zeros = np.where(y == 0)[0]
    if len(zeros) > 0:
        result = np.sign(y)
        result[zeros] = np.random.choice([-1, 1], size=len(zeros))
        return result
    return np.sign(y)

def _greedy_polish(x, max_iters=30000, deadline=None):
    """Greedy flip: try all positions, flip the one that most reduces peak sidelobe."""
    x = x.copy()
    n = len(x)
    acf = _autocorr(x)
    best_peak = np.max(np.abs(acf[1:]))
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if best_peak < 1:
            break
        
        best_gain = 0
        best_idx = -1
        
        # Try all positions (N=400 is small enough)
        for i in range(n):
            s = x[i]
            x[i] = -s
            new_acf = _autocorr(x)
            new_peak = np.max(np.abs(new_acf[1:]))
            x[i] = s
            
            gain = best_peak - new_peak
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx >= 0:
            x[best_idx] = -x[best_idx]
            acf = _autocorr(x)
            best_peak = np.max(np.abs(acf[1:]))
        else:
            # No improvement, break
            break
    
    return x, best_peak

def _anneal(x, deadline=None, max_iters=100000):
    """Simulated annealing to minimize peak sidelobe."""
    x = x.copy()
    n = len(x)
    best_x = x.copy()
    best_peak = _peak_sidelobe(x)
    curr_peak = best_peak
    T = 5.0
    T_min = 0.001
    cooling = 0.9995
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        i = np.random.randint(n)
        x[i] = -x[i]
        new_peak = _peak_sidelobe(x)
        delta = new_peak - curr_peak
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, T_min)):
            curr_peak = new_peak
            if new_peak < best_peak:
                best_peak = new_peak
                best_x = x.copy()
        else:
            x[i] = -x[i]  # revert
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_x, best_peak

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    np.random.seed(42)
    
    best_x = None
    best_peak = np.inf
    
    # Phase 1: Continuous gradient descent to find good binary seeds
    n_seeds = 5
    for trial in range(n_seeds):
        if time.time() > deadline:
            break
        
        # Random continuous initialization
        x0 = np.random.choice([-1, 1], size=N).astype(float)
        
        # Gradient descent on continuous relaxation
        y_cont = _continuous_gradient_descent(x0, n_iter=3000, lr=0.05, deadline=deadline)
        
        if time.time() > deadline:
            break
        
        # Threshold to binary
        x_bin = _binary_from_continuous(y_cont)
        
        # Greedy polish
        x_polished, peak = _greedy_polish(x_bin, max_iters=10000, deadline=deadline)
        
        if peak < best_peak:
            best_peak = peak
            best_x = x_polished.copy()
        
        if time.time() > deadline:
            break
    
    # Phase 2: Simulated annealing from best
    if best_x is not None and time.time() < deadline:
        x_annealed, peak_a = _anneal(best_x, deadline=deadline, max_iters=50000)
        if peak_a < best_peak:
            best_peak = peak_a
            best_x = x_annealed.copy()
    
    # Phase 3: More greedy polish if time remains
    if best_x is not None and time.time() < deadline:
        x_final, peak_f = _greedy_polish(best_x, max_iters=10000, deadline=deadline)
        if peak_f < best_peak:
            best_peak = peak_f
            best_x = x_final.copy()
    
    if best_x is None:
        best_x = np.random.choice([-1, 1], size=N)
    
    return best_x
# EVOLVE-BLOCK-END
