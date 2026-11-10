"""Find flattest plus-minus-one polynomials with 24 coefficients."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def _eval_poly(c, z):
    """Evaluate polynomial sum c_k z^k at points z (array)."""
    # c is length 24, z is array of complex numbers
    # Use Horner's method or direct evaluation
    # For efficiency with many points, use matrix multiplication
    # z^k for k=0..23
    n = len(c)
    # Build Vandermonde-like: V[i,k] = z[i]^k
    # But for memory, do it in chunks if z is large
    # For our use, z size is manageable
    V = np.ones((len(z), n), dtype=np.complex128)
    for k in range(1, n):
        V[:, k] = V[:, k-1] * z
    return V @ c

def _sup_norm_refined(c, n_points=2000, refine_iters=30):
    """Compute sup norm of |P(z)| on unit circle with golden-section refinement."""
    # Dense sampling
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    z = np.exp(1j * theta)
    vals = np.abs(_eval_poly(c, z))
    
    # Find top few peaks
    top_k = 20
    if len(vals) < top_k:
        top_k = len(vals)
    idx = np.argpartition(vals, -top_k)[-top_k:]
    
    max_val = 0.0
    for i in idx:
        # Golden section refinement around theta[i]
        # We want to maximize |P(z)| near theta[i]
        lo = theta[i] - 2*np.pi/n_points
        hi = theta[i] + 2*np.pi/n_points
        # Handle wrap-around
        if lo < 0:
            lo += 2*np.pi
            hi += 2*np.pi
        if hi > 2*np.pi:
            hi -= 2*np.pi
            lo -= 2*np.pi
        
        def neg_mag(t):
            zt = np.exp(1j * t)
            return -abs(_eval_poly(c, np.array([zt]))[0])
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2
        a, b = lo, hi
        c1 = b - gr*(b-a)
        c2 = a + gr*(b-a)
        f1 = neg_mag(c1)
        f2 = neg_mag(c2)
        for _ in range(refine_iters):
            if f1 < f2:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + gr*(b-a)
                f2 = neg_mag(c2)
            else:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - gr*(b-a)
                f1 = neg_mag(c1)
        
        peak_val = max(-f1, -f2)
        if peak_val > max_val:
            max_val = peak_val
    
    return max_val

def _score(c):
    """Score = sup_norm / sqrt(24)."""
    return _sup_norm_refined(c) / np.sqrt(24)

def _local_search(c, n_iterations=5000, timeout=None):
    """Local search: flip one coefficient at a time if it reduces the score."""
    c = list(c)
    n = len(c)
    best_score = _score(c)
    
    # Precompute for speed
    theta = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    z = np.exp(1j * theta)
    
    def quick_score(cc):
        vals = np.abs(_eval_poly(cc, z))
        return vals.max()
    
    best_quick = quick_score(c)
    improved = True
    start_time = time.time()
    
    while improved and (timeout is None or time.time() - start_time < timeout):
        improved = False
        for k in range(n):
            # Try flipping coefficient k
            old_val = c[k]
            c[k] = -old_val
            new_quick = quick_score(c)
            if new_quick < best_quick - 1e-12:
                best_quick = new_quick
                improved = True
            else:
                c[k] = old_val
    
    # Final precise score
    final_score = _score(c)
    return c, final_score

def _hill_climbing(c, max_evals=50000):
    """More aggressive local search with random restarts of single flips."""
    c = list(c)
    n = len(c)
    
    theta = np.linspace(0, 2*np.pi, 2000, endpoint=False)
    z = np.exp(1j * theta)
    
    def quick_score(cc):
        vals = np.abs(_eval_poly(cc, z))
        return vals.max()
    
    best_quick = quick_score(c)
    evals = 0
    start_time = time.time()
    timeout = 100  # seconds
    
    while evals < max_evals and time.time() - start_time < timeout:
        # Try all single flips
        best_k = -1
        best_new_quick = best_quick
        for k in range(n):
            old_val = c[k]
            c[k] = -old_val
            evals += 1
            new_quick = quick_score(c)
            if new_quick < best_new_quick:
                best_new_quick = new_quick
                best_k = k
            c[k] = old_val
            if time.time() - start_time > timeout:
                break
        
        if best_k >= 0:
            c[best_k] = -c[best_k]
            best_quick = best_new_quick
        else:
            break
    
    final_score = _score(c)
    return c, final_score

def _simulated_annealing(c, start_temp=1.0, end_temp=0.01, n_iterations=10000):
    """Simulated annealing to escape local minima."""
    c = list(c)
    n = len(c)
    
    theta = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    z = np.exp(1j * theta)
    
    def quick_score(cc):
        vals = np.abs(_eval_poly(cc, z))
        return vals.max()
    
    current_score = quick_score(c)
    best_c = list(c)
    best_score = current_score
    
    start_time = time.time()
    timeout = 80
    
    for i in range(n_iterations):
        if time.time() - start_time > timeout:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (i / n_iterations)
        
        # Random single flip
        k = np.random.randint(n)
        old_val = c[k]
        c[k] = -old_val
        new_score = quick_score(c)
        
        delta = new_score - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / (temp + 1e-10)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_c = list(c)
        else:
            c[k] = old_val
    
    final_score = _score(best_c)
    return best_c, final_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_c = p[:24]
    best_score = _score(best_c)
    
    # Local search from GS
    c_ls, score_ls = _local_search(best_c, timeout=20)
    if score_ls < best_score:
        best_c = c_ls
        best_score = score_ls
    
    # Hill climbing
    if time.time() < deadline - 30:
        c_hc, score_hc = _hill_climbing(best_c, max_evals=20000)
        if score_hc < best_score:
            best_c = c_hc
            best_score = score_hc
    
    # Simulated annealing from best
    if time.time() < deadline - 20:
        np.random.seed(42)
        c_sa, score_sa = _simulated_annealing(best_c, start_temp=0.5, end_temp=0.01, n_iterations=5000)
        if score_sa < best_score:
            best_c = c_sa
            best_score = score_sa
    
    # More SA with different seeds
    for seed in [1, 7, 13, 42, 99]:
        if time.time() > deadline - 10:
            break
        np.random.seed(seed)
        c_sa, score_sa = _simulated_annealing(best_c, start_temp=0.3, end_temp=0.005, n_iterations=3000)
        if score_sa < best_score:
            best_c = c_sa
            best_score = score_sa
    
    # Random restarts
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        c_rand = list(np.random.choice([-1, 1], size=24))
        c_ls, score_ls = _local_search(c_rand, timeout=5)
        if score_ls < best_score:
            best_c = c_ls
            best_score = score_ls
    
    # Final polish on best
    if time.time() < deadline - 5:
        c_hc, score_hc = _hill_climbing(best_c, max_evals=10000)
        if score_hc < best_score:
            best_c = c_hc
            best_score = score_hc
    
    return best_c
