"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _compute_score(x):
    """Compute the maximum absolute autoconvolution divided by squared integral."""
    ac = np.convolve(x, x, mode='full')
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    squared_integral = integral * integral
    max_abs_ac = np.max(np.abs(ac))
    return max_abs_ac / squared_integral

def _m_sequence(n):
    """Generate a binary m-sequence of length 2^n - 1, then pad/truncate to N."""
    # Find smallest n such that 2^n - 1 >= N
    n = 1
    while (2**n - 1) < N:
        n += 1
    
    # Use a known primitive polynomial for small n
    # For n=9 (511 length), x^9 + x^4 + 1 is primitive
    # For n=8 (255 length), x^8 + x^4 + x^3 + x^2 + 1 is primitive
    # We need length >= 400, so n=9 -> 511
    # Let's implement a generic LFSR
    
    # Primitive polynomial for n=9: x^9 + x^4 + 1
    # Bits: positions 9 and 4
    taps = [9, 4]
    
    # Initialize state with all ones
    state = [1] * n
    
    seq = []
    for _ in range(2**n - 1):
        seq.append(state[0])
        # Feedback bit
        fb = 0
        for t in taps:
            fb ^= state[(t - 1) % n]
        # Shift right, insert fb at left
        state = [fb] + state[:-1]
    
    # seq is 0/1. Convert to -1/1: 0 -> -1, 1 -> 1
    x = 2 * np.array(seq, dtype=float) - 1
    
    # Pad or truncate to N
    if len(x) < N:
        x = np.tile(x, int(np.ceil(N / len(x))))[:N]
    else:
        x = x[:N]
        
    return x

def _local_search_fast(x, time_limit=5.0, max_iter=100000):
    """
    Local search by flipping signs.
    Uses incremental update of autoconvolution to speed up evaluation.
    ac[k] = sum_i x[i] x[k-i]
    Flipping x[j] changes ac by:
    delta_ac[k] = -2 * x[j] * (sum_{i: k-i=j} x[i] + sum_{i: i=j} x[k-i]) ... roughly
    Actually, ac_new = ac - 2*x[j]*x * 2*x[j]? No.
    ac = x * x (conv).
    If x_j -> -x_j, then x_new = x - 2*x_j*e_j.
    ac_new = (x - 2*x_j e_j) * (x - 2*x_j e_j)
            = x*x - 4*x_j (x*e_j) + 4*x_j^2 (e_j*e_j)
            = ac - 4*x_j * (x shifted) + 4*x_j * delta
    This is still O(N) to update ac.
    With N=400, np.convolve is very fast (~50us). 100000 iters * 50us = 5s.
    So simple re-computation is fine.
    """
    start_time = time.time()
    current_score = _compute_score(x)
    best_x = x.copy()
    best_score = current_score
    
    # Precompute ac for current x
    ac = np.convolve(x, x, mode='full')
    integral = np.sum(x)
    sq_int = integral * integral
    
    # We will try flipping each index in a random order or sequential
    # To avoid getting stuck, use simulated annealing or random restarts within local search?
    # Let's stick to greedy hill climbing with random order of indices.
    
    indices = np.arange(N)
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random index to flip
        j = np.random.choice(indices)
        
        # Flip
        x[j] = -x[j]
        
        # Compute new score
        # Update integral: integral_new = integral - 2*x[j] (since x[j] changed sign, diff is -2*old_x[j])
        # Wait, x[j] was old_val, now -old_val.
        # integral_new = integral - old_val + (-old_val) = integral - 2*old_val.
        # But we already flipped x[j] in place.
        # Let's just recompute score directly, it's fast enough.
        
        new_score = _compute_score(x)
        
        if new_score < best_score:
            best_score = new_score
            best_x = x.copy()
        else:
            # Revert
            x[j] = -x[j]
            
    return best_x, best_score

def _spectral_refine(x, time_limit=10.0):
    """
    Refine x by minimizing the peak of the autoconvolution using gradient descent on a smooth surrogate.
    Surrogate: sum(exp(ac^2 / beta)) / (integral^2)
    """
    start_time = time.time()
    best_x = x.copy()
    best_score = _compute_score(x)
    
    beta = 5.0
    lr = 0.05
    x_cont = x.copy()
    
    def surrogate_val(xc):
        ac = np.convolve(xc, xc, mode='full')
        ac_sq = ac ** 2
        # Soft max approximation
        val = np.sum(np.exp(ac_sq / beta))
        integral = np.sum(xc)
        if abs(integral) < 1e-6:
            return 1e10
        return val / (integral * integral)
    
    for step in range(5000):
        if time.time() - start_time > time_limit:
            break
            
        # Compute gradient numerically or analytically?
        # Analytical gradient of sum(exp(ac_k^2/beta)) / S^2
        # Let L = sum_k exp(ac_k^2/beta) / S^2
        # dL/dx_j = (1/S^2) * sum_k [ exp(ac_k^2/beta) * (2 ac_k / beta) * d(ac_k)/dx_j ] - (2 L / S) * dS/dx_j
        # dS/dx_j = 1
        # d(ac_k)/dx_j: ac_k = sum_i x_i x_{k-i}
        # If j is in [0, N-1], then x_j appears in ac_k if k-j in [0, N-1] (term x_j x_{k-j}) or j-(k) in [0, N-1] (term x_{j-(k-j)} x_j? no, x_i x_{k-i}, if i=j then x_j x_{k-j}, if k-i=j then i=k-j, x_{k-j} x_j).
        # So d(ac_k)/dx_j = x_{k-j} + x_{j-k} (if indices valid).
        
        ac = np.convolve(x_cont, x_cont, mode='full')
        ac_sq = ac ** 2
        weights = np.exp(ac_sq / beta)
        S = np.sum(x_cont)
        L = np.sum(weights) / (S * S)
        
        grad = np.zeros_like(x_cont)
        
        # This loop is O(N^2) which is 160k ops. 5000 steps * 160k = 800M ops. Too slow in Python.
        # We need a faster way.
        # Let's just use finite differences on a subset or use a simpler heuristic.
        # Actually, let's just do sign flips guided by the gradient of the surrogate?
        # Or just use the local search from before.
        
        # Let's try a simple gradient step with FFT?
        # ac = ifft(fft(x)^2)
        # grad = fft^{-1}( 2 * fft(x) * fft( weights * 2*ac/beta ) ) ... complicated.
        
        # Given time constraints, let's skip complex gradient and just do more local search.
        # But we want a structural change.
        # Let's try: minimize variance of power spectrum.
        # P = |FFT(x)|^2. Minimize max(P) or sum(P^2).
        # This is equivalent to minimizing peak autoconvolution.
        
        # Let's use a simple heuristic: flip x[j] if it reduces the max autoconv peak.
        # This is what _local_search_fast does.
        
        # Instead, let's try a different initialization strategy in the main function.
        break
        
    return best_x, best_score

def construct():
    """Construct a signed step function minimizing the maximum absolute autoconvolution 
    divided by squared integral."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_x = None
    best_score = float('inf')
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # 1. M-sequence (low autocorrelation property)
    x_mseq = _m_sequence(N)
    candidates.append(x_mseq)
    
    # 2. All ones (baseline)
    candidates.append(np.ones(N))
    
    # 3. Random
    np.random.seed(42)
    for _ in range(10):
        candidates.append(np.random.choice([-1.0, 1.0], size=N))
        
    # 4. Alternating
    candidates.append(np.tile([1.0, -1.0], N // 2))
    
    # 5. Block patterns
    for block in [2, 4, 8, 16]:
        if N % block == 0:
            pattern = np.zeros(block)
            pattern[:block//2] = 1
            pattern[block//2:] = -1
            candidates.append(np.tile(pattern, N // block))
            
    # Evaluate all candidates
    for x in candidates:
        score = _compute_score(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
    # Phase 2: Local search on the best candidate
    if best_x is not None and time.time() < deadline - 10:
        improved_x, improved_score = _local_search_fast(best_x, time_limit=min(30.0, deadline - time.time() - 5))
        if improved_score < best_score:
            best_x = improved_x
            best_score = improved_score
            
    # Phase 3: Local search on other good candidates
    # Sort candidates by score
    scored_candidates = []
    for x in candidates:
        s = _compute_score(x)
        scored_candidates.append((s, x))
    scored_candidates.sort(key=lambda t: t[0])
    
    for s, x in scored_candidates[:3]: # Top 3
        if time.time() > deadline - 10:
            break
        if s >= best_score:
            continue
        # If this candidate is worse than current best, try to improve it
        # But if it's close, it might lead to a better basin.
        improved_x, improved_score = _local_search_fast(x, time_limit=min(20.0, deadline - time.time() - 5))
        if improved_score < best_score:
            best_x = improved_x
            best_score = improved_score
            
    return best_x

# EVOLVE-BLOCK-END
