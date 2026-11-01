import numpy as np
import time
from scipy.optimize import minimize

N = 400
N_FFT = 2 * N

def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length 2*N.
    """
    F = np.fft.rfft(h, n=N_FFT)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=N_FFT)
    return ac

def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes.
    """
    return np.max(np.abs(ac[1:N]))

def _metric(h):
    """
    Compute the C3 upper bound metric.
    Assumed to be max_sidelobe / N.
    """
    ac = _compute_autocorr(h)
    return _max_sidelobe(ac) / N

def _mobius_function(n):
    """Compute Möbius function for 1..n."""
    mu = np.ones(n + 1, dtype=np.int8)
    primes = []
    for i in range(2, n + 1):
        if mu[i] == 1:  # i is prime
            primes.append(i)
            for j in range(2 * i, n + 1, i):
                mu[j] *= -1
            for j in range(i * i, n + 1, i * i):
                mu[j] = 0
    return mu[1:n+1]

def _legendre_symbol(n, p):
    """Compute Legendre symbol for 1..n modulo prime p."""
    # p must be prime > n
    # Find smallest prime > n
    cand = n + 1
    while True:
        is_prime = True
        for i in range(2, int(np.sqrt(cand)) + 1):
            if cand % i == 0:
                is_prime = False
                break
        if is_prime:
            p = cand
            break
        cand += 1
    
    # Compute (a/p) using Euler's criterion: a^((p-1)/2) % p
    # Or simpler: since we just need a low autocorrelation sequence, 
    # we can just use the quadratic residue property.
    # (a/p) = 1 if a is quadratic residue, -1 if not, 0 if a%p==0.
    # For a=1..n, a%p != 0 since p > n.
    
    # Efficient calculation
    # a^((p-1)/2) % p
    exp = (p - 1) // 2
    res = np.ones(n, dtype=np.int8)
    for i in range(1, n + 1):
        # pow(i, exp, p)
        val = pow(i, exp, p)
        if val == 1:
            res[i-1] = 1
        else:
            res[i-1] = -1
    return res

def _local_search_sa_fast(h, time_limit=10.0):
    """
    Simulated Annealing with fast autocorrelation update.
    h: array of +/- 1
    """
    start_time = time.time()
    # Initialize ac
    ac_full = _compute_autocorr(h)
    ac_pos = ac_full[1:N].copy() # lags 1..N-1
    
    current_val = np.max(np.abs(ac_pos))
    
    T = 0.5
    T_end = 0.01
    
    # Precompute indices for fast update?
    # For a flip at i, we update ac_pos[k-1] for k where 0 <= i+k < N or 0 <= i-k < N.
    
    while time.time() - start_time < time_limit:
        T = max(T_end, T * 0.995)
        
        i = np.random.randint(N)
        old_val = h[i]
        h[i] = -old_val
        
        # Update ac_pos
        # Delta for lag k (index k-1 in ac_pos) is:
        # -2 * old_val * (h[i+k] if i+k < N else 0 + h[i-k] if i-k >= 0 else 0)
        
        # Vectorized update for all k
        # k ranges from 1 to N-1
        # Term 1: h[i+k] is valid for k <= N-1-i. Indices in h: i+1 .. N-1
        # Term 2: h[i-k] is valid for k <= i. Indices in h: 0 .. i-1
        
        # Let's construct the delta vector
        delta_ac = np.zeros(N-1)
        
        # Contributions from h[i+k]
        # k=1..N-1-i
        # h indices: i+1 .. N-1
        if i < N - 1:
            delta_ac[:N-1-i] -= 2 * old_val * h[i+1:]
            
        # Contributions from h[i-k]
        # k=1..i
        # h indices: i-1 .. 0 (reversed)
        if i > 0:
            delta_ac[:i] -= 2 * old_val * h[:i][::-1]
            
        ac_pos += delta_ac
        
        new_val = np.max(np.abs(ac_pos))
        
        delta = current_val - new_val
        if delta > 0 or np.random.rand() < np.exp(-delta / T):
            current_val = new_val
        else:
            # Revert
            h[i] = old_val
            ac_pos -= delta_ac
            
    return h, current_val / N

def _local_search_steepest_fast(h, time_limit=10.0):
    """
    Steepest Descent with fast autocorrelation update.
    """
    start_time = time.time()
    ac_full = _compute_autocorr(h)
    ac_pos = ac_full[1:N].copy()
    current_val = np.max(np.abs(ac_pos))
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        best_delta = 0
        best_idx = -1
        best_ac_pos_new = None
        best_val = current_val
        
        for i in range(N):
            old_val = h[i]
            # h[i] = -old_val
            
            delta_ac = np.zeros(N-1)
            if i < N - 1:
                delta_ac[:N-1-i] -= 2 * old_val * h[i+1:]
            if i > 0:
                delta_ac[:i] -= 2 * old_val * h[:i][::-1]
                
            ac_pos_trial = ac_pos + delta_ac
            new_val = np.max(np.abs(ac_pos_trial))
            
            delta = current_val - new_val
            if delta > best_delta:
                best_delta = delta
                best_idx = i
                best_ac_pos_new = ac_pos_trial
                best_val = new_val
        
        if best_idx != -1:
            h[best_idx] = -h[best_idx]
            ac_pos = best_ac_pos_new
            current_val = best_val
            improved = True
            
    return h, current_val / N

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Generate diverse initializations
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N, dtype=np.float64))
    
    # 2. Alternating
    alt = np.ones(N, dtype=np.float64)
    alt[1::2] = -1.0
    inits.append(alt)
    
    # 3. Random
    np.random.seed(42)
    inits.append(np.random.choice([-1.0, 1.0], size=N))
    
    # 4. Möbius
    mu = _mobius_function(N)
    # Map 0 to 1 (or -1, doesn't matter much, but 0 is neutral)
    mu[mu == 0] = 1.0
    inits.append(mu.astype(np.float64))
    
    # 5. Legendre
    leg = _legendre_symbol(N, 1000003) # Use a large prime
    inits.append(leg.astype(np.float64))
    
    # Evaluate inits
    for h_init in inits:
        val = _metric(h_init)
        if val < best_val:
            best_val = val
            best_h = h_init.copy()
            
    # Refine the best few initializations
    # Sort inits by initial metric
    scored_inits = []
    for h_init in inits:
        val = _metric(h_init)
        scored_inits.append((val, h_init))
    scored_inits.sort(key=lambda x: x[0])
    
    # Take top 3
    for val_init, h_init in scored_inits[:3]:
        if time.time() >= deadline - 20.0:
            break
            
        # Steepest Descent
        remaining = deadline - time.time()
        if remaining > 10.0:
            t_limit = min(10.0, remaining - 10.0)
            h_ref, val_ref = _local_search_steepest_fast(h_init.copy(), time_limit=t_limit)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref
        
        # Simulated Annealing
        remaining = deadline - time.time()
        if remaining > 10.0:
            t_limit = min(10.0, remaining - 5.0)
            h_sa, val_sa = _local_search_sa_fast(h_init.copy(), time_limit=t_limit)
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa
                
        # Final Polish
        remaining = deadline - time.time()
        if remaining > 5.0:
            t_limit = min(5.0, remaining - 1.0)
            h_final, val_final = _local_search_steepest_fast(best_h.copy(), time_limit=t_limit)
            if val_final < best_val:
                best_val = val_final
                best_h = h_final

    if best_h is None:
        return np.ones(N)
        
    return best_h
