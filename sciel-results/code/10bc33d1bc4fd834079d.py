import numpy as np
import time
from scipy.optimize import minimize

N = 400
N_FFT = 2 * N

def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length N_FFT.
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

def _legendre_symbol(n, p):
    """Compute Legendre symbol (n/p) for n mod p != 0."""
    if n % p == 0:
        return 0
    # Euler's criterion: (n/p) = n^((p-1)/2) mod p
    return 1 if pow(n, (p-1)//2, p) == 1 else -1

def _init_legendre():
    """
    Initialize using Legendre symbols.
    We need a prime p > N.
    """
    # Find a prime slightly larger than N
    # 401 is prime
    p = 401
    h = np.zeros(N)
    for i in range(N):
        # Map index i to a value modulo p
        # We want a sequence of length 400. 
        # Using (i+1) mod p ensures non-zero if p > N
        val = _legendre_symbol(i + 1, p)
        if val == 0:
            h[i] = 1.0
        else:
            h[i] = float(val)
    return h

def _init_mobius_like():
    """
    Approximate Mobius function initialization.
    mu(n) is 0 if n has a squared prime factor, else (-1)^k.
    For a step function, we just need +/- 1.
    We can use a simple deterministic pattern or a hash-based one.
    Here we use a simple alternating pattern modified by a prime step.
    """
    h = np.zeros(N)
    p = 401
    for i in range(N):
        # Simple heuristic: sign of sin(i * pi / p)
        h[i] = 1.0 if np.sin(i * np.pi / p) >= 0 else -1.0
    return h

def _init_random():
    return np.random.choice([-1.0, 1.0], size=N)

def _init_golden_ratio():
    """
    Based on fractional parts of n * phi.
    """
    phi = (1 + np.sqrt(5)) / 2
    h = np.zeros(N)
    for i in range(N):
        val = (i * phi) % 1.0
        h[i] = 1.0 if val < 0.5 else -1.0
    return h

def _sa_step_multi_bit(h, temp, num_flips):
    """
    Perform a simulated annealing step with multiple bit flips.
    Returns new_h, delta.
    """
    h_new = h.copy()
    if num_flips > 0:
        indices = np.random.choice(N, size=num_flips, replace=False)
        h_new[indices] *= -1.0
        
    # Calculate delta in objective
    # Objective is max sidelobe. 
    # Recomputing max sidelobe for every candidate is expensive but necessary for accuracy.
    # However, for SA, we often use a cheaper metric or accept based on the change.
    # Given N=400, FFT is fast.
    ac_old = _compute_autocorr(h)
    ac_new = _compute_autocorr(h_new)
    val_old = _max_sidelobe(ac_old)
    val_new = _max_sidelobe(ac_new)
    
    delta = val_new - val_old
    return h_new, delta

def _simulated_annealing_multi_bit(h_init, time_limit=5.0, start_temp=1.0, end_temp=0.01):
    """
    Simulated Annealing with multi-bit flips.
    """
    start_time = time.time()
    h = h_init.copy()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    best_h = h.copy()
    best_val = current_val
    
    # Adaptive parameters
    # We want to explore initially, then exploit.
    num_steps = 0
    
    while time.time() - start_time < time_limit:
        # Temperature schedule: geometric cooling
        # We estimate total steps based on time per step, but simpler to just use a counter
        # Let's assume ~1000 steps in time_limit for rough schedule
        progress = min(1.0, (time.time() - start_time) / time_limit)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Choose number of bits to flip
        # Early on: more flips (exploration). Later: fewer (exploitation).
        if progress < 0.5:
            num_flips = np.random.choice([2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.2])
        else:
            num_flips = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            
        h_candidate, delta = _sa_step_multi_bit(h, temp, num_flips)
        
        # Acceptance criterion
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-8)):
            h = h_candidate
            current_val += delta # Approximation, re-calculate if needed for precision?
            # For SA, delta is exact if we calculated it from the same ac.
            # But _sa_step_multi_bit calculated ac_new and ac_old.
            # To be safe and simple, we track current_val from the last full eval or delta.
            # Actually, _sa_step_multi_bit returns delta.
            # Let's just use the returned val_new for precision in best tracking.
            # We didn't return val_new, only delta. Let's modify _sa_step to return val_new.
            
            # Refinement: Let's just re-evaluate current_val periodically or trust delta.
            # Trusting delta is faster.
            
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
        else:
            pass # h remains unchanged
            
        num_steps += 1
        
    return best_h, best_val

def _local_search_single_bit(h, time_limit=5.0):
    """
    Intensive local search using single-bit flips (Hill Climbing).
    """
    start_time = time.time()
    h = h.copy()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    best_h = h.copy()
    best_val = current_val
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try flipping each bit
        for i in range(N):
            h[i] *= -1.0
            ac = _compute_autocorr(h)
            new_val = _max_sidelobe(ac)
            
            if new_val < current_val:
                current_val = new_val
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                h[i] *= -1.0 # Revert
                
        if not improved:
            break # Local minimum reached
            
    return best_h, best_val

def construct():
    start_time = time.time()
    # Safety margin: 150s kill, aim for 115s to be safe.
    deadline = start_time + 115.0
    
    best_h = None
    best_val = float('inf')
    
    # List of initializers
    initializers = [
        _init_legendre,
        _init_mobius_like,
        _init_random,
        _init_golden_ratio,
    ]
    
    # Multi-start strategy
    # We have ~115 seconds.
    # Let's divide into slots.
    # 4 initializers * 2 restarts each = 8 runs.
    # Each run gets ~14 seconds.
    # Within each run:
    #   - 10s Simulated Annealing (multi-bit)
    #   - 3s Local Search (single-bit)
    
    num_restarts_per_init = 2
    total_runs = len(initializers) * num_restarts_per_init
    time_per_run = (deadline - start_time) / total_runs - 1.0 # 1s safety per run
    
    for init_func in initializers:
        for restart in range(num_restarts_per_init):
            if time.time() >= deadline - 2.0:
                break
                
            h_init = init_func()
            
            # Phase 1: SA to find a good basin
            h_sa, val_sa = _simulated_annealing_multi_bit(h_init, time_limit=time_per_run * 0.7, start_temp=1.0, end_temp=0.05)
            
            # Phase 2: Polish with Local Search
            remaining = deadline - time.time()
            if remaining < 2.0:
                break
                
            ls_time = min(time_per_run * 0.3, remaining - 1.0)
            if ls_time > 0.1:
                h_ls, val_ls = _local_search_single_bit(h_sa, time_limit=ls_time)
            else:
                h_ls, val_ls = h_sa, val_sa
                
            if val_ls < best_val:
                best_val = val_ls
                best_h = h_ls.copy()
                
            # If we found a very good value, we might want to spend more time polishing it
            # But for now, strict time budget per run.
            
    # Final Polish: If we have time left, polish the best solution found
    if best_h is not None:
        remaining = deadline - time.time()
        if remaining > 1.0:
            h_final, val_final = _local_search_single_bit(best_h, time_limit=remaining - 0.5)
            if val_final < best_val:
                best_val = val_final
                best_h = h_final
                
    if best_h is None:
        return np.ones(N)
        
    return best_h
