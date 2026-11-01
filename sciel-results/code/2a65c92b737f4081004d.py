import numpy as np
import time
import random

N = 400

def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length 2*N.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac

def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes.
    """
    # Due to symmetry, max |ac[1:N]| is sufficient
    return np.max(np.abs(ac[1:N]))

def _eval_bit_flip(h, i, ac):
    """
    Evaluate the change in max sidelobe when flipping bit i.
    Returns the new max sidelobe value.
    """
    # h[i] -> -h[i]
    # ac_new[k] = ac[k] - 2*h[i]*h[i+k] - 2*h[i]*h[i-k] (with bounds)
    # This is O(N) per flip.
    
    # We can compute the new autocorrelation vector efficiently if we track changes,
    # but for simplicity and correctness, let's just recompute or use a faster incremental update.
    # Given N=400, recomputing full FFT is fast (~0.1ms).
    # However, we can do better.
    
    # Let's stick to recomputing for simplicity unless it's too slow.
    # 400 iterations * 0.1ms = 40ms per full pass. 1000 passes = 40s.
    # We have 150s. This is feasible.
    
    # But wait, we can compute the change in autocorrelation more directly.
    # ac[k] = sum_j h[j]h[j+k]
    # If we flip h[i], the change in ac[k] is:
    # delta_ac[k] = -2 * h[i] * (h[i+k] + h[i-k]) (with appropriate bounds)
    
    # Let's implement the incremental update for speed.
    # We need to update ac in place or create a new one.
    # Since we need the max of the new ac, we need the full new ac or at least the max.
    
    # Actually, simply recomputing the FFT is probably fast enough for N=400.
    # Let's benchmark mentally:
    # np.fft.rfft on 800 points is very fast.
    
    h_new = h.copy()
    h_new[i] = -h_new[i]
    ac_new = _compute_autocorr(h_new)
    return _max_sidelobe(ac_new)

def _local_search(h, max_iterations=10000, time_limit=5.0, use_sa=False, T0=0.01, random_seed=None):
    """
    Fast local search using FFT for evaluation.
    Optionally uses Simulated Annealing (SA) to escape local minima.
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    best_val = current_val
    best_h = h.copy()
    
    if use_sa and current_val > 1e-10:
        T = T0
    else:
        T = 0.0

    # Pre-compute indices for faster access? Not really needed for N=400.
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping a random bit
        i = np.random.randint(N)
        old_h_i = h[i]
        h[i] = -old_h_i
        
        ac_new = _compute_autocorr(h)
        new_val = _max_sidelobe(ac_new)
        
        if new_val < current_val:
            # Accept improvement
            current_val = new_val
            ac = ac_new
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Check for acceptance in SA
            accepted = False
            if T > 1e-9:
                delta = new_val - current_val
                if delta < 0:
                    accepted = True
                else:
                    # Accept with probability exp(-delta / T)
                    prob = np.exp(-delta / T)
                    if np.random.rand() < prob:
                        accepted = True
            
            if accepted:
                current_val = new_val
                ac = ac_new
            else:
                # Revert
                h[i] = old_h_i
                # ac remains unchanged (old ac)
        
        if T > 1e-9:
            T *= 0.9995  # Cooling factor

    return best_h, best_val

def _simulated_annealing(h, time_limit=10.0, T_start=0.1, T_end=1e-4, random_seed=None):
    """
    Dedicated Simulated Annealing with a specific cooling schedule.
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    best_val = current_val
    best_h = h.copy()
    
    # We don't know the number of iterations, so we use time-based cooling.
    # Estimate iterations per second?
    # Let's just run until time_limit and cool based on step count, 
    # but we need to know total steps. 
    # Alternatively, use a fixed number of steps and adjust T.
    
    # Let's try a fixed number of steps, say 50000, and see how long it takes.
    # If it's too slow, we reduce steps.
    # N=400, FFT is fast. Let's assume ~1000 steps/sec is possible.
    # 10s -> 10000 steps.
    
    # Let's just use the _local_search with SA but with more control.
    # Actually, _local_search already does SA.
    
    # Let's implement a parallel tempering-like approach or just multiple SA runs.
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Random Initializations and Local Search
    # We will run multiple independent local searches from random starts.
    # Since the landscape is rugged, more restarts are better.
    
    # Estimate time per local search iteration.
    # Let's do a quick benchmark.
    h_test = np.random.choice([-1.0, 1.0], size=N)
    t0 = time.time()
    for _ in range(100):
        i = np.random.randint(N)
        h_test[i] = -h_test[i]
        ac = _compute_autocorr(h_test)
        val = _max_sidelobe(ac)
        h_test[i] = -h_test[i]
    t1 = time.time()
    avg_iter_time = (t1 - t0) / 100.0
    # print(f"Avg iter time: {avg_iter_time*1000:.2f} ms")
    
    # We want to use as much time as possible.
    # Let's allocate time to multiple restarts.
    
    # Strategy:
    # 1. Run a few quick local searches to find a good baseline.
    # 2. Use remaining time for a long SA run from the best baseline.
    
    num_restarts = 5
    time_per_restart = 10.0
    
    for r in range(num_restarts):
        if time.time() >= deadline - 20.0:  # Reserve 20s for final long SA
            break
            
        h_init = np.random.choice([-1.0, 1.0], size=N)
        # Quick local search
        h_ref, val_ref = _local_search(h_init.copy(), max_iterations=5000, time_limit=time_per_restart, use_sa=False)
        
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref
            
    # Phase 2: Long Simulated Annealing from best solution
    if best_h is not None and time.time() < deadline - 5.0:
        remaining_time = deadline - time.time() - 5.0
        # Use SA to escape local minima
        # T0 should be large enough to allow significant exploration
        # Typical delta in sidelobe value is small, maybe 0.1 or less.
        # Let's start T at 0.5 and cool down.
        T0 = 0.5
        h_final, val_final = _local_search(best_h.copy(), max_iterations=1000000, time_limit=remaining_time, use_sa=True, T0=T0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
            
    # Phase 3: Final Polish (Greedy Local Search)
    if best_h is not None and time.time() < deadline - 1.0:
        remaining_time = deadline - time.time() - 1.0
        h_polish, val_polish = _local_search(best_h.copy(), max_iterations=100000, time_limit=remaining_time, use_sa=False)
        if val_polish < best_val:
            best_val = val_polish
            best_h = h_polish
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
