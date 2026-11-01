"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _max_sidelobe(h):
    """Compute max |autocorrelation| at non-zero lags using FFT."""
    n = len(h)
    n_fft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(h.astype(np.float64), n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:n]
    return np.max(np.abs(acf[1:]))


def _flip_delta(h, i):
    """
    Compute the change in max sidelobe when flipping bit i.
    Returns the new max sidelobe value after flipping bit i.
    Uses FFT-based computation for efficiency.
    """
    n = len(h)
    n_fft = 1 << (2 * n - 1).bit_length()
    
    # Create flipped version
    h_flip = h.copy()
    h_flip[i] = -h_flip[i]
    
    F = np.fft.rfft(h_flip.astype(np.float64), n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:n]
    return np.max(np.abs(acf[1:]))


def _simulated_annealing(h, deadline, initial_temp=10.0, cooling=0.95):
    """
    Simulated annealing for discrete bit-flip search.
    """
    n = len(h)
    current_val = _max_sidelobe(h)
    best_val = current_val
    best_h = h.copy()
    
    temp = initial_temp
    rng = random.Random(12345)
    
    while time.time() < deadline:
        # Pick a random position to flip
        i = rng.randrange(n)
        
        # Compute new value
        h[i] = -h[i]
        new_val = _max_sidelobe(h)
        
        # Accept or reject
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Accept with probability exp(-(new_val - current_val) / temp)
            delta = new_val - current_val
            if delta > 0 and rng.random() < np.exp(-delta / max(temp, 1e-10)):
                current_val = new_val
            else:
                h[i] = -h[i]  # Revert
        
        # Cool down
        temp *= cooling
        
        # If temperature is very low, do a few more iterations and stop
        if temp < 0.01:
            # Do a local search pass
            for j in range(n):
                if time.time() > deadline:
                    break
                h[j] = -h[j]
                val_j = _max_sidelobe(h)
                h[j] = -h[j]
                if val_j < current_val:
                    h[j] = -h[j]
                    current_val = val_j
                    if current_val < best_val:
                        best_val = current_val
                        best_h = h.copy()
            break
    
    return best_h, best_val


def _greedy_local_search(h, deadline):
    """
    Greedy local search: repeatedly find the best single flip.
    """
    n = len(h)
    current_val = _max_sidelobe(h)
    best_val = current_val
    best_h = h.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_i = -1
        best_new_val = current_val
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            h[i] = -h[i]
            val = _max_sidelobe(h)
            h[i] = -h[i]
            
            if val < best_new_val:
                best_new_val = val
                best_i = i
        
        if best_i >= 0 and best_new_val < current_val:
            h[best_i] = -h[best_i]
            current_val = best_new_val
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
            improved = True
    
    return best_h, best_val


def _random_restart_search(deadline, num_restarts=3):
    """
    Multiple random restarts with greedy local search.
    """
    best_h = None
    best_val = np.inf
    
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        # Random initial sequence
        h = np.random.choice([-1.0, 1.0], size=N).astype(np.float64)
        
        # Greedy local search
        h, val = _greedy_local_search(h, time.time() + time_per_restart - 1)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    # Phase 1: Random restarts with greedy local search
    best_h, best_val = _random_restart_search(deadline - 30, num_restarts=2)
    
    # Phase 2: Simulated annealing from the best solution found
    if time.time() < deadline - 5:
        sa_deadline = deadline - 5
        sa_h, sa_val = _simulated_annealing(best_h.copy(), sa_deadline, 
                                            initial_temp=max(0.1, best_val * 0.1),
                                            cooling=0.98)
        if sa_val < best_val:
            best_val = sa_val
            best_h = sa_h
    
    # Phase 3: Final greedy local search polish
    if time.time() < deadline - 1:
        final_h, final_val = _greedy_local_search(best_h.copy(), deadline - 1)
        if final_val < best_val:
            best_val = final_val
            best_h = final_h
    
    return best_h
# EVOLVE-BLOCK-END
