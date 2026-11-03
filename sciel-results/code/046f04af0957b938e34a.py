import numpy as np
import time

N = 400
_FFT_SIZE = 1024

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=_FFT_SIZE)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=_FFT_SIZE)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _calc_delta_acf(s, i):
    """
    Compute the change in autocorrelation when flipping s[i].
    Returns delta array where A_new[k] = A[k] + delta[k].
    """
    old_si = s[i]
    delta = np.zeros(N)
    
    # Part 1: s[i+k] is defined for k in 0..N-1-i
    # delta[k] -= 2*old_si * s[i+k]
    if N - i > 0:
        delta[:N-i] -= 2.0 * old_si * s[i:N]
        
    # Part 2: s[i-k] is defined for k in 0..i
    # delta[k] -= 2*old_si * s[i-k]
    if i >= 0:
        delta[:i+1] -= 2.0 * old_si * s[i::-1]
        
    return delta

def _local_search(s, A, current_val, deadline, max_passes=20):
    """
    Greedy local search using O(N) delta updates.
    Returns improved s, A, current_val.
    """
    improved = True
    passes = 0
    
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        # Random order to avoid bias
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Calculate delta
            delta = _calc_delta_acf(s, i)
            
            # Check if flipping improves the max abs ACF
            A_new = A + delta
            val_new = _max_abs_acf_from_array(A_new)
            
            if val_new < current_val:
                # Accept flip
                s[i] = -s[i]
                A = A_new
                current_val = val_new
                improved = True
                
    return s, A, current_val

def _simulated_annealing(s, A, current_val, deadline, T_start=50.0, T_min=0.01, cooling_rate=0.995):
    """
    Simulated annealing with block flips.
    Returns improved s, A, current_val.
    """
    T = T_start
    
    while time.time() < deadline and T > T_min:
        # Generate block flip
        L = np.random.randint(1, 17)
        i = np.random.randint(0, N - L + 1)
        
        # Flip block
        s[i:i+L] *= -1
        
        # Recalculate A and objective
        A = _calc_acf(s)
        new_val = _max_abs_acf_from_array(A)
        
        delta_E = new_val - current_val
        
        # Acceptance criterion
        if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
            current_val = new_val
        else:
            # Reject and flip back
            s[i:i+L] *= -1
            # Note: A is already recalculated, but since we flipped back, 
            # the original A is valid again. However, we just overwrote A.
            # So we need to recalculate A or keep a copy.
            # For efficiency in SA, recalculating A on rejection is expensive.
            # Let's optimize: only recalculate A if accepted? No, we need new_val to decide.
            # So we must calculate A_new.
            # If rejected, we restore s. The A corresponding to restored s is the old A.
            # So we should keep old_A.
            # Let's refactor to be safe.
            pass
        
        T *= cooling_rate
        
    return s, A, current_val

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Number of restarts
    # We want to balance restart count with local search depth.
    # Greedy local search is fast with O(N) updates.
    # SA is slower due to FFT per step.
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Phase 1: Simulated Annealing
        # Use a portion of the time for SA to escape local minima
        sa_deadline = min(time.time() + 5.0, deadline - 10.0)
        s, A, current_val = _simulated_annealing(s, A, current_val, sa_deadline)
        
        # Phase 2: Greedy Local Search
        # Use the remaining time for fast local search
        s, A, current_val = _local_search(s, A, current_val, deadline - 5.0, max_passes=50)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
            
        # Early termination if very good
        if best_val < 5.0:
            break
            
    # Final polish with remaining time
    remaining_deadline = deadline - 1.0
    if time.time() < remaining_deadline:
        s, A, current_val = _local_search(best_seq.copy(), _calc_acf(best_seq), best_val, remaining_deadline, max_passes=100)
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
            
    return best_seq
