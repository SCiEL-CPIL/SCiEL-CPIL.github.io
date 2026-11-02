import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1] using FFT."""
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=_FFT_SIZE)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=_FFT_SIZE)
    return np.real(acf[:N])

def _hard_score(s):
    """Calculate the hard peak sidelobe magnitude."""
    A = _calc_acf(s)
    return float(np.max(np.abs(A[1:])))

def _soft_score(s, beta=10.0):
    """Calculate a smooth surrogate for the peak sidelobe using log-sum-exp."""
    A = _calc_acf(s)
    abs_A = np.abs(A[1:])
    # Square the values to emphasize large peaks more strongly in the soft max
    v = abs_A ** 2
    if len(v) == 0:
        return 0.0
    v_max = np.max(v)
    if not np.isfinite(v_max):
        return 0.0
    # Shift to prevent overflow
    shifted = v - v_max
    # Clip to prevent underflow issues in exp, though usually fine
    shifted = np.clip(shifted, -50, 50)
    exp_sum = np.sum(np.exp(beta * shifted))
    if exp_sum <= 0:
        return np.sqrt(v_max)
    # The soft max of v is approx v_max + (1/beta) * log(sum(exp(...)))
    # We want the soft max of abs_A, so we take sqrt of the soft max of v
    soft_v = v_max + (1.0 / beta) * np.log(exp_sum)
    return float(np.sqrt(soft_v))

def _vdc_init(seed=0):
    """Generate a +-1 sequence based on the van der Corput sequence.
    
    Low-discrepancy sequences provide good spectral flatness, which is
    desirable for minimizing peak autocorrelation.
    """
    vdc_vals = np.zeros(N)
    for i in range(N):
        val = 0.0
        div = 1
        n = i
        while n > 0:
            div *= 2
            if n % 2 == 1:
                val += 1.0 / div
            n //= 2
        vdc_vals[i] = val
        
    # Map [0, 1) to {-1, 1}
    # Shift the sequence to vary the phase
    shift = seed % N
    vdc_vals = np.roll(vdc_vals, shift)
    
    h = np.where(vdc_vals < 0.5, -1.0, 1.0)
    return h

def _sine_init(freq, phase, seed):
    """Generate a +-1 sequence from zero crossings of a sine wave with noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi * freq, N)
    signal = np.sin(x + phase)
    # Add small noise to break perfect periodicity
    noise = rng.normal(0, 0.1, size=N)
    h = np.sign(signal + noise)
    h[h == 0] = 1.0
    return h

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    best_val = float('inf')
    best_seq = np.ones(N, dtype=np.int8)
    
    # Initialization strategies
    def _get_initializations():
        inits = []
        # 1. Van der Corput with different shifts
        for shift in range(0, N, 50):
            inits.append(_vdc_init(shift))
        
        # 2. Sine waves with different frequencies
        for freq in [2, 3, 5, 7, 10, 15]:
            for phase in [0, np.pi/2, np.pi]:
                inits.append(_sine_init(freq, phase, seed=int(freq*10+phase*10)))
        
        # 3. Random (fallback)
        rng = np.random.default_rng(42)
        inits.append(rng.choice([-1, 1], size=N))
        
        return inits

    def _simulated_annealing(s_init, time_budget):
        """SA with soft objective for exploration, then hard for exploitation."""
        s = s_init.copy()
        current_score = _soft_score(s, beta=10.0)
        best_s = s.copy()
        best_score = current_score
        
        T = 50.0
        T_min = 0.1
        cooling = 0.995
        rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
        
        # Block sizes for flips
        block_sizes = [1, 2, 4, 8, 16, 32]
        
        while time.time() < start_time + time_budget and T > T_min:
            # Choose block size
            if rng.random() < 0.3:
                block_size = 1
            else:
                block_size = block_sizes[rng.randint(1, len(block_sizes))]
            
            # Choose start position
            start_idx = rng.randint(0, N - block_size + 1)
            
            # Flip block
            old_block = s[start_idx:start_idx + block_size].copy()
            s[start_idx:start_idx + block_size] = -old_block
            
            new_score = _soft_score(s, beta=10.0)
            delta = new_score - current_score
            
            if delta < 0:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_s = s.copy()
            else:
                # Accept with probability exp(-delta/T)
                if rng.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    s[start_idx:start_idx + block_size] = old_block
            
            T *= cooling
            
        return best_s

    def _greedy_polish(s, time_budget):
        """Greedy local search with hard objective."""
        s = s.copy()
        current_score = _hard_score(s)
        
        # Try single bit flips
        improved = True
        while improved and time.time() < start_time + time_budget:
            improved = False
            for i in range(N):
                if time.time() > start_time + time_budget:
                    break
                s[i] = -s[i]
                new_score = _hard_score(s)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    s[i] = -s[i] # revert
                    
        # Try small block flips (size 2, 4) if time permits
        if time.time() < start_time + time_budget - 5.0:
            for block_size in [2, 4]:
                for _ in range(100): # Limited iterations
                    if time.time() > start_time + time_budget:
                        break
                    start_idx = np.random.randint(0, N - block_size + 1)
                    old_block = s[start_idx:start_idx + block_size].copy()
                    s[start_idx:start_idx + block_size] = -old_block
                    new_score = _hard_score(s)
                    if new_score < current_score:
                        current_score = new_score
                    else:
                        s[start_idx:start_idx + block_size] = old_block
                        
        return s, current_score

    # Main loop
    inits = _get_initializations()
    
    # Sort inits by initial soft score to prioritize good starts
    # This costs some time upfront but can save search time
    scored_inits = []
    for s in inits:
        score = _soft_score(s, beta=10.0)
        scored_inits.append((score, s))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Allocate time budget
    # Reserve 10s for final polish and safety
    search_time = deadline - 10.0
    num_restarts = len(scored_inits)
    time_per_restart = max(2.0, (search_time - 5.0) / num_restarts)
    
    for score, s_init in scored_inits:
        if time.time() > deadline - 10.0:
            break
            
        # Run SA
        s_sa = _simulated_annealing(s_init, time_per_restart)
        
        # Evaluate hard score
        hard_val = _hard_score(s_sa)
        
        if hard_val < best_val:
            best_val = hard_val
            best_seq = s_sa.copy()
            
        # If we have time, do a quick greedy polish on this candidate
        if time.time() < deadline - 5.0:
            s_polished, val_polished = _greedy_polish(s_sa, 2.0)
            if val_polished < best_val:
                best_val = val_polished
                best_seq = s_polished.copy()

    # Final polish on the best sequence found
    if time.time() < deadline - 2.0:
        best_seq, best_val = _greedy_polish(best_seq, deadline - time.time() - 1.0)

    return best_seq
