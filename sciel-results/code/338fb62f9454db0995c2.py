"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac[:size]))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin: 150s kill, leave 5s buffer
    
    # Precompute FFT parameters
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
        
    rng = random.Random(123)
    
    def _fast_flip_score(h, i, H):
        """Flip sign at position i, return new score and updated H."""
        v = h[i]
        h[i] = -v
        delta = -2 * v
        H_new = H + delta * np.exp(-1j * 2 * np.pi * i * np.arange(fsize // 2 + 1) / fsize)
        ac = np.fft.irfft(H_new * H_new, fsize)
        new_ac_max = np.max(np.abs(ac[:size]))
        s = np.sum(h)
        if abs(s) < 1e-9:
            new_score = 1e10
        else:
            new_score = new_ac_max / (s * s)
        return new_score, H_new
    
    def _local_search(h, max_sweeps=100, time_budget=None):
        """
        Greedy local search with sign flips.
        Uses precomputed twiddles for efficiency.
        """
        h = h.copy()
        H = np.fft.rfft(h, fsize)
        twiddles = np.exp(-1j * 2 * np.pi * np.outer(np.arange(N), np.arange(fsize // 2 + 1)) / fsize)
        
        current_val = _score(h)
        t_local_start = time.time()
        
        for sweep in range(max_sweeps):
            if time.time() > deadline - 5:
                break
            if time_budget is not None and (time.time() - t_local_start) > time_budget:
                break
            
            improved = False
            # Random order to avoid bias
            order = rng.sample(range(N), N)
            
            for i in order:
                if time.time() > deadline - 5:
                    return h, current_val
                
                v = h[i]
                h[i] = -v
                H += -2 * v * twiddles[i]
                
                ac = np.fft.irfft(H * H, fsize)
                new_ac_max = np.max(np.abs(ac[:size]))
                s = np.sum(h)
                if abs(s) < 1e-9:
                    new_score = 1e10
                else:
                    new_score = new_ac_max / (s * s)
                
                if new_score < current_val:
                    current_val = new_score
                    improved = True
                else:
                    h[i] = v
                    H -= -2 * v * twiddles[i]
            
            if not improved:
                break
        
        return h, current_val
    
    def _simulated_annealing(h, T_init=0.1, T_final=0.001, max_iter=5000):
        """Simulated annealing with sign flips."""
        h = h.copy()
        H = np.fft.rfft(h, fsize)
        twiddles = np.exp(-1j * 2 * np.pi * np.outer(np.arange(N), np.arange(fsize // 2 + 1)) / fsize)
        
        current_val = _score(h)
        best_h = h.copy()
        best_val = current_val
        T = T_init
        cooling = (T_final / T_init) ** (1.0 / max_iter)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            i = rng.randint(0, N - 1)
            v = h[i]
            h[i] = -v
            H += -2 * v * twiddles[i]
            
            ac = np.fft.irfft(H * H, fsize)
            new_ac_max = np.max(np.abs(ac[:size]))
            s = np.sum(h)
            if abs(s) < 1e-9:
                new_score = 1e10
            else:
                new_score = new_ac_max / (s * s)
            
            delta_score = new_score - current_val
            if delta_score < 0 or rng.random() < np.exp(-delta_score / max(T, 1e-10)):
                current_val = new_score
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                h[i] = v
                H -= -2 * v * twiddles[i]
            
            T *= cooling
        
        return best_h, best_val
    
    # Generate diverse initial candidates
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Random signs
    for _ in range(5):
        inits.append(np.array([1.0 if rng.random() > 0.5 else -1.0 for _ in range(N)]))
    
    # 3. Chirp patterns
    x = np.arange(N)
    inits.append(np.sign(np.cos(np.pi * x**2 / (2 * N))))
    inits.append(np.sign(np.sin(np.pi * x / N)))
    inits.append(np.sign(np.cos(2 * np.pi * x / N)))
    
    # 4. Block patterns with varying block sizes
    for block_size in [5, 10, 20, 40]:
        h = np.ones(N)
        for start in range(0, N, block_size * 2):
            h[start:start+block_size] = -1
        inits.append(h)
    
    # 5. Walsh-like patterns
    for k in [1, 2, 4, 8]:
        pattern = np.array([(1 if ((x >> j) & 1) else -1) for x in range(N) for j in range(k)]).reshape(N, k)
        # Use XOR-like combination
        h = np.ones(N)
        for j in range(k):
            h *= pattern[:, j]
        inits.append(h)
    
    # 6. Barker-like sequences (approximate for N=400)
    # Use a few structured patterns
    for period in [16, 32, 64]:
        h = np.ones(N)
        base = np.sign(np.cos(np.pi * np.arange(period) / period))
        for start in range(0, N, period):
            end = min(start + period, N)
            h[start:end] = base[:end - start]
        inits.append(h)
    
    # Evaluate all initial candidates
    best_h = inits[0].copy()
    best_val = _score(best_h)
    
    # Run local search on each initial candidate with time management
    time_per_candidate = max(5.0, (deadline - t_start - 15.0) / len(inits))
    
    for idx, h_init in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - 15 - time.time()
        time_budget = min(time_per_candidate, remaining)
        
        h_ls, val_ls = _local_search(h_init, max_sweeps=50, time_budget=time_budget)
        if val_ls < best_val:
            best_val = val_ls
            best_h = h_ls.copy()
    
    # Run simulated annealing on the best solution found so far
    if time.time() < deadline - 10:
        sa_time_budget = deadline - 10 - time.time()
        max_sa_iter = min(10000, int(sa_time_budget * 500))  # ~2ms per iter estimate
        h_sa, val_sa = _simulated_annealing(best_h, T_init=0.05, T_final=0.001, max_iter=max_sa_iter)
        if val_sa < best_val:
            best_val = val_sa
            best_h = h_sa.copy()
    
    # Final local search polish
    if time.time() < deadline - 5:
        h_final, val_final = _local_search(best_h, max_sweeps=200, time_budget=deadline - 5 - time.time())
        if val_final < best_val:
            best_val = val_final
            best_h = h_final.copy()
    
    return best_h

# EVOLVE-BLOCK-END
