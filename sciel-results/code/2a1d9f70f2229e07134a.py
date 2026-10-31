"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    def autoconv_peak(x):
        """Compute max absolute autoconvolution value."""
        if np.all(x == 0):
            return 0.0
        ac = fftconvolve(x, x, mode='full')
        return np.max(np.abs(ac))

    def score(x):
        """max|autoconv| / (sum)^2 - lower is better."""
        s = np.sum(x)
        if abs(s) < 1e-12:
            return float('inf')
        return autoconv_peak(x) / (s * s)

    def score_incremental(x, ac, idx, old_val, new_val):
        """Update autoconvolution after changing x[idx] from old_val to new_val.
        Returns (new_ac, new_score) without full FFT."""
        delta = new_val - old_val
        if abs(delta) < 1e-15:
            return ac, score(x)
        
        # ac[k] = sum_i x[i]*x[i+k-1] for full convolution
        # Changing x[idx] by delta affects ac at positions related to idx
        # ac_full has length 2*N-1, index k corresponds to lag k-(N-1)
        
        # For full convolution: ac[m] = sum_i x[i]*x[m-i+1-N+1]... 
        # Actually, let's just compute the change directly
        # ac_new = ac + delta * (x shifted appropriately)
        
        # The autoconvolution at lag l: sum_i x[i]*x[i+l]
        # If we change x[idx] by delta, then:
        #   ac_new[l] = ac[l] + delta*x[idx+l] + delta*x[idx-l] (for valid indices)
        # In full convolution indexing, lag l corresponds to index l + (N-1)
        
        n = len(x)
        ac_new = ac.copy()
        
        # Positive lags: l = 0, 1, ..., n-1
        for l in range(n):
            # ac index for lag l is l + (n-1)
            ac_idx = l + (n - 1)
            # Contribution: delta * x[idx+l] if idx+l < n
            if idx + l < n:
                ac_new[ac_idx] += delta * x[idx + l]
            # Contribution: delta * x[idx-l] if idx-l >= 0 and l > 0
            if l > 0 and idx - l >= 0:
                ac_new[ac_idx] += delta * x[idx - l]
        
        # Update x
        x[idx] = new_val
        s = np.sum(x)
        if abs(s) < 1e-12:
            return ac_new, float('inf')
        peak = np.max(np.abs(ac_new))
        return ac_new, peak / (s * s)

    def local_search_binary(x, max_iters=200, batch_size=50):
        """Local search on binary {-1,1} sequence by flipping signs with incremental updates."""
        ac = fftconvolve(x, x, mode='full')
        current_val = score(x)
        improved = True
        it = 0
        while improved and it < max_iters:
            if time.time() > deadline:
                break
            improved = False
            it += 1
            
            non_zero_idx = np.where(x != 0)[0]
            if len(non_zero_idx) == 0:
                break
            
            # Try multiple flips per iteration
            for _ in range(min(batch_size, len(non_zero_idx))):
                if time.time() > deadline:
                    break
                
                trial_idx = rng.choice(non_zero_idx)
                orig = x[trial_idx]
                new_val = -orig
                
                ac_new, new_val_score = score_incremental(x, ac, trial_idx, orig, new_val)
                
                if new_val_score < current_val - 1e-12:
                    ac = ac_new
                    current_val = new_val_score
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_x = x.copy()
                else:
                    # Revert
                    x[trial_idx] = orig

        return current_val

    def generate_chirp(n, phase=0.0):
        """Generate a chirp signal."""
        t = np.arange(n) / n
        freq = 0.5 + 2.0 * t  # linearly increasing frequency
        return np.sign(np.sin(2 * np.pi * (freq * t + phase)))

    def generate_block(n, n_blocks, block_len):
        """Generate block pattern."""
        x = np.zeros(n)
        pos = 0
        sign = 1.0
        for _ in range(n_blocks):
            end = min(pos + block_len, n)
            x[pos:end] = sign
            sign = -sign
            pos = end
            if pos >= n:
                break
        return x

    best_x = np.ones(N)
    best_val = score(best_x)

    rng = np.random.default_rng(42)

    # Generate diverse initializations
    initializations = []
    
    # Random sparse
    for density in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        mask = rng.random(N) < density
        x = np.zeros(N)
        x[mask] = rng.choice([-1.0, 1.0], size=np.sum(mask))
        if abs(np.sum(x)) > 1e-12:
            initializations.append(x)
    
    # Chirp patterns
    for phase in [0.0, 0.25, 0.5, 0.75]:
        initializations.append(generate_chirp(N, phase))
    
    # Block patterns
    for n_blocks in [4, 8, 16, 32]:
        block_len = max(1, N // n_blocks)
        initializations.append(generate_block(N, n_blocks, block_len))
    
    # All ones and all alternating
    initializations.append(np.ones(N))
    initializations.append(np.ones(N) * np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))

    # Evaluate all initializations and run local search on best ones
    init_scores = []
    for x in initializations:
        if time.time() > deadline:
            break
        s = np.sum(x)
        if abs(s) < 1e-12:
            continue
        val = score(x)
        init_scores.append((val, x))
    
    init_scores.sort(key=lambda t: t[0])
    
    # Run local search on top initializations
    n_ls = min(10, len(init_scores))
    for val, x in init_scores[:n_ls]:
        if time.time() > deadline:
            break
        if val >= best_val:
            break
        local_search_binary(x.copy(), max_iters=150, batch_size=30)

    # Final polish on best solution
    if time.time() < deadline - 5:
        x = best_x.copy()
        ac = fftconvolve(x, x, mode='full')
        current_val = score(x)
        for trial in range(500):
            if time.time() > deadline - 2:
                break
            idx = rng.choice(N)
            orig = x[idx]
            new_val = -orig if orig != 0 else rng.choice([-1.0, 1.0])
            
            ac_new, new_val_score = score_incremental(x, ac, idx, orig, new_val)
            
            if new_val_score < current_val - 1e-12:
                ac = ac_new
                current_val = new_val_score
                if current_val < best_val:
                    best_val = current_val
                    best_x = x.copy()
            else:
                x[idx] = orig

    return best_x
# EVOLVE-BLOCK-END
