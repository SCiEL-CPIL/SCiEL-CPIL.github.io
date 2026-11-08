import numpy as np
import time

N = 400

def _calc_metrics(h):
    """
    Calculate the autoconvolution, integral, and the objective score.
    Returns: max_abs_autoconv, integral, score
    """
    C = np.convolve(h, h, mode='full')
    max_abs_sid = np.max(np.abs(np.concatenate([C[:N-1], C[N:]])))
    
    integral = np.sum(h)
    int_sq = integral * integral
    
    if int_sq == 0:
        return max_abs_sid, integral, float('inf')
        
    score = max_abs_sid / int_sq
    return max_abs_sid, integral, score

def _rudin_shapiro(n):
    """Generate Rudin-Shapiro sequence of length n (padded/truncated)."""
    # Rudin-Shapiro: a(0)=1, a(1)=1, a(2n)=a(n), a(2n+1)=-a(n+1)
    # This gives low autocorrelation properties
    seq = []
    a = {0: 1, 1: 1}
    for i in range(n):
        if i not in a:
            if i % 2 == 0:
                a[i] = a[i // 2]
            else:
                a[i] = -a[(i + 1) // 2]
        seq.append(a[i])
    return np.array(seq, dtype=np.int8)

def _block_flip(h, i, block_size, rng):
    """Flip a block of consecutive elements."""
    n = len(h)
    end = min(i + block_size, n)
    h[i:end] = -h[i:end]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Strategy 1: Rudin-Shapiro initialization
    h_rs = _rudin_shapiro(N)
    _, _, score_rs = _calc_metrics(h_rs)
    
    # Strategy 2: All ones (baseline)
    h_ones = np.ones(N, dtype=np.int8)
    _, _, score_ones = _calc_metrics(h_ones)
    
    # Strategy 3: Random
    rng = np.random.default_rng(42)
    h_rand = rng.choice([-1, 1], size=N).astype(np.int8)
    _, _, score_rand = _calc_metrics(h_rand)
    
    # Pick best initial
    candidates = [(h_rs, score_rs), (h_ones, score_ones), (h_rand, score_rand)]
    candidates.sort(key=lambda x: x[1])
    h = candidates[0][0].copy()
    current_score = candidates[0][1]
    best_score = current_score
    best_h = h.copy()
    
    # Phase 1: Block-level simulated annealing
    T = 5.0
    T_min = 1e-5
    cooling_rate = 0.9998
    block_sizes = [1, 2, 4, 8]
    
    while time.time() < deadline and T > T_min:
        block_size = rng.choice(block_sizes)
        i = rng.integers(0, max(1, N - block_size + 1))
        
        old_block = h[i:i+block_size].copy()
        h[i:i+block_size] = -old_block
        
        _, _, new_score = _calc_metrics(h)
        delta = new_score - current_score
        
        if delta < 0:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            if rng.random() < np.exp(-delta / T):
                current_score = new_score
            else:
                h[i:i+block_size] = old_block
        
        T *= cooling_rate
    
    # Phase 2: Local hill climbing with single-bit flips
    h = best_h.copy()
    _, _, current_score = _calc_metrics(h)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_flip_idx = -1
        best_flip_score = current_score
        
        for i in range(N):
            if time.time() > deadline:
                break
            h[i] *= -1
            _, _, s = _calc_metrics(h)
            if s < best_flip_score:
                best_flip_score = s
                best_flip_idx = i
                improved = True
            else:
                h[i] *= -1
        
        if improved:
            h[best_flip_idx] *= -1
            current_score = best_flip_score
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            break
    
    # Phase 3: Try block flips from best state
    h = best_h.copy()
    _, _, current_score = _calc_metrics(h)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_move = None
        best_move_score = current_score
        
        for block_size in [2, 4, 8]:
            for i in range(0, N - block_size + 1, block_size):
                if time.time() > deadline:
                    break
                old_block = h[i:i+block_size].copy()
                h[i:i+block_size] = -old_block
                _, _, s = _calc_metrics(h)
                h[i:i+block_size] = old_block
                
                if s < best_move_score:
                    best_move_score = s
                    best_move = (i, block_size)
                    improved = True
        
        if improved:
            i, block_size = best_move
            h[i:i+block_size] = -h[i:i+block_size]
            current_score = best_move_score
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            break
    
    return best_h
