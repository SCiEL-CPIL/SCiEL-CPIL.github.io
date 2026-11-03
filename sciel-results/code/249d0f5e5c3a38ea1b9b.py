"""Rewritten program for signed autocorrelation maximum absolute-value variant."""
import numpy as np
import time

N = 400

def _autoconv_full(x):
    """Compute linear autoconvolution (x * x) using FFT for speed."""
    n = len(x)
    size = 2 * n
    # Next power of 2 for efficient FFT
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    X = np.fft.rfft(x, n=fft_size)
    conv = np.fft.irfft(X * X, n=fft_size)
    return conv[:size]  # Full convolution has length 2n-1, but we take 2n to be safe (last is 0)

def _score(x):
    """
    Compute the raw value: max(|autoconv|) / (sum(x))^2.
    Lower is better.
    """
    conv = _autoconv_full(x)
    max_abs = np.max(np.abs(conv))
    integral = np.sum(x)
    if integral == 0:
        return float('inf')
    return max_abs / (integral * integral)

def _smooth_obj(x, beta=10.0):
    """
    A smooth surrogate for the max-abs objective to allow gradient-based or
    continuous optimization steps, though we stick to discrete search primarily.
    This might be useful if we implement a local search that allows non-binary values
    temporarily, but for now we use it just as a metric that correlates well.
    Actually, for binary search, exact score is best.
    """
    conv = _autoconv_full(x)
    # Soft max approximation
    return np.logsumexp(beta * np.abs(conv)) / beta

def _gen_random_binary():
    return np.random.choice([-1, 1], N)

def _gen_barker_like():
    """
    Generate a sequence with low auto-correlation properties.
    Barker codes have optimal peak-to-sidelobe ratios for small N.
    For larger N, we can use recursive constructions or random search guided by correlation.
    Here, we try a few known low-correlation structures or random restarts.
    """
    # Simple random
    return _gen_random_binary()

def _local_search_flip(x, max_iters=1000, deadline=None):
    """
    Local search by flipping individual bits.
    """
    current = x.copy()
    current_score = _score(current)
    
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        # Pick a random position to flip
        idx = np.random.randint(N)
        old_val = current[idx]
        current[idx] = -old_val
        
        new_score = _score(current)
        
        if new_score < current_score:
            current_score = new_score
        else:
            # Revert
            current[idx] = old_val
            
    return current, current_score

def _local_search_block(x, max_iters=500, block_sizes=[1, 2, 3, 5, 8], deadline=None):
    """
    Local search by flipping blocks of bits.
    """
    current = x.copy()
    current_score = _score(current)
    
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        # Pick a random block size and position
        bsize = np.random.choice(block_sizes)
        if bsize > N:
            bsize = N
        start = np.random.randint(0, N - bsize + 1)
        
        # Flip the block
        current[start:start+bsize] *= -1
        
        new_score = _score(current)
        
        if new_score < current_score:
            current_score = new_score
        else:
            # Revert
            current[start:start+bsize] *= -1
            
    return current, current_score

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We should stop well before.
    # Let's aim to finish by 140s to be safe.
    deadline = start_time + 140.0
    
    best_x = None
    best_score = float('inf')
    
    # 1. Initial Candidates
    candidates = []
    
    # All ones (bad, but a baseline)
    candidates.append(np.ones(N, dtype=int))
    
    # Alternating
    alt = np.ones(N, dtype=int)
    alt[1::2] = -1
    candidates.append(alt)
    
    # Random restarts
    for _ in range(50):
        candidates.append(_gen_random_binary())
    
    # Evaluate initial candidates
    for x in candidates:
        score = _score(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
    if time.time() > deadline:
        return best_x
        
    # 2. Local Search Phase 1: Single bit flips
    # We run local search on the best candidate found so far.
    # We might also run it on a few other good candidates in parallel (sequentially).
    
    # Let's keep a list of top candidates to explore
    # For simplicity, we just refine the best one found so far,
    # and occasionally restart local search from a new random point if stuck.
    
    stagnation_count = 0
    last_best_score = best_score
    
    while time.time() < deadline:
        # Pick a starting point for local search
        # Mostly use the best so far, but sometimes a random one to escape local minima
        if np.random.random() < 0.8:
            start_x = best_x.copy()
        else:
            start_x = _gen_random_binary()
            
        # Run local search with flips
        x_refined, score_refined = _local_search_flip(start_x, max_iters=2000, deadline=deadline)
        
        if score_refined < best_score:
            best_score = score_refined
            best_x = x_refined
            stagnation_count = 0
        else:
            stagnation_count += 1
            
        # If we've been stagnant for a while, try block flips on the best
        if stagnation_count > 5:
            x_block_refined, score_block = _local_search_block(best_x, max_iters=1000, block_sizes=[1, 2, 4, 8, 16], deadline=deadline)
            if score_block < best_score:
                best_score = score_block
                best_x = x_block_refined
                stagnation_count = 0
        
        # Check time
        if time.time() > deadline - 5:
            break
            
    return best_x

# Self-test
if __name__ == "__main__":
    x = construct()
    val = _score(x)
    print(f"Best raw value: {val}")
