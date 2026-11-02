import numpy as np
import time

N = 400
DEADLINE = 140.0  # safety margin before 150s kill

def _compute_c3_bound(signs):
    """
    Compute an upper bound on C3 for a signed step function.
    Uses L1 norm of autoconvolution as the metric.
    """
    dx = 0.5 / N
    n_fft = 2 * N
    F = np.fft.rfft(signs.astype(float), n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft) * dx
    return np.sum(np.abs(conv)) * dx

def _greedy_flip_optimize(signs, max_rounds=50):
    """
    Greedy coordinate descent: flip individual signs if it reduces the C3 bound.
    Multiple passes until no improvement or max rounds reached.
    """
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    
    for _ in range(max_rounds):
        improved = False
        # Try flipping each position
        for i in range(N):
            old_val = current[i]
            current[i] = -old_val
            new_score = _compute_c3_bound(current)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
        if not improved:
            break
    
    return current, current_score

def _block_flip_move(signs, block_size, rng):
    """
    Flip a random contiguous block of signs.
    Returns new signs array.
    """
    new_signs = signs.copy()
    start = rng.randint(0, N - block_size + 1)
    new_signs[start:start + block_size] = -new_signs[start:start + block_size]
    return new_signs

def _simulated_annealing(signs, start_temp=1.0, end_temp=0.01, n_iter=5000, block_sizes=None):
    """
    Simulated annealing with block-flip moves.
    """
    if block_sizes is None:
        block_sizes = [1, 2, 4, 8, 16, 32]
    
    rng = np.random.RandomState(12345)
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    best = current.copy()
    best_score = current_score
    
    for i in range(n_iter):
        # Linear cooling schedule
        temp = start_temp * (end_temp / start_temp) ** (i / max(n_iter - 1, 1))
        
        # Choose a random block size with bias toward smaller blocks
        bs = block_sizes[rng.choice(len(block_sizes), p=[0.25, 0.2, 0.15, 0.15, 0.15, 0.1])]
        if bs > N:
            bs = N
        
        new_signs = _block_flip_move(current, bs, rng)
        new_score = _compute_c3_bound(new_signs)
        
        delta = new_score - current_score
        if delta < 0 or (temp > 0 and rng.random() < np.exp(-delta / max(temp, 1e-10))):
            current = new_signs
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best = current.copy()
    
    return best, best_score

def _pairwise_swap_optimize(signs, max_rounds=20):
    """
    Try swapping pairs of signs to find improvements.
    More effective than single flips for certain structures.
    """
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    rng = np.random.RandomState(999)
    
    for _ in range(max_rounds):
        improved = False
        # Try random pairs
        for _ in range(500):
            i, j = rng.choice(N, size=2, replace=False)
            if current[i] == current[j]:
                continue
            # Swap signs at i and j
            current[i], current[j] = current[j], current[i]
            new_score = _compute_c3_bound(current)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current[i], current[j] = current[j], current[i]
        if not improved:
            break
    
    return current, current_score

def construct():
    start_time = time.time()
    
    # Start with a few diverse initializations
    rng = np.random.RandomState(42)
    
    initializations = []
    # All ones
    initializations.append(np.ones(N))
    # Alternating
    initializations.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)]))
    # Random
    initializations.append(rng.choice([1, -1], size=N))
    # Block structure: +1 -1 +1 -1
    p4 = np.ones(N)
    p4[N//4:N//2] = -1
    p4[3*N//4:] = -1
    initializations.append(p4)
    # Sine pattern
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    p5 = np.sign(np.cos(4 * np.pi * x))
    p5[p5 == 0] = 1
    initializations.append(p5)
    # Low-frequency pattern
    p7 = np.sign(np.cos(2 * np.pi * x * 3))
    p7[p7 == 0] = 1
    initializations.append(p7)
    
    best_pattern = initializations[0]
    best_score = _compute_c3_bound(best_pattern)
    
    for init in initializations[1:]:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
        s = _compute_c3_bound(init)
        if s < best_score:
            best_score = s
            best_pattern = init
    
    # Run greedy optimization on the best initial pattern
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        optimized, opt_score = _greedy_flip_optimize(best_pattern, max_rounds=30)
        if opt_score < best_score:
            best_score = opt_score
            best_pattern = optimized
    
    # Run simulated annealing to escape local minima
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 40:
        optimized_sa, sa_score = _simulated_annealing(best_pattern, start_temp=0.5, end_temp=0.01, n_iter=3000)
        if sa_score < best_score:
            best_score = sa_score
            best_pattern = optimized_sa
    
    # Run pairwise swap optimization
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        optimized2, opt_score2 = _pairwise_swap_optimize(best_pattern, max_rounds=10)
        if opt_score2 < best_score:
            best_score = opt_score2
            best_pattern = optimized2
    
    # Try simulated annealing from a different random start if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 40:
        rand_start = rng.choice([1, -1], size=N)
        rand_score = _compute_c3_bound(rand_start)
        if rand_score < best_score:
            best_score = rand_score
            best_pattern = rand_start
        optimized3, opt_score3 = _simulated_annealing(best_pattern, start_temp=0.3, end_temp=0.01, n_iter=2000)
        if opt_score3 < best_score:
            best_score = opt_score3
            best_pattern = optimized3
    
    # Final greedy polish
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20:
        optimized4, opt_score4 = _greedy_flip_optimize(best_pattern, max_rounds=10)
        if opt_score4 < best_score:
            best_score = opt_score4
            best_pattern = optimized4
    
    # Ensure we return integer signs
    return np.sign(best_pattern).astype(int)
