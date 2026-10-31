"""Optimized signed step function to minimize max |autoconv| / (integral)^2."""
import numpy as np
import time

N = 400
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds


def _autoconv(a):
    """Compute full autoconvolution of a."""
    # For real signals, autoconvolution is the same as autocorrelation with flip
    # But actually autoconvolution is (a * a)[k] = sum_j a[j] a[k-j]
    # This is the same as the convolution of a with a
    return np.convolve(a, a, mode='full')


def _score(a):
    """Compute the C3' score: max |autoconv| / (integral)^2. Lower is better."""
    integral = np.sum(a)
    if abs(integral) < 1e-12:
        return np.inf
    ac = _autoconv(a)
    max_abs_ac = np.max(np.abs(ac))
    return max_abs_ac / (integral ** 2)


def _random_signed_step(rng, N):
    """Generate a random signed step function."""
    return np.ones(N) * (2 * rng.random(N) - 1)


def _greedy_local_search(a, max_iters=5000, deadline=None):
    """
    Greedy local search: flip individual elements to reduce the score.
    Use a time budget to stop early.
    """
    current_score = _score(a)
    a = a.copy()
    n = len(a)
    
    # Precompute for faster updates - we'll use a simpler approach
    # Track which indices have been tried recently
    tried = np.zeros(n, dtype=bool)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Randomly pick an index to try flipping
        idx = np.random.randint(n)
        
        # Try flipping
        old_val = a[idx]
        a[idx] = -old_val
        new_score = _score(a)
        
        if new_score < current_score:
            current_score = new_score
            tried[idx] = True
        else:
            a[idx] = old_val
    
    return a, current_score


def _pairwise_local_search(a, max_iters=2000, deadline=None):
    """
    Local search that flips pairs of elements.
    """
    current_score = _score(a)
    a = a.copy()
    n = len(a)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick two distinct indices
        i, j = np.random.choice(n, 2, replace=False)
        
        # Try flipping both
        old_i, old_j = a[i], a[j]
        a[i] = -old_i
        a[j] = -old_j
        new_score = _score(a)
        
        if new_score < current_score:
            current_score = new_score
        else:
            a[i] = old_i
            a[j] = old_j
    
    return a, current_score


def construct():
    """
    Construct a signed step function that minimizes the C3' score.
    Returns an array of N signed values.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    rng = np.random.RandomState(42)
    
    # Start with a random signed step function
    best_a = _random_signed_step(rng, N)
    best_score = _score(best_a)
    
    # Multiple restarts with local search
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate a new candidate
        a = _random_signed_step(rng, N)
        score = _score(a)
        
        # Greedy local search
        a, score = _greedy_local_search(a, max_iters=3000, deadline=deadline)
        
        if score < best_score:
            best_score = score
            best_a = a.copy()
        
        # Pairwise local search on the best
        if time.time() > deadline:
            break
        a2, score2 = _pairwise_local_search(best_a, max_iters=1000, deadline=deadline)
        if score2 < best_score:
            best_score = score2
            best_a = a2.copy()
    
    # Final polish with more iterations
    if time.time() < deadline:
        best_a, final_score = _greedy_local_search(best_a, max_iters=5000, deadline=deadline)
        if final_score < best_score:
            best_score = final_score
    
    return best_a


# Test
if __name__ == "__main__":
    a = construct()
    print(f"Score: {_score(a)}")
    print(f"Shape: {a.shape}")
    print(f"Integral: {np.sum(a)}")
