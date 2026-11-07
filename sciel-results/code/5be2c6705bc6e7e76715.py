"""Rewritten program: discrete assignment with greedy local search + continuous polish."""
import numpy as np
import time

N = 100

def _objective(h):
    """Compute the evaluation value (lower is better)."""
    # value = max(np.correlate(h, 1 - h, mode="full")) / len(h) * 2
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / len(h) * 2.0

def _is_valid(h):
    """Check constraints: finite, in [0,1], sum ~ N/2."""
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1.0 + 1e-9):
        return False
    if abs(np.sum(h) - N / 2.0) > 1e-9 * N:
        return False
    return True

def _clamp_and_normalize(h):
    """Clamp to [0,1] and adjust sum to be exactly N/2."""
    h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(h)
    # Distribute the difference uniformly, then re-clamp
    h = h + diff / N
    h = np.clip(h, 0.0, 1.0)
    # If still off, do a second pass on the non-clamped entries
    diff2 = N / 2.0 - np.sum(h)
    mask = (h > 0) & (h < 1)
    if np.sum(mask) > 0:
        h[mask] += diff2 / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def _greedy_swap(h, max_swaps=5000):
    """Greedy local search: swap pairs to reduce max correlation."""
    best_val = _objective(h)
    h = h.copy()
    for _ in range(max_swaps):
        improved = False
        # Try random swaps
        for _attempt in range(200):
            i, j = np.random.randint(0, N, 2)
            if i == j:
                continue
            h[i], h[j] = h[j], h[i]
            new_val = _objective(h)
            if new_val < best_val - 1e-12:
                best_val = new_val
                improved = True
                break
            else:
                h[i], h[j] = h[j], h[i]
        if not improved:
            # Try all pairs for a more thorough search (expensive but N=100)
            found = False
            for i in range(N):
                for j in range(i + 1, N):
                    h[i], h[j] = h[j], h[i]
                    new_val = _objective(h)
                    if new_val < best_val - 1e-12:
                        best_val = new_val
                        found = True
                        break
                    else:
                        h[i], h[j] = h[j], h[i]
                if found:
                    break
            if not found:
                break
    return h, best_val

def _continuous_polish(h, max_iters=2000):
    """Gradient-based polish using projected gradient descent on the max-correlation objective."""
    h = h.astype(float).copy()
    best_h = h.copy()
    best_val = _objective(h)
    
    lr = 0.01
    for it in range(max_iters):
        # Compute gradient of max correlation
        corr = np.correlate(h, 1.0 - h, mode="full")
        max_idx = np.argmax(corr)
        
        # Gradient of corr[k] with respect to h
        # corr[k] = sum_i h[i] * (1-h[i+k-99]) for mode="full"
        # d corr[k] / d h[j] = (1 - h[j + k - 99]) if 0 <= j+k-99 < N
        #                    - h[j - (99 - k)] if 0 <= j - (99-k) < N
        #                    (both terms if applicable)
        # For full mode, lag k ranges from -(N-1) to (N-1), index in corr array is k + (N-1)
        # corr_idx = max_idx, lag = max_idx - (N-1)
        lag = max_idx - (N - 1)
        
        grad = np.zeros(N)
        for j in range(N):
            # Term 1: h[j] * (1 - h[j+lag]) if 0 <= j+lag < N
            if 0 <= j + lag < N:
                grad[j] += (1.0 - h[j + lag])
            # Term 2: (1 - h[j]) * h[j - lag] if 0 <= j - lag < N
            if 0 <= j - lag < N:
                grad[j] += -h[j - lag]
        
        # Gradient descent with projection
        h_new = h - lr * grad
        h_new = _clamp_and_normalize(h_new)
        
        new_val = _objective(h_new)
        if new_val < best_val:
            best_val = new_val
            best_h = h_new.copy()
            h = h_new
            lr = min(lr * 1.05, 0.1)
        else:
            lr = max(lr * 0.8, 1e-5)
            if lr < 1e-6:
                break
    
    return best_h, best_val

def construct():
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    # Start with binary assignment: 50 zeros and 50 ones
    h = np.zeros(N)
    h[:N//2] = 1.0
    # Shuffle
    np.random.shuffle(h)
    
    best_h = h.copy()
    best_val = _objective(h)
    
    # Phase 1: Greedy swap local search
    h_swapped, val_swapped = _greedy_swap(h, max_swaps=3000)
    if val_swapped < best_val:
        best_val = val_swapped
        best_h = h_swapped.copy()
    
    if time.time() > deadline - 30:
        return _clamp_and_normalize(best_h)
    
    # Phase 2: Continuous polish from the binary solution
    h_polished, val_polished = _continuous_polish(h_swapped, max_iters=3000)
    if val_polished < best_val:
        best_val = val_polished
        best_h = h_polished.copy()
    
    if time.time() > deadline - 10:
        return _clamp_and_normalize(best_h)
    
    # Phase 3: Try a few more random restarts if time permits
    for restart in range(5):
        if time.time() > deadline - 20:
            break
        
        # Random binary assignment
        h_r = np.zeros(N)
        h_r[:N//2] = 1.0
        np.random.shuffle(h_r)
        
        # Quick greedy swap
        h_r, val_r = _greedy_swap(h_r, max_swaps=1000)
        
        # Quick polish
        h_r, val_r = _continuous_polish(h_r, max_iters=1000)
        
        if val_r < best_val:
            best_val = val_r
            best_h = h_r.copy()
    
    # Final validation and normalization
    best_h = _clamp_and_normalize(best_h)
    
    # Ensure validity
    if not _is_valid(best_h):
        # Fallback to seed
        return np.full(N, 0.5)
    
    return best_h
