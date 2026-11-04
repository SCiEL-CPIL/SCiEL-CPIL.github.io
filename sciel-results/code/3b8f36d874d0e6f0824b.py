import numpy as np
import time
from itertools import combinations

N = 50
DEADLINE = 135.0

def _elapsed(start):
    return time.time() - start

def _time_left(start):
    return DEADLINE - _elapsed(start)

def _autoconv(h):
    """Compute discrete autoconvolution of h (length N) -> length 2N-1."""
    return np.convolve(h, h, mode='full')

def _compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    conv_max = np.max(conv)
    if conv_max < 1e-15:
        return 0.0
    return l2_sq / conv_max

def _heights_from_mask(mask):
    """Given a binary frequency mask (length N), compute h = |IDFT(mask)|^2."""
    # Use the mask as the DFT of h, so h = |IDFT(mask)|^2
    # For real non-negative h, we need Hermitian symmetry in frequency
    # Let's work with complex: H = mask (real, symmetric), h = |IDFT(H)|^2
    # Actually, let's define: H is the DFT of some complex signal z, and h = |z|^2
    # If H is real and even, then z is real, and h = z^2 >= 0
    
    # For a binary mask with Hermitian symmetry:
    # H[k] = 1 for k in S, H[N-k] = 1 for k in S (to maintain reality)
    # H[0] = 1 if 0 in S
    
    # Simpler: just take h = |IDFT(mask_complex)|^2 where mask_complex is constructed
    # from the binary mask with proper symmetry
    
    H = np.zeros(N, dtype=complex)
    for k in range(N):
        if mask[k]:
            H[k] = 1.0
            # Ensure Hermitian symmetry for real output
            if k != 0 and k != N // 2:
                H[(N - k) % N] = 1.0
    
    z = np.fft.ifft(H)
    h = np.abs(z) ** 2
    return h

def _sidon_score(frequencies):
    """
    Score a set of frequencies by the minimum pairwise difference.
    Higher is better (more spread out = better Sidon-like property).
    """
    freqs = sorted(frequencies)
    if len(freqs) < 2:
        return float('inf')
    min_diff = float('inf')
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            diff = abs(freqs[j] - freqs[i])
            # Also consider wrapped differences for cyclic group
            wrapped = min(diff, N - diff)
            min_diff = min(min_diff, wrapped)
    return min_diff

def _find_sidon_sets(K, max_sets=200, seed=42):
    """
    Find frequency sets that are good Sidon sets in Z_N.
    A Sidon set has all pairwise sums (or differences) distinct.
    We approximate by maximizing the minimum pairwise distance.
    """
    rng = np.random.RandomState(seed)
    candidates = []
    
    # Try to build sets greedily with maximum separation
    # Start with evenly spaced sets
    for start in range(N):
        freqs = [(start + i * (N // K)) % N for i in range(K)]
        if len(set(freqs)) == K:
            candidates.append((tuple(sorted(freqs)), _sidon_score(freqs)))
    
    # Random search for better sets
    for _ in range(5000):
        freqs = list(rng.choice(N, size=K, replace=False))
        score = _sidon_score(freqs)
        candidates.append((tuple(sorted(freqs)), score))
    
    # Sort by score (descending) and return top unique sets
    candidates.sort(key=lambda x: -x[1])
    unique_sets = []
    seen = set()
    for freqs, score in candidates:
        if freqs not in seen:
            seen.add(freqs)
            unique_sets.append(freqs)
        if len(unique_sets) >= max_sets:
            break
    
    return unique_sets

def _local_optimize_mask(mask, max_iters=50):
    """
    Local optimization of the binary mask via swap operations.
    Try swapping active/inactive bins to improve the ratio.
    """
    mask = mask.copy()
    current_h = _heights_from_mask(mask)
    current_ratio = _compute_ratio(current_h)
    
    active = [i for i in range(N) if mask[i]]
    inactive = [i for i in range(N) if not mask[i]]
    
    for _ in range(max_iters):
        improved = False
        for a in active:
            for b in inactive:
                # Try swap
                mask[a] = 0
                mask[b] = 1
                h_new = _heights_from_mask(mask)
                ratio_new = _compute_ratio(h_new)
                if ratio_new > current_ratio + 1e-10:
                    current_ratio = ratio_new
                    active.remove(a)
                    active.append(b)
                    inactive.remove(b)
                    inactive.append(a)
                    improved = True
                    break
                else:
                    # Revert
                    mask[a] = 1
                    mask[b] = 0
            if improved:
                break
        if not improved:
            break
    
    return mask, current_ratio

def construct():
    start_time = time.time()
    best_heights = [1.0] * N
    best_score = 0.0
    
    rng = np.random.RandomState(123)
    
    # Try different K values (number of active frequency bins)
    K_values = [3, 4, 5, 6, 7, 8, 10, 12, 15]
    
    for K in K_values:
        if _time_left(start_time) < 10:
            break
        
        # Find good Sidon-like sets
        sidon_sets = _find_sidon_sets(K, max_sets=100, seed=42 + K)
        
        for freq_set in sidon_sets:
            if _time_left(start_time) < 5:
                break
            
            # Create mask
            mask = np.zeros(N, dtype=bool)
            for f in freq_set:
                mask[f] = True
            
            # Compute heights
            h = _heights_from_mask(mask)
            ratio = _compute_ratio(h)
            
            if ratio > best_score:
                best_score = ratio
                best_heights = h.tolist()
            
            # Local optimization
            if _time_left(start_time) > 15:
                mask_opt, ratio_opt = _local_optimize_mask(mask, max_iters=30)
                if ratio_opt > best_score:
                    h_opt = _heights_from_mask(mask_opt)
                    best_score = ratio_opt
                    best_heights = h_opt.tolist()
        
        # Also try random masks for this K
        if _time_left(start_time) > 20:
            for _ in range(50):
                if _time_left(start_time) < 5:
                    break
                mask = np.zeros(N, dtype=bool)
                indices = rng.choice(N, size=K, replace=False)
                mask[indices] = True
                
                h = _heights_from_mask(mask)
                ratio = _compute_ratio(h)
                
                if ratio > best_score:
                    best_score = ratio
                    best_heights = h.tolist()
                
                if _time_left(start_time) > 10:
                    mask_opt, ratio_opt = _local_optimize_mask(mask, max_iters=20)
                    if ratio_opt > best_score:
                        h_opt = _heights_from_mask(mask_opt)
                        best_score = ratio_opt
                        best_heights = h_opt.tolist()
    
    # Also try some non-Sidon structured masks
    if _time_left(start_time) > 10:
        # Geometric progression in frequency
        for ratio_freq in [2, 3, 5]:
            for start in range(0, N, max(1, N // 20)):
                if _time_left(start_time) < 5:
                    break
                mask = np.zeros(N, dtype=bool)
                f = start
                for _ in range(min(10, N)):
                    mask[f] = True
                    f = (f * ratio_freq) % N
                    if f == start:
                        break
                if np.sum(mask) >= 2:
                    h = _heights_from_mask(mask)
                    ratio = _compute_ratio(h)
                    if ratio > best_score:
                        best_score = ratio
                        best_heights = h.tolist()
    
    # Final safety
    if best_score <= 0:
        best_heights = [1.0] * N
    
    return best_heights
