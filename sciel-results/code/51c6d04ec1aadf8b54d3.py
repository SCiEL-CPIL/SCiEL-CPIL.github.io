"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600
START = -0.25
END = 0.25
W = (END - START) / N

# EVOLVE-BLOCK-START
def _compute_ratio_continuous(h):
    """
    Compute the continuous ratio max(f*f) / ||f||_2^2 for a step function h.
    Uses FFT on a fine grid to approximate the continuous autoconvolution.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf')
    
    # Fine grid for accurate peak detection
    n_grid = 4096
    x_fine = np.linspace(START, END, n_grid, endpoint=False)
    # Map fine grid points to step function intervals
    cell_idx = np.clip(((x_fine - START) / W).astype(np.int32), 0, N - 1)
    f_fine = h[cell_idx]
    
    # FFT convolution
    n_fft = 2 * n_grid
    F = np.fft.rfft(f_fine, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx_fine = (END - START) / n_grid
    conv *= dx_fine
    
    max_conv = np.max(conv)
    
    l2_sq = np.sum(f_fine ** 2) * dx_fine
    if l2_sq < 1e-15:
        return float('inf')
        
    return max_conv / l2_sq

def _generate_sidon_set(n_max, min_gap=1):
    """
    Generate a Sidon set (B2 sequence) of indices in [0, n_max).
    Ensures all pairwise sums are unique, which implies unique differences.
    """
    indices = []
    used_sums = set()
    
    for i in range(n_max):
        # Check if i can be added
        ok = True
        for j in indices:
            s = i + j
            if s in used_sums:
                ok = False
                break
        if ok:
            # Add all new sums involving i
            for j in indices:
                used_sums.add(i + j)
            used_sums.add(2 * i)
            indices.append(i)
            
    return indices

def construct():
    deadline = time.time() + 140.0
    
    # 1. Generate a large Sidon set
    # We want to maximize the number of disjoint spikes.
    # A Sidon set ensures that the autoconvolution peaks (from cross terms) are minimized.
    # Specifically, for a 0/1 Sidon set, the max autoconvolution is determined by the self-convolution of the individual spikes.
    # If spikes are disjoint (separated by > 0), the max is just the peak of one spike.
    # Ratio = 1 / M.
    
    sidon_indices = _generate_sidon_set(N)
    
    # Build the sparse function
    best_f = np.zeros(N)
    for idx in sidon_indices:
        best_f[idx] = 1.0
    
    # Evaluate
    best_ratio = _compute_ratio_continuous(best_f)
    
    # 2. Try to improve by local search or different sets?
    # The Sidon set is likely near-optimal for sparse functions.
    # We can try to add more spikes if the greedy method stopped early?
    # Or try a different construction?
    
    # Alternative: Random sparse sets
    np.random.seed(42)
    for _ in range(100):
        if time.time() > deadline:
            break
        # Pick random subset
        M = np.random.randint(10, 50)
        idxs = np.random.choice(N, M, replace=False)
        f_rand = np.zeros(N)
        f_rand[idxs] = 1.0
        r = _compute_ratio_continuous(f_rand)
        if r < best_ratio:
            best_ratio = r
            best_f = f_rand.copy()

    return list(best_f)
# EVOLVE-BLOCK-END
