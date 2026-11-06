import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    MIN_N = 1000
    MAX_N = 1000000
    W = 24
    # primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # Precompute smooth part (factor composed of primes < 24) for 1..MAX_N+W
    Nmax = MAX_N + W
    # Use Python list for precomputation as it's fast for this specific logic
    smooth = [1] * (Nmax + 1)
    for p in primes:
        for m in range(p, Nmax + 1, p):
            temp = m
            acc = 1
            while temp % p == 0:
                temp //= p
                acc *= p
            smooth[m] *= acc

    # Convert to NumPy array for vectorized processing
    smooth_arr = np.array(smooth[MIN_N + 1 : MIN_N + 1 + (MAX_N - MIN_N + 1) + W - 1], dtype=np.int64)
    # The window for n starts at index n+1.
    # We want windows for n in [MIN_N, MAX_N].
    # Number of windows = MAX_N - MIN_N + 1.
    # Window i corresponds to n = MIN_N + i.
    # Values are smooth[n+1] ... smooth[n+W].
    # In smooth_arr, index 0 is smooth[MIN_N+1].
    # Window i starts at smooth_arr[i] and ends at smooth_arr[i+W-1].
    
    num_windows = MAX_N - MIN_N + 1
    
    # Create a view of the sliding windows
    # Shape: (num_windows, W)
    # We use a strided view to avoid copying memory if possible, but sort will copy.
    # Let's just slice and reshape if we were doing it in chunks, but here we do it all at once.
    # Creating a 2D array of shape (num_windows, W)
    # This will take ~200MB.
    
    # Check memory safety
    if num_windows * W * 8 > 500 * 1024 * 1024:
        # Fallback to Python loop if memory is a concern
        # (But 200MB is usually safe)
        pass

    windows = np.lib.stride_tricks.as_strided(
        smooth_arr,
        shape=(num_windows, W),
        strides=(smooth_arr.strides[0], smooth_arr.strides[0])
    )
    
    # Sort each window
    # This creates a new array
    windows_sorted = np.sort(windows, axis=1)
    
    # Calculate distinct counts
    # Distinct count = 1 + number of times adjacent elements differ
    diffs = np.diff(windows_sorted, axis=1)
    counts = 1 + np.sum(diffs != 0, axis=1)
    
    # Find the minimum count and the corresponding index
    min_count = np.min(counts)
    best_idx = np.argmin(counts)
    best_n = MIN_N + best_idx
    
    return best_n
# EVOLVE-BLOCK-END
