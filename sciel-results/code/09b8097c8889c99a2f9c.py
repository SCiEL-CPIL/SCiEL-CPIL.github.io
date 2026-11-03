<<<<<<< SEARCH
def _local_search_flip(h, max_passes=10, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Works on binary {-1, 1} sequences.
    """
    h = h.copy()
    # Ensure binary
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = compute_raw_value(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val

def _local_search_block(h, deadline=None, block_sizes=[3,5,7,10]):
    """
    Block flip local search: flip contiguous blocks of signs.
    Helps escape local minima of single-bit flips.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for bs in block_sizes:
            for start in range(0, N - bs + 1, max(1, (N - bs) // 50)):
                if deadline is not None and time.time() > deadline:
                    break
                h[start:start+bs] = -h[start:start+bs]
                new_val = compute_raw_value(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[start:start+bs] = -h[start:start+bs]  # revert
    return h, best_val
=======
def _local_search_flip(h, max_passes=10, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Works on binary {-1, 1} sequences.
    """
    return _fast_local_search_flip(h, max_passes, deadline)

def _local_search_block(h, deadline=None, block_sizes=[3,5,7,10]):
    """
    Block flip local search: flip contiguous blocks of signs.
    Helps escape local minima of single-bit flips.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    n = len(h)
    fsize, size = _get_fft_size(n)
    
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    best_val = np.max(np.abs(ac)) / (np.sum(h) ** 2)
    best_h = h.copy()
    best_H = H.copy()
    best_sum = np.sum(h)
    
    k = np.arange(fsize // 2 + 1)
    phase_factors = np.exp(-1j * 2 * np.pi * np.outer(np.arange(n), k) / fsize)
    
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for bs in block_sizes:
            for start in range(0, n - bs + 1, max(1, (n - bs) // 50)):
                if deadline is not None and time.time() > deadline:
                    break
                
                # Flip block [start, start+bs)
                old_vals = h[start:start+bs].copy()
                h[start:start+bs] = -old_vals
                
                delta_H = -2 * np.sum(old_vals * phase_factors[start:start+bs], axis=0)
                new_H = best_H + delta_H
                new_ac = np.fft.irfft(new_H * new_H, fsize)[:size]
                new_sum = best_sum - 2 * np.sum(old_vals)
                
                new_val = np.max(np.abs(new_ac)) / (new_sum ** 2)
                
                if new_val < best_val:
                    best_val = new_val
                    best_h = h.copy()
                    best_H = new_H.copy()
                    best_sum = new_sum
                    improved = True
                else:
                    h[start:start+bs] = old_vals
    
    return best_h, best_val
>>>>>>> REPLACE

