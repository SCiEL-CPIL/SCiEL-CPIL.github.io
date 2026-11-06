"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

N = 256
DEADLINE = 115.0

def _count_mono4(c):
    """Count monochromatic 4-term APs in a 0/1 numpy array of length N (0-indexed)."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += np.count_nonzero(mask)
    return count

def _thue_morse():
    """Generate Thue-Morse sequence for 0..255."""
    c = np.zeros(N, dtype=np.int8)
    for i in range(N):
        c[i] = bin(i).count('1') % 2
    return c

def _delta_flip(c, i):
    """Compute change in mono-4AP count if we flip position i."""
    delta = 0
    # For each d and each role k in {0,1,2,3}, check if i = a + k*d
    for d in range(1, N // 3 + 1):
        for k in range(4):
            a = i - k * d
            if a < 0 or a + 3 * d >= N:
                continue
            # AP is (a, a+d, a+2d, a+3d)
            idxs = [a, a + d, a + 2 * d, a + 3 * d]
            # Before flip: all same?
            old_vals = [c[j] for j in idxs]
            was_mono = (old_vals[0] == old_vals[1] == old_vals[2] == old_vals[3])
            # After flip: position i has flipped
            new_vals = [c[j] if j != i else 1 - c[j] for j in idxs]
            is_mono = (new_vals[0] == new_vals[1] == new_vals[2] == new_vals[3])
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
    return delta

def _local_search_delta(c, time_limit):
    """Improve coloring via single-flip local search using delta computation."""
    start = time.time()
    current = _count_mono4(c)
    improved = True
    while improved:
        improved = False
        for i in range(N):
            if time.time() - start > time_limit:
                break
            delta = _delta_flip(c, i)
            if delta < 0:
                c[i] = 1 - c[i]
                current += delta
                improved = True
        if time.time() - start > time_limit:
            break
    return c, current

def construct():
    start_time = time.time()
    
    # Start with Thue-Morse
    c = _thue_morse()
    best_c = c.copy()
    best_count = _count_mono4(c)
    
    # Local search from Thue-Morse using delta
    time_remaining = DEADLINE - (time.time() - start_time)
    if time_remaining > 5:
        c, cnt = _local_search_delta(c, time_remaining - 5)
        if cnt < best_count:
            best_count = cnt
            best_c = c.copy()
    
    # Try random restarts if time permits
    rng = np.random.default_rng(42)
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 20:
        num_restarts = min(int(remaining // 10), 15)
        for r in range(num_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
            c_rand = rng.integers(0, 2, size=N).astype(np.int8)
            t_start = time.time()
            time_left = DEADLINE - (time.time() - start_time) - 5
            if time_left < 3:
                break
            c_rand, cnt_rand = _local_search_delta(c_rand, min(10, time_left))
            if cnt_rand < best_count:
                best_count = cnt_rand
                best_c = c_rand.copy()
    
    return [int(x) for x in best_c]
# EVOLVE-BLOCK-END
