"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

N = 256

def _count_mono4(c):
    """Count monochromatic 4-term APs in a 0/1 numpy array of length 256."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(0, max_a):
            if c[a] == c[a + d] == c[a + 2 * d] == c[a + 3 * d]:
                count += 1
    return count

def _count_mono4_fast(c):
    """Faster count using numpy vectorization per d."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        # indices: 0..max_a-1, d..d+max_a-1, 2d..2d+max_a-1, 3d..3d+max_a-1
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2*d:2*d + max_a]
        c3 = c[3*d:3*d + max_a]
        # All four equal means all equal to c0 (since binary)
        # c0==c1 and c0==c2 and c0==c3
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += np.sum(mask)
    return int(count)

def _thue_morse():
    """Generate Thue-Morse sequence of length 256."""
    tm = np.zeros(N, dtype=np.int8)
    for i in range(1, N):
        tm[i] = tm[i >> 1] ^ (i & 1)
    return tm

def construct():
    deadline = time.time() + 145.0  # 150s limit, 5s safety margin
    
    # Start with Thue-Morse
    best = _thue_morse()
    best_score = _count_mono4_fast(best)
    
    # Local search: try flipping each bit, keep if improves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            # Try flipping bit i
            old_val = best[i]
            best[i] = 1 - old_val
            new_score = _count_mono4_fast(best)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best[i] = old_val
    
    # Return as list of ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
