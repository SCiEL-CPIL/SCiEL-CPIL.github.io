"""Find the longest interval in [1,1008] with no modular inverse pair (incl. diagonal)."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80.0  # safety margin under 90s wall-clock
    MOD = 1009

    # Self-inverse residues: x^2 % MOD == 1 -> x in {1, MOD-1} = {1, 1008}.
    # These cannot appear in any valid interval (diagonal check x*x % MOD != 1 fails).
    # So restrict to [2, 1007].
    lo, hi = 2, 1007

    # Build inverse map for x in [2, 1007] using NumPy for vectorized computation.
    xs = np.arange(lo, hi + 1, dtype=np.int64)
    # Compute modular inverse using Fermat's little theorem: x^(MOD-2) mod MOD
    # Use numpy's power function with modular reduction
    inv_arr = np.zeros(len(xs), dtype=np.int64)
    for i in range(len(xs)):
        inv_arr[i] = pow(int(xs[i]), MOD - 2, MOD)
    
    # Convert to a dictionary for easy lookup
    inv = {int(xs[i]): int(inv_arr[i]) for i in range(len(xs))}
    
    # Partition [2,1007] into 2-cycles. Assign each cycle an id.
    cycle_id = {}
    seen = set()
    next_id = 0
    for x in range(lo, hi + 1):
        if x in seen:
            continue
        y = inv[x]
        if x < y:
            a, b = x, y
        else:
            a, b = y, x
        cycle_id[a] = next_id
        cycle_id[b] = next_id
        seen.add(a)
        seen.add(b)
        next_id += 1
    num_cycles = next_id

    # Sliding window with 'bad' counter for O(n) solution.
    count = [0] * num_cycles
    bad = 0  # number of cycles with count == 2
    best_start = lo
    best_len = 0
    left = lo
    for right in range(lo, hi + 1):
        c = cycle_id[right]
        count[c] += 1
        if count[c] == 2:
            bad += 1
        while bad > 0:
            cl = cycle_id[left]
            count[cl] -= 1
            if count[cl] == 1:
                bad -= 1
            left += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
        cur_len = right - left + 1
        if cur_len > best_len:
            best_len = cur_len
            best_start = left

    # Fallback if nothing found (shouldn't happen).
    if best_len <= 0:
        return [2, 1]

    # Verify validity of the chosen interval (defensive, cheap).
    start = best_start
    length = best_len
    present = set(range(start, start + length))
    for x in present:
        if x * x % MOD == 1:
            # self-inverse present -> invalid, shouldn't happen
            return [2, 1]
        r = inv.get(x)
        if r is not None and r in present:
            # full cycle present -> invalid, shouldn't happen
            return [2, 1]

    return [start, length]
# EVOLVE-BLOCK-END
