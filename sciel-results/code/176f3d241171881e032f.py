"""Size-efficient sorting network for 16 wires.
Uses a hardcoded 60-comparator baseline with fast vectorized verification
and an aggressive pruning pass to find a smaller valid network.
"""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START

# ---- 8-wire optimal sorting network (19 comparators) ------------------------
def _net8():
    """Return a 19-comparator sorting network for 8 wires."""
    return [
        (0, 4), (1, 5), (2, 6), (3, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 1), (2, 3), (4, 5), (6, 7),
        (1, 4), (2, 5), (3, 6),
        (1, 2), (3, 4), (2, 3), (5, 6),
    ]

# ---- 16-wire network: two 8-wire sorts + cross merge ------------------------
def _construct_60():
    """Construct a 60-comparator sorting network for 16 wires."""
    net = _net8()
    net += [(i + 8, j + 8) for i, j in _net8()]

    # 22-comparator merge network for two sorted 8-wire halves
    merge = [
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14),
        (2, 10), (3, 11), (4, 12), (5, 13),
        (3, 11), (4, 12), (5, 13),
        (5, 13),
    ]

    net += merge
    return net

# ---- Fast vectorized verification with precomputed popcount ----------------
def _precompute_popcount():
    """Precompute popcount for all 2^16 values."""
    popcount = np.zeros(1 << N, dtype=np.int64)
    for m in range(1, 1 << N):
        popcount[m] = popcount[m >> 1] + (m & 1)
    return popcount

_POPCOUNT = _precompute_popcount()

# Precompute expected sorted states
_EXP_SORTED = np.zeros(1 << N, dtype=np.int64)
for _k in range(N + 1):
    _exp_val = 0
    for _w in range(N - _k, N):
        _exp_val |= (1 << _w)
    _EXP_SORTED[_POPCOUNT == _k] = _exp_val

def _verify_network_vectorized(comparators, deadline):
    """
    Verify using zero-one principle with numpy vectorization.
    Processes all 65536 inputs in parallel.
    Returns True if valid, False if invalid, None on timeout.
    """
    start = time.time()

    # All 2^16 input masks
    states = np.arange(1 << N, dtype=np.int64)

    # Simulate
    for idx, (i, j) in enumerate(comparators):
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        swap_mask = (bi > bj)

        if np.any(swap_mask):
            states[swap_mask] &= ~(1 << i)
            states[swap_mask] |= (1 << j)

        # Check time every 10 comparators to reduce overhead
        if idx % 10 == 0 and time.time() - start > deadline:
            return None

    if np.array_equal(states, _EXP_SORTED):
        return True
    return False

def _verify_network_fast(comparators, num_samples=1000, deadline=1.0):
    """
    Fast approximate verification using random samples.
    Returns True if valid on samples, False if invalid, None on timeout.
    """
    start = time.time()
    # Generate random samples
    # Use a fixed seed for reproducibility? No, random is better for coverage.
    samples = np.random.choice(1 << N, size=num_samples, replace=False)
    states = samples.astype(np.int64)
    
    # Compute expected for these samples
    exp_samples = _EXP_SORTED[samples]
    
    for idx, (i, j) in enumerate(comparators):
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        swap_mask = (bi > bj)
        
        if np.any(swap_mask):
            states[swap_mask] &= ~(1 << i)
            states[swap_mask] |= (1 << j)
            
        if idx % 10 == 0 and time.time() - start > deadline:
            return None

    if np.array_equal(states, exp_samples):
        return True
    return False

# ---- Fallback: odd-even transposition (120 comparators) ---------------------
def _odd_even_transposition():
    """Return the 120-comparator odd-even transposition network."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

# ---- Main construct function ------------------------------------------------
def construct():
    """
    Constructs a size-efficient sorting network for 16 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    deadline = 85.0  # 5s safety margin before 90s kill

    # Start with the 60-comparator network
    network = _construct_60()

    # Verify the initial network
    is_valid = _verify_network_vectorized(network, deadline - 5.0)

    if is_valid is None:
        # Timeout during verification, return what we have
        return network

    if not is_valid:
        # Fall back to 120-comparator odd-even transposition
        return _odd_even_transposition()

    # Attempt to prune redundant comparators if time permits
    time_left = deadline - time.time()

    if time_left > 5.0:
        removals = 0
        max_removals = 20  # Allow more removals
        
        # Use a set of indices to try, or just scan
        # We will try to remove one by one, but use fast check first
        
        while removals < max_removals and time.time() < deadline - 5.0:
            found_removal = False
            # Scan from the end to the beginning
            # To avoid getting stuck, we can randomize the order of checks
            # But sequential is safer for finding "redundant" ones at the end
            for idx in range(len(network) - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break

                candidate = network[:idx] + network[idx + 1:]
                
                # Fast check first
                fast_valid = _verify_network_fast(candidate, num_samples=200, deadline=0.5)
                
                if fast_valid is None:
                    continue # Timeout on fast check, skip
                
                if fast_valid is False:
                    continue # Invalid on samples, skip
                
                # Full check
                is_valid = _verify_network_vectorized(candidate, deadline - 5.0)

                if is_valid is True:
                    network = candidate
                    removals += 1
                    found_removal = True
                    break  # Restart scan with new shorter list

            if not found_removal:
                # If no removal found in a full pass, break
                # Or we could try a different strategy (e.g. random removals)
                # For now, break to save time
                break

    return network

# EVOLVE-BLOCK-END
