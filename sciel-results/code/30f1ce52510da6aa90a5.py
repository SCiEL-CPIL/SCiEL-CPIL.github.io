"""Size-efficient sorting networks on 16 wires."""
import numpy as np
import time

N = 16
ALL_INPUTS = (1 << N) - 1

# EVOLVE-BLOCK-START

def _verify_full(nets):
    """Verify network against all 2^N zero-one inputs.
    Returns True if all inputs are sorted."""
    if not nets:
        # Empty network only sorts if N <= 1
        return N <= 1
    # Apply all comparators to all inputs at once using numpy
    # inputs: shape (65536, 16)
    inputs = np.array([[1 if (k >> i) & 1 else 0 for i in range(N)] for k in range(1 << N)], dtype=np.uint8)
    for (i, j) in nets:
        # swap: min to i, max to j
        # If inputs[:, i] > inputs[:, j], swap
        mask = inputs[:, i] > inputs[:, j]
        tmp = inputs[mask, i].copy()
        inputs[mask, i] = inputs[mask, j]
        inputs[mask, j] = tmp
    # Check if all rows are sorted (non-decreasing)
    # sorted means inputs[:, k] <= inputs[:, k+1] for all k
    diffs = np.diff(inputs, axis=1)
    return np.all(diffs >= 0)

def _verify_sample(nets, sample_inputs):
    """Verify network against a subset of inputs."""
    if not nets:
        return np.all(np.diff(sample_inputs, axis=1) >= 0) if len(sample_inputs) > 0 else True
    inputs = sample_inputs.copy()
    for (i, j) in nets:
        mask = inputs[:, i] > inputs[:, j]
        tmp = inputs[mask, i].copy()
        inputs[mask, i] = inputs[mask, j]
        inputs[mask, j] = tmp
    diffs = np.diff(inputs, axis=1)
    return np.all(diffs >= 0)

def _random_zero_one(n):
    return (np.random.randint(0, 2, size=n) * (1 << np.arange(n))).astype(np.uint64)

def _odd_even_transposition(n=N):
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def _batcher_odd_even_merge(n):
    """Batcher's odd-even merge sort for n wires.
    Returns list of (i, j) comparators."""
    if n <= 1:
        return []
    half = n // 2
    # Sort left and right halves
    comps = _batcher_odd_even_merge(half) + _batcher_odd_even_merge(half)
    # Interleave: left wires are even indices, right wires are odd indices
    # Odd-even merge network for merging two sorted lists of length n/2
    # The interleave step: compare (i, i+1) for appropriate i
    # Standard Batcher odd-even merge for 2m elements:
    # Step 1: compare (i, i+1) for i in {0, 2, 4, ...} (even i)
    # Step 2: compare (i, i+1) for i in {1, 3, 5, ...} (odd i)
    # Recursively merge the even and odd subsequences
    comps.append((i, i+1) for i in range(0, n-1, 2))  # even i
    # Now merge the two subsequences of length n/2
    # Even subsequence: wires 0, 2, 4, ..., n-2
    # Odd subsequence: wires 1, 3, 5, ..., n-1
    # Apply Batcher merge to these subsequences
    even_merge = _batcher_odd_even_merge(n // 2)
    for (a, b) in even_merge:
        comps.append((2*a, 2*b))
    odd_merge = _batcher_odd_even_merge(n // 2)
    for (a, b) in odd_merge:
        comps.append((2*a+1, 2*b+1))
    return comps

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin before 90s kill

    # Start with odd-even transposition network (120 comparators)
    best_nets = _odd_even_transposition(N)
    best_count = len(best_nets)

    # Try Batcher's odd-even merge as an alternative starting point
    batcher_nets = _batcher_odd_even_merge(N)
    if _verify_full(batcher_nets):
        if len(batcher_nets) < best_count:
            best_nets = batcher_nets
            best_count = len(batcher_nets)

    # Greedy removal: try to remove each comparator and check if network still sorts
    # Use a randomized order for multiple passes
    np.random.seed(42)
    
    removed = set()
    for pass_num in range(20):
        if time.time() > deadline - 10.0:
            break
        # Try removing comparators in random order
        indices = list(range(len(best_nets)))
        np.random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline - 5.0:
                break
            # Try removing comparator at index idx
            test_nets = best_nets[:idx] + best_nets[idx+1:]
            if _verify_full(test_nets):
                # Removal works
                best_nets = test_nets
                best_count = len(best_nets)
                # Shift remaining indices
                # We can't easily shift, so just break and re-shuffle
                break
        # If no removal was made in this pass, try a different strategy
        # Swap-based local search
        if time.time() > deadline - 5.0:
            break
        # Try swapping adjacent comparators
        improved = False
        for i in range(len(best_nets) - 1):
            if time.time() > deadline - 5.0:
                break
            test_nets = best_nets[:]
            test_nets[i], test_nets[i+1] = best_nets[i+1], best_nets[i]
            if _verify_full(test_nets):
                # Check if this leads to a removable comparator
                # Just accept if it's still valid
                best_nets = test_nets
                improved = True
                break
        if not improved and pass_num > 5:
            # Try adding a random comparator and then removing others
            if time.time() > deadline - 10.0:
                break
            # Add a random comparator
            while True:
                i, j = np.random.randint(0, N, size=2)
                if i < j:
                    break
            test_nets = best_nets + [(i, j)]
            if _verify_full(test_nets):
                best_nets = test_nets
                best_count = len(best_nets)
                # Now try to remove some comparators
                for idx in range(len(best_nets) - 1, -1, -1):
                    if time.time() > deadline - 5.0:
                        break
                    test2 = best_nets[:idx] + best_nets[idx+1:]
                    if _verify_full(test2):
                        best_nets = test2
                        best_count = len(best_nets)
                        break

    return best_nets
# EVOLVE-BLOCK-END
