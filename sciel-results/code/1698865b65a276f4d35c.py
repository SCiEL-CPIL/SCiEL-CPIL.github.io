"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network on 18 wires with as few comparators as possible.
    
    Strategy:
    1. Start from the odd-even transposition network (153 comparators).
    2. Use a vectorized NumPy verifier to check all 2^18 Boolean inputs at once.
    3. Iteratively remove redundant comparators (multiple passes).
    4. Also try swapping comparator order to find more removable comparators.
    """
    start_time = time.time()
    deadline = start_time + 82  # Leave 8s safety margin for the 90s limit

    # Start with odd-even transposition network
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]

    # Precompute the sorted target: for each input mask, the sorted output
    # has all 0s followed by all 1s. The number of 1s = popcount(mask).
    # Sorted mask = (1 << popcount) - 1 (ones in the lowest positions)
    masks = np.arange(1 << N, dtype=np.uint32)
    # Compute popcount for all masks using bit tricks
    # popcount via np.unpackbits trick
    popcount = np.zeros(1 << N, dtype=np.uint8)
    for shift in range(N):
        popcount += (masks >> shift) & 1
    sorted_target = (1 << popcount) - 1  # all 1s in lowest popcount positions

    def verify_fast(net):
        """
        Vectorized verification: simulate all 2^18 Boolean inputs simultaneously.
        Returns True if the network sorts all inputs.
        """
        vec = masks.copy()  # shape (2^N,)
        for i, j in net:
            vi = (vec >> i) & 1
            vj = (vec >> j) & 1
            # If vi > vj, swap: set bit i to 0, bit j to 1
            swap_mask = (vi & ~vj)  # 1 where vi=1 and vj=0
            # Clear bit i where swap needed
            vec = np.where(swap_mask, vec & ~(1 << i), vec)
            # Set bit j where swap needed
            vec = np.where(swap_mask, vec | (1 << j), vec)
        # Check if all sorted: vec should equal sorted_target
        return np.array_equal(vec, sorted_target)

    # Verify initial network
    if not verify_fast(network):
        return network

    # Iterative removal: try to remove each comparator, repeat until no improvement
    improved = True
    while improved:
        if time.time() > deadline - 3:
            break
        improved = False
        # Try removing each comparator from the end (more likely to be redundant)
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline - 3:
                break
            candidate = network[:idx] + network[idx + 1:]
            if verify_fast(candidate):
                network = candidate
                improved = True
                break

    # Second pass: try swapping adjacent comparators to unlock more removals
    # (structural change: different search method)
    if time.time() < deadline - 5:
        for _pass in range(3):
            if time.time() > deadline - 3:
                break
            progress = False
            for idx in range(len(network) - 1):
                if time.time() > deadline - 3:
                    break
                # Try swapping network[idx] and network[idx+1]
                candidate = network.copy()
                candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
                if verify_fast(candidate):
                    # Check if we can now remove a comparator
                    for ridx in range(len(candidate) - 1, -1, -1):
                        if time.time() > deadline - 3:
                            break
                        candidate2 = candidate[:ridx] + candidate[ridx + 1:]
                        if verify_fast(candidate2):
                            network = candidate2
                            progress = True
                            break
                if progress:
                    break
            if not progress:
                break

    return network
# EVOLVE-BLOCK-END
