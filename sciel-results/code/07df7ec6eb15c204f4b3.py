import time
import random
import numpy as np

N = 16
DEADLINE = time.time() + 85

def _bitonic_sort_network(n):
    """
    Generates a correct Bitonic Sort network for n wires (power of 2).
    Returns a list of (i, j) comparators.
    """
    def generate(start, length, direction):
        if length <= 1:
            return []
        half = length // 2
        comps = []
        
        # Sort first half ascending
        comps.extend(generate(start, half, 1))
        # Sort second half descending
        comps.extend(generate(start + half, half, -1))
        
        # Bitonic Merge
        # 1. Compare (start+i, start+half+i)
        for i in range(half):
            a = start + i
            b = start + half + i
            if direction == 1:
                comps.append((a, b))
            else:
                comps.append((b, a))
        
        # 2. Recursively sort the two halves in the target direction
        comps.extend(generate(start, half, direction))
        comps.extend(generate(start + half, half, direction))
        
        return comps

    return generate(0, n, 1)

def _verify_full(network):
    """
    Full verification using numpy over all 2^N states.
    Returns True if the network sorts all inputs.
    """
    states = np.arange(1 << N, dtype=np.int64)
    for i, j in network:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        # If bi > bj, we need to swap to put min at i and max at j
        swap = bi > bj
        # Flip bits i and j where swap is True
        mask = (1 << i) | (1 << j)
        states = np.where(swap, states ^ mask, states)
    
    # Check if all states are sorted (0s followed by 1s)
    # A number is sorted if (state + 1) & state == 0
    return bool(np.all(((states + 1) & states) == 0))

def _verify_sample(network, n_samples=10000, rng=None):
    """
    Sampled verification for speed during search.
    """
    if rng is None:
        rng = random.Random(42)
    for _ in range(n_samples):
        state = rng.randrange(1 << N)
        s = state
        for i, j in network:
            if ((s >> i) & 1) > ((s >> j) & 1):
                s ^= (1 << i) | (1 << j)
        if (s + 1) & s != 0:
            return False
    return True

def construct():
    start_time = time.time()
    
    # 1. Generate Bitonic Sort Network (56 comparators for N=16)
    net = _bitonic_sort_network(N)
    best = list(net)
    
    # Verify initial network
    if not _verify_full(best):
        # Fallback: Odd-Even Transposition (120 comparators)
        best = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    rng = random.Random(12345)
    
    # 2. Greedy Removal: Try to remove each comparator if network still sorts
    improved = True
    while improved and time.time() < DEADLINE - 10:
        improved = False
        order = list(range(len(best)))
        rng.shuffle(order)
        
        for idx in order:
            if time.time() > DEADLINE - 10:
                break
            
            candidate = best[:idx] + best[idx+1:]
            
            # Use sampled verification for speed
            if _verify_sample(candidate, n_samples=20000, rng=rng):
                # Confirm with full verification if it's getting small
                if len(candidate) < 60:
                    if _verify_full(candidate):
                        best = candidate
                        improved = True
                else:
                    best = candidate
                    improved = True
    
    # Final full verification
    if _verify_full(best):
        return best
    else:
        # If the greedy search corrupted the network, fall back to the initial valid one
        # We know the initial Bitonic or Transposition was valid
        initial_net = _bitonic_sort_network(N)
        if _verify_full(initial_net):
            return initial_net
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
