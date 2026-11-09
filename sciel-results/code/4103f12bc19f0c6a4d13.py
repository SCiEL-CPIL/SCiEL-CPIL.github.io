"""Size-efficient sorting network on 17 wires."""
import time
import random
import numpy as np

N = 17
NUM_INPUTS = 1 << N  # 131072

# Precompute the initial state matrix for verification
# state[w][k] = bit k of input k, i.e., (k >> w) & 1
# We store as uint8 arrays for fast bitwise ops
_initial_state = None

def _get_initial_state():
    global _initial_state
    if _initial_state is None:
        # Create array of shape (N, NUM_INPUTS) with uint8
        # state[w][k] = (k >> w) & 1
        k = np.arange(NUM_INPUTS, dtype=np.uint32)
        _initial_state = np.empty((N, NUM_INPUTS), dtype=np.uint8)
        for w in range(N):
            _initial_state[w] = (k >> w) & 1
    return _initial_state

def _verify_network(network):
    """
    Verify sorting network using the zero-one principle.
    Uses numpy for fast bitwise operations.
    Returns True if network sorts all 2^N boolean inputs.
    """
    if not network:
        return N <= 1
    
    state = _get_initial_state().copy()
    
    for i, j in network:
        # min goes to i, max goes to j
        # new_i = state[i] & state[j]
        # new_j = state[i] | state[j]
        si = state[i]
        sj = state[j]
        state[i] = si & sj
        state[j] = si | sj
    
    # Check sorted: for all w, state[w] <= state[w+1] element-wise
    # i.e., state[w] & ~state[w+1] should be all zeros
    for w in range(N - 1):
        if np.any(state[w] & ~state[w + 1]):
            return False
    return True

def _verify_network_fast(network):
    """
    Fast verification using Python big integers.
    Each wire holds an integer mask where bit k is 1 if input k has a 1 on that wire.
    """
    if not network:
        return N <= 1
    
    # Precompute initial masks as big integers
    global _initial_masks
    if not hasattr(_verify_network_fast, '_initial_masks'):
        _verify_network_fast._initial_masks = []
        for w in range(N):
            mask = 0
            # For input k, bit w is set if (k >> w) & 1
            # Build mask efficiently
            period = 1 << (w + 1)
            half = 1 << w
            # Pattern: 0*half 1*half repeated
            block = (1 << half) - 1
            # Number of full periods in 2^N inputs
            num_periods = (1 << N) // period
            # Build the mask
            # Each period contributes: block shifted to the upper half
            # So within each period of length 'period', bits [half, half+1, ..., half+half-1] are 1
            # That means the block is (2^half - 1) << half
            block_in_period = ((1 << half) - 1) << half
            # Repeat this num_periods times
            # We can build by OR-ing shifted copies
            # But for large N this could be slow. Let's use a different approach.
            # Actually for N=17, 2^17 = 131072 bits, so the integer is about 16KB.
            # Let's just build it with a loop over periods.
            mask = 0
            shift = 0
            for _ in range(num_periods):
                mask |= block_in_period << shift
                shift += period
            _verify_network_fast._initial_masks.append(mask)
    
    masks = _verify_network_fast._initial_masks[:]
    
    for i, j in network:
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    
    for w in range(N - 1):
        if masks[w] & ~masks[w + 1]:
            return False
    return True

def _odd_even_transposition_network(n):
    """Generate the odd-even transposition sorting network."""
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    deadline = time.time() + 55  # Safety margin before 90s limit
    
    # Start with the odd-even transposition network (136 comparators)
    current_network = _odd_even_transposition_network(N)
    best_network = list(current_network)
    best_size = len(current_network)
    
    random.seed(42)
    
    # Phase 1: Greedy removal - try removing each comparator in reverse order
    # This is the most effective phase for finding removable comparators
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            # Try removing comparator at idx
            candidate = current_network[:idx] + current_network[idx+1:]
            if _verify_network_fast(candidate):
                current_network = candidate
                best_size = len(current_network)
                best_network = list(current_network)
                changed = True
                break  # Restart as list changed
    
    # Phase 2: Greedy removal - forward order
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for idx in range(len(current_network)):
            if time.time() > deadline - 10:
                break
            candidate = current_network[:idx] + current_network[idx+1:]
            if _verify_network_fast(candidate):
                current_network = candidate
                best_size = len(current_network)
                best_network = list(current_network)
                changed = True
                break  # Restart as list changed
    
    # Phase 3: Randomized local search
    if time.time() < deadline - 3:
        while time.time() < deadline - 3:
            if random.random() < 0.7 and len(current_network) > 0:
                # Remove a random comparator
                idx = random.randint(0, len(current_network) - 1)
                candidate = current_network[:idx] + current_network[idx+1:]
            else:
                # Swap two adjacent comparators (if they don't conflict)
                if len(current_network) < 2:
                    continue
                idx = random.randint(0, len(current_network) - 2)
                candidate = list(current_network)
                candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
                # Check if the swap is valid (wires don't overlap in a way that breaks order)
                # Actually just verify
            
            if _verify_network_fast(candidate):
                current_network = candidate
                if len(candidate) < best_size:
                    best_size = len(candidate)
                    best_network = list(candidate)
    
    return best_network
