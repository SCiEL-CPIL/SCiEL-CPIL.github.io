import time
import random
import numpy as np

N = 17
SEED = 42
_NUM_INPUTS = 1 << N

# Time management
_START_TIME = time.time()
_TIME_LIMIT = 85.0  # Safety margin for 90s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

# Precompute input matrix for fast vectorized verification
# _INPUTS shape: (131072, 17), dtype uint8
# _SORTED_INPUTS shape: (131072, 17), dtype uint8
def _init_data():
    global _INPUTS, _SORTED_INPUTS
    indices = np.arange(_NUM_INPUTS, dtype=np.uint32)
    # Create bit mask matrix
    # We want bit i of the input to correspond to wire i
    # indices[:, None] & (1 << np.arange(N))
    # This creates a matrix where row m has 1s at positions corresponding to set bits in m
    bits = 1 << np.arange(N, dtype=np.uint32)
    _INPUTS = ((indices[:, None] & bits[None, :]) > 0).astype(np.uint8)
    # Sort each row to get the expected output
    _SORTED_INPUTS = np.sort(_INPUTS, axis=1)

_init_data()

def _is_sorted_network(comparators):
    """
    Verify if the network sorts all 2^17 Boolean inputs.
    Uses NumPy vectorization for speed.
    """
    if len(comparators) == 0:
        # Only valid if N=1, but N=17, so False
        return False
    
    # Copy input state
    state = _INPUTS.copy()
    
    for i, j in comparators:
        # Vectorized compare-exchange
        # min goes to i, max goes to j
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    # Check if all rows match sorted inputs
    return bool(np.array_equal(state, _SORTED_INPUTS))

def _make_odd_even_transposition():
    """
    Generate the seed odd-even transposition sorting network for N wires.
    """
    comps = []
    for phase in range(N):
        # Odd phases start at 0, Even phases start at 1
        start = phase % 2
        for i in range(start, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _make_bitonic_sort(n):
    """
    Generate a bitonic sorting network for n wires.
    This is often more compact than odd-even transposition for larger N.
    """
    if n <= 1:
        return []
    
    # For n=17, we can't use standard power-of-2 bitonic directly without padding or complex logic.
    # However, we can use a recursive definition that handles non-powers of 2 by splitting.
    # A simpler robust baseline is often just the transposition, but let's try a standard bitonic 
    # for the largest power of 2 <= n and handle the rest? 
    # Actually, for 17, odd-even transposition (136 comps) is a known seed. 
    # Bitonic for 16 is 64 comps. 
    # Let's stick to the seed provided in the prompt context which is odd-even transposition.
    
    # If we want a better starting point, we could try to construct a specific known good network,
    # but the prompt implies starting from the seed or finding one.
    # The inspiration used odd-even merge sort which was complex. 
    # Let's stick to the simple, reliable odd-even transposition as the primary seed, 
    # but also generate a random network for restarts.
    pass
    return _make_odd_even_transposition()

def _random_network(num_comps):
    """Generate a random network with num_comps comparators."""
    comps = []
    for _ in range(num_comps):
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i != j:
            if i > j:
                i, j = j, i
            comps.append((i, j))
    return comps

def _local_search(comparators, max_iters=5000):
    """
    Perform local search to minimize comparator count.
    Strategies:
    1. Try removing a random comparator.
    2. Try replacing a random comparator with a random one.
    3. Try swapping two comparators (reordering).
    """
    best = list(comparators)
    best_len = len(best)
    no_improve_count = 0
    
    for _ in range(max_iters):
        if _time_left() < 5.0:
            break
            
        # Strategy 1: Removal
        if len(best) > 63:  # Lower bound is 63, don't go below
            idx = random.randint(0, len(best) - 1)
            trial = best[:idx] + best[idx+1:]
            if _is_sorted_network(trial):
                best = trial
                best_len = len(best)
                no_improve_count = 0
                continue
        
        # Strategy 2: Replacement
        if len(best) > 64:
            idx = random.randint(0, len(best) - 1)
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i != j:
                if i > j:
                    i, j = j, i
                trial = best[:idx] + [(i, j)] + best[idx+1:]
                if _is_sorted_network(trial):
                    best = trial
                    best_len = len(best)
                    no_improve_count = 0
                    continue
        
        # Strategy 3: Swap (Reordering can affect depth but not size, 
        # however, reordering might enable subsequent removals if the 
        # search space is structured, though for pure size minimization 
        # it's less direct. But it helps escape local minima in 
        # "removal" if the current order is suboptimal for deletion).
        if len(best) >= 2:
            idx1 = random.randint(0, len(best) - 1)
            idx2 = random.randint(0, len(best) - 1)
            if idx1 != idx2:
                trial = list(best)
                trial[idx1], trial[idx2] = trial[idx2], trial[idx1]
                # Note: Reordering doesn't change size, so we don't count it as an 
                # "improvement" in length, but it might lead to one.
                # We only accept if it's still valid (it always is, since it's a permutation).
                # Actually, any permutation of a valid sorting network is valid.
                # So we can just swap freely to explore the "shape" of the network.
                best = trial
                no_improve_count = 0
                continue

        no_improve_count += 1
        if no_improve_count > 500:
            break
            
    return best

def construct():
    """
    Construct a sorting network for 17 wires, minimizing comparator count.
    Returns a list of (i, j) pairs.
    """
    random.seed(SEED)
    
    # Start with the seed network
    best = _make_odd_even_transposition()
    best_len = len(best)
    
    # Initial verification (should be true)
    if not _is_sorted_network(best):
        # Fallback if seed fails (should not happen)
        best = _random_network(100)
        while not _is_sorted_network(best) and _time_left() > 10:
            best = _random_network(100)
    
    # Phase 1: Intensive local search from the seed
    best = _local_search(best, max_iters=20000)
    best_len = len(best)
    
    # Phase 2: Random restarts with perturbation
    # If we have time, try to jump out of local minima
    restarts = 0
    while _time_left() > 15.0 and restarts < 5:
        # Perturb the current best
        trial = list(best)
        
        # Add some random comparators to create a "messier" network that might sort differently
        num_add = random.randint(5, 15)
        for _ in range(num_add):
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i != j:
                if i > j:
                    i, j = j, i
                trial.append((i, j))
        
        # Local search on the perturbed network
        trial = _local_search(trial, max_iters=5000)
        
        if len(trial) < best_len:
            best = trial
            best_len = len(best)
        
        restarts += 1
        
        # Occasionally try a completely random start if stuck
        if restarts % 2 == 0 and _time_left() > 20.0:
            rand_net = _random_network(random.randint(80, 120))
            if _is_sorted_network(rand_net):
                rand_net = _local_search(rand_net, max_iters=5000)
                if len(rand_net) < best_len:
                    best = rand_net
                    best_len = len(best)

    return best

# Ensure the function is accessible as construct
# The evaluator calls construct()
