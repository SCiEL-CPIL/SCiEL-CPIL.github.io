"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
DEADLINE = 85.0  # safety margin before 90s kill
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _apply_network(net, x):
    """Apply network to a single input vector x (numpy array)."""
    x = x.copy()
    for i, j in net:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _verify_network(net):
    """Verify network sorts all 2^17 Boolean inputs."""
    n_comp = len(net)
    # Pre-compute the effect of each comparator as a mask
    # For each boolean input, we can simulate quickly
    # Use the zero-one principle: check all 2^N inputs
    for mask in range(1 << N):
        x = np.array([(mask >> i) & 1 for i in range(N)], dtype=np.int8)
        for i, j in net:
            if x[i] > x[j]:
                x[i], x[j] = x[j], x[i]
        # Check sorted: x should be non-decreasing
        for k in range(N - 1):
            if x[k] > x[k + 1]:
                return False
    return True

def _verify_network_fast(net):
    """Faster verification using vectorized approach over all masks."""
    n_comp = len(net)
    if n_comp == 0:
        return N <= 1
    # Represent all 2^N masks as a 2D array: shape (2^N, N)
    masks = np.arange(1 << N, dtype=np.uint32)
    x = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        x[:, i] = (masks >> i) & 1
    
    for i, j in net:
        # Where x[:,i] > x[:,j], swap
        swap = x[:, i] > x[:, j]
        x[swap, i] = 1 - x[swap, i]
        x[swap, j] = 1 - x[swap, j]
    
    # Check sorted: for each k, x[:,k] <= x[:,k+1]
    for k in range(N - 1):
        if np.any(x[:, k] > x[:, k + 1]):
            return False
    return True

def _compute_depth(net):
    """Compute parallel depth of the network."""
    if not net:
        return 0
    wire_depth = [0] * N
    for i, j in net:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
    return max(wire_depth)

def _make_odd_even_transposition():
    """Odd-even transposition sorting network: N*(N-1)/2 comparators for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _make_batcher_odd_even():
    """
    Batcher's odd-even mergesort network.
    For N=17, we can pad to 32 (power of 2) or use a recursive construction.
    Actually, let's use a simpler approach: Batcher's bitonic sort won't work for arbitrary N easily.
    Let's use the odd-even merge sort network construction.
    
    For now, let's just use odd-even transposition as baseline and try to improve.
    """
    return _make_odd_even_transposition()

def _try_remove_comparator(net, idx):
    """Try removing comparator at index idx and verify."""
    new_net = net[:idx] + net[idx+1:]
    if _verify_network_fast(new_net):
        return new_net
    return None

def _try_swap_comparators(net, i, j):
    """Try swapping comparators i and j."""
    if i == j:
        return None
    new_net = net[:]
    new_net[i], new_net[j] = new_net[j], new_net[i]
    if _verify_network_fast(new_net):
        return new_net
    return None

def _random_local_search(net, max_iters=2000):
    """Random local search: try random removals and swaps."""
    best_net = net[:]
    best_len = len(best_net)
    rng = random.Random(42)
    
    for _ in range(max_iters):
        if _elapsed() > DEADLINE:
            break
        if best_len <= 63:  # lower bound
            break
        
        # Try random removal
        if rng.random() < 0.7:
            idx = rng.randint(0, best_len - 1)
            new_net = _try_remove_comparator(best_net, idx)
            if new_net is not None:
                best_net = new_net
                best_len = len(best_net)
        else:
            # Try random swap
            i = rng.randint(0, best_len - 1)
            j = rng.randint(0, best_len - 1)
            if i != j:
                new_net = _try_swap_comparators(best_net, i, j)
                if new_net is not None:
                    best_net = new_net
                    best_len = len(best_net)
    
    return best_net

def _systematic_removal(net):
    """Systematically try removing each comparator."""
    best_net = net[:]
    best_len = len(best_net)
    changed = True
    while changed and _elapsed() < DEADLINE:
        changed = False
        for idx in range(len(best_net)):
            if _elapsed() > DEADLINE:
                break
            new_net = _try_remove_comparator(best_net, idx)
            if new_net is not None:
                best_net = new_net
                best_len = len(best_net)
                changed = True
                break  # restart from beginning since indices shifted
    return best_net

def _build_better_initial_network():
    """
    Try to build a better initial network than odd-even transposition.
    Use a combination of approaches:
    1. Start with odd-even transposition
    2. Try to use known smaller networks or construct better ones
    
    For 17 wires, the best known is 71 comparators.
    Let's try a few different initial constructions.
    """
    # Odd-even transposition: 17*16/2 = 136 comparators
    net_oe = _make_odd_even_transposition()
    
    # Try a different approach: use a recursive Batcher-style network
    # For N=17, we can split into 8+9 or use other decompositions
    
    # Let's try: merge sort network using odd-even merge
    # This is complex, so let's just use the transposition and optimize
    
    return net_oe

def construct():
    """Construct a sorting network on 17 wires with as few comparators as possible."""
    global START_TIME
    START_TIME = time.time()
    
    # Start with odd-even transposition
    net = _make_odd_even_transposition()
    initial_len = len(net)
    
    # Verify initial network
    if not _verify_network_fast(net):
        # Should not happen, but just in case
        return net
    
    # Phase 1: Systematic removal
    net = _systematic_removal(net)
    
    # Phase 2: Random local search
    net = _random_local_search(net, max_iters=5000)
    
    # Phase 3: Another round of systematic removal
    net = _systematic_removal(net)
    
    # Final verification
    if not _verify_network_fast(net):
        # Fallback to original
        return _make_odd_even_transposition()
    
    return net

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
