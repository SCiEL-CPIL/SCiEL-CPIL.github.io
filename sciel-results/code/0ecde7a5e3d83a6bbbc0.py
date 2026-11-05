"""Sorting network for 18 wires - optimized with fast verification and local search."""
import time
import random
import numpy as np

N = 18
NUM_INPUTS = 1 << N  # 262144

# EVOLVE-BLOCK-START

def _build_bitonic_network(n):
    """Build a bitonic sorting network for n wires."""
    comps = []
    def _bitonic_merge(lo, hi, count, direction):
        if count <= 1:
            return
        half = count // 2
        for i in range(half):
            if direction:
                comps.append((lo + i, lo + i + half))
            else:
                comps.append((lo + i + half, lo + i))
        _bitonic_merge(lo, lo + half, half, direction)
        _bitonic_merge(lo + half, hi, half, direction)
    def _bitonic_sort(lo, hi, count, direction):
        if count <= 1:
            return
        half = count // 2
        _bitonic_sort(lo, lo + half, half, True)
        _bitonic_sort(lo + half, hi, half, False)
        _bitonic_merge(lo, hi, count, direction)
    _bitonic_sort(0, n, n, True)
    return comps

def _build_odd_even_network(n):
    """Build odd-even transposition sort network."""
    comps = []
    for phase in range(n):
        if phase % 2 == 0:
            for i in range(0, n - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, n - 1, 2):
                comps.append((i, i + 1))
    return comps

def _verify_network(net):
    """Verify network sorts all 2^N Boolean inputs using precomputed bit vectors."""
    # Precompute: for each wire position, a bit vector of length 2^N
    # where bit k is 1 if wire has value 1 on input k
    # Input k has wire i = 1 iff bit i of k is 1
    # So wire i's bit vector = all inputs where bit i is set
    
    # We'll track for each wire a numpy array of uint8 (or use int64 bitmasks)
    # Actually, let's use a different approach: track the set of inputs where
    # the network is "broken" (not sorted) at each wire position.
    
    # Fast approach: for each input (0 to 2^N-1), simulate the network.
    # But that's 262144 * len(net) operations which is too slow for many calls.
    
    # Better: use numpy to process all inputs at once.
    # For each wire, maintain a numpy array of shape (2^N,) with 0/1 values.
    
    if not net:
        return N <= 1
    
    # Initialize: wire i has value = bit i of the input
    # input k: wire i = (k >> i) & 1
    inputs = np.arange(NUM_INPUTS, dtype=np.int64)
    wires = np.array([(inputs >> i) & 1 for i in range(N)], dtype=np.int8)  # (N, 2^N)
    
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        wires[i] = np.minimum(wi, wj)
        wires[j] = np.maximum(wi, wj)
    
    # Check sorted: wire[k] <= wire[k+1] for all k
    for k in range(N - 1):
        if np.any(wires[k] > wires[k + 1]):
            return False
    return True

def _verify_network_fast(net):
    """Faster verification using incremental tracking of violations."""
    if not net:
        return N <= 1
    
    # For each pair of adjacent wires (k, k+1), track which inputs violate the order.
    # An input violates at position k if wire[k] > wire[k+1] for that input.
    # We need all violations to be empty at the end.
    
    # Initialize: wire i = bit i of input
    # For adjacent pair (k, k+1): violation if bit k > bit k+1, i.e., bit k=1 and bit k+1=0
    # This is inputs where (k-th bit is 1) AND ((k+1)-th bit is 0)
    
    # Represent each wire's state as a set of inputs (as a numpy boolean array)
    # where that wire has value 1.
    
    # Actually the simplest fast approach: just do the numpy thing but cache it.
    # The numpy approach processes 262144 values per comparator, which is fast.
    
    inputs = np.arange(NUM_INPUTS, dtype=np.int64)
    wires = np.array([(inputs >> i) & 1 for i in range(N)], dtype=np.int8)
    
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        wires[i] = np.minimum(wi, wj)
        wires[j] = np.maximum(wi, wj)
    
    for k in range(N - 1):
        if np.any(wires[k] > wires[k + 1]):
            return False
    return True

def _depth_of_network(net):
    """Compute parallel depth of the network."""
    if not net:
        return 0
    depth = [0] * N
    for i, j in net:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth)

def _greedy_removal(net, deadline, max_iterations=50000):
    """Try to remove comparators one at a time, keeping the network valid."""
    net = list(net)
    start = time.time()
    removed = 0
    consecutive_fails = 0
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        if not net:
            break
        
        # Try removing each comparator, starting from random positions
        # Use a random order to escape local minima
        order = list(range(len(net)))
        random.shuffle(order)
        
        found_removal = False
        for idx in order:
            if time.time() > deadline:
                return net
            if idx >= len(net):
                continue
            
            candidate = net[:idx] + net[idx+1:]
            if _verify_network_fast(candidate):
                net = candidate
                removed += 1
                found_removal = True
                consecutive_fails = 0
                break
        
        if not found_removal:
            consecutive_fails += 1
            if consecutive_fails >= 3:
                break
    
    return net

def _randomized_greedy_removal(net, deadline, max_rounds=200):
    """More aggressive: in each round, try all positions but with random perturbation."""
    net = list(net)
    start = time.time()
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        if not net:
            break
        
        # Shuffle and try to remove one comparator
        order = list(range(len(net)))
        random.shuffle(order)
        
        found = False
        for idx in order:
            if time.time() > deadline:
                return net
            if idx >= len(net):
                continue
            candidate = net[:idx] + net[idx+1:]
            if _verify_network_fast(candidate):
                net = candidate
                found = True
                break
        
        if not found:
            # Try a different strategy: remove 2 comparators at random
            if len(net) >= 2 and random.random() < 0.3:
                idx1, idx2 = random.sample(range(len(net)), 2)
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                candidate = net[:idx1] + net[idx1+1:idx2] + net[idx2+1:]
                if _verify_network_fast(candidate):
                    net = candidate
                    found = True
        
        if not found:
            break
    
    return net

def _swap_optimize(net, deadline, max_rounds=100):
    """Try swapping adjacent comparators to enable further removals."""
    net = list(net)
    start = time.time()
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        if len(net) < 2:
            break
        
        improved = False
        for i in range(len(net) - 1):
            if time.time() > deadline:
                return net
            # Try swapping positions i and i+1
            candidate = net[:i] + [net[i+1], net[i]] + net[i+2:]
            if _verify_network_fast(candidate):
                # Check if this enables a removal
                for j in range(len(candidate)):
                    if time.time() > deadline:
                        return net
                    candidate2 = candidate[:j] + candidate[j+1:]
                    if _verify_network_fast(candidate2):
                        net = candidate2
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return net

def _construct_network(deadline):
    """Main construction: build initial network and optimize."""
    # Start with bitonic sort (63 comparators) - much better than odd-even (153)
    net = _build_bitonic_network(N)
    
    # Verify initial network
    if not _verify_network_fast(net):
        # Fallback to odd-even
        net = _build_odd_even_network(N)
    
    best_net = list(net)
    best_len = len(net)
    
    # Phase 1: Greedy removal
    remaining = deadline - time.time()
    if remaining > 2.0:
        net = _greedy_removal(net, deadline - 2.0)
        if len(net) < best_len:
            best_net = list(net)
            best_len = len(net)
    
    # Phase 2: Randomized greedy removal with restarts
    remaining = deadline - time.time()
    if remaining > 2.0:
        for restart in range(5):
            if time.time() > deadline - 2.0:
                break
            # Start from best known
            net = list(best_net)
            # Add some random noise: insert a few random comparators
            for _ in range(random.randint(0, 3)):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                pos = random.randint(0, len(net))
                net = net[:pos] + [(i, j)] + net[pos:]
            
            net = _randomized_greedy_removal(net, deadline - 2.0, max_rounds=50)
            if len(net) < best_len:
                best_net = list(net)
                best_len = len(net)
    
    # Phase 3: More aggressive search - try removing multiple comparators
    remaining = deadline - time.time()
    if remaining > 3.0:
        for attempt in range(10):
            if time.time() > deadline - 3.0:
                break
            net = list(best_net)
            # Try removing 2-3 random comparators at once
            for _ in range(50):
                if time.time() > deadline - 3.0:
                    break
                k = random.randint(1, min(3, len(net)))
                indices = sorted(random.sample(range(len(net)), k))
                candidate = []
                for idx in range(len(net)):
                    if idx not in indices:
                        candidate.append(net[idx])
                if _verify_network_fast(candidate):
                    net = candidate
                    if len(net) < best_len:
                        best_net = list(net)
                        best_len = len(net)
    
    # Phase 4: Final greedy cleanup
    remaining = deadline - time.time()
    if remaining > 1.0:
        net = _greedy_removal(best_net, deadline - 0.5)
        if len(net) < best_len:
            best_net = list(net)
            best_len = len(net)
    
    # Final safety check
    if not _verify_network_fast(best_net):
        # Shouldn't happen, but fallback
        best_net = _build_bitonic_network(N)
    
    # Ensure all pairs are valid
    final_net = []
    for i, j in best_net:
        if 0 <= i < j < N:
            final_net.append((i, j))
    
    return final_net

def construct():
    """Construct a sorting network for N wires."""
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for 90s limit
    
    try:
        net = _construct_network(deadline)
    except Exception:
        # Fallback: return a known valid network
        net = _build_bitonic_network(N)
    
    # Final validation
    if not _verify_network_fast(net):
        net = _build_odd_even_network(N)
    
    return net

# EVOLVE-BLOCK-END
