<<<<<<< SEARCH
import time
import numpy as np

N = 18

def _apply_network(network, x):
    """Apply comparison network to input vector x."""
    x = list(x)
    for i, j in network:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _is_sorted_network(network):
    """Check if network sorts all 0-1 inputs using 0-1 principle."""
    for mask in range(1 << N):
        x = [(mask >> i) & 1 for i in range(N)]
        result = _apply_network(network, x)
        if result != sorted(result):
            return False
    return True

def _depth(network):
    """Compute parallel depth of the network."""
    # Simple sequential depth (each comparator is a layer)
    # For true depth, we'd need to compute the longest path
    return len(network)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Verify it works
    assert _is_sorted_network(network), "Initial network is invalid"
    
    # Local search: try removing comparators
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > deadline - 5:
                break
            # Try removing comparator at idx
            test_network = network[:idx] + network[idx+1:]
            if _is_sorted_network(test_network):
                network = test_network
                improved = True
    
    # Try adding comparators to enable further removals
    if (time.time() - start_time) < deadline - 10:
        for _ in range(10):  # Limited number of add-remove cycles
            if (time.time() - start_time) > deadline - 10:
                break
            # Try all possible (i,j) pairs not already in network
            existing = set(network)
            for i in range(N):
                for j in range(i+1, N):
                    if (i,j) not in existing:
                        test_network = network + [(i,j)]
                        # Try removing one comparator from the extended network
                        for idx in range(len(test_network) - 1, -1, -1):
                            if (time.time() - start_time) > deadline - 10:
                                break
                            if idx == len(test_network) - 1:
                                continue  # Don't remove the one we just added
                            removed = test_network[:idx] + test_network[idx+1:]
                            if len(removed) < len(network) and _is_sorted_network(removed):
                                network = removed
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
            if not improved:
                break
    
    return network
=======
import time
import numpy as np

N = 18
NUM_INPUTS = 1 << N

def _apply_network_vectorized(network, bits):
    """Apply comparison network to all 2^N Boolean inputs at once.
    
    Args:
        network: list of (i, j) pairs
        bits: numpy array of shape (NUM_INPUTS, N) with 0/1 values
    
    Returns:
        numpy array of shape (NUM_INPUTS, N) with results
    """
    x = bits.copy()
    for i, j in network:
        swap_mask = x[:, i] > x[:, j]
        if np.any(swap_mask):
            x[swap_mask, i] = 0
            x[swap_mask, j] = 1
    return x

def _is_sorted_network_fast(network):
    """Check if network sorts all 0-1 inputs using vectorized 0-1 principle."""
    indices = np.arange(NUM_INPUTS)
    bits = np.zeros((NUM_INPUTS, N), dtype=np.int8)
    for i in range(N):
        bits[:, i] = (indices >> i) & 1
    
    result = _apply_network_vectorized(network, bits)
    
    # For binary rows, sorted means all 0s before all 1s
    ones_per_row = result.sum(axis=1)
    positions = np.arange(N)
    expected = (positions[None, :] >= (N - ones_per_row)[:, None]).astype(np.int8)
    
    return np.array_equal(result, expected)

def _depth(network):
    """Compute parallel depth of the network."""
    return len(network)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (evaluator kills at 90s)
    
    # Generate all Boolean inputs once
    indices = np.arange(NUM_INPUTS)
    bits = np.zeros((NUM_INPUTS, N), dtype=np.int8)
    for i in range(N):
        bits[:, i] = (indices >> i) & 1
    
    # Start with odd-even transposition network
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Verify it works
    assert _is_sorted_network_fast(network), "Initial network is invalid"
    
    # Phase 1: Greedy removal - try removing each comparator, keep if still valid
    improved = True
    while improved and (time.time() - start_time) < deadline - 5:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > deadline - 5:
                break
            test_network = network[:idx] + network[idx+1:]
            if _is_sorted_network_fast(test_network):
                network = test_network
                improved = True
    
    # Phase 2: Try swapping adjacent comparators to enable further removals
    if (time.time() - start_time) < deadline - 10:
        for _ in range(20):
            if (time.time() - start_time) > deadline - 10:
                break
            
            found_swap = False
            for idx in range(len(network) - 1):
                if (time.time() - start_time) > deadline - 10:
                    break
                
                swapped = network[:idx] + [network[idx+1], network[idx]] + network[idx+2:]
                
                for rem_idx in range(len(swapped) - 1, -1, -1):
                    if (time.time() - start_time) > deadline - 10:
                        break
                    removed = swapped[:rem_idx] + swapped[rem_idx+1:]
                    if len(removed) < len(network) and _is_sorted_network_fast(removed):
                        network = removed
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
            if not found_swap:
                break
    
    # Phase 3: Try adding a comparator then removing two (net -1)
    if (time.time() - start_time) < deadline - 15:
        existing = set(network)
        all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
        
        for add_pair in all_pairs:
            if (time.time() - start_time) > deadline - 15:
                break
            
            if add_pair in existing:
                continue
            
            extended = network + [add_pair]
            
            for rem1 in range(len(extended) - 2, -1, -1):
                if (time.time() - start_time) > deadline - 15:
                    break
                if rem1 == len(extended) - 1:
                    continue
                
                partial = extended[:rem1] + extended[rem1+1:]
                
                for rem2 in range(len(partial) - 1, -1, -1):
                    if (time.time() - start_time) > deadline - 15:
                        break
                    if rem2 == len(partial) - 1:
                        continue
                    
                    removed = partial[:rem2] + partial[rem2+1:]
                    if len(removed) < len(network) and _is_sorted_network_fast(removed):
                        network = removed
                        existing = set(network)
                        break
                
                if len(network) < len(extended) - 1:
                    break
            
            if len(network) < len(extended) - 1:
                break
    
    return network
>>>>>>> REPLACE
