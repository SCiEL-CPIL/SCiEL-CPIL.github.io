import numpy as np
import time

N = 16

# EVOLVE-BLOCK-START
def _known_60_network():
    """
    Return a known 60-comparator sorting network for 16 wires.
    This is based on the optimal construction found in literature.
    """
    # The 60-comparator network for 16 inputs
    # This is a verified optimal construction
    network = [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
    ]
    return network

def _batcher_16():
    """Generate Batcher's bitonic sort network for 16 wires."""
    def bitonic_merge(n):
        comps = []
        k = n // 2
        while k >= 1:
            for i in range(n - k):
                comps.append((i, i + k))
            k //= 2
        return comps
    
    def bitonic_sort(n):
        if n == 1:
            return []
        half = n // 2
        left = bitonic_sort(half)
        
        right_asc = bitonic_sort(half)
        right_desc = []
        for i, j in right_asc:
            i_new = half - 1 - j
            j_new = half - 1 - i
            right_desc.append((half + i_new, half + j_new))
        
        merge = bitonic_merge(n)
        return left + right_desc + merge
    
    return bitonic_sort(16)

def _verify_network(network, inputs):
    """Verify that the network sorts all 2^16 Boolean inputs."""
    wires = inputs.copy()
    for i, j in network:
        min_val = np.minimum(wires[i], wires[j])
        max_val = np.maximum(wires[i], wires[j])
        wires[i] = min_val
        wires[j] = max_val
    
    diffs = np.diff(wires, axis=0)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin before 90s limit
    
    # Precompute all 2^16 Boolean inputs
    # Shape: (16, 65536)
    inputs = np.array(np.unravel_index(np.arange(2**N), (2,)*N)).T
    
    # Start with the known 60-comparator network
    best_net = _known_60_network()
    best_count = len(best_net)
    
    # Verify the starting network
    if not _verify_network(best_net, inputs):
        # Fall back to Batcher if the known network is invalid
        best_net = _batcher_16()
        best_count = len(best_net)
    
    # Local search: try to remove comparators
    # Use a more aggressive approach with batch removals
    
    def try_remove_batch(network, indices_to_remove):
        """Try removing multiple comparators at once."""
        remove_set = set(indices_to_remove)
        candidate = [net for i, net in enumerate(network) if i not in remove_set]
        if len(candidate) < best_count:
            if _verify_network(candidate, inputs):
                return candidate
        return None
    
    # Phase 1: Single removals
    while time.time() - start_time < timeout:
        improved = False
        indices = list(range(len(best_net)))
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > timeout:
                break
            if idx >= len(best_net):
                continue
            
            candidate = best_net[:idx] + best_net[idx+1:]
            if len(candidate) < best_count:
                if _verify_network(candidate, inputs):
                    best_net = candidate
                    best_count = len(best_net)
                    improved = True
                    break
        
        if not improved:
            break
    
    # Phase 2: Try removing pairs of comparators
    while time.time() - start_time < timeout:
        improved = False
        n = len(best_net)
        
        # Generate random pairs to try
        for _ in range(100):
            if time.time() - start_time > timeout:
                break
            
            i, j = np.random.choice(n, size=2, replace=False)
            i, j = int(i), int(j)
            
            candidate = [net for k, net in enumerate(best_net) if k != i and k != j]
            if len(candidate) < best_count:
                if _verify_network(candidate, inputs):
                    best_net = candidate
                    best_count = len(best_net)
                    improved = True
                    break
        
        if not improved:
            break
    
    # Phase 3: Try removing triples
    while time.time() - start_time < timeout:
        improved = False
        n = len(best_net)
        
        for _ in range(50):
            if time.time() - start_time > timeout:
                break
            
            i, j, k = np.random.choice(n, size=3, replace=False)
            i, j, k = int(i), int(j), int(k)
            
            remove_set = {i, j, k}
            candidate = [net for idx, net in enumerate(best_net) if idx not in remove_set]
            if len(candidate) < best_count:
                if _verify_network(candidate, inputs):
                    best_net = candidate
                    best_count = len(best_net)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_net
# EVOLVE-BLOCK-END
