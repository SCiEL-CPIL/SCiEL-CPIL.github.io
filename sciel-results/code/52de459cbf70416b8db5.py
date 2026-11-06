import numpy as np
import time

N = 18
_ALL_INPUTS = None

def _init_inputs():
    global _ALL_INPUTS
    indices = np.arange(2**N, dtype=np.uint32)
    shifts = np.arange(N, dtype=np.uint32)
    _ALL_INPUTS = ((indices[:, None] >> shifts) & 1).astype(np.uint8)

def _is_sorted_network(network):
    if not _ALL_INPUTS:
        _init_inputs()
        
    x = _ALL_INPUTS.copy()
    
    for i, j in network:
        min_vals = np.minimum(x[:, i], x[:, j])
        max_vals = np.maximum(x[:, i], x[:, j])
        x[:, i] = min_vals
        x[:, j] = max_vals
        
    if np.any(x[:, :-1] > x[:, 1:]):
        return False
    return True

def _depth(network):
    return len(network)

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    _init_inputs()
    
    # Odd-even transposition
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
            
    if not _is_sorted_network(network):
        # Should not happen
        pass
        
    rng = np.random.RandomState(42)
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        
        # 1. Greedy Removal
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > deadline - 2:
                break
            test_network = network[:idx] + network[idx+1:]
            if _is_sorted_network(test_network):
                network = test_network
                improved = True
                break
        
        if not improved:
            # 2. Swaps
            swaps_tried = 0
            while swaps_tried < 10 and (time.time() - start_time) < deadline - 2:
                swaps_tried += 1
                if len(network) < 2:
                    break
                i = rng.randint(0, len(network))
                j = rng.randint(0, len(network))
                if i == j:
                    continue
                network[i], network[j] = network[j], network[i]
                if _is_sorted_network(network):
                    improved = True
                    break
                else:
                    network[i], network[j] = network[j], network[i]
                    
            if not improved:
                # 3. Add-Remove (2-opt)
                inserts_tried = 0
                while inserts_tried < 5 and (time.time() - start_time) < deadline - 5:
                    inserts_tried += 1
                    pos = rng.randint(0, len(network) + 1)
                    i = rng.randint(0, N)
                    j = rng.randint(i + 1, N)
                    test_network = network[:pos] + [(i, j)] + network[pos:]
                    
                    if _is_sorted_network(test_network):
                        # Try to remove one to see if we can get smaller or different
                        for r_idx in range(len(test_network) - 1, -1, -1):
                            if (time.time() - start_time) > deadline - 5:
                                break
                            candidate = test_network[:r_idx] + test_network[r_idx+1:]
                            # We only accept if it's strictly better (smaller) 
                            # or if it's the same size but we are exploring?
                            # For now, only accept if smaller.
                            if len(candidate) < len(network) and _is_sorted_network(candidate):
                                network = candidate
                                improved = True
                                break
                        if improved:
                            break

    return network
