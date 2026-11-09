import numpy as np
import time

N = 16

def construct():
    start_time = time.time()
    time_limit = 85.0
    
    # Generate Odd-Even Transposition Network
    # This is a known correct sorting network with N*(N-1)/2 comparators?
    # No, OET has N phases.
    # Phase p: comparators (i, i+1) where i % 2 == p % 2.
    # Total comparators: sum of counts in each phase.
    # For N=16:
    # Even phases (8 of them): 8 comparators each.
    # Odd phases (8 of them): 7 comparators each.
    # Total: 8*8 + 8*7 = 64 + 56 = 120.
    
    current_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            current_network.append((i, i + 1))
            
    # Verify initial network
    if not _verify_network(current_network):
        # Should not happen, but if it does, we have a problem.
        # Return a trivial valid network? No, OET is valid.
        pass

    # Pruning Phase
    changed = True
    while changed and (time.time() - start_time) < time_limit:
        changed = False
        # Try to remove each comparator
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            new_network = current_network[:idx] + current_network[idx+1:]
            if _verify_network(new_network):
                current_network = new_network
                changed = True
                break
        
    # Swap Phase (optional, if time remains)
    if time.time() - start_time < time_limit - 5: # Leave 5s for final checks
        improved = True
        while improved and (time.time() - start_time) < time_limit - 5:
            improved = False
            for i in range(len(current_network) - 1):
                if time.time() - start_time > time_limit - 5:
                    break
                new_network = current_network.copy()
                new_network[i], new_network[i+1] = new_network[i+1], new_network[i]
                if _verify_network(new_network):
                    # Try to prune the swapped network
                    pruned = _prune_network(new_network, time_limit, start_time)
                    if len(pruned) < len(current_network):
                        current_network = pruned
                        improved = True
                        break
                        
    return current_network

def _verify_network(network):
    if not network:
        return False
        
    idx = np.arange(2**16, dtype=np.uint32)
    inputs = ((idx[:, None] >> np.arange(16, dtype=np.uint32)) & 1).astype(np.uint8)
    
    for i, j in network:
        min_val = np.minimum(inputs[:, i], inputs[:, j])
        max_val = np.maximum(inputs[:, i], inputs[:, j])
        inputs[:, i] = min_val
        inputs[:, j] = max_val
        
    diffs = np.diff(inputs, axis=1)
    if np.any(diffs < 0):
        return False
    return True

def _prune_network(network, time_limit, start_time):
    current = network
    changed = True
    while changed and (time.time() - start_time) < time_limit:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                return current
            new_net = current[:idx] + current[idx+1:]
            if _verify_network(new_net):
                current = new_net
                changed = True
                break
    return current
