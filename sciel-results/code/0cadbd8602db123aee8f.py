import numpy as np
import time

N = 17
MAX_TIME = 55.0

def _get_inputs():
    if not hasattr(_get_inputs, 'inputs'):
        _get_inputs.inputs = np.zeros((2**N, N), dtype=np.uint8)
        for i in range(N):
            _get_inputs.inputs[:, i] = (np.arange(2**N) >> (N - 1 - i)) & 1
    return _get_inputs.inputs

def is_valid(net):
    x = _get_inputs().copy()
    for i, j in net:
        vi = x[:, i]
        vj = x[:, j]
        x[:, i] = np.minimum(vi, vj)
        x[:, j] = np.maximum(vi, vj)
    return np.all(np.diff(x, axis=1) >= 0)

def seed_network():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    best_net = seed_network()
    best_len = len(best_net)
    
    # Optimization loop
    while time.time() - start_time < MAX_TIME:
        current_net = list(best_net)
        
        # 1. Greedy Removal
        improved = True
        while improved and time.time() - start_time < MAX_TIME:
            improved = False
            # Randomize order of removal attempts
            indices = list(range(len(current_net)))
            np.random.shuffle(indices)
            
            for idx in indices:
                if idx >= len(current_net):
                    continue
                # Try removing current_net[idx]
                test_net = current_net[:idx] + current_net[idx+1:]
                if is_valid(test_net):
                    current_net = test_net
                    improved = True
                    break # Restart the while loop with new length
            
        if len(current_net) < best_len:
            best_net = current_net
            best_len = len(current_net)
            
        # 2. Local Search with Swaps
        # Try to swap comparators to potentially enable further removals
        # or just to find a different local minimum
        if time.time() - start_time < MAX_TIME:
            swaps_attempted = 0
            max_swaps = 200 # Limit swaps per restart to ensure progress
            
            while swaps_attempted < max_swaps and time.time() - start_time < MAX_TIME:
                if len(current_net) < 2:
                    break
                
                i, j = np.random.choice(len(current_net), 2, replace=False)
                
                # Perform swap
                current_net[i], current_net[j] = current_net[j], current_net[i]
                
                if is_valid(current_net):
                    # If valid, check if we can remove more
                    # Just update best if length is same (it is) or if we find a shorter one later
                    # Actually, swap doesn't change length.
                    # But it might lead to a state where removal is possible.
                    # So we keep the swap and continue.
                    swaps_attempted += 1
                else:
                    # Revert
                    current_net[i], current_net[j] = current_net[j], current_net[i]
                    swaps_attempted += 1
                    
            # After swaps, try greedy removal again
            improved = True
            while improved and time.time() - start_time < MAX_TIME:
                improved = False
                indices = list(range(len(current_net)))
                np.random.shuffle(indices)
                
                for idx in indices:
                    if idx >= len(current_net):
                        continue
                    test_net = current_net[:idx] + current_net[idx+1:]
                    if is_valid(test_net):
                        current_net = test_net
                        improved = True
                        break
                        
            if len(current_net) < best_len:
                best_net = current_net
                best_len = len(current_net)

    return best_net
