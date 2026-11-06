"""Size-efficient sorting network on 18 wires."""
import time
import numpy as np

N = 18

def _verify_network_fast(comparators):
    """Verify network sorts all 2^N Boolean inputs using 0-1 principle with NumPy."""
    total = 1 << N
    indices = np.arange(total, dtype=np.int32)
    
    # input_mat[i, x] = (x >> i) & 1
    input_mat = np.empty((N, total), dtype=np.uint8)
    for i in range(N):
        input_mat[i] = (indices >> i) & 1
    
    for (i, j) in comparators:
        swap_mask = (input_mat[i] > input_mat[j])
        if np.any(swap_mask):
            tmp = input_mat[i][swap_mask].copy()
            input_mat[i][swap_mask] = input_mat[j][swap_mask]
            input_mat[j][swap_mask] = tmp
    
    # Check sorted: each column non-decreasing
    diffs = np.diff(input_mat, axis=0)
    return not np.any(diffs < 0)

def _odd_even_transposition():
    """Standard odd-even transposition network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _greedy_reduce(comparators, time_budget=45.0):
    """Try to remove comparators one by one while maintaining validity."""
    start = time.time()
    comps = list(comparators)
    
    # Verify initial validity
    if not _verify_network_fast(comps):
        return comps
    
    improved = True
    while improved:
        improved = False
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() - start > time_budget:
                break
            trial = comps[:idx] + comps[idx+1:]
            if _verify_network_fast(trial):
                comps = trial
                improved = True
                break
        
        if time.time() - start > time_budget:
            break
    
    return comps

def _local_search(comparators, time_budget=30.0):
    """Local search: try swaps and removals to find smaller valid networks."""
    start = time.time()
    comps = list(comparators)
    
    while time.time() - start < time_budget:
        found = False
        
        # Try removing any comparator
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() - start > time_budget:
                break
            trial = comps[:idx] + comps[idx+1:]
            if _verify_network_fast(trial):
                comps = trial
                found = True
                break
        
        if found:
            continue
        
        # Try swapping adjacent comparators
        for idx in range(len(comps) - 1):
            if time.time() - start > time_budget:
                break
            trial = comps[:]
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            if _verify_network_fast(trial):
                # After swap, try to remove
                for jdx in range(len(trial) - 1, -1, -1):
                    if time.time() - start > time_budget:
                        break
                    trial2 = trial[:jdx] + trial[jdx+1:]
                    if _verify_network_fast(trial2):
                        comps = trial2
                        found = True
                        break
                if found:
                    break
        
        if not found:
            break
    
    return comps

def construct():
    """Construct a sorting network for 18 wires."""
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Start with odd-even transposition
    comps = _odd_even_transposition()
    
    # Phase 1: Greedy reduction
    time_left = deadline - time.time()
    if time_left > 15.0:
        budget = min(time_left - 10.0, 40.0)
        comps = _greedy_reduce(comps, time_budget=budget)
    
    # Phase 2: Local search for further improvements
    time_left = deadline - time.time()
    if time_left > 15.0:
        budget = min(time_left - 10.0, 35.0)
        comps = _local_search(comps, time_budget=budget)
    
    # Final safety check
    if time.time() < deadline - 3.0:
        if not _verify_network_fast(comps):
            comps = _odd_even_transposition()
    
    return comps
