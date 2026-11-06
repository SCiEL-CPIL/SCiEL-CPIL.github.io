import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 55  # Safety margin before 60s kill

def _depth(net):
    """Compute the parallel depth of a comparator network."""
    if not net:
        return 0
    layer = [0] * N
    for i, j in net:
        layer[i] = max(layer[i], layer[j]) + 1
    return max(layer) if layer else 0

def _is_sorted_network(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    Uses the 0-1 principle with vectorized NumPy operations.
    Returns True if the network sorts all inputs.
    """
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    if n == 0:
        return True
    
    # Generate all 2^N boolean inputs as a matrix
    all_masks = np.arange(1 << N, dtype=np.int32)
    wires = np.zeros((1 << N, N), dtype=np.int8)
    for i in range(N):
        wires[:, i] = (all_masks >> i) & 1
    
    # Apply comparators using vectorized min/max
    for i, j in net:
        wi = wires[:, i]
        wj = wires[:, j]
        min_val = np.minimum(wi, wj)
        max_val = np.maximum(wi, wj)
        wires[:, i] = min_val
        wires[:, j] = max_val
    
    # Check if sorted: wires[:, k] <= wires[:, k+1] for all k
    if np.any(wires[:, :-1] > wires[:, 1:]):
        return False
    return True

def _is_sorted_network_fast(net, base_wires=None):
    """
    Faster verification that can reuse a precomputed base wire matrix.
    If base_wires is provided, it must be a copy of the initial state.
    """
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    if n == 0:
        return True
    
    if base_wires is None:
        all_masks = np.arange(1 << N, dtype=np.int32)
        wires = np.zeros((1 << N, N), dtype=np.int8)
        for i in range(N):
            wires[:, i] = (all_masks >> i) & 1
    else:
        wires = base_wires
    
    # Apply comparators using vectorized min/max
    for i, j in net:
        wi = wires[:, i]
        wj = wires[:, j]
        min_val = np.minimum(wi, wj)
        max_val = np.maximum(wi, wj)
        wires[:, i] = min_val
        wires[:, j] = max_val
    
    # Check if sorted: wires[:, k] <= wires[:, k+1] for all k
    if np.any(wires[:, :-1] > wires[:, 1:]):
        return False
    return True

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _count_comparator_activity(net):
    """
    For each comparator in the network, count how many of the 2^N inputs
    actually have wire[i] > wire[j] (i.e., the comparator 'fires').
    Comparators with low activity are more likely to be removable.
    """
    all_masks = np.arange(1 << N, dtype=np.int32)
    wires = np.zeros((1 << N, N), dtype=np.int8)
    for i in range(N):
        wires[:, i] = (all_masks >> i) & 1
    
    activity = np.zeros(len(net), dtype=np.int32)
    for idx, (i, j) in enumerate(net):
        wi = wires[:, i]
        wj = wires[:, j]
        # Count how many inputs have wi > wj
        activity[idx] = np.sum(wi > wj)
        # Apply the comparator
        min_val = np.minimum(wi, wj)
        max_val = np.maximum(wi, wj)
        wires[:, i] = min_val
        wires[:, j] = max_val
    
    return activity

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with odd-even transposition network (153 comparators for N=18)
    net = _odd_even_transposition()
    best_net = net[:]
    best_count = len(net)
    
    # Precompute base wires for fast verification
    all_masks = np.arange(1 << N, dtype=np.int32)
    base_wires = np.zeros((1 << N, N), dtype=np.int8)
    for i in range(N):
        base_wires[:, i] = (all_masks >> i) & 1
    
    def verify(net_to_check):
        """Verify using a fresh copy of base wires."""
        wires = base_wires.copy()
        n = len(net_to_check)
        if n > MAX_COMPARATORS:
            return False
        if n == 0:
            return True
        for i, j in net_to_check:
            wi = wires[:, i]
            wj = wires[:, j]
            min_val = np.minimum(wi, wj)
            max_val = np.maximum(wi, wj)
            wires[:, i] = min_val
            wires[:, j] = max_val
        if np.any(wires[:, :-1] > wires[:, 1:]):
            return False
        return True
    
    # Phase 1: Greedy removal with activity-based ordering
    # First pass: compute activity to identify likely-removable comparators
    if time.time() < deadline - 10:
        activity = _count_comparator_activity(net)
        # Sort indices by activity (lowest first - most likely removable)
        sorted_indices = sorted(range(len(net)), key=lambda x: activity[x])
        
        # Try removing in order of lowest activity first
        for idx in sorted_indices:
            if time.time() > deadline - 3:
                break
            
            new_net = net[:idx] + net[idx+1:]
            if verify(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                # Recompute activity after removal
                activity = _count_comparator_activity(net)
                sorted_indices = sorted(range(len(net)), key=lambda x: activity[x])
    
    # Phase 2: Continue greedy removal end-to-start, then start-to-end
    changed = True
    while changed:
        if time.time() > deadline - 5:
            break
        changed = False
        
        # Try removing from end to start
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 3:
                return best_net
            
            new_net = net[:idx] + net[idx+1:]
            if verify(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                changed = True
                break
        
        # If no change from end-to-start, try start-to-end
        if not changed:
            for idx in range(len(net)):
                if time.time() > deadline - 3:
                    return best_net
                
                new_net = net[:idx] + net[idx+1:]
                if verify(new_net):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break
    
    # Phase 3: Local search - try replacing comparators to enable further removal
    if time.time() < deadline - 8:
        attempts = 0
        max_attempts = 200
        while time.time() < deadline - 3 and attempts < max_attempts:
            attempts += 1
            if len(net) < 2:
                break
            
            # Pick a random comparator to replace
            idx = random.randint(0, len(net) - 1)
            i, j = net[idx]
            
            # Try a random different comparator
            ni = random.randint(0, N - 2)
            nj = random.randint(ni + 1, N - 1)
            if (ni, nj) == (i, j):
                continue
            
            new_net = net[:]
            new_net[idx] = (ni, nj)
            
            if verify(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                attempts = 0  # Reset on success
                # After a successful replacement, try to remove more
                changed = True
                while changed and time.time() < deadline - 3:
                    changed = False
                    for ridx in range(len(net) - 1, -1, -1):
                        if time.time() > deadline - 3:
                            break
                        new_net2 = net[:ridx] + net[ridx+1:]
                        if verify(new_net2):
                            net = new_net2
                            best_net = net[:]
                            best_count = len(net)
                            changed = True
                            break
    
    # Final validation of best_net
    if not verify(best_net):
        # Fallback to original odd-even transposition
        best_net = _odd_even_transposition()
    
    return best_net
