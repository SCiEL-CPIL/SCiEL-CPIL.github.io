import time
import random
import math

N = 16
NUM_INPUTS = 1 << N  # 65536

# Pre-compute initial states for bit-parallel verification
_initial_states = None

def _get_initial_states():
    global _initial_states
    if _initial_states is None:
        _initial_states = []
        for i in range(N):
            val = 0
            period = 1 << (i + 1)
            half = 1 << i
            period_val = ((1 << half) - 1) << half
            num_periods = NUM_INPUTS // period
            shift = 0
            for _ in range(num_periods):
                val |= period_val << shift
                shift += period
            _initial_states.append(val)
    return _initial_states

def fast_verify(network):
    """
    Verify a sorting network using bit-parallel zero-one principle.
    Returns (is_sorted, comparator_count).
    """
    if not network:
        # Empty network is sorted only if N=1, but N=16 so False
        return False, 0
        
    state = _get_initial_states()[:]
    
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    # Check if sorted: state[i] <= state[i+1] for all i
    # In bit-parallel, state[i] <= state[i+1] means (state[i] & ~state[i+1]) == 0
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False, len(network)
            
    return True, len(network)

def bitonic_sorter(n):
    """
    Generate Batcher's bitonic sorting network for n wires (n must be power of 2).
    """
    if n <= 1:
        return []
    
    comparators = []
    
    def bitonic_sort(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        bitonic_sort(half, start, ascending)
        bitonic_sort(half, start + half, not ascending)
        bitonic_merge(half, start, ascending)
    
    def bitonic_merge(size, start, ascending):
        if size <= 1:
            return
        for i in range(size):
            a = start + i
            b = start + i + size
            if ascending:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        bitonic_merge(size // 2, start, ascending)
        bitonic_merge(size // 2, start + size, ascending)
    
    bitonic_sort(n, 0, True)
    return comparators

def construct():
    deadline = time.time() + 50  # 50s for search
    
    # Start from Batcher (64 comparators)
    net = bitonic_sorter(N)
    best_net = list(net)
    best_count = len(net)
    
    # Verify initial
    is_sorted, count = fast_verify(net)
    if not is_sorted:
        # Fallback, though Batcher should be sorted
        return best_net
        
    random.seed(42)
    
    T = 10.0  # Initial temperature
    cooling = 0.995
    
    # Current energy
    curr_energy = count + (0 if is_sorted else 1000)
    
    while time.time() < deadline:
        if T < 0.1:
            break
            
        # Generate neighbor
        new_net = list(net)
        
        # Choose move type
        move_type = random.random()
        
        if len(new_net) > 0:
            if move_type < 0.5:
                # Remove random comparator
                idx = random.randint(0, len(new_net) - 1)
                new_net.pop(idx)
            elif move_type < 0.8:
                # Swap two comparators
                if len(new_net) >= 2:
                    i = random.randint(0, len(new_net) - 1)
                    j = random.randint(0, len(new_net) - 1)
                    if i != j:
                        new_net[i], new_net[j] = new_net[j], new_net[i]
            elif move_type < 0.95:
                # Move comparator to random position
                i = random.randint(0, len(new_net) - 1)
                comp = new_net.pop(i)
                j = random.randint(0, len(new_net))
                new_net.insert(j, comp)
            else:
                # Add random comparator (to escape local minima)
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                pos = random.randint(0, len(new_net))
                new_net.insert(pos, (i, j))
        else:
            # If empty, add
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            new_net.append((i, j))

        # Verify new network
        is_new_sorted, new_count = fast_verify(new_net)
        new_energy = new_count + (0 if is_new_sorted else 1000)
        
        # Metropolis acceptance
        delta = new_energy - curr_energy
        if delta < 0 or random.random() < math.exp(-delta / T):
            net = new_net
            curr_energy = new_energy
            
            # Update best if it's sorted and smaller
            if is_new_sorted and new_count < best_count:
                best_net = list(net)
                best_count = new_count
                # Optional: cool down faster if we found a new best?
                # T *= 0.9
        
        T *= cooling
    
    # Final cleanup: try to remove any remaining redundant comparators greedily
    # This ensures we don't return a network with unnecessary comparators
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for idx in range(len(best_net) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            candidate = best_net[:idx] + best_net[idx+1:]
            is_sorted, count = fast_verify(candidate)
            if is_sorted:
                best_net = candidate
                best_count = count
                changed = True
                break
                
    return best_net
