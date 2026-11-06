"""Size-efficient sorting network search for N=16 wires."""
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
            # For input j (0..65535), wire i has value (j >> i) & 1
            # Build a 65536-bit integer where bit j = (j >> i) & 1
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
    """Verify a sorting network using bit-parallel zero-one principle."""
    if not network:
        return N == 1
        
    state = _get_initial_states()[:]
    
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    # Check if sorted: state[i] should be subset of state[i+1] for all i
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def bitonic_sorter(n):
    """Generate Batcher's bitonic sorting network for n wires."""
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

def odd_even_network():
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def shell_sort_network():
    """Shell sort variant network."""
    comparators = []
    gaps = [8, 4, 2, 1]
    for gap in gaps:
        for i in range(gap, N):
            comparators.append((i - gap, i))
    return comparators

def merge_sort_network(n):
    """Merge sort network."""
    if n <= 1:
        return []
    comparators = []
    
    def merge_sort(size, start):
        if size <= 1:
            return
        half = size // 2
        merge_sort(half, start)
        merge_sort(half, start + half)
        merge(half, start)
    
    def merge(size, start):
        if size <= 1:
            return
        for i in range(size):
            a = start + i
            b = start + i + size
            comparators.append((a, b))
        merge(size // 2, start)
        merge(size // 2, start + size)
    
    merge_sort(n, 0)
    return comparators

def compute_states(net):
    """Returns list of states before each comparator."""
    states = [None] * (len(net) + 1)
    current_state = _get_initial_states()[:]
    states[0] = current_state[:]
    
    for k, (i, j) in enumerate(net):
        next_state = current_state[:]
        min_val = current_state[i] & current_state[j]
        max_val = current_state[i] | current_state[j]
        next_state[i] = min_val
        next_state[j] = max_val
        states[k+1] = next_state
        current_state = next_state
        
    return states

def remove_redundant(net):
    """Remove comparators that are no-ops."""
    if not net:
        return []
        
    states = compute_states(net)
    to_remove = set()
    
    for k, (i, j) in enumerate(net):
        # If wire i <= wire j for all inputs before this comparator, it's redundant
        if states[k][i] & ~states[k][j] == 0:
            to_remove.add(k)
            
    if not to_remove:
        return net
        
    return [net[k] for k in range(len(net)) if k not in to_remove]

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s to leave 10s margin
    
    random.seed(42)
    
    # Generate starting networks
    start_nets = []
    
    bn = bitonic_sorter(N)
    bn_clean = remove_redundant(bn)
    if fast_verify(bn_clean):
        start_nets.append(bn_clean)
    
    oen = odd_even_network()
    oen_clean = remove_redundant(oen)
    if fast_verify(oen_clean):
        start_nets.append(oen_clean)
    
    sn = shell_sort_network()
    sn_clean = remove_redundant(sn)
    if fast_verify(sn_clean):
        start_nets.append(sn_clean)
    
    mn = merge_sort_network(N)
    mn_clean = remove_redundant(mn)
    if fast_verify(mn_clean):
        start_nets.append(mn_clean)
    
    if not start_nets:
        # Fallback: return odd-even if nothing else works
        return oen
    
    best_net = min(start_nets, key=len)
    best_count = len(best_net)
    
    # Simulated annealing
    current_net = best_net[:]
    current_count = len(current_net)
    
    T = 100.0
    cooling_rate = 0.999
    min_T = 0.1
    
    while time.time() < deadline and T > min_T:
        # Choose a mutation
        r = random.random()
        
        if r < 0.4 and current_count > 1:
            # Remove a random comparator
            idx = random.randint(0, len(current_net) - 1)
            candidate = current_net[:idx] + current_net[idx+1:]
        elif r < 0.7:
            # Add a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(current_net))
            candidate = current_net[:pos] + [(i, j)] + current_net[pos:]
        elif r < 0.85:
            # Swap two comparators
            if len(current_net) >= 2:
                i, j = random.sample(range(len(current_net)), 2)
                candidate = current_net[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
            else:
                continue
        else:
            # Mutate a comparator's endpoints
            if not current_net:
                continue
            idx = random.randint(0, len(current_net) - 1)
            i, j = current_net[idx]
            # Slightly perturb
            if random.random() < 0.5:
                new_i = max(0, min(N - 2, i + random.choice([-1, 0, 1])))
                new_j = min(N - 1, max(new_i + 1, j + random.choice([-1, 0, 1])))
                if new_i < new_j:
                    candidate = current_net[:]
                    candidate[idx] = (new_i, new_j)
                else:
                    continue
            else:
                new_j = max(new_i + 1, min(N - 1, j + random.choice([-1, 0, 1])))
                new_i = min(N - 2, max(0, i + random.choice([-1, 0, 1])))
                if new_i < new_j:
                    candidate = current_net[:]
                    candidate[idx] = (new_i, new_j)
                else:
                    continue
        
        # Verify candidate
        if not fast_verify(candidate):
            continue
        
        cand_count = len(candidate)
        delta = cand_count - current_count
        
        # Accept or reject
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_net = candidate
            current_count = cand_count
            
            if current_count < best_count:
                best_net = current_net[:]
                best_count = current_count
        
        T *= cooling_rate
        
        # Periodically remove redundant comparators
        if random.random() < 0.05:
            cleaned = remove_redundant(current_net)
            if fast_verify(cleaned):
                current_net = cleaned
                current_count = len(current_net)
                if current_count < best_count:
                    best_net = current_net[:]
                    best_count = current_count
    
    return best_net
