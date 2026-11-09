"""Size-efficient sorting network on 18 wires via SAT-inspired greedy synthesis."""
import numpy as np
import time
import random
from itertools import combinations

N = 18
TOTAL_INPUTS = 1 << N  # 262144

def _apply_network_to_inputs(inputs, network):
    """
    Apply a sorting network to a batch of inputs.
    inputs: (B, N) int8 array
    network: list of (i, j) comparators
    Returns: sorted (B, N) int8 array
    """
    if len(network) == 0:
        return inputs.copy()
    
    result = inputs.copy()
    for i, j in network:
        mask = result[:, i] > result[:, j]
        if np.any(mask):
            temp = result[mask, i].copy()
            result[mask, i] = result[mask, j]
            result[mask, j] = temp
    return result

def _is_sorted(inputs):
    """Check if all rows are sorted (non-decreasing)."""
    if inputs.shape[0] == 0:
        return True
    return np.all(np.diff(inputs, axis=1) >= 0)

def _verify_network(network):
    """
    Verify if the network sorts all 2^N Boolean inputs using the 0-1 principle.
    """
    if len(network) == 0:
        return N <= 1
    
    chunk_size = 1 << 15  # 32768
    for start in range(0, TOTAL_INPUTS, chunk_size):
        end = min(start + chunk_size, TOTAL_INPUTS)
        masks = np.arange(start, end, dtype=np.int64)
        inputs = (masks[:, None] >> np.arange(N)[None, :]) & 1
        inputs = inputs.astype(np.int8)
        
        sorted_inputs = _apply_network_to_inputs(inputs, network)
        if not _is_sorted(sorted_inputs):
            return False
    
    return True

def _get_failing_inputs(network, sample_size=8192):
    """
    Get a sample of failing inputs for a given network.
    Returns a set of input indices that are not sorted.
    """
    if len(network) == 0:
        # All inputs with more than one 1 are failing
        return set(range(1, TOTAL_INPUTS))
    
    # Sample some inputs to check
    sample_indices = random.sample(range(TOTAL_INPUTS), min(sample_size, TOTAL_INPUTS))
    sample_indices.sort()
    
    failing = set()
    chunk_size = 2048
    for start in range(0, len(sample_indices), chunk_size):
        end = min(start + chunk_size, len(sample_indices))
        masks = np.array(sample_indices[start:end], dtype=np.int64)
        inputs = (masks[:, None] >> np.arange(N)[None, :]) & 1
        inputs = inputs.astype(np.int8)
        
        sorted_inputs = _apply_network_to_inputs(inputs, network)
        not_sorted_mask = np.any(np.diff(sorted_inputs, axis=1) < 0, axis=1)
        
        for idx in range(len(masks)):
            if not_sorted_mask[idx]:
                failing.add(sample_indices[start + idx])
    
    return failing

def _compute_comparator_impact(comparator, network, sample_size=4096):
    """
    Compute how many failing inputs a comparator would help fix.
    This is used in the greedy set-cover approach.
    """
    # For now, we estimate impact by checking if adding this comparator
    # reduces the number of failing inputs in a sample
    test_network = network + [comparator]
    
    # Use a smaller sample for speed
    sample_indices = random.sample(range(TOTAL_INPUTS), min(sample_size, TOTAL_INPUTS))
    sample_indices.sort()
    
    # First, get failing inputs for current network
    masks = np.array(sample_indices, dtype=np.int64)
    inputs = (masks[:, None] >> np.arange(N)[None, :]) & 1
    inputs = inputs.astype(np.int8)
    
    # Apply current network
    current_sorted = _apply_network_to_inputs(inputs, network)
    current_failing = np.any(np.diff(current_sorted, axis=1) < 0, axis=1)
    
    # Apply network with comparator
    new_sorted = _apply_network_to_inputs(inputs, test_network)
    new_failing = np.any(np.diff(new_sorted, axis=1) < 0, axis=1)
    
    # Count how many inputs are fixed
    fixed = np.sum(current_failing & ~new_failing)
    # Count how many inputs are broken (should be 0 if valid, but may happen)
    broken = np.sum(~current_failing & new_failing)
    
    return int(fixed), int(broken)

def construct():
    """
    Construct a sorting network on N=18 wires using SAT-inspired greedy synthesis.
    
    Approach:
    1. Start with a seed network (odd-even transposition)
    2. Use greedy removal to reduce size
    3. Use greedy insertion with impact estimation to find useful comparators
    4. Local search with simulated annealing for refinement
    """
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer
    
    random.seed(42)
    
    # Phase 1: Generate seed network (odd-even transposition)
    def odd_even_transposition():
        net = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                net.append((i, i + 1))
        return net
    
    network = odd_even_transposition()
    best_network = network[:]
    best_size = len(network)
    
    # Verify seed
    if not _verify_network(network):
        raise ValueError("Seed network is invalid")
    
    # Phase 2: Greedy removal (20% of time)
    greedy_time_limit = time_limit * 0.2
    while time.time() - start_time < greedy_time_limit:
        removed_any = False
        indices = list(range(len(network)))
        random.shuffle(indices)
        
        for idx in indices:
            if idx >= len(network):
                continue
            new_network = network[:idx] + network[idx+1:]
            if _verify_network(new_network):
                network = new_network
                best_network = network[:]
                best_size = len(network)
                removed_any = True
                break  # Restart scan
        
        if not removed_any:
            break
    
    # Phase 3: Greedy insertion with impact estimation (30% of time)
    # Generate candidate comparator positions
    all_comparators = list(combinations(range(N), 2))
    
    insertion_time_limit = greedy_time_limit + time_limit * 0.3
    current_size = len(network)
    
    while time.time() - start_time < insertion_time_limit:
        # Find the comparator that fixes the most failing inputs
        best_comparator = None
        best_fixed = -1
        best_broken = 0
        
        # Sample a subset of candidates for speed
        candidates = random.sample(all_comparators, min(200, len(all_comparators)))
        
        for comp in candidates:
            fixed, broken = _compute_comparator_impact(comp, network, sample_size=2048)
            
            # We want to maximize fixed and minimize broken
            score = fixed - 2 * broken
            
            if score > best_fixed or (score == best_fixed and broken < best_broken):
                best_fixed = score
                best_comparator = comp
                best_broken = broken
        
        if best_comparator is None or best_fixed <= 0:
            break
        
        # Insert the best comparator
        pos = random.randint(0, len(network))
        network = network[:pos] + [best_comparator] + network[pos:]
        current_size = len(network)
        
        # Verify
        if _verify_network(network):
            if current_size < best_size:
                best_network = network[:]
                best_size = current_size
    
    # Phase 4: Local search with simulated annealing (remaining time)
    T = 1.0  # Temperature
    current_size = len(network)
    
    while time.time() - start_time < time_limit:
        # Generate neighbor
        r = random.random()
        new_network = network[:]
        
        if r < 0.3:
            # Remove a random comparator
            if len(network) > 0:
                idx = random.randint(0, len(network) - 1)
                new_network = network[:idx] + network[idx+1:]
        elif r < 0.6:
            # Insert a random comparator
            i, j = random.sample(range(N), 2)
            if i > j:
                i, j = j, i
            pos = random.randint(0, len(network))
            new_network = network[:pos] + [(i, j)] + network[pos:]
        else:
            # Swap two comparators
            if len(network) > 1:
                i, j = random.sample(range(len(network)), 2)
                new_network = network[:]
                new_network[i], new_network[j] = new_network[j], new_network[i]
        
        new_size = len(new_network)
        
        # Verify
        if _verify_network(new_network):
            # Accept if better, or with some probability if same/worse
            if new_size < current_size:
                network = new_network
                best_network = network[:]
                best_size = new_size
                current_size = new_size
                T = max(0.1, T * 0.99)  # Cool down
            else:
                delta = new_size - current_size
                if delta == 0:
                    p = 0.1  # Accept same size with low prob to shuffle
                else:
                    p = np.exp(-delta / T)
                
                if random.random() < p:
                    network = new_network
                    current_size = new_size
                    T = max(0.1, T * 0.99)
    
    return best_network

# Cache the result
_cached_network = None

def construct():
    """
    Construct a sorting network on N=18 wires.
    Uses caching to avoid recomputation.
    """
    global _cached_network
    if _cached_network is None:
        _cached_network = _construct_impl()
    return _cached_network

def _construct_impl():
    """Internal implementation of construct."""
    # Re-execute the logic above
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer
    
    random.seed(42)
    
    def odd_even_transposition():
        net = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                net.append((i, i + 1))
        return net
    
    network = odd_even_transposition()
    best_network = network[:]
    best_size = len(network)
    
    if not _verify_network(network):
        raise ValueError("Seed network is invalid")
    
    greedy_time_limit = time_limit * 0.2
    while time.time() - start_time < greedy_time_limit:
        removed_any = False
        indices = list(range(len(network)))
        random.shuffle(indices)
        
        for idx in indices:
            if idx >= len(network):
                continue
            new_network = network[:idx] + network[idx+1:]
            if _verify_network(new_network):
                network = new_network
                best_network = network[:]
                best_size = len(network)
                removed_any = True
                break
        
        if not removed_any:
            break
    
    all_comparators = list(combinations(range(N), 2))
    insertion_time_limit = greedy_time_limit + time_limit * 0.3
    current_size = len(network)
    
    while time.time() - start_time < insertion_time_limit:
        best_comparator = None
        best_fixed = -1
        best_broken = 0
        
        candidates = random.sample(all_comparators, min(200, len(all_comparators)))
        
        for comp in candidates:
            fixed, broken = _compute_comparator_impact(comp, network, sample_size=2048)
            score = fixed - 2 * broken
            
            if score > best_fixed or (score == best_fixed and broken < best_broken):
                best_fixed = score
                best_comparator = comp
                best_broken = broken
        
        if best_comparator is None or best_fixed <= 0:
            break
        
        pos = random.randint(0, len(network))
        network = network[:pos] + [best_comparator] + network[pos:]
        current_size = len(network)
        
        if _verify_network(network):
            if current_size < best_size:
                best_network = network[:]
                best_size = current_size
    
    T = 1.0
    current_size = len(network)
    
    while time.time() - start_time < time_limit:
        r = random.random()
        new_network = network[:]
        
        if r < 0.3:
            if len(network) > 0:
                idx = random.randint(0, len(network) - 1)
                new_network = network[:idx] + network[idx+1:]
        elif r < 0.6:
            i, j = random.sample(range(N), 2)
            if i > j:
                i, j = j, i
            pos = random.randint(0, len(network))
            new_network = network[:pos] + [(i, j)] + network[pos:]
        else:
            if len(network) > 1:
                i, j = random.sample(range(len(network)), 2)
                new_network = network[:]
                new_network[i], new_network[j] = new_network[j], new_network[i]
        
        new_size = len(new_network)
        
        if _verify_network(new_network):
            if new_size < current_size:
                network = new_network
                best_network = network[:]
                best_size = new_size
                current_size = new_size
                T = max(0.1, T * 0.99)
            else:
                delta = new_size - current_size
                if delta == 0:
                    p = 0.1
                else:
                    p = np.exp(-delta / T)
                
                if random.random() < p:
                    network = new_network
                    current_size = new_size
                    T = max(0.1, T * 0.99)
    
    return best_network
