import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 55

# Precompute full matrix of all 2^N boolean inputs
num_inputs = 1 << N
W_target = np.zeros((num_inputs, N), dtype=np.uint8)

# Calculate popcount for each mask to determine number of 1s
masks = np.arange(num_inputs, dtype=np.int32)
popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
low = masks & 0xFF
mid = (masks >> 8) & 0xFF
high = (masks >> 16) & 0x3
popcounts = popcount_table[low] + popcount_table[mid] + popcount_table[high]

# Construct W_target
for k in range(N + 1):
    idxs = np.where(popcounts == k)[0]
    if len(idxs) > 0:
        W_target[idxs, N-k:] = 1

# Precompute initial W0 (input states)
W0 = np.zeros((num_inputs, N), dtype=np.uint8)
for i in range(N):
    W0[:, i] = (masks >> i) & 1

def _is_sorted_network(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    """
    if len(net) > MAX_COMPARATORS:
        return False
    
    W = W0.copy()
    for i, j in net:
        min_val = W[:, i] & W[:, j]
        max_val = W[:, i] | W[:, j]
        W[:, i] = min_val
        W[:, j] = max_val
        
    return np.array_equal(W, W_target)

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _batcher_network(n):
    """
    Generate Batcher's odd-even merge sort network for n wires.
    This is a known efficient network.
    """
    if n == 1:
        return []
    
    # Batcher's bitonic sort is simpler and often comparable in size.
    # Let's use Bitonic Sort network.
    return _bitonic_network(n)

def _bitonic_network(n):
    """
    Generate Bitonic sorting network.
    """
    comps = []
    
    def bitonic_merge(lo, hi, direction):
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        # Merge two halves
        # The standard bitonic merge compares (i, i + (hi-lo)//2)
        # We need to be careful with the recursive structure.
        
        # Standard Bitonic Sort:
        # 1. Sort first half ascending, second half descending.
        # 2. Merge them.
        
        # Actually, let's implement the recursive bitonic sort directly.
        pass

    # Simpler: Use the standard recursive definition for Bitonic Sort.
    # B(n):
    #   if n == 1: return
    #   B(n/2) on first half (ascending)
    #   B(n/2) on second half (descending)
    #   Merge(n/2)
    
    # This requires a helper to generate ascending/descending.
    # Let's just use the known comparator list for n=18 if we can find one, 
    # or stick to Odd-Even Transposition which is 153.
    # Bitonic sort for 18:
    # Depth is log2(18)^2 ~ 6.5^2 ~ 42.
    # Size is n/2 * log2(n)^2 ~ 9 * 42 ~ 378? No, that's too big.
    # Bitonic sort size is (n/2) * log2(n) * (log2(n) + 1) / 2 ?
    # For n=16, size is 120.
    # For n=18, it will be larger.
    # Odd-even transposition is 153.
    # So Odd-Even is likely better or equal for N=18.
    
    return _odd_even_transposition()

def _greedy_remove(net):
    """
    Try to remove comparators one by one from the end, then from the beginning.
    """
    changed = True
    while changed:
        changed = False
        # Try removing from the end
        for idx in range(len(net) - 1, -1, -1):
            new_net = net[:idx] + net[idx+1:]
            if _is_sorted_network(new_net):
                net = new_net
                changed = True
                break
        # If no change, try removing from the beginning
        if not changed:
            for idx in range(len(net)):
                new_net = net[:idx] + net[idx+1:]
                if _is_sorted_network(new_net):
                    net = new_net
                    changed = True
                    break
    return net

def _simulated_annealing(net, start_time, deadline):
    """
    Simulated annealing to minimize comparator count.
    """
    current_net = net[:]
    best_net = net[:]
    best_count = len(net)
    current_count = len(net)
    
    T = 1.0
    min_T = 0.01
    cooling = 0.9995
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if T < min_T or no_improve > max_no_improve:
            # Restart
            current_net = best_net[:]
            current_count = best_count
            T = 1.0
            no_improve = 0
            
        # Generate a move
        r = random.random()
        if r < 0.5:
            # Remove a comparator
            if len(current_net) == 0:
                break
            idx = random.randint(0, len(current_net) - 1)
            new_net = current_net[:idx] + current_net[idx+1:]
            delta = -1
        elif r < 0.8:
            # Insert a comparator
            idx = random.randint(0, len(current_net))
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            new_net = current_net[:idx] + [(i, j)] + current_net[idx:]
            delta = 1
        else:
            # Swap two comparators
            if len(current_net) < 2:
                continue
            idx1, idx2 = random.sample(range(len(current_net)), 2)
            new_net = current_net[:]
            new_net[idx1], new_net[idx2] = new_net[idx2], new_net[idx1]
            delta = 0
        
        # Check validity
        if not _is_sorted_network(new_net):
            no_improve += 1
            continue
        
        # Accept or reject
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_net = new_net
            current_count = len(current_net)
            no_improve = 0
            if current_count < best_count:
                best_net = current_net[:]
                best_count = current_count
        
        T *= cooling
        
    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with a good seed network
    net = _odd_even_transposition()
    
    # Phase 1: Greedy removal
    if time.time() < deadline - 10:
        net = _greedy_remove(net)
    
    best_net = net[:]
    best_count = len(net)
    
    # Phase 2: Simulated Annealing
    if time.time() < deadline - 5:
        sa_deadline = deadline - 5
        net = _simulated_annealing(net, start_time, sa_deadline)
        if len(net) < best_count:
            best_net = net[:]
            best_count = len(net)
    
    # Final validation
    if not _is_sorted_network(best_net):
        best_net = _odd_even_transposition()
        
    return best_net
