import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 55  # Safety margin before 60s kill

# Precompute full integer representation of all 2^N boolean inputs
all_masks = np.arange(1 << N, dtype=np.int32)
W0_int = all_masks.copy()

# Precompute popcounts for the initial inputs
# Using a lookup table for 8-bit chunks for speed
popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.int8)
# Split W0_int into 4 bytes
W0_bytes = W0_int[:, None] // (256 ** np.arange(4)) % 256
popcounts = popcount_table[W0_bytes].sum(axis=1)

# Precompute target values for sorted outputs
# Sorted output for k ones: 1s are in the most significant k bits (wires 18-k to 17)
# Value = 2^18 - 2^(18-k)
targets = np.zeros(N + 1, dtype=np.int32)
for k in range(N + 1):
    if k == 0:
        targets[k] = 0
    else:
        targets[k] = (1 << N) - (1 << (N - k))

def _odd_even_transposition_n(n):
    """Generate odd-even transposition for n wires (wires 0..n-1)."""
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def _odd_even_merge(n):
    """
    Odd-even merge: merges two sorted n-wire sequences into 2n sorted.
    Returns comparators for the merge step.
    """
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n, 2):
            comps.append((i, n + i))
    if n > 1:
        even_comps = _odd_even_merge(n // 2)
        for i, j in even_comps:
            comps.append((2 * i, 2 * j))
        odd_comps = _odd_even_merge(n // 2)
        for i, j in odd_comps:
            comps.append((2 * i + 1, 2 * j + 1))
    return comps

def _batcher_merge_sort_18():
    """Odd-even merge sort: split 18 into 9+9, sort each, merge."""
    half = 9
    left = _odd_even_transposition_n(half)
    right = [(i + half, j + half) for i, j in _odd_even_transposition_n(half)]
    merge = _odd_even_merge(half)
    return left + right + merge

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
    Uses fast bitwise operations on int32 arrays.
    """
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    if n == 0:
        # Empty network sorts only if N<=1, which is false for N=18
        return False 
        
    W = W0_int.copy()
    
    for i, j in net:
        # Extract bits
        ai = (W >> i) & 1
        aj = (W >> j) & 1
        
        # Min and Max
        mn = ai & aj
        mx = ai | aj
        
        # Clear bits i and j
        mask_i = ~(1 << i)
        mask_j = ~(1 << j)
        
        W = (W & mask_i & mask_j) | (mn << i) | (mx << j)
        
    # Check if W matches the sorted target for each input's popcount
    expected = targets[popcounts]
    
    return np.all(W == expected)

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _three_group_merge():
    """Split 18 into 3 groups of 6, sort each, merge pairwise."""
    gs = 6
    net = []
    for g in range(3):
        off = g * gs
        for i, j in _odd_even_transposition_n(gs):
            net.append((i + off, j + off))
    # Merge group 0 and 1
    m01 = _odd_even_merge(gs)
    for i, j in m01:
        net.append((i, j + gs))
    # Merge result (12 wires) with group 2
    m12 = _odd_even_merge(12)
    for i, j in m12:
        net.append((i, j + 12))
    return net

def _better_seed():
    """Pick the shortest valid seed from multiple constructions."""
    candidates = [
        _batcher_merge_sort_18(),
        _three_group_merge(),
        _odd_even_transposition(),
    ]
    best = min(candidates, key=len)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with the best seed from multiple constructions
    net = _better_seed()
    best_net = net[:]
    best_count = len(net)
    
    # Phase 1: Greedy removal (multiple passes until no more removals)
    for _pass in range(100):
        if time.time() > deadline - 15:
            break
        changed = False
        
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 15:
                break
            new_net = net[:idx] + net[idx+1:]
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                changed = True
                break
        
        if not changed:
            for idx in range(len(net)):
                if time.time() > deadline - 15:
                    break
                new_net = net[:idx] + net[idx+1:]
                if _is_sorted_network(new_net):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break
        
        if not changed:
            break
    
    # Phase 2: Local search with substitution move
    if time.time() < deadline - 3:
        T = 2.0
        min_T = 0.1
        cooling = 0.9995
        current_count = len(net)
        all_comps = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        while time.time() < deadline - 1:
            if T < min_T:
                net = best_net[:]
                current_count = best_count
                T = 2.0
            
            r = random.random()
            if r < 0.25:
                # Remove
                if len(net) == 0:
                    break
                idx = random.randint(0, len(net) - 1)
                new_net = net[:idx] + net[idx+1:]
            elif r < 0.45:
                # Insert
                idx = random.randint(0, len(net))
                comp = random.choice(all_comps)
                new_net = net[:idx] + [comp] + net[idx:]
            elif r < 0.70:
                # Substitute: replace one comparator with another
                if len(net) == 0:
                    continue
                idx = random.randint(0, len(net) - 1)
                new_comp = random.choice(all_comps)
                new_net = net[:]
                new_net[idx] = new_comp
            elif r < 0.85:
                # Swap
                if len(net) < 2:
                    continue
                idx1 = random.randint(0, len(net) - 1)
                idx2 = random.randint(0, len(net) - 1)
                if idx1 == idx2:
                    continue
                new_net = net[:]
                new_net[idx1], new_net[idx2] = new_net[idx2], new_net[idx1]
            else:
                # Move
                if len(net) == 0:
                    continue
                idx1 = random.randint(0, len(net) - 1)
                idx2 = random.randint(0, len(net) - 1)
                if idx1 == idx2:
                    continue
                new_net = net[:]
                comp = new_net.pop(idx1)
                new_net.insert(idx2, comp)
            
            if not _is_sorted_network(new_net):
                continue
                
            new_count = len(new_net)
            delta = new_count - current_count
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                net = new_net
                current_count = new_count
                if current_count < best_count:
                    best_net = net[:]
                    best_count = current_count
                    
            T *= cooling
    
    # Final validation
    if not _is_sorted_network(best_net):
        best_net = _odd_even_transposition()
        
    return best_net
