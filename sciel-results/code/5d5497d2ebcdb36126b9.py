import time
import random

N = 16
TOTAL_BITS = 1 << N  # 65536
MASK_ALL = (1 << TOTAL_BITS) - 1

# Precompute initial wire states as Python big integers
# Wire i has bit k set iff (k >> i) & 1 == 1
def _build_initial_wires():
    wires = [0] * N
    for i in range(N):
        w = 0
        # Bit i of k is 1 for k in ranges: [2^i*(2m+1), 2^i*(2m+1) + 2^i - 1]
        # Simpler: build by pattern
        period = 1 << (i + 1)
        half = 1 << i
        # For each period, bits [half, 2*half) within the period are set
        # We can build this efficiently
        for start in range(half, TOTAL_BITS, period):
            # Set bits from start to start+half-1
            # Create mask for half consecutive bits
            seg_mask = (1 << half) - 1
            w |= seg_mask << start
        wires[i] = w
    return wires

INITIAL_WIRES = _build_initial_wires()

def _verify_network(nets):
    """Verify network sorts all 65536 inputs using bit-parallel Python ints."""
    wires = list(INITIAL_WIRES)
    for (i, j) in nets:
        # swap_mask: bits where wire_i=1 and wire_j=0
        swap_mask = wires[i] & (~wires[j]) & MASK_ALL
        wires[i] = wires[i] ^ swap_mask
        wires[j] = wires[j] ^ swap_mask
    # Check sorted: for each input k, output should have top popcount(k) bits as 1
    # Equivalent: wire[0] should have all 1s where any wire has 1s, etc.
    # Faster check: the sorted output for input k has bits [N-popcount(k), N) set.
    # So wire[N-1] should equal the OR of all wires (the "any 1" mask per input)
    # And wire[0] should have the same popcount pattern... 
    # Actually simplest: check that wires are monotonically non-increasing.
    # wire[0] >= wire[1] >= ... >= wire[N-1] in the sense that wire[i] & ~wire[i+1] == 0
    # i.e., wire[i+1] is a subset of wire[i] (as bitsets)
    for i in range(N - 1):
        if wires[i + 1] & ~wires[i] & MASK_ALL:
            return False
    return True

def _batcher_odd_merge(n):
    """Generate Batcher's Odd-Merge sorting network for n inputs."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    result = []
    
    # 1. Recursively sort the two halves
    first_half = _batcher_odd_merge(half)
    result.extend(first_half)
    
    second_half = _batcher_odd_merge(half)
    result.extend([(i + half, j + half) for (i, j) in second_half])
    
    # 2. Merge the two sorted halves using Odd-Merge
    result.extend(_odd_even_merge(n))
    
    return result

def _odd_even_merge(n):
    """Generate the merge network for two sorted halves of total size n."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    result = []
    
    # Step 1: Cross compares (i, i+half) for i in 0..half-1
    for i in range(half):
        result.append((i, i + half))
    
    # Step 2: Recursively merge the odd-indexed and even-indexed subsequences
    # Odd indices: 1, 3, 5, ..., n-1 (size half)
    # Even indices: 0, 2, 4, ..., n-2 (size half)
    
    # Generate merge network for size half, then remap
    sub_merge = _odd_even_merge(half)
    
    # Remap for odd indices: local k -> global 2*k + 1
    for (i, j) in sub_merge:
        result.append((2 * i + 1, 2 * j + 1))
    
    # Remap for even indices: local k -> global 2*k
    for (i, j) in sub_merge:
        result.append((2 * i, 2 * j))
    
    return result

def _odd_even_transposition():
    """Generate the 120-comparator odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    # Generate candidate networks
    batcher_net = _batcher_odd_merge(N)
    transposition_net = _odd_even_transposition()
    
    # Verify both are correct
    batcher_valid = _verify_network(batcher_net)
    trans_valid = _verify_network(transposition_net)
    
    print(f"Batcher: {len(batcher_net)} comparators, valid={batcher_valid}")
    print(f"Transposition: {len(transposition_net)} comparators, valid={trans_valid}")
    
    # Start with the smaller valid network
    if batcher_valid:
        best = list(batcher_net)
    elif trans_valid:
        best = list(transposition_net)
    else:
        # Fallback: should not happen
        return transposition_net
    
    print(f"Initial best: {len(best)} comparators")
    
    # Pruning phase: iteratively try to remove comparators
    net = list(best)
    random.seed(42)
    
    while time.time() < deadline:
        improved = False
        indices = list(range(len(net)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            # Try removing net[idx]
            candidate = net[:idx] + net[idx + 1:]
            if _verify_network(candidate):
                net = candidate
                improved = True
                if len(net) < len(best):
                    best = list(net)
                    print(f"  Improved to {len(net)} comparators")
        
        if not improved:
            break
    
    # Try adding and re-pruning with different random seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        if len(net) >= len(best):
            break
        
        random.seed(seed)
        trial = list(net)
        
        # Add a random comparator
        i, j = random.sample(range(N), 2)
        if i > j:
            i, j = j, i
        trial.append((i, j))
        
        # Re-prune
        for _ in range(3):  # A few passes
            if time.time() > deadline:
                break
            improved = False
            indices = list(range(len(trial)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                candidate = trial[:idx] + trial[idx + 1:]
                if _verify_network(candidate):
                    trial = candidate
                    improved = True
            if not improved:
                break
        
        if len(trial) < len(best):
            best = list(trial)
            net = list(trial)
            print(f"  After add+prune: {len(net)} comparators")
    
    # Final verification
    final_valid = _verify_network(best)
    if not final_valid:
        # Fallback to a known valid network
        if batcher_valid:
            return batcher_net
        return transposition_net
    
    return best

# Test
if __name__ == "__main__":
    t0 = time.time()
    net = construct()
    t1 = time.time()
    print(f"Final: {len(net)} comparators, time={t1-t0:.2f}s")
    print(f"Valid: {_verify_network(net)}")
