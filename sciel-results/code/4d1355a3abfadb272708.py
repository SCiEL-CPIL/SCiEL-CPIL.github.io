"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import time

def _batcher_omm(n):
    """Generate Batcher's Odd-Even Merge Network for n elements (n must be even)."""
    if n == 2:
        return [(0, 1)]
    
    comps = []
    # Step 1: Compare (0,1), (2,3), ..., (n-2, n-1)
    for i in range(0, n, 2):
        comps.append((i, i + 1))
    
    # Step 2: Recursively merge the odd-indexed elements
    # The odd indices are 1, 3, ..., n-1
    # We apply OMM(n//2) to these indices.
    # Virtual index k maps to global index 2*k + 1
    sub_net = _batcher_omm(n // 2)
    for i, j in sub_net:
        gi = 2 * i + 1
        gj = 2 * j + 1
        comps.append((gi, gj))
        
    return comps

def _batcher_om(n):
    """Generate Batcher's Odd-Even Merge Sort Network for n elements."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    comps = []
    # 1. Recursively sort even indices
    even_net = _batcher_om(n // 2)
    for i, j in even_net:
        comps.append((2 * i, 2 * j))
        
    # 2. Recursively sort odd indices
    odd_net = _batcher_om(n // 2)
    for i, j in odd_net:
        comps.append((2 * i + 1, 2 * j + 1))
        
    # 3. Merge the two sorted subsequences using OMM(n)
    merge_net = _batcher_omm(n)
    comps.extend(merge_net)
    
    return comps

def _is_valid_sorter(net, N=18, timeout=30, quick_check=True):
    """Check if the network sorts all 2^N Boolean inputs."""
    start_time = time.time()
    
    # Precompute sorted targets for each possible count of 1s
    # target[k] is the bitmask with k 1s in the most significant bits
    targets = [0] * (N + 1)
    for k in range(1, N + 1):
        targets[k] = ((1 << k) - 1) << (N - k)
        
    # Quick check with specific patterns to fail fast
    if quick_check:
        # Check all 0s, all 1s
        # Check single 1 at each position
        # Check single 0 at each position
        # Check half 1s, half 0s (various patterns)
        test_inputs = []
        test_inputs.append(0)
        test_inputs.append((1 << N) - 1)
        for i in range(N):
            test_inputs.append(1 << i)
            test_inputs.append(((1 << N) - 1) ^ (1 << i))
        # Add some randoms
        import random
        random.seed(42)
        for _ in range(50):
            test_inputs.append(random.getrandbits(N))
            
        for mask in test_inputs:
            state = mask
            for i, j in net:
                bit_i = (state >> i) & 1
                bit_j = (state >> j) & 1
                if bit_i > bit_j:
                    state ^= (1 << i) | (1 << j)
            k = bin(state).count('1')
            if state != targets[k]:
                return False
                
    # Full check
    for mask in range(1 << N):
        if time.time() - start_time > timeout:
            return False
            
        state = mask
        for i, j in net:
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1
            if bit_i > bit_j:
                state ^= (1 << i) | (1 << j)
                
        k = bin(state).count('1')
        if state != targets[k]:
            return False
            
    return True

def construct():
    """Construct a sorting network on 18 wires using Batcher's Odd-Even Merge Sort
    and local search to remove redundant comparators."""
    
    start_time = time.time()
    timeout = 55  # Leave 5s for safety
    
    # Generate Batcher network
    net = _batcher_om(N)
    
    # Verify initial network
    if not _is_valid_sorter(net, N, timeout=10):
        # Fallback to Odd-Even Transposition if Batcher fails (should not happen)
        net = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
        
    # Local Search: Try to remove comparators
    # We iterate until no more comparators can be removed
    improved = True
    while improved and time.time() - start_time < timeout - 5:
        improved = False
        # Iterate backwards so we don't skip indices when removing
        for k in range(len(net) - 1, -1, -1):
            if time.time() - start_time > timeout - 5:
                break
                
            # Try removing net[k]
            candidate_net = net[:k] + net[k+1:]
            
            # Check validity
            if _is_valid_sorter(candidate_net, N, timeout=5):
                net = candidate_net
                improved = True
                # Restart the loop to try removing more from the new list
                # Breaking the inner loop to restart the outer while loop
                break
                
    return net
# EVOLVE-BLOCK-END
