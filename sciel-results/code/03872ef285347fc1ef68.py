import time

N = 16

# EVOLVE-BLOCK-START
def _build_batcher_oms(n):
    """Build Batcher's Odd-Even Merge Sort network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    k = n // 2
    left_net = _build_batcher_oms(k)
    right_net = _build_batcher_oms(k)
    
    comparators = []
    # Apply left network to wires 0..k-1
    for i, j in left_net:
        comparators.append((i, j))
    # Apply right network to wires k..n-1
    for i, j in right_net:
        comparators.append((i + k, j + k))
    
    # Odd-even merge step
    # 1. Compare (i, i+k) for all i in 0..k-1
    for i in range(k):
        comparators.append((i, i + k))
    
    # 2. Recursively merge even and odd subsequences
    even_sub = _build_batcher_oms(k)
    odd_sub = _build_batcher_oms(k)
    
    for i, j in even_sub:
        comparators.append((2*i, 2*j))
    for i, j in odd_sub:
        comparators.append((2*i + 1, 2*j + 1))
        
    return comparators

def construct():
    start_time = time.time()
    deadline = 85.0  # seconds
    
    # 1. Generate Batcher's Odd-Even Merge Sort network
    network = _build_batcher_oms(N)
    
    # 2. Pre-compute initial S[w]
    INIT_S = []
    for w in range(N):
        mask = 0
        for i in range(1 << N):
            if (i >> w) & 1:
                mask |= (1 << i)
        INIT_S.append(mask)
        
    # 3. Local search optimization
    current_network = network
    changed = True
    while changed:
        changed = False
        new_network = []
        
        # Reset S to initial state
        S = INIT_S[:]  # Shallow copy of list of ints (ints are immutable)
        
        for i, j in current_network:
            # Check redundancy
            if (S[i] & ~S[j]) == 0:
                changed = True
            else:
                new_network.append((i, j))
                old_i = S[i]
                old_j = S[j]
                S[i] = old_i & old_j
                S[j] = old_i | old_j
                
        current_network = new_network
        
        if time.time() - start_time > deadline:
            break

    return current_network
# EVOLVE-BLOCK-END
