import time

N = 17

def _verify_fast(network):
    """Fast verification using 0-1 principle."""
    for mask in range(1 << N):
        # Simulate
        wires = [0] * N
        m = mask
        for i in range(N):
            wires[i] = m & 1
            m >>= 1
        for (i, j) in network:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        # Check sorted
        for k in range(N - 1):
            if wires[k] > wires[k + 1]:
                return False
    return True

def _build_merge_network(n):
    """Build a sorting network for n wires using recursive merge."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 3:
        return [(0, 1), (1, 2), (0, 1)]
    
    # Split into two halves
    mid = n // 2
    left_n = mid
    right_n = n - mid
    
    # Sort left and right
    left_net = _build_merge_network(left_n)
    right_net = _build_merge_network(right_n)
    
    # Merge
    merge_net = _build_merge_network_merge(left_n, right_n)
    
    # Combine
    network = []
    for (i, j) in left_net:
        network.append((i, j))
    for (i, j) in right_net:
        network.append((left_n + i, left_n + j))
    for (i, j) in merge_net:
        network.append((i, j))
    
    return network

def _build_merge_network_merge(a, b):
    """Build merge network for two sorted lists of size a and b."""
    # Simple merge: compare-exchange adjacent elements in a specific pattern
    # For now, use a simple approach: do a+b-1 compare-exchanges
    network = []
    # This is a placeholder; need proper merge
    # Use a simple bubble-sort-like merge
    for phase in range(a + b - 1):
        for i in range(phase % 2, a + b - 1, 2):
            if i < a + b:
                network.append((i, i + 1))
    return network

def construct():
    start = time.time()
    network = _build_merge_network(N)
    
    # Verify and prune
    deadline = start + 75
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(network)):
            if time.time() >= deadline:
                break
            candidate = network[:idx] + network[idx+1:]
            if _verify_fast(candidate):
                network = candidate
                improved = True
                break
    return network
