"""Size-efficient sorting network on 16 wires."""
N = 16

# EVOLVE-BLOCK-START

def _verify(net, n=N):
    """Verify correctness using zero-one principle. Returns True if net sorts all 2^n inputs."""
    for mask in range(1 << n):
        inputs = [(mask >> i) & 1 for i in range(n)]
        for (i, j) in net:
            if inputs[i] > inputs[j]:
                inputs[i], inputs[j] = inputs[j], inputs[i]
        for k in range(n - 1):
            if inputs[k] > inputs[k + 1]:
                return False
    return True

def _find_redundant(net, n=N):
    """Find a redundant comparator in net (if any) by removing it and re-verifying."""
    m = len(net)
    for idx in range(m):
        candidate = list(net[:idx]) + list(net[idx+1:])
        if _verify(candidate, n):
            return idx, candidate
    return None, None

def construct():
    # Start with a known good network (Batcher odd-even-merge sort for N=16)
    def batcher_oom(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        h = n // 2
        left = batcher_oom(h)
        right = batcher_oom(h)
        right = [(i + h, j + h) for (i, j) in right]
        base = left + right
        merge = _odd_even_merge(n)
        return base + merge

    def _odd_even_merge(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        h = n // 2
        comparators = []
        # Odd merge
        for i in range(1, h, 2):
            comparators.append((i, i + h))
        # Even merge
        for i in range(0, h, 2):
            comparators.append((i, i + h))
        # Recursively merge odd and even subsequences
        odd_merge = _odd_even_merge(h)
        for (a, b) in odd_merge:
            comparators.append((2 * a + 1, 2 * b + 1))
        even_merge = _odd_even_merge(h)
        for (a, b) in even_merge:
            comparators.append((2 * a, 2 * b))
        return comparators

    current_net = batcher_oom(N)
    
    # Iteratively remove redundant comparators
    # We keep trying until no more redundant comparators can be found
    # or we hit a time limit / size limit
    import time
    start_time = time.time()
    timeout = 55  # Leave 5 seconds for overhead
    
    while time.time() - start_time < timeout:
        idx, candidate = _find_redundant(current_net, N)
        if idx is None:
            break
        current_net = candidate
        # After removing one, restart the search from the beginning
        # to find the next redundant one in the new network
    
    return current_net

# EVOLVE-BLOCK-END
