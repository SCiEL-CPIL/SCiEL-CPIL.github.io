"""Search for a large graph with no regular induced subgraph on 6+ vertices."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s kill

    rng = random.Random(42)
    fallback = _make_fallback(rng)

    best = fallback
    best_n = fallback['n']

    # Try to find larger graphs via local search with bitmask validation
    for target_n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if time.time() > deadline - 15:
            break
        result = _local_search(target_n, deadline - 5, rng)
        if result is not None and result['n'] > best_n:
            best = result
            best_n = result['n']

    return best


def _make_fallback(rng):
    """Generate a verified valid graph on 10 vertices."""
    n = 10
    for _ in range(10000):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if _is_valid_bitmask(adj, n):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    return {'n': 5, 'edges': []}


def _is_valid_bitmask(adj, n):
    """Check if no subset of size >= 6 induces a regular graph using bitmasks."""
    # Precompute all 6-subsets
    all_subsets = list(combinations(range(n), 6))
    for subset in all_subsets:
        if _is_regular_subset_bitmask(adj, subset):
            return False
    # Check larger subsets
    for size in range(7, n + 1):
        for subset in combinations(range(n), size):
            if _is_regular_subset_bitmask(adj, subset):
                return False
    return True


def _is_regular_subset_bitmask(adj, subset):
    """Check if induced subgraph on subset is regular, using bitmask adjacency."""
    s = len(subset)
    if s < 2:
        return True
    degrees = [0] * s
    for i in range(s):
        ui = subset[i]
        for j in range(i + 1, s):
            uj = subset[j]
            if adj[ui] & (1 << uj):
                degrees[i] += 1
                degrees[j] += 1
    d0 = degrees[0]
    for i in range(1, s):
        if degrees[i] != d0:
            return False
    return True


def _find_violation_bitmask(adj, n):
    """Find a subset of size >= 6 that induces a regular graph. Returns subset or None."""
    all_subsets = list(combinations(range(n), 6))
    for subset in all_subsets:
        if _is_regular_subset_bitmask(adj, subset):
            return subset
    for size in range(7, n + 1):
        for subset in combinations(range(n), size):
            if _is_regular_subset_bitmask(adj, subset):
                return subset
    return None


def _local_search(target_n, deadline, rng):
    """Local search with bitmask adjacency to find a valid graph on target_n vertices."""
    n = target_n

    # Initialize with random graph using bitmask
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Check if already valid
    if _is_valid_bitmask(adj, n):
        return _to_result_bitmask(adj, n)

    # Precompute all 6-subsets for fast violation checking
    all_6_subsets = list(combinations(range(n), 6))

    for iteration in range(20000):
        if time.time() > deadline:
            break

        # Find a violation
        viol = None
        for subset in all_6_subsets:
            if _is_regular_subset_bitmask(adj, subset):
                viol = subset
                break
        if viol is None:
            # Check larger subsets
            for size in range(7, n + 1):
                for subset in combinations(range(n), size):
                    if _is_regular_subset_bitmask(adj, subset):
                        viol = subset
                        break
                if viol is not None:
                    break

        if viol is None:
            # No violation found, graph is valid!
            return _to_result_bitmask(adj, n)

        # Pick a random edge within the violation subset to flip
        s = len(viol)
        i, j = rng.sample(range(s), 2)
        u, v = viol[i], viol[j]

        # Flip the edge
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        # Occasionally flip a random edge outside the subset for diversity
        if rng.random() < 0.2:
            a, b = rng.sample(range(n), 2)
            if adj[a] & (1 << b):
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
            else:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)

    # Final check
    if _is_valid_bitmask(adj, n):
        return _to_result_bitmask(adj, n)
    return None


def _to_result_bitmask(adj, n):
    """Convert bitmask adjacency to result format."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
