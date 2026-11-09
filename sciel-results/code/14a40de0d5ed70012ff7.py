"""Search for a large graph with no regular induced subgraph on 6+ vertices."""
import random
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    rng = random.Random(12345)

    # Start with a known-good small graph as fallback
    fallback = _make_small_fallback(rng)
    best = fallback
    best_n = fallback['n']

    # Try incremental construction: build up from small valid graphs
    for target_n in [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 15:
            break
        result = _incremental_construct(target_n, deadline - 5, rng)
        if result is not None and result['n'] > best_n:
            best = result
            best_n = result['n']
        if best_n >= 20:
            break

    # If incremental didn't find large graphs, try local search from random init
    if best_n < 20:
        for target_n in [20, 19, 18, 17, 16, 15]:
            if time.time() > deadline - 10:
                break
            result = _local_search(target_n, deadline - 5, rng)
            if result is not None and result['n'] > best_n:
                best = result
                best_n = result['n']
            if best_n >= 20:
                break

    return best


def _make_small_fallback(rng):
    """Generate a verified valid graph on 10 vertices."""
    n = 10
    for _ in range(20000):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if _is_valid(adj, n):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    return {'n': 5, 'edges': []}


def _is_valid(adj, n):
    """Check if no subset of size >= 6 induces a regular graph."""
    # Check all subsets of size 6 first (most common and most numerous)
    for subset in itertools.combinations(range(n), 6):
        if _is_regular_subset(adj, subset):
            return False
    # Check subsets of size 7+
    for size in range(7, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_subset(adj, subset):
                return False
    return True


def _is_regular_subset(adj, subset):
    """Check if induced subgraph on subset is regular."""
    s = len(subset)
    mask = 0
    for v in subset:
        mask |= (1 << v)
    d0 = (adj[subset[0]] & mask).bit_count()
    for v in subset[1:]:
        if (adj[v] & mask).bit_count() != d0:
            return False
    return True


def _count_violations_6(adj, n):
    """Count the number of 6-vertex subsets that induce regular graphs."""
    count = 0
    for subset in itertools.combinations(range(n), 6):
        if _is_regular_subset(adj, subset):
            count += 1
    return count


def _find_violation_6(adj, n):
    """Find one 6-vertex subset that induces a regular graph."""
    for subset in itertools.combinations(range(n), 6):
        if _is_regular_subset(adj, subset):
            return subset
    return None


def _find_any_violation(adj, n):
    """Find any subset of size >= 6 that induces a regular graph."""
    # Check size 6 first
    for subset in itertools.combinations(range(n), 6):
        if _is_regular_subset(adj, subset):
            return subset
    # Check larger sizes
    for size in range(7, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_subset(adj, subset):
                return subset
    return None


def _incremental_construct(target_n, deadline, rng):
    """Build a graph incrementally, adding vertices one at a time."""
    n = target_n
    if n < 6:
        # Trivially valid
        return {'n': n, 'edges': []}

    # Start with a valid small graph
    # Try to find a valid graph on min(n, 12) vertices first
    base_n = min(n, 12)
    base_adj = None
    for _ in range(5000):
        adj = [0] * base_n
        for i in range(base_n):
            for j in range(i + 1, base_n):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if _is_valid(adj, base_n):
            base_adj = adj
            break

    if base_adj is None:
        return None

    # Now add vertices one at a time
    adj = base_adj[:]
    current_n = base_n

    while current_n < n:
        if time.time() > deadline:
            break

        new_v = current_n
        adj.append(0)

        # Try all possible edge patterns to the new vertex
        # There are 2^current_n possible patterns
        best_pattern = None
        best_score = float('inf')

        # Use a heuristic: try patterns that minimize 6-vertex violations
        # Sample some patterns rather than trying all
        num_tries = min(2 ** current_n, 200)
        for _ in range(num_tries):
            # Random pattern
            pattern = 0
            for i in range(current_n):
                if rng.random() < 0.5:
                    pattern |= (1 << i)

            # Apply pattern
            adj[new_v] = pattern
            for i in range(current_n):
                if pattern & (1 << i):
                    adj[i] |= (1 << new_v)
                else:
                    adj[i] &= ~(1 << new_v)

            # Score: number of 6-vertex violations involving the new vertex
            # Plus a small penalty for total violations
            score = 0
            # Check 6-subsets that include new_v
            for subset in itertools.combinations(range(current_n), 5):
                full_subset = tuple(sorted(subset + (new_v,)))
                if _is_regular_subset(adj, full_subset):
                    score += 1

            if score < best_score:
                best_score = score
                best_pattern = pattern

                if score == 0:
                    break

        # Apply the best pattern
        adj[new_v] = best_pattern
        for i in range(current_n):
            if best_pattern & (1 << i):
                adj[i] |= (1 << new_v)
            else:
                adj[i] &= ~(1 << new_v)

        current_n += 1

        # Check if the graph is still valid
        if not _is_valid(adj, current_n):
            # Try to fix with local search
            fixed = _local_search(current_n, deadline, rng)
            if fixed is not None:
                # Convert back to adj
                adj = [0] * current_n
                for u, v in fixed['edges']:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            else:
                # Revert
                adj = adj[:current_n - 1]
                break

    if _is_valid(adj, current_n):
        return _to_result(adj, current_n)
    return None


def _local_search(target_n, deadline, rng):
    """Local search to find a valid graph on target_n vertices."""
    n = target_n

    for restart in range(5):
        if time.time() > deadline:
            break

        # Initialize with random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

        # Check if already valid
        if _is_valid(adj, n):
            return _to_result(adj, n)

        # Local search: iteratively flip edges to reduce violations
        for iteration in range(500):
            if time.time() > deadline:
                break

            # Find a violation
            violation = _find_any_violation(adj, n)
            if violation is None:
                return _to_result(adj, n)

            # Pick a random pair within the violation to flip
            i, j = rng.sample(range(len(violation)), 2)
            u, v = violation[i], violation[j]

            # Flip edge (u, v)
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

            # Occasionally flip a random edge outside the violation
            if rng.random() < 0.3:
                a, b = rng.sample(range(n), 2)
                if adj[a] & (1 << b):
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                else:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)

        # Final check
        if _is_valid(adj, n):
            return _to_result(adj, n)

    return None


def _to_result(adj, n):
    """Convert adjacency bitmask to result format."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
