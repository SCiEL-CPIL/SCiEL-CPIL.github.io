"""Search for a large graph with no regular induced subgraph on 6+ vertices."""
import random
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s kill

    rng = random.Random(12345)

    # Start with a known-good small graph as fallback
    fallback = _make_small_fallback(rng)
    best = fallback
    best_n = fallback['n']

    # Try to find larger graphs
    for target_n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 8:
            break
        result = _search_graph(target_n, deadline - 3, rng)
        if result is not None and result['n'] > best_n:
            best = result
            best_n = result['n']

    return best


def _make_small_fallback(rng):
    """Generate a verified valid graph on 10 vertices."""
    n = 10
    for _ in range(5000):
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if _is_valid_fast(adj, n):
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    # Trivially valid
    return {'n': 5, 'edges': []}


def _is_valid_fast(adj, n):
    """Check if no subset of size >= 6 induces a regular graph.
    adj[i] is a bitmask of neighbors of i."""
    # Check all subsets of size 6 first
    for subset in itertools.combinations(range(n), 6):
        if _is_regular_6(adj, subset):
            return False
    # Check subsets of size 7+
    for size in range(7, n+1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_subset(adj, subset):
                return False
    return True


def _is_regular_6(adj, subset):
    """Fast check for 6-vertex regular induced subgraph."""
    s0, s1, s2, s3, s4, s5 = subset
    mask = (1 << s0) | (1 << s1) | (1 << s2) | (1 << s3) | (1 << s4) | (1 << s5)
    d0 = bin(adj[s0] & mask).count('1')
    d1 = bin(adj[s1] & mask).count('1')
    d2 = bin(adj[s2] & mask).count('1')
    d3 = bin(adj[s3] & mask).count('1')
    d4 = bin(adj[s4] & mask).count('1')
    d5 = bin(adj[s5] & mask).count('1')
    return d0 == d1 == d2 == d3 == d4 == d5


def _is_regular_subset(adj, subset):
    """Check if induced subgraph on subset is regular."""
    s = len(subset)
    mask = 0
    for v in subset:
        mask |= (1 << v)
    d0 = bin(adj[subset[0]] & mask).count('1')
    for v in subset[1:]:
        if bin(adj[v] & mask).count('1') != d0:
            return False
    return True


def _find_violations(adj, n):
    """Find all subsets of size >= 6 that induce regular graphs."""
    violations = []
    for subset in itertools.combinations(range(n), 6):
        if _is_regular_6(adj, subset):
            violations.append(subset)
    for size in range(7, n+1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_subset(adj, subset):
                violations.append(subset)
    return violations


def _search_graph(target_n, deadline, rng):
    """Local search to find a valid graph on target_n vertices."""
    n = target_n
    max_iter = 2000

    for restart in range(3):
        if time.time() > deadline:
            break

        # Initialize with random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

        # Check if already valid
        if _is_valid_fast(adj, n):
            return _to_result(adj, n)

        # Local search
        for iteration in range(max_iter):
            if time.time() > deadline:
                break

            # Find violations
            violations = _find_violations(adj, n)
            if not violations:
                return _to_result(adj, n)

            # Pick a random violation
            subset = rng.choice(violations)
            # Pick a random pair within the subset to flip
            i, j = rng.sample(range(len(subset)), 2)
            u, v = subset[i], subset[j]

            # Flip edge (u, v)
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

            # Occasionally flip a random edge outside the violation
            if rng.random() < 0.2:
                a, b = rng.sample(range(n), 2)
                if adj[a] & (1 << b):
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                else:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)

        # Final check
        if _is_valid_fast(adj, n):
            return _to_result(adj, n)

    return None


def _to_result(adj, n):
    """Convert adjacency bitmask to result format."""
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
