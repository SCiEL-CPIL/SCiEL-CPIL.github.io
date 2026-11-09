"""Search for a large graph with no regular induced subgraph on 6+ vertices."""
import random
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    rng = random.Random(42)

    # Try greedy sequential construction for various target sizes
    best = None
    best_n = 0

    for target_n in [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 20:
            break
        result = _greedy_construct(target_n, deadline - 15, rng)
        if result is not None and result['n'] > best_n:
            best = result
            best_n = result['n']

    # If greedy didn't find anything good, try local search
    if best is None or best_n < 10:
        fallback = _make_fallback(rng)
        if fallback['n'] > best_n:
            best = fallback
            best_n = fallback['n']

    # Try local search on the best found graph to potentially improve
    if best is not None and best_n < 20 and time.time() < deadline - 30:
        improved = _local_search(best_n + 1, deadline - 10, rng)
        if improved is not None and improved['n'] > best_n:
            best = improved

    return best


def _make_fallback(rng):
    """Generate a verified valid graph on 10 vertices."""
    n = 10
    for _ in range(5000):
        adj = [[0]*n for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i][j] = adj[j][i] = 1
                    edges.append([i, j])
        if _is_valid_fast(adj, n):
            return {'n': n, 'edges': edges}
    return {'n': 5, 'edges': []}


def _is_valid_fast(adj, n):
    """Check if no subset of size >= 6 induces a regular graph using bitmasks."""
    # Precompute adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                adj_mask[i] |= (1 << j)

    # Check all subsets of size 6 first (most likely to be violations)
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_subset_fast(adj_mask, subset):
                return False
    return True


def _is_regular_subset_fast(adj_mask, subset):
    """Check if induced subgraph on subset is regular using bitmask operations."""
    s = len(subset)
    if s < 2:
        return True

    # Build mask for the subset
    subset_mask = 0
    for v in subset:
        subset_mask |= (1 << v)

    # Count degrees within the induced subgraph
    degrees = []
    for v in subset:
        # Count neighbors of v that are also in the subset
        d = bin(adj_mask[v] & subset_mask).count('1')
        # Subtract 1 if v is connected to itself (it shouldn't be, but just in case)
        if adj_mask[v] & (1 << v):
            d -= 1
        degrees.append(d)

    # Check if all degrees are equal
    d0 = degrees[0]
    for i in range(1, s):
        if degrees[i] != d0:
            return False
    return True


def _find_violation_fast(adj_mask, n):
    """Find a subset of size >= 6 that induces a regular graph. Returns subset or None."""
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_subset_fast(adj_mask, subset):
                return subset
    return None


def _greedy_construct(target_n, deadline, rng):
    """Greedy sequential vertex addition: add vertices one by one,
    choosing edges to minimize new regular induced subgraphs of size 6+."""
    n = target_n
    start_time = time.time()

    # Start with empty graph
    adj = [[0]*n for _ in range(n)]
    edges = []

    # Build adjacency mask
    adj_mask = [0] * n

    # For each new vertex, try all possible edge configurations to existing vertices
    # and pick the one that minimizes new violations
    for new_v in range(n):
        if time.time() > deadline:
            break

        existing = list(range(new_v))
        if not existing:
            continue

        # Try different edge patterns to existing vertices
        best_pattern = None
        best_violation_count = float('inf')
        best_adj_mask = None

        # Number of existing vertices
        m = len(existing)

        # If m < 5, any pattern is fine (can't form size-6 subgraph yet)
        if m < 5:
            # Randomly connect
            for u in existing:
                if rng.random() < 0.5:
                    adj[new_v][u] = adj[u][new_v] = 1
                    adj_mask[new_v] |= (1 << u)
                    adj_mask[u] |= (1 << new_v)
                    edges.append([min(new_v, u), max(new_v, u)])
            continue

        # Try a few random patterns plus some structured ones
        patterns_to_try = []

        # All zeros
        patterns_to_try.append(0)
        # All ones
        all_ones = (1 << m) - 1
        patterns_to_try.append(all_ones)

        # Random patterns
        for _ in range(min(50, 2**m - 2)):
            pat = 0
            for i in range(m):
                if rng.random() < 0.5:
                    pat |= (1 << i)
            if pat not in patterns_to_try:
                patterns_to_try.append(pat)

        # Evaluate each pattern
        for pat in patterns_to_try:
            if time.time() > deadline - 5:
                break

            # Create a temporary state
            temp_adj_mask = adj_mask[:]
            # Apply pattern
            for i, u in enumerate(existing):
                if pat & (1 << i):
                    temp_adj_mask[new_v] |= (1 << u)
                    temp_adj_mask[u] |= (1 << new_v)

            # Count new violations: check subsets of size 6 that include new_v
            # and subsets of size 6+ that don't include new_v (but these were already checked)
            # Actually, we need to check all subsets of size 6+ in the new graph
            # But for efficiency, focus on subsets that include new_v

            violation_count = 0

            # Check subsets of size 6 that include new_v
            # We need to choose 5 from existing vertices
            if m >= 5:
                for combo in itertools.combinations(range(m), 5):
                    subset = tuple(existing[i] for i in combo) + (new_v,)
                    if _is_regular_subset_fast(temp_adj_mask, subset):
                        violation_count += 1
                        if violation_count > best_violation_count:
                            break

            # Also check larger subsets if time permits
            if violation_count < best_violation_count and m >= 6:
                # Check subsets of size 7 that include new_v
                for combo in itertools.combinations(range(m), 6):
                    subset = tuple(existing[i] for i in combo) + (new_v,)
                    if _is_regular_subset_fast(temp_adj_mask, subset):
                        violation_count += 1
                        if violation_count > best_violation_count:
                            break

            if violation_count < best_violation_count:
                best_violation_count = violation_count
                best_pattern = pat
                best_adj_mask = temp_adj_mask

            if best_violation_count == 0:
                break

        # Apply the best pattern
        if best_pattern is not None:
            for i, u in enumerate(existing):
                if best_pattern & (1 << i):
                    adj[new_v][u] = adj[u][new_v] = 1
                    adj_mask[new_v] |= (1 << u)
                    adj_mask[u] |= (1 << new_v)
                    edges.append([min(new_v, u), max(new_v, u)])
        else:
            # Default: no edges
            pass

    # Final verification
    if _is_valid_fast(adj, n):
        return {'n': n, 'edges': edges}

    return None


def _local_search(target_n, deadline, rng):
    """Local search to find a valid graph on target_n vertices."""
    n = target_n

    # Initialize with random graph
    adj = [[0]*n for _ in range(n)]
    adj_mask = [0] * n
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < 0.5:
                adj[i][j] = adj[j][i] = 1
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                edges.append([i, j])

    # Check if already valid
    if _is_valid_fast(adj, n):
        return {'n': n, 'edges': edges}

    # Local search: find violations and flip edges to break them
    for iteration in range(3000):
        if time.time() > deadline:
            break

        # Find a violation
        viol = _find_violation_fast(adj_mask, n)
        if viol is None:
            return {'n': n, 'edges': edges}

        subset = viol
        size = len(subset)

        # Pick a random edge within the subset to flip
        i, j = rng.sample(range(size), 2)
        u, v = subset[i], subset[j]

        # Flip the edge
        old_val = adj[u][v]
        adj[u][v] = adj[v][u] = 1 - old_val
        if old_val:
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            # Remove from edges list
            for k in range(len(edges)-1, -1, -1):
                if (edges[k][0] == u and edges[k][1] == v) or (edges[k][0] == v and edges[k][1] == u):
                    edges.pop(k)
        else:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edges.append([min(u, v), max(u, v)])

        # Optionally, also try flipping a random edge outside the subset
        if rng.random() < 0.3:
            a, b = rng.sample(range(n), 2)
            old_val2 = adj[a][b]
            adj[a][b] = adj[b][a] = 1 - old_val2
            if old_val2:
                adj_mask[a] &= ~(1 << b)
                adj_mask[b] &= ~(1 << a)
                for k in range(len(edges)-1, -1, -1):
                    if (edges[k][0] == a and edges[k][1] == b) or (edges[k][0] == b and edges[k][1] == a):
                        edges.pop(k)
            else:
                adj_mask[a] |= (1 << b)
                adj_mask[b] |= (1 << a)
                edges.append([min(a, b), max(a, b)])

    # Final check
    if _is_valid_fast(adj, n):
        return {'n': n, 'edges': edges}
    return None
# EVOLVE-BLOCK-END
