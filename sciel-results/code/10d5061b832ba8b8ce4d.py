import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    def get_vertices(mask):
        """Extract vertex indices from bitmask."""
        verts = []
        m = mask
        while m:
            v = (m & -m).bit_length() - 1
            m &= m - 1
            verts.append(v)
        return verts

    def is_regular_fast(adj, mask):
        """Check if induced subgraph on mask is regular using fast bit counting."""
        verts = get_vertices(mask)
        if len(verts) < 6:
            return False
        # Compute degrees within the induced subgraph
        first_deg = bin(adj[verts[0]] & mask).count('1')
        for v in verts[1:]:
            if bin(adj[v] & mask).count('1') != first_deg:
                return False
        return True

    def full_verify(adj, n):
        """Verify no subset of size >= 6 induces a regular subgraph."""
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if is_regular_fast(adj, mask):
                    return False
        return True

    def find_violation(adj, n, k=6):
        """Find one regular induced subgraph of size k, or None."""
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            if is_regular_fast(adj, mask):
                return combo
        return None

    def edges_from_adj(adj, n):
        """Convert adjacency list to edge list."""
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Try n=20 first, then smaller
    for n in [20, 19, 18, 17, 16, 15, 12, 10]:
        if time.time() > deadline - 15:
            break

        rng = random.Random(n * 7919 + 13)

        # Strategy 1: Random init + local search with fast violation detection
        for attempt in range(50):
            if time.time() > deadline - 10:
                break

            # Initialize with random graph (slightly biased to reduce regularity)
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.45:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            # Local search
            max_steps = 3000
            steps = 0
            stuck = 0
            while steps < max_steps and time.time() < deadline - 5:
                steps += 1

                # Find a violation of size 6 (most common)
                combo = find_violation(adj, n, 6)
                if combo is None:
                    # No size-6 violation; check larger sizes
                    found_larger = False
                    for k in [7, 8, 9, 10]:
                        if k > n:
                            break
                        combo = find_violation(adj, n, k)
                        if combo is not None:
                            found_larger = True
                            break
                    if not found_larger:
                        # Valid!
                        if full_verify(adj, n):
                            return {'n': n, 'edges': edges_from_adj(adj, n)}
                        break

                # Flip a random edge within the violating set
                u, v = rng.sample(combo, 2)
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

                # Occasionally flip an edge outside the set to escape local minima
                if steps % 10 == 0:
                    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
                    u2, v2 = rng.choice(all_edges)
                    if adj[u2] & (1 << v2):
                        adj[u2] &= ~(1 << v2)
                        adj[v2] &= ~(1 << u2)
                    else:
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)

        # Strategy 2: Start from sparse graph and add edges carefully
        if time.time() > deadline - 10:
            break

        for attempt in range(10):
            if time.time() > deadline - 5:
                break

            adj = [0] * n
            edge_order = [(i, j) for i in range(n) for j in range(i+1, n)]
            rng.shuffle(edge_order)

            for (u, v) in edge_order:
                if time.time() > deadline - 5:
                    break
                # Tentatively add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                # Check if this creates any regular induced subgraph of size 6
                # that includes both u and v
                violation = False
                for combo in itertools.combinations(range(n), 6):
                    if u in combo and v in combo:
                        mask = 0
                        for x in combo:
                            mask |= (1 << x)
                        if is_regular_fast(adj, mask):
                            violation = True
                            break
                if violation:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

            if full_verify(adj, n):
                return {'n': n, 'edges': edges_from_adj(adj, n)}

    # Fallback: verified small graph
    n = 10
    rng = random.Random(42)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    if full_verify(adj, n):
        return {'n': n, 'edges': edges_from_adj(adj, n)}

    return {'n': 5, 'edges': []}
