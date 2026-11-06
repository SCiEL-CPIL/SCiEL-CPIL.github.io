import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def check_violation(adj, n):
        """
        Returns (True, mask, k) if a regular induced subgraph of size >= 6 is found.
        Otherwise (False, None, None).
        """
        # Check sizes 6 to n
        # Optimization: If we find a violation of size 6, we can stop?
        # No, we need to ensure NO violation. So we must check all.
        # But for repair, we just need ONE violation.
        
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                
                # Check if regular
                # Get degrees
                # Using bit_count is faster than bin().count('1')
                degs = [ (adj[v] & mask).bit_count() for v in combo ]
                if len(set(degs)) == 1:
                    return True, mask, k
        return False, None, None

    def full_verify(adj, n):
        """
        Returns True if valid (no regular induced subgraph >= 6).
        """
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                degs = [ (adj[v] & mask).bit_count() for v in combo ]
                if len(set(degs)) == 1:
                    return False
        return True

    # Try n=21 first, then 20
    for n in [21, 20]:
        if time.time() > deadline - 10:
            continue

        for attempt in range(50):
            if time.time() > deadline - 5:
                break

            rng = random.Random(n * 1000 + attempt)
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            # Local search
            # We want to eliminate violations.
            # Loop until no violation found or time runs out
            while time.time() < deadline - 5:
                found, mask, k = check_violation(adj, n)
                if not found:
                    # Valid!
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                
                # Repair: flip a random edge in the violating set
                # Extract vertices from mask
                verts = []
                m = mask
                while m:
                    v = (m & -m).bit_length() - 1
                    m &= m - 1
                    verts.append(v)
                
                # Pick two random vertices from the set
                u, v = rng.sample(verts, 2)
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

    # Fallback: return a small known valid graph or empty
    # n=5 is always valid since min size is 6
    return {'n': 5, 'edges': []}
