import random
import time

def construct():
    deadline = time.time() + 110

    # Try n from 40 down to 28
    for n in range(40, 27, -1):
        if time.time() > deadline:
            break

        max_restarts = 200
        for r in range(max_restarts):
            if time.time() > deadline:
                break

            # Random bipartite initialization
            adj = [0] * n
            partition = [random.random() < 0.5 for _ in range(n)]

            # Add edges between partitions with moderate probability
            p = 0.3
            for u in range(n):
                if partition[u]:
                    for v in range(u + 1, n):
                        if not partition[v]:
                            if random.random() < p:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)

            # Local search
            for step in range(500):
                if time.time() > deadline:
                    break

                # Fast alpha check
                if not _has_independent_set(adj, n, 10):
                    edge_list = []
                    for u in range(n):
                        mask = adj[u]
                        while mask:
                            lsb = mask & -mask
                            v = lsb.bit_length() - 1
                            if u < v:
                                edge_list.append([u, v])
                            mask ^= lsb
                    return {'n': n, 'edges': edge_list}

                # Find an independent set of size 10 to target
                iso = _find_independent_set(adj, n, 10)
                if iso is None:
                    break

                iso_verts = [i for i in range(n) if (iso >> i) & 1]

                # Find safe edges: (u,v) non-adjacent with no common neighbors
                candidates = []
                # Prioritize edges within iso_verts
                for i in range(len(iso_verts)):
                    u = iso_verts[i]
                    for j in range(i + 1, len(iso_verts)):
                        v = iso_verts[j]
                        if not (adj[u] >> v) & 1:
                            if (adj[u] & adj[v]) == 0:
                                candidates.append((u, v))

                if not candidates:
                    for u in range(n):
                        for v in range(u + 1, n):
                            if not (adj[u] >> v) & 1:
                                if (adj[u] & adj[v]) == 0:
                                    candidates.append((u, v))

                if not candidates:
                    break

                u, v = random.choice(candidates)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    # Fallback: small random search for n=26-25
    for n in range(26, 24, -1):
        for r in range(100):
            if time.time() > deadline:
                break
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.1:
                        if (adj[u] & adj[v]) == 0:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
            if not _has_independent_set(adj, n, 10):
                edge_list = []
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            edge_list.append([u, v])
                        mask ^= lsb
                return {'n': n, 'edges': edge_list}

    # Absolute fallback
    return {'n': 1, 'edges': []}

def _has_independent_set(adj, n, k):
    return _find_independent_set(adj, n, k) is not None

def _find_independent_set(adj, n, k):
    """
    Returns bitmask of independent set of size k, or None.
    Uses branch and bound with bitsets.
    """
    full_mask = (1 << n) - 1

    def backtrack(curr, cands):
        if bin(curr).count('1') == k:
            return curr
        if bin(cands).count('1') < k - bin(curr).count('1'):
            return None

        lsb = cands & -cands
        v = lsb.bit_length() - 1

        # Include v
        new_cands = (cands & ~lsb) & ~adj[v]
        res = backtrack(curr | lsb, new_cands)
        if res is not None:
            return res

        # Exclude v
        res = backtrack(curr, cands & ~lsb)
        if res is not None:
            return res

        return None

    return backtrack(0, full_mask)
