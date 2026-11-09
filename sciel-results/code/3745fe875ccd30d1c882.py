"""Cochromatic partition via iterative deepening with strong pruning."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}


def construct():
    n = P['n']
    edges = P['edges']

    # Build adjacency as bitmask
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    t0 = time.time()
    DEADLINE = t0 + 80.0

    def is_clique(verts_mask):
        """Check if vertices in mask form a clique."""
        m = verts_mask
        while m:
            b = m & -m
            v = b.bit_length() - 1
            if (adj[v] & verts_mask) != (verts_mask ^ b):
                return False
            m ^= b
        return True

    def is_independent(verts_mask):
        """Check if vertices in mask form an independent set."""
        m = verts_mask
        while m:
            b = m & -m
            v = b.bit_length() - 1
            if adj[v] & verts_mask:
                return False
            m ^= b
        return True

    def can_extend_clique(verts_mask, v):
        """Can vertex v be added to clique verts_mask?"""
        return (adj[v] & verts_mask) == verts_mask

    def can_extend_independent(verts_mask, v):
        """Can vertex v be added to independent set verts_mask?"""
        return (adj[v] & verts_mask) == 0

    # Order vertices by degree for better pruning
    order = sorted(range(n), key=lambda v: -bin(adj[v]).count('1'))

    def search(K):
        """Try to partition into exactly K parts (each clique or independent set)."""
        nonlocal DEADLINE
        if time.time() > DEADLINE:
            return None

        # parts[i] = (mask, type) where type: 1=clique, 2=independent, 0=undetermined
        # We use symmetry breaking: only allow new parts to be opened, and
        # for undetermined parts, only try one type first to reduce symmetry.
        
        best = [None]

        def backtrack(pos, parts):
            if time.time() > DEADLINE:
                return
            if pos == n:
                # Convert to list of lists
                result = []
                for mask, ptype in parts:
                    verts = []
                    m = mask
                    while m:
                        b = m & -m
                        verts.append(b.bit_length() - 1)
                        m ^= b
                    result.append(verts)
                best[0] = result
                return

            v = order[pos]
            bm = 1 << v

            # Try adding v to each existing part
            for i in range(len(parts)):
                pmask, ptype = parts[i]
                if ptype == 1:  # clique
                    if can_extend_clique(pmask, v):
                        parts[i] = (pmask | bm, 1)
                        backtrack(pos + 1, parts)
                        parts[i] = (pmask, 1)
                        if best[0] is not None:
                            return
                elif ptype == 2:  # independent
                    if can_extend_independent(pmask, v):
                        parts[i] = (pmask | bm, 2)
                        backtrack(pos + 1, parts)
                        parts[i] = (pmask, 2)
                        if best[0] is not None:
                            return
                else:  # undetermined (size 1)
                    u_mask = pmask  # single vertex
                    u = (u_mask & -u_mask).bit_length() - 1
                    # Try as clique
                    if can_extend_clique(pmask, v):
                        parts[i] = (pmask | bm, 1)
                        backtrack(pos + 1, parts)
                        parts[i] = (pmask, 0)
                        if best[0] is not None:
                            return
                    # Try as independent
                    if can_extend_independent(pmask, v):
                        parts[i] = (pmask | bm, 2)
                        backtrack(pos + 1, parts)
                        parts[i] = (pmask, 0)
                        if best[0] is not None:
                            return

            # Try opening a new part (symmetry breaking: only if we haven't already
            # tried opening a new part at this level with the same structure)
            if len(parts) < K:
                parts.append((bm, 0))
                backtrack(pos + 1, parts)
                parts.pop()
                if best[0] is not None:
                    return

        backtrack(0, [])
        return best[0]

    # Try K = 1, 2, 3, ...
    for K in range(1, n + 1):
        if time.time() > DEADLINE:
            break
        sol = search(K)
        if sol is not None:
            # Ensure Python ints in lists
            return [list(map(int, part)) for part in sol]

    # Fallback: trivial partition
    return [[v] for v in range(n)]
