"""Rewritten: exact cochromatic partition via chromatic-number lower bound + branch-and-bound."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _fallback():
    return [[v] for v in range(P['n'])]

def construct():
    n = P['n']
    edges = P['edges']

    # Build adjacency bitmask and neighbor set
    adj = [0] * n
    nbrs = [set() for _ in range(n)]
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
        nbrs[a].add(b)
        nbrs[b].add(a)

    t0 = time.time()
    DEADLINE = t0 + 48.0  # safety margin well under 90s

    # ---------- Exact chromatic number via bitmask subset DP ----------
    # chrom[mask] = chromatic number of the induced subgraph on 'mask'
    # This equals the number of color classes needed; a cochromatic partition
    # must use at least this many parts (each part is a clique or independent
    # set, and both are color classes in some proper coloring).
    full = (1 << n) - 1
    chrom = [0] * (1 << n)
    for mask in range(1, 1 << n):
        # pick an arbitrary vertex v in mask (lowest set bit)
        v = (mask & -mask).bit_length() - 1
        # Try all subsets S of mask that contain v and are independent sets
        # (i.e., no edge inside S). The chromatic number of mask is
        #   1 + min over such S of chrom[mask ^ S]
        # Optimization: iterate over submasks of the "non-neighborhood" part.
        # A set S containing v is independent iff S \ {v} is a subset of
        # (mask ^ (1<<v)) \ N(v) AND pairwise independent within itself.
        # Simpler: enumerate submasks of (mask & ~adj[v] & ~(1<<v)) that are
        # independent, then add v.
        # For n=14 this is fast enough.
        rest = mask & ~(1 << v)
        # Independent subsets of 'rest' that don't include neighbors of v
        # and are themselves independent. We'll iterate all submasks of
        # (rest & ~adj[v]) and test independence.
        cand_base = rest & ~adj[v]
        best = n + 1
        # Enumerate all submasks of cand_base
        sub = cand_base
        while True:
            # S = sub | (1<<v); check S is independent (v has no neighbor in sub by construction,
            # but sub itself must be independent)
            # Quick independence test for sub: for each pair... use bit trick:
            # sub is independent iff for all u in sub, (adj[u] & sub) == 0
            # We can check: (union of adj over u in sub) & sub == 0
            # Faster: precompute? For n=14, just do a loop.
            indep = True
            s = sub
            while s:
                b = s & -s
                u = b.bit_length() - 1
                if adj[u] & sub:
                    indep = False
                    break
                s ^= b
            if indep:
                val = 1 + chrom[mask ^ (sub | (1 << v))]
                if val < best:
                    best = val
            if sub == 0:
                break
            sub = (sub - 1) & cand_base
        chrom[mask] = best

    chi = chrom[full]

    # ---------- Branch-and-bound search for cochromatic partition ----------
    # Try to assign vertices 0..n-1 into at most K parts, each part being
    # all-clique or all-independent. Track per part:
    #   part_vertices: list of vertex indices
    #   part_type: 0 = undetermined, 1 = clique, 2 = independent
    #   For clique: every new vertex must be adjacent to all existing.
    #   For independent: every new vertex must be non-adjacent to all existing.
    #   Undetermined (size 1): can go either way.

    def try_k(K):
        """Try to find a partition into <= K parts. Returns list of lists or None."""
        nonlocal t0, DEADLINE
        parts_verts = []   # list of lists
        parts_type = []    # 0=undetermined, 1=clique, 2=independent
        parts_nbr_mask = []  # bitmask of neighbors within the part (for quick check)
        # Actually, for clique check: new vertex v is OK iff (adj[v] & part_mask) == part_mask
        # For independent check: new vertex v is OK iff (adj[v] & part_mask) == 0
        # Store part_mask (bitmask of vertices in the part) for fast checks.
        parts_mask = []

        # Order vertices by descending degree for better pruning
        order = sorted(range(n), key=lambda v: -bin(adj[v]).count('1'))
        # Map position in order to actual vertex
        # We'll assign in order[0], order[1], ...

        best_sol = [None]

        def backtrack(pos):
            if time.time() > DEADLINE:
                return
            if pos == n:
                if best_sol[0] is None:
                    best_sol[0] = [list(p) for p in parts_verts]
                return

            v = order[pos]
            bm = 1 << v

            # Try placing v in each existing part
            for i in range(len(parts_verts)):
                ptype = parts_type[i]
                pmask = parts_mask[i]
                if ptype == 1:  # clique
                    if (adj[v] & pmask) == pmask:
                        parts_verts[i].append(v)
                        parts_mask[i] = pmask | bm
                        backtrack(pos + 1)
                        parts_verts[i].pop()
                        parts_mask[i] = pmask
                elif ptype == 2:  # independent
                    if (adj[v] & pmask) == 0:
                        parts_verts[i].append(v)
                        parts_mask[i] = pmask | bm
                        backtrack(pos + 1)
                        parts_verts[i].pop()
                        parts_mask[i] = pmask
                else:  # undetermined (size 1)
                    u = parts_verts[i][0]
                    # Can become clique if adjacent, independent if not
                    if adj[u] & bm:
                        # try clique
                        parts_verts[i].append(v)
                        parts_type[i] = 1
                        parts_mask[i] = pmask | bm
                        backtrack(pos + 1)
                        parts_verts[i].pop()
                        parts_type[i] = 0
                        parts_mask[i] = pmask
                    if not (adj[u] & bm):
                        # try independent
                        parts_verts[i].append(v)
                        parts_type[i] = 2
                        parts_mask[i] = pmask | bm
                        backtrack(pos + 1)
                        parts_verts[i].pop()
                        parts_type[i] = 0
                        parts_mask[i] = pmask

            # Try opening a new part (if we have room)
            if len(parts_verts) < K:
                parts_verts.append([v])
                parts_type.append(0)
                parts_mask.append(bm)
                backtrack(pos + 1)
                parts_verts.pop()
                parts_type.pop()
                parts_mask.pop()

        backtrack(0)
        return best_sol[0]

    # Search for K = chi, chi+1, chi+2 (bounded by n)
    result = None
    for K in range(chi, min(chi + 3, n + 1)):
        if time.time() > DEADLINE:
            break
        sol = try_k(K)
        if sol is not None:
            result = sol
            break

    if result is None:
        # Fallback: trivial partition (each vertex its own part) is always valid
        result = _fallback()

    # Ensure return type: list of lists of Python ints
    return [list(map(int, part)) for part in result]
