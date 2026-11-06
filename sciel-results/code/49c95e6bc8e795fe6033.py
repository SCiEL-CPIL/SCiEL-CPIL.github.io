"""Fresh design: binary search over edge prefix K for a Hamilton cycle in a 24-vertex
complete graph under a pinned edge order, then extract a cycle with pruned DFS."""
import time
import sys

sys.setrecursionlimit(100000)

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
M = len(EDGE_ORDER)


def _build_prefix(K):
    """Adjacency bitmasks using only the first K edges (0-based count)."""
    adj = [0] * N
    for i in range(K):
        a, b = EDGE_ORDER[i]
        adj[a] |= 1 << b
        adj[b] |= 1 << a
    return adj


def _dfs_has_cycle(adj):
    """Return a permutation (cycle) if the prefix graph has a Hamilton cycle, else None.
    Uses bitmask DP over subsets with a fixed start vertex (0) and a fixed end vertex
    (the earliest-arriving neighbor of 0), plus a cutset connectivity prune."""
    # Choose end vertex as the neighbor of 0 with the smallest edge index.
    end = None
    best_idx = M + 1
    for i, (a, b) in enumerate(EDGE_ORDER):
        if a == 0:
            v = b
        elif b == 0:
            v = a
        else:
            continue
        if i < best_idx:
            best_idx = i
            end = v
    if end is None:
        return None

    start = 0
    start_bit = 1 << start
    end_bit = 1 << end
    full_mask = (1 << N) - 1
    all_except_end = full_mask & ~end_bit

    # Precompute: for each vertex, mask of neighbors that are NOT end.
    nonend_neighbors = [adj[v] & ~end_bit for v in range(N)]

    # dp[mask][v]: reachable (start -> ... -> v) using exactly vertices in mask.
    # Use a list of dicts: mask -> set of vertices.
    dp = [None] * (1 << N)
    dp[start_bit] = {start}

    # Iterate masks by increasing popcount.
    # For efficiency, iterate all masks and propagate.
    # We only care about masks that contain start.
    for mask in range(1 << N):
        if not (mask & start_bit):
            continue
        if dp[mask] is None:
            continue
        verts = dp[mask]
        # If we already reached end with all vertices, we're done (but we keep end out until last).
        # Propagate to next vertices.
        # The number of remaining vertices:
        rem = all_except_end & ~mask
        if not rem:
            # mask == start_bit | (all non-end) -> check if any v in verts connects to end
            for v in verts:
                if adj[v] & end_bit:
                    # Build cycle: need the actual path. We'll reconstruct separately.
                    return (mask, v)
            continue
        # Prune: the unvisited non-end vertices must be reachable from the current
        # frontier (verts) to avoid disconnected components.
        # Compute union of neighbors of verts (excluding end), restricted to rem.
        reach = 0
        for v in verts:
            reach |= nonend_neighbors[v]
        reach &= rem
        if reach == 0:
            continue
        # Actually, we need every unvisited vertex to be reachable eventually; a stronger
        # prune: rem must be subset of (reach expanded). We do a BFS reachability from
        # frontier within the graph induced by (mask|rem) non-end vertices.
        # Quick prune: if rem has a component not touching reach, fail.
        # Compute connected components of rem under nonend_neighbors, and check each
        # component intersects reach.
        unvisited = rem
        while unvisited:
            # take lowest bit
            lsb = unvisited & -unvisited
            # BFS from lsb within rem
            comp = lsb
            frontier = lsb
            unvisited &= ~lsb
            while frontier:
                fbit = frontier & -frontier
                frontier &= ~fbit
                fidx = fbit.bit_length() - 1
                nb = nonend_neighbors[fidx] & rem & ~comp
                if nb:
                    comp |= nb
                    unvisited &= ~nb
                    frontier |= nb
            # comp is a connected component of rem; it must intersect reach.
            if not (comp & reach):
                break  # disconnected component -> prune
            if not unvisited:
                break
        else:
            pass
        if not unvisited:
            # all components reachable; proceed
            pass
        else:
            # a component not reachable from frontier -> prune this mask
            continue

        # Propagate
        for v in verts:
            nb = nonend_neighbors[v] & rem
            while nb:
                lsb = nb & -nb
                nb &= ~lsb
                nmask = mask | lsb
                if dp[nmask] is None:
                    dp[nmask] = set()
                dp[nmask].add(v)
                # We also need to record that we came from v to the new vertex; for
                # reconstruction we store parent pointers.
    return None


def _reconstruct(adj, end):
    """Given we know a Hamilton cycle exists with start=0 and end=end, reconstruct it
    using DFS with memoization on (mask, v)."""
    start = 0
    start_bit = 1 << start
    end_bit = 1 << end
    full_mask = (1 << N) - 1
    all_except_end = full_mask & ~end_bit
    nonend_neighbors = [adj[v] & ~end_bit for v in range(N)]

    # Memo: (mask, v) -> next vertex or None.
    memo = {}

    def solve(mask, v):
        # returns a list of vertices from v onward (excluding v) to complete the path to end
        key = (mask, v)
        if key in memo:
            return memo[key]
        rem = all_except_end & ~mask
        if not rem:
            # Must go to end.
            if adj[v] & end_bit:
                memo[key] = [end]
                return [end]
            memo[key] = None
            return None
        # Try neighbors in rem, ordered by degree in rem (ascending) for speed.
        nb = nonend_neighbors[v] & rem
        cands = []
        while nb:
            lsb = nb & -nb
            nb &= ~lsb
            u = lsb.bit_length() - 1
            deg = (nonend_neighbors[u] & rem).bit_count()
            cands.append((deg, u))
        cands.sort()
        for deg, u in cands:
            sub = solve(mask | (1 << u), u)
            if sub is not None:
                memo[key] = [u] + sub
                return [u] + sub
        memo[key] = None
        return None

    res = solve(start_bit, start)
    if res is None:
        return None
    return [start] + res


def _has_cycle(adj):
    """Return a cycle permutation or None."""
    # Determine end vertex.
    end = None
    best_idx = M + 1
    for i, (a, b) in enumerate(EDGE_ORDER):
        if a == 0:
            v = b
        elif b == 0:
            v = a
        else:
            continue
        if i < best_idx:
            best_idx = i
            end = v
    if end is None:
        return None
    cyc = _reconstruct(adj, end)
    return cyc


def _score_of(cycle):
    """Score = largest one-based arrival index among the cycle's edges."""
    # Build edge -> index map.
    idx = {}
    for i, (a, b) in enumerate(EDGE_ORDER):
        idx[(min(a, b), max(a, b))] = i + 1
    n = len(cycle)
    worst = 0
    for i in range(n):
        a = cycle[i]
        b = cycle[(i + 1) % n]
        key = (min(a, b), max(a, b))
        val = idx.get(key, M + 1)
        if val > worst:
            worst = val
    return worst


def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill

    # Binary search for the smallest K with a Hamilton cycle.
    lo, hi = N, M  # lo = known infeasible-ish lower bound (need at least N edges)
    # First, ensure hi is feasible.
    best_cycle = None
    adj_hi = _build_prefix(M)
    cyc = _has_cycle(adj_hi)
    if cyc is None:
        # Should not happen for a complete graph, but fallback.
        return list(range(N))
    best_cycle = cyc
    hi = M

    # Binary search
    while lo < hi:
        if time.time() > deadline - 10.0:
            break
        mid = (lo + hi) // 2
        if mid < N:
            lo = mid + 1
            continue
        adj_mid = _build_prefix(mid)
        cyc = _has_cycle(adj_mid)
        if cyc is not None:
            hi = mid
            best_cycle = cyc
        else:
            lo = mid + 1

    # Final check at hi.
    if time.time() < deadline - 5.0:
        adj_final = _build_prefix(hi)
        cyc = _has_cycle(adj_final)
        if cyc is not None:
            best_cycle = cyc

    return best_cycle if best_cycle is not None else list(range(N))
# EVOLVE-BLOCK-END
