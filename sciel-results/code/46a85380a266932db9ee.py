import time
import sys

# Precompute edge positions
N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build edge position map: (min(u,v), max(u,v)) -> 1-based position
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1

def get_pos(u, v):
    a, b = min(u, v), max(u, v)
    return edge_pos[(a, b)]

def cycle_score(cycle):
    """Max 1-based arrival index of the cycle's edges (with implicit closing edge)."""
    m = len(cycle)
    best = 0
    for i in range(m):
        u = cycle[i]
        v = cycle[(i + 1) % m]
        p = get_pos(u, v)
        if p > best:
            best = p
    return best

# ---------------------------------------------------------------------------
# Fast feasibility test: does the prefix graph G_T have a 2-factor?
# (A Hamilton cycle implies a 2-factor; we use this as a fast necessary test
#  to binary-search the minimal feasible T.)
# ---------------------------------------------------------------------------
def has_2factor(adj, n):
    """Return True if graph (adjacency list) on n vertices has a 2-factor.
    Uses bipartite perfect matching (Hopcroft-Karp) on the directed
    out-neighborhood of each vertex."""
    # Quick degree check
    for v in range(n):
        if len(adj[v]) < 2:
            return False

    # Build out-neighbor lists (u -> v means u is matched to v as its successor)
    # We need a permutation p where p[v] is a successor of v and p is a bijection,
    # i.e. a perfect matching in the bipartite graph (L=vertices, R=vertices,
    # edge u-L -- v-R iff v in adj[u]).
    from collections import deque

    # Hopcroft-Karp
    matchL = [-1] * n   # matchL[u] = v (u's matched successor)
    matchR = [-1] * n   # matchR[v] = u (v's matched predecessor)
    dist = [-1] * n

    def bfs():
        q = deque()
        found = False
        for u in range(n):
            if matchL[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = matchR[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = matchR[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                matchL[u] = v
                matchR[v] = u
                return True
        dist[u] = -1
        return False

    matching = 0
    while bfs():
        for u in range(n):
            if matchL[u] == -1 and dfs(u):
                matching += 1
    return matching == n

# ---------------------------------------------------------------------------
# Greedy Hamilton cycle construction with 2-opt refinement.
# ---------------------------------------------------------------------------
def greedy_cycle(adj, n, deadline):
    """Build a Hamilton cycle via backtracking with Warnsdorff ordering,
    then refine with 2-opt. Returns cycle list or None."""
    # Sort each vertex's neighbors by (degree, edge position) for pruning
    neigh = []
    for v in range(n):
        lst = list(adj[v])
        lst.sort(key=lambda x: (len(adj[x]), get_pos(v, x)))
        neigh.append(lst)

    path = [0]
    used = [False] * n
    used[0] = True

    sys.setrecursionlimit(100000)

    def bt():
        if time.time() > deadline:
            return None
        if len(path) == n:
            last = path[-1]
            if 0 in adj[last]:
                return path[:]
            return None
        last = path[-1]
        for nxt in neigh[last]:
            if not used[nxt]:
                # Prune: if one vertex left, it must connect to last and 0
                if len(path) == n - 2:
                    remaining = [v for v in range(n) if not used[v]]
                    r = remaining[0]
                    if r not in adj[nxt] or 0 not in adj[r]:
                        continue
                used[nxt] = True
                path.append(nxt)
                res = bt()
                if res is not None:
                    return res
                path.pop()
                used[nxt] = False
        return None

    cycle = bt()
    if cycle is None:
        return None
    return cycle

def two_opt_refine(cycle, n, max_iter=2000, deadline=None):
    """Improve a Hamilton cycle by 2-opt: replace two edges to reduce max edge position.
    Objective: minimize the maximum edge position (bottleneck)."""
    cycle = cycle[:]
    m = len(cycle)

    def score(c):
        best = 0
        for i in range(m):
            p = get_pos(c[i], c[(i + 1) % m])
            if p > best:
                best = p
        return best

    cur = score(cycle)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        if deadline is not None and time.time() > deadline:
            break
        for i in range(m - 1):
            for j in range(i + 2, m):
                if j == m - 1 and i == 0:
                    continue  # same edge pair
                a, b = cycle[i], cycle[i + 1]
                c, d = cycle[j], cycle[(j + 1) % m]
                # Candidate new edges: (a,c),(b,d) or (a,d),(b,c)
                for (e1, e2) in [((a, c), (b, d)), ((a, d), (b, c))]:
                    # compute new max
                    # remove edges (a,b) and (c,d), add (e1),(e2)
                    newmax = 0
                    for k in range(m):
                        u = cycle[k]
                        v = cycle[(k + 1) % m]
                        if (u == a and v == b) or (u == b and v == a):
                            continue
                        if (u == c and v == d) or (u == d and v == c):
                            continue
                        p = get_pos(u, v)
                        if p > newmax:
                            newmax = p
                    for (x, y) in (e1, e2):
                        p = get_pos(x, y)
                        if p > newmax:
                            newmax = p
                    if newmax < cur:
                        # apply: reverse segment i+1..j
                        cycle[i + 1:j + 1] = reversed(cycle[i + 1:j + 1])
                        cur = newmax
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return cycle

# ---------------------------------------------------------------------------
# Merge a 2-factor (list of cycles) into a single Hamilton cycle.
# ---------------------------------------------------------------------------
def extract_2factor_cycles(adj, n):
    """Return list of cycles (each a list of vertices) forming a 2-factor,
    using the matching from a perfect-matching solver."""
    from collections import deque
    matchL = [-1] * n
    matchR = [-1] * n
    dist = [-1] * n

    def bfs():
        q = deque()
        found = False
        for u in range(n):
            if matchL[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = matchR[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = matchR[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                matchL[u] = v
                matchR[v] = u
                return True
        dist[u] = -1
        return False

    while bfs():
        for u in range(n):
            if matchL[u] == -1:
                dfs(u)

    # Build cycles from successor relation matchL
    visited = [False] * n
    cycles = []
    for s in range(n):
        if not visited[s]:
            cyc = []
            u = s
            while not visited[u]:
                visited[u] = True
                cyc.append(u)
                u = matchL[u]
            cycles.append(cyc)
    return cycles

def merge_cycles(cycles, adj, n, deadline):
    """Merge multiple cycles into one Hamilton cycle by 2-opt-style edge swaps.
    Returns a single cycle list or None."""
    if len(cycles) == 1:
        return cycles[0][:]

    # Flatten into a single ordering, then do 2-opt to merge.
    # Build a permutation that concatenates cycles.
    perm = []
    for c in cycles:
        perm.extend(c)
    # Now perm is a Hamiltonian path/cycle? No — it's a set of cycles concatenated.
    # The "cycle" formed by perm has extra edges between cycle boundaries that may
    # not exist. Instead, we treat it as a tour and run 2-opt to remove boundary edges.

    # We need a valid tour to start. Build one by connecting cycles via existing edges.
    # Use 2-opt on the concatenated order; but initial tour may be invalid.
    # Alternative: repeatedly pick two cycles and try to merge via a swap of two edges
    # where one edge from each cycle is replaced by two cross edges that exist.

    cur_cycles = [c[:] for c in cycles]
    changed = True
    while changed and len(cur_cycles) > 1 and time.time() < deadline:
        changed = False
        for i in range(len(cur_cycles)):
            for j in range(i + 1, len(cur_cycles)):
                A = cur_cycles[i]
                B = cur_cycles[j]
                # Try all pairs of edges (a1,a2) in A and (b1,b2) in B
                for ia in range(len(A)):
                    a1, a2 = A[ia], A[(ia + 1) % len(A)]
                    for ib in range(len(B)):
                        b1, b2 = B[ib], B[(ib + 1) % len(B)]
                        # Option 1: replace (a1,a2),(b1,b2) with (a1,b1),(a2,b2)
                        if b1 in adj[a1] and b2 in adj[a2]:
                            # merge
                            newA = A[:ia + 1] + B[ib:] + B[:ib] + A[ia + 1:]
                            # careful with wrap; simpler: build new cycle
                            # A: a1..(before a2) then jump to B
                            # Let's construct properly
                            Aseg1 = A[ia + 1:] + A[:ia + 1]  # rotation so a2 is last? 
                            # Simpler: new cycle = A[0..ia] + B[ib..end]+B[0..ib-1] reversed appropriately
                            # We'll just build: start at a1, go through A to a2, then into B
                            newcyc = []
                            # traverse A from a1 to a2 (inclusive of a1, up to a2)
                            idx = ia
                            while True:
                                newcyc.append(A[idx])
                                if A[idx] == a2:
                                    break
                                idx = (idx + 1) % len(A)
                            # now go into B from b1 to b2
                            idx = ib
                            while True:
                                newcyc.append(B[idx])
                                if B[idx] == b2:
                                    break
                                idx = (idx + 1) % len(B)
                            # newcyc should have len A + len B
                            cur_cycles[i] = newcyc
                            del cur_cycles[j]
                            changed = True
                            break
                        # Option 2: replace with (a1,b2),(a2,b1)
                        if b2 in adj[a1] and b1 in adj[a2]:
                            newcyc = []
                            idx = ia
                            while True:
                                newcyc.append(A[idx])
                                if A[idx] == a2:
                                    break
                                idx = (idx + 1) % len(A)
                            idx = ib
                            while True:
                                newcyc.append(B[idx])
                                if B[idx] == b1:
                                    break
                                idx = (idx + 1) % len(B)
                            cur_cycles[i] = newcyc
                            del cur_cycles[j]
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break
    if len(cur_cycles) == 1:
        return cur_cycles[0][:]
    return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin (evaluator kills at 90s)

    n = N
    total_edges = n * (n - 1) // 2  # 276

    # Precompute adjacency for each prefix threshold T (edges with pos <= T)
    # We'll build incrementally.

    # First, get a quick feasible cycle via greedy on the full graph to have a fallback.
    full_adj = [[] for _ in range(n)]
    for (u, v) in EDGE_ORDER:
        full_adj[u].append(v)
        full_adj[v].append(u)

    fallback = greedy_cycle(full_adj, n, deadline)
    if fallback is not None:
        fallback = two_opt_refine(fallback, n, max_iter=5000, deadline=deadline)

    best_cycle = fallback
    best_score = cycle_score(fallback) if fallback is not None else total_edges + 1

    # Binary search for minimal T such that G_T has a 2-factor (necessary for Hamilton cycle)
    lo, hi = n, total_edges  # 24 .. 276
    # We know full graph is feasible.
    feasible_T = total_edges
    found_feasible = False

    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        # Build adj for threshold mid
        adj = [[] for _ in range(n)]
        for (u, v), p in edge_pos.items():
            if p <= mid:
                adj[u].append(v)
                adj[v].append(u)
        if has_2factor(adj, n):
            feasible_T = mid
            found_feasible = True
            hi = mid - 1
        else:
            lo = mid + 1

    # Now try to find an actual Hamilton cycle at threshold feasible_T
    if found_feasible and time.time() < deadline:
        T = feasible_T
        adj = [[] for _ in range(n)]
        for (u, v), p in edge_pos.items():
            if p <= T:
                adj[u].append(v)
                adj[v].append(u)

        # Try greedy backtracking directly
        cycle = greedy_cycle(adj, n, deadline)
        if cycle is not None:
            cycle = two_opt_refine(cycle, n, max_iter=2000, deadline=deadline)
            s = cycle_score(cycle)
            if s < best_score:
                best_score = s
                best_cycle = cycle
        else:
            # Greedy failed; try 2-factor + merge
            cycles = extract_2factor_cycles(adj, n)
            if cycles and len(cycles) > 0:
                merged = merge_cycles(cycles, adj, n, deadline)
                if merged is not None:
                    merged = two_opt_refine(merged, n, max_iter=2000, deadline=deadline)
                    s = cycle_score(merged)
                    if s < best_score:
                        best_score = s
                        best_cycle = merged

    # Also try a few thresholds slightly above feasible_T for better cycles
    if found_feasible and time.time() < deadline:
        for T in range(feasible_T, min(feasible_T + 15, total_edges + 1)):
            if time.time() > deadline:
                break
            adj = [[] for _ in range(n)]
            for (u, v), p in edge_pos.items():
                if p <= T:
                    adj[u].append(v)
                    adj[v].append(u)
            cycle = greedy_cycle(adj, n, deadline)
            if cycle is not None:
                cycle = two_opt_refine(cycle, n, max_iter=1500, deadline=deadline)
                s = cycle_score(cycle)
                if s < best_score:
                    best_score = s
                    best_cycle = cycle

    # Final refinement on best_cycle with more 2-opt iterations
    if best_cycle is not None and time.time() < deadline:
        best_cycle = two_opt_refine(best_cycle, n, max_iter=20000, deadline=deadline)
        s = cycle_score(best_cycle)
        if s < best_score:
            best_score = s

    if best_cycle is not None:
        return best_cycle
    return list(range(n))
