"""
New design: construct a 5-regular graph of girth at least 9 using a
probabilistic/randomized search over the line graph of a 3-regular graph
with girth at least 14.

Key insight: If G is a 3-regular graph with girth >= 14, then its line
graph L(G) is 4-regular (not 5-regular). So that direct mapping doesn't
work for k=5.

Alternative approach: Use a generalized Petersen-type or algebraic construction
for a 5-regular bipartite graph with girth >= 9.

A 5-regular bipartite graph with bipartition (X, Y) where |X| = |Y| = n
has 2n vertices and 5n edges. Girth >= 9 means no cycles of length 2, 4, 6, or 8.

Strategy: Use a circulant-type bipartite graph. Let vertices on left be
(0, j) for j in range(n) and on right be (1, j) for j in range(n).
Connect (0, j) to (1, j + s mod n) for s in a set S of size 5.

For a (5, 9)-cage-like graph, we need to find n and shift set S such that
the resulting bipartite graph has girth at least 9.

A cycle of length 2m in this bipartite circulant graph corresponds to a
sequence of m steps from X to Y and back, where each step uses one of the
shifts. A cycle of length 2m exists if there's a sequence of shifts
s_1, ..., s_m (each in S, with signs ±1 for direction) such that the
sum s_1 + s_2 + ... + s_m ≡ 0 (mod n), with no proper prefix sum ≡ 0 (mod n)
(for a simple cycle), and the walk is non-backtracking.

For girth >= 9 (i.e., no cycles of length 2, 4, 6, or 8 in the bipartite
graph), we need:
- No cycle of length 2: trivially satisfied if S has no 0 (no multiple edges)
- No cycle of length 4: no two distinct shifts s1, s2 with s1 - s2 ≡ 0 or
  s1 + s2 ≡ 0 (mod n), i.e., S ∩ (-S) = ∅ and |S| = 5 means all pairwise
  differences non-zero (automatic) and no s1 + s2 ≡ 0
- No cycle of length 6: no three shifts s1, s2, s3 with s1 + s2 + s3 ≡ 0
- No cycle of length 8: no four shifts with sum ≡ 0

Actually, for bipartite circulants, a 2m-cycle exists iff there is a sequence
of m elements from S (with repetition allowed, since we can use the same
shift multiple times in a cycle) whose signed sum is 0 mod n. More precisely,
a cycle of length 2m corresponds to choosing m edges from X to Y and m edges
from Y to X forming a cycle. In the circulant case, a closed non-backtracking
walk of length 2m in the bipartite graph corresponds to a sequence of m
"effective steps" where each step is in {s, -s : s in S}, and the sum is
0 mod n, with the constraint that consecutive steps don't immediately reverse.

This is getting complex. Let me use a different, more reliable approach.

Use the known construction: For a 5-regular graph of girth 9, we can use
a voltage graph / covering construction, or simply do a randomized search
over sparse random 5-regular graphs and check girth.

Given the 120s time limit, a practical approach:
1. Try a few algebraic constructions (circulant bipartite graphs with carefully
   chosen shift sets).
2. For each candidate, verify girth using BFS from each vertex (limited depth).
3. Return the smallest valid graph found.

For a 5-regular bipartite graph on 2n vertices, the Moore bound for girth 9 is:
n >= 1 + 5 + 5*4 = 26 (for one side), so 2n >= 52. But actual cages are much larger.

Let me try circulant bipartite graphs with shift sets of size 5. For n up to
a few hundred, I can enumerate shift sets and check girth.

A simpler approach: use the known (5,9)-cage lower bound region and try
random regular graphs with girth checking.

Actually, let me try a direct approach: generate random 5-regular graphs
on increasing n vertices, check if girth >= 9 using BFS, and return the
first one found. This is feasible for n in the hundreds within 120s if we
check efficiently.

For girth checking: BFS from each vertex up to depth 4 (since girth 9 means
no cycle of length <= 8, and in a bipartite graph all cycles are even, so
we check for cycles of length 2, 4, 6, 8). In a non-bipartite 5-regular graph,
we need to check odd cycles too (lengths 3, 5, 7).

Since the task doesn't require bipartite, let me just do general girth checking.

Girth >= 9 means no cycles of length 3, 4, 5, 6, 7, or 8.

For a 5-regular graph on n vertices, checking girth >= 9:
- From each vertex, do BFS up to depth 4. If we find a cycle of length < 9, reject.
- More precisely, from vertex v, do a non-backtracking BFS. If at any point
  we encounter a previously visited vertex at depth d where d + 1 < 9, we have
  a short cycle.

Actually, the standard way: from each vertex v, do BFS. Track (parent, depth).
If we reach a vertex u at depth d via parent p, and u was previously reached
at depth d' via parent p', and u != p and p != parent of u in first visit,
and d + d' + 1 < 9, then we have a cycle of length d + d' + 1 < 9.

This is O(n * 5^4) per check which is O(n) for fixed depth, very fast.

Let me implement:
1. Generate random 5-regular graphs using configuration model / Havel-Hakimi.
2. Check girth >= 9.
3. Return the smallest n that works.

For random 5-regular graph generation: use the configuration model (pair up
5n half-edges randomly), then check for simplicity (no loops/multi-edges)
and connectivity.
"""

import time
import random
from collections import deque

def _generate_random_5_reg(n, max_retries=50):
    """Generate a random 5-regular simple graph on n vertices using configuration model."""
    for _ in range(max_retries):
        half_edges = []
        for v in range(n):
            for _ in range(5):
                half_edges.append(v)
        random.shuffle(half_edges)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, n * 5, 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if not valid:
            continue
        ok = True
        for v in range(n):
            if len(adj[v]) != 5:
                ok = False
                break
        if not ok:
            continue
        # Check connectivity
        visited = [False] * n
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            u = queue.popleft()
            for w in adj[u]:
                if not visited[w]:
                    visited[w] = True
                    count += 1
                    queue.append(w)
        if count != n:
            continue
        return [list(s) for s in adj]
    return None

def _check_girth_nb(adj, g, n):
    """
    Exact girth check via non-backtracking walks from every vertex.
    A cycle of length < g exists iff there is a non-backtracking closed walk
    of length < g. We check by doing BFS on the directed line graph (states
    are directed edges (u,v)) up to depth g-1, looking for a return to the
    starting vertex via a different incoming edge.
    
    More efficiently: from each vertex s, do a BFS where state is (current, prev).
    We track the minimum depth at which each (current, prev) state is first reached.
    If we reach state (s, x) at depth d where d < g and x != start_prev (i.e., we
    returned to s from a different direction), that's a cycle of length d.
    Actually simpler: BFS on directed edges. State = (u, v) meaning we're at v
    having come from u. Start: for each neighbor w of s, state (s, w) at depth 1.
    Transition: (u, v) -> (v, w) for w in adj[v], w != u.
    If we reach a state (v, s) where v != s... no.
    
    Let me use the standard approach: from start vertex s, maintain dist[state]
    where state = (prev, curr). Initialize dist[(s, w)] = 1 for each w in adj[s].
    BFS: pop (prev, curr) at depth d. For each next in adj[curr], next != prev:
      new_state = (curr, next), new_depth = d + 1.
      If next == s and new_depth < g: cycle of length new_depth found -> return False.
      If dist[new_state] not set and new_depth < g: set and enqueue.
    If no cycle found from s, try next s.
    """
    for s in range(n):
        # dist[(prev, curr)] = depth
        dist = {}
        queue = deque()
        for w in adj[s]:
            state = (s, w)
            dist[state] = 1
            queue.append(state)
        while queue:
            prev, curr = queue.popleft()
            d = dist[(prev, curr)]
            if d >= g - 1:
                continue
            for nxt in adj[curr]:
                if nxt == prev:
                    continue
                nd = d + 1
                if nd >= g:
                    continue
                if nxt == s:
                    return False  # cycle of length nd < g
                nstate = (curr, nxt)
                if nstate not in dist:
                    dist[nstate] = nd
                    queue.append(nstate)
    return True

def _try_circulant_bipartite(k, g, deadline):
    """
    Try to construct a k-regular bipartite circulant graph with girth >= g.
    Bipartite graph with parts X and Y, each of size n.
    X[i] connected to Y[(i+s) % n] for s in S, |S| = k.
    Total vertices = 2n.
    """
    if k % 2 == 1:
        # For odd k, bipartite circulant: each vertex has k neighbors on other side
        # n must be such that we can pick k shifts
        # Try various n values
        for n in range(k + 1, 3000):
            if time.time() > deadline:
                break
            # Try a few random shift sets
            for attempt in range(3):
                if time.time() > deadline:
                    break
                # Pick k random distinct shifts from 1..n-1
                if n <= k:
                    continue
                S = random.sample(range(1, n), k)
                # Build adjacency and check girth
                adj = _build_circulant_bipartite(n, S)
                if adj is not None:
                    total_n = 2 * n
                    if _check_girth_nb(adj, g, total_n):
                        edges = []
                        for u in range(total_n):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        return total_n, edges
    return None

def _build_circulant_bipartite(n, S):
    """Build k-regular bipartite circulant graph. Parts: 0..n-1 (X), n..2n-1 (Y)."""
    k = len(S)
    adj = [[] for _ in range(2 * n)]
    for i in range(n):
        for s in S:
            j = (i + s) % n
            adj[i].append(n + j)
            adj[n + j].append(i)
    # Verify simplicity (no duplicate edges) and degree
    for i in range(2 * n):
        if len(adj[i]) != k:
            return None
        if len(set(adj[i])) != k:
            return None
    # Check no self-loops
    for i in range(2 * n):
        if i in adj[i]:
            return None
    return adj

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # leave 5s margin for 120s limit
    
    if k != 5:
        # Fallback: use the original LUW construction for other k values
        return _luw_fallback(k, g)
    
    # For k=5, g=9: search for 5-regular graph with girth >= 9
    # Moore bound: for girth g=9 (odd, r=4): n >= 1 + 5 + 5*4 + 5*4^2 + 5*4^3 = 426
    
    best_n = None
    best_adj = None
    
    # Strategy 1: Try circulant bipartite constructions (deterministic, fast)
    result = _try_circulant_bipartite(k, g, deadline)
    if result is not None:
        return result
    
    # Strategy 2: Random 5-regular graphs
    # Expected girth of random k-regular graph ~ 2*log_{k-1}(n)
    # For girth 9: 2*log_4(n) >= 9 => n >= 4^4.5 ~ 1024
    # Try n values around and above this
    
    # Try a range of n values, larger first (more likely to have high girth)
    n_candidates = []
    # Start near expected girth threshold and go up
    for n in range(1024, 1200, 4):
        n_candidates.append(n)
    # Also try some smaller values
    for n in range(426, 1024, 8):
        n_candidates.append(n)
    # Shuffle smaller ones to explore
    random.shuffle(n_candidates)
    
    for n in n_candidates:
        if time.time() > deadline:
            break
        # Number of attempts based on remaining time
        remaining = deadline - time.time()
        if remaining < 2:
            break
        max_attempts = max(1, min(30, int(remaining / 0.5)))
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            adj = _generate_random_5_reg(n, max_retries=10)
            if adj is None:
                continue
            if _check_girth_nb(adj, g, n):
                best_n = n
                best_adj = adj
                break
        if best_n is not None:
            break
    
    if best_n is not None:
        edges = []
        for u in range(best_n):
            for v in best_adj[u]:
                if u < v:
                    edges.append((u, v))
        return best_n, edges
    
    # Fallback: return the LUW construction
    return _luw_fallback(k, g)

def _luw_fallback(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
