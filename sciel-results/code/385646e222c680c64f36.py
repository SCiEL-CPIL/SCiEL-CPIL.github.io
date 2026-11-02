import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    def luw_construction():
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

    def nonbacktracking_girth(adj, g):
        """Return True if the graph has a cycle shorter than g.
        Uses non-backtracking walks from every vertex to certify girth exactly."""
        n = len(adj)
        # For each starting vertex s and first step to neighbor t,
        # do BFS in the non-backtracking graph: state is (current, prev).
        # A cycle of length L < g exists iff we can reach (s, prev_of_s_in_walk)
        # where prev_of_s_in_walk is the neighbor we came from to reach s,
        # and the walk length is L.
        # More precisely: a non-backtracking walk of length L from s back to s
        # (with the last step coming from some neighbor u of s, and L >= 3)
        # corresponds to a cycle of length L (if the walk is closed and non-backtracking).
        # Actually, a closed non-backtracking walk of length L starting and ending at s
        # with first step to a and last step from b (a != b, and the walk doesn't backtrack)
        # gives a cycle of length L if L >= 3 and the walk is simple. But non-backtracking
        # walks can repeat vertices. However, the shortest non-backtracking closed walk
        # from s to s has length equal to the girth if we consider all s.
        # Standard result: the girth is the minimum length of a closed non-backtracking
        # walk (over all starting vertices).
        # So we check: for each s, for each neighbor a of s, do BFS in non-backtracking
        # space starting from state (a, s) [at vertex a, came from s], and see if we can
        # reach state (s, b) for any neighbor b of s with b != a, with walk length < g.
        # The walk length is the number of edges traversed. Starting at (a,s) means we've
        # traversed 1 edge (s->a). To close the cycle at s, we need to be at s having come
        # from some b != a. The total walk length would be the BFS distance + 1 (for the
        # final step into s)... let me think more carefully.
        # State: (v, p) means we are at vertex v, having arrived from p.
        # Initial state for starting vertex s, first step to a: (a, s), distance 1.
        # We want to reach (s, b) for some b in adj[s], b != a. The distance to (s, b)
        # is the number of edges in the non-backtracking walk from s to s via a...b.
        # If dist[(s, b)] < g, then there's a closed non-backtracking walk of length dist[(s,b)]
        # which implies a cycle of length <= dist[(s,b)] < g.
        for s in range(n):
            for a in adj[s]:
                # BFS from state (a, s)
                dist = {}
                dist[(a, s)] = 1
                q = deque([(a, s)])
                while q:
                    v, p = q.popleft()
                    d = dist[(v, p)]
                    if d >= g - 1:
                        continue
                    for w in adj[v]:
                        if w == p:
                            continue
                        new_state = (w, v)
                        if new_state not in dist:
                            dist[new_state] = d + 1
                            if w == s and v != a:
                                # We've returned to s from v, and v != a (non-backtracking)
                                # Cycle length = d + 1
                                if d + 1 < g:
                                    return True
                            q.append(new_state)
        return False

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    def is_valid_graph(n, edges, k, g):
        if n <= 0:
            return False
        adj = [[] for _ in range(n)]
        seen = set()
        for i, j in edges:
            if i == j:
                return False
            if i < 0 or i >= n or j < 0 or j >= n:
                return False
            key = (min(i, j), max(i, j))
            if key in seen:
                return False
            seen.add(key)
            adj[i].append(j)
            adj[j].append(i)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        if nonbacktracking_girth(adj, g):
            return False
        return True

    def random_k_regular(n, k, seed=None):
        """Generate a random k-regular graph on n vertices using configuration model."""
        if n * k % 2 != 0:
            return None
        rng = random.Random(seed)
        # Build half-edges
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        rng.shuffle(half_edges)
        # Pair them up
        edges = set()
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                return None  # self-loop, retry
            key = (min(u, v), max(u, v))
            if key in edges:
                return None  # multi-edge, retry
            edges.add(key)
            adj[u].append(v)
            adj[v].append(u)
        return sorted([list(e) for e in edges])

    def try_local_search(n_target, k, g, deadline):
        """Try to find a k-regular graph on n_target vertices with girth >= g."""
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            edges = random_k_regular(n_target, k, seed=attempt)
            if edges is None:
                continue
            adj = build_adj(edges, n_target)
            if not nonbacktracking_girth(adj, g):
                return n_target, edges
        return None

    # First, get the LUW construction as baseline
    n_luw, edges_luw = luw_construction()
    best_n, best_edges = n_luw, edges_luw

    # Try to find a smaller graph
    # Moore bound for (4,10): 2 * sum_{i=0}^{4} 3^i = 2*(1+3+9+27+81) = 2*121 = 242
    # So n must be >= 242 (and even, since 4-regular)
    moore_bound = 2 * sum(3**i for i in range(g // 2))
    # n must be even for k-regular with even k... actually n*k must be even, so n*4 even always
    # But for 4-regular, n can be any value >= 4 such that n*4 is even (always true)
    # Try values from moore_bound to n_luw - 1
    search_n = moore_bound
    if search_n % 2 != 0:
        search_n += 1

    # Try a few candidate sizes
    candidates = []
    for n_try in range(search_n, best_n, 2):
        candidates.append(n_try)
    # Also try some values closer to the known best
    for n_try in [250, 260, 280, 300, 310, 318]:
        if search_n <= n_try < best_n:
            candidates.append(n_try)
    candidates = sorted(set(candidates))

    # Limit the number of candidates to try
    max_candidates = 20
    if len(candidates) > max_candidates:
        # Sample from candidates
        candidates = candidates[::max(1, len(candidates) // max_candidates)][:max_candidates]

    for n_try in candidates:
        if time.time() > deadline - 10:
            break
        result = try_local_search(n_try, k, g, deadline)
        if result is not None:
            n_found, edges_found = result
            if n_found < best_n:
                best_n, best_edges = n_found, edges_found
                break  # Found a smaller one, stop

    return best_n, best_edges
