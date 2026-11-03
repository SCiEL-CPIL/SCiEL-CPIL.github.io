"""Construct small 4-regular graphs of high girth (cage problem n(4,10))."""
import time
import random
from collections import deque
import numpy as np

# EVOLVE-BLOCK-START

def _build_adj(n, k, edges):
    """Build adjacency list and verify degree/simplicity. Returns None if invalid."""
    adj = [[] for _ in range(n)]
    seen = set()
    for a, b in edges:
        if a == b or a < 0 or a >= n or b < 0 or b >= n:
            return None
        key = (min(a, b), max(a, b))
        if key in seen:
            return None
        seen.add(key)
        adj[a].append(b)
        adj[b].append(a)
    for v in range(n):
        if len(adj[v]) != k:
            return None
    return adj


def _is_valid(k, g, n, edges):
    """Check degree, simplicity, and girth via non-backtracking BFS from every vertex."""
    adj = _build_adj(n, k, edges)
    if adj is None:
        return False
    return _check_girth(k, g, n, adj)


def _check_girth(k, g, n, adj):
    """Fast girth check using bitmask-based non-backtracking BFS."""
    # Precompute neighbor bitmasks for fast intersection
    if n <= 64:
        masks = [0] * n
        for v in range(n):
            m = 0
            for u in adj[v]:
                m |= (1 << u)
            masks[v] = m
        # BFS from each vertex using bitmasks
        for s in range(n):
            # State: set of (cur, prev) pairs encoded as bitmasks
            # Level 1: all neighbors of s
            # We track for each node 'cur', the set of 'prev' nodes we've visited
            # prev_sets[cur] = bitmask of prev nodes
            prev_sets = [0] * n
            # Initialize: from s, go to each neighbor v. State (v, s)
            for v in adj[s]:
                prev_sets[v] |= (1 << s)
            frontier = list(adj[s])  # nodes to expand at current level
            dist = 1
            while frontier and dist < g - 1:
                next_frontier = []
                next_prev_sets = [0] * n
                for cur in frontier:
                    prev_mask = prev_sets[cur]
                    # Next nodes: neighbors of cur, excluding prev
                    nxt_mask = masks[cur]
                    # Remove prev nodes
                    candidates = nxt_mask & ~prev_mask
                    # For each candidate nxt, add cur as a prev
                    while candidates:
                        lb = candidates & (-candidates)
                        nxt = lb.bit_length() - 1
                        candidates ^= lb
                        if nxt == s:
                            # Found cycle of length dist + 1
                            if dist + 1 < g:
                                return False
                        if not (next_prev_sets[nxt] & (1 << cur)):
                            next_prev_sets[nxt] |= (1 << cur)
                            next_frontier.append(nxt)
                prev_sets = next_prev_sets
                frontier = next_frontier
                dist += 1
    else:
        # Fallback: standard BFS with sets (for larger graphs)
        for s in range(n):
            visited = set()
            q = deque()
            for v in adj[s]:
                visited.add((v, s))
                q.append((v, s, 1))
            while q:
                cur, prev, dist = q.popleft()
                if dist >= g - 1:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    new_state = (nxt, cur)
                    if new_state in visited:
                        continue
                    if nxt == s:
                        if dist + 1 < g:
                            return False
                    visited.add(new_state)
                    q.append((nxt, cur, dist + 1))
    return True


def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar construction (baseline)."""
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


def _random_k_regular(n, k, seed=0):
    """Generate a random k-regular graph on n vertices using configuration model + dedup."""
    if n * k % 2 != 0:
        return None
    rng = random.Random(seed)
    # Configuration model
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    while True:
        rng.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edges:
                valid = False
                break
            edges.add(key)
        if valid and len(edges) == n * k // 2:
            return list(edges)
    return None


def _edge_swap_improve(n, k, g, edges, deadline, max_iters=5000, seed=0):
    """Try to reduce girth violations via edge swaps (2-opt style). Returns (n, edges) or original."""
    rng = random.Random(seed)
    adj = _build_adj(n, k, edges)
    if adj is None:
        return n, edges
    
    # Build edge set for quick lookup
    edge_set = set()
    for a, b in edges:
        edge_set.add((min(a, b), max(a, b)))
    
    # Current graph might have girth < g. Try to fix via swaps.
    # A swap: remove edges (a,b) and (c,d), add (a,c) and (b,d) (or (a,d) and (b,c))
    # This preserves k-regularity.
    
    edges_list = list(edges)
    best_n, best_edges = n, edges_list
    
    # Check if already valid
    if _check_girth(k, g, n, adj):
        return n, edges_list
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Pick two random edges to swap
        e1 = rng.randint(0, len(edges_list) - 1)
        e2 = rng.randint(0, len(edges_list) - 1)
        if e1 == e2:
            continue
        a, b = edges_list[e1]
        c, d = edges_list[e2]
        # Ensure no shared vertices (otherwise swap doesn't preserve regularity properly)
        if len({a, b, c, d}) < 4:
            continue
        # Try swap: (a,c) and (b,d)
        if (min(a,c), max(a,c)) in edge_set or (min(b,d), max(b,d)) in edge_set:
            continue
        # Perform swap
        edge_set.discard((min(a,b), max(a,b)))
        edge_set.discard((min(c,d), max(c,d)))
        edge_set.add((min(a,c), max(a,c)))
        edge_set.add((min(b,d), max(b,d)))
        edges_list[e1] = (a, c)
        edges_list[e2] = (b, d)
        
        # Rebuild adj and check girth
        adj2 = _build_adj(n, k, edges_list)
        if adj2 is None:
            # Revert
            edge_set.discard((min(a,c), max(a,c)))
            edge_set.discard((min(b,d), max(b,d)))
            edge_set.add((min(a,b), max(a,b)))
            edge_set.add((min(c,d), max(c,d)))
            edges_list[e1] = (a, b)
            edges_list[e2] = (c, d)
            continue
        
        if _check_girth(k, g, n, adj2):
            return n, edges_list
        # Keep the swap (accept even if not yet valid, to explore)
        # But revert if it makes things worse? For simplicity, always accept.
    
    return n, edges_list


def _try_improve(n, k, g, edges, deadline, seed=0):
    """Try to reduce n by finding a smaller valid graph."""
    best_n, best_edges = n, edges
    moore = 2 * (k - 1) ** (g // 2 - 1) if g % 2 == 0 else k * (k - 1) ** ((g - 2) // 2)
    
    # First, try edge-swap improvement on current graph to see if we can fix girth
    if not _is_valid(k, g, n, edges):
        n, edges = _edge_swap_improve(n, k, g, edges, deadline, max_iters=3000, seed=seed)
        best_n, best_edges = n, edges
    
    # Try smaller sizes via random generation
    # Focus on sizes near the known best (320) and below
    target_sizes = []
    for n_try in range(max(moore, 200), best_n):
        if time.time() > deadline:
            break
        target_sizes.append(n_try)
    
    # Try in order from smallest, but limit attempts per size
    for n_try in target_sizes:
        if time.time() > deadline:
            break
        # More attempts for sizes near known best
        attempts = 5 if n_try > 280 else 3
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            cand_edges = _random_k_regular(n_try, k, seed=seed + n_try * 100 + attempt)
            if cand_edges is None:
                continue
            if _is_valid(k, g, n_try, cand_edges):
                return n_try, cand_edges
    return best_n, best_edges


def construct(k, g):
    deadline = time.time() + 110  # Safety margin before 120s limit
    # Start with LUW construction
    n, edges = _luw_graph(k, g)
    # Verify it's valid
    if not _is_valid(k, g, n, edges):
        # Try edge-swap repair
        n, edges = _edge_swap_improve(n, k, g, edges, deadline, max_iters=5000, seed=0)
        if not _is_valid(k, g, n, edges):
            # Fallback: try random
            for attempt in range(30):
                if time.time() > deadline:
                    break
                cand = _random_k_regular(n, k, seed=attempt)
                if cand and _is_valid(k, g, n, cand):
                    edges = cand
                    break
    # Try to improve
    n, edges = _try_improve(n, k, g, edges, deadline)
    return n, edges

# EVOLVE-BLOCK-END
