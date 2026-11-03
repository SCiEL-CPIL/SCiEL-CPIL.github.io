"""Improved Bipartite Random Greedy with Balanced Layering."""
import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Returns (n, edges).
    """
    deadline = time.time() + 112  # Safety margin

    # 1) Fast algebraic LZW construction (deterministic, valid upper bound)
    best = None
    try:
        lzw = _lzw_construct(k, g)
        if lzw is not None:
            best = lzw
    except Exception:
        best = None

    # 2) Improved Bipartite Random Greedy
    if time.time() < deadline:
        base = _moore_bound(k, g)
        cap = best[0] if best else 2000
        
        # Generate candidate n values
        candidates = []
        n = max(base + 2, 500)
        while n <= cap and time.time() < deadline:
            candidates.append(n)
            n = int(n * 1.08) + 2
            
        # Add specific strategic points
        for extra in [700, 900, 1100, 1300, 1500]:
            if extra <= cap and extra not in candidates:
                candidates.append(extra)
                
        candidates = sorted(set(candidates))

        for n in candidates:
            if time.time() > deadline - 5:
                break
            res = _bipartite_greedy_balanced(k, g, n, deadline)
            if res is not None:
                if best is None or res[0] < best[0]:
                    best = res
                # Since we search upwards, the first valid n is the smallest for this strategy
                break

    if best is not None:
        return best

    # 3) Fallback
    return _lzw_construct(k, g)

def _moore_bound(k, g):
    if g % 2 == 0:
        d = g // 2
        n = k * (k - 1) ** (d - 1)
    else:
        d = (g - 1) // 2
        n = 1 + k * sum((k - 1) ** i for i in range(d))
    return n

def _bipartite_greedy_balanced(k, g, n, deadline):
    """
    Bipartite random greedy with balanced degree progression.
    """
    n_half = n // 2
    if 2 * n_half != n:
        n = 2 * n_half
        n_half = n // 2

    # Girth constraint for bipartite graph:
    # Cycle length = 2 * (dist + 1). Need 2 * (dist + 1) >= g => dist >= ceil(g/2) - 1.
    min_dist = -(-g // 2) - 1

    max_restarts = 5
    for _ in range(max_restarts):
        if time.time() > deadline - 2:
            return None

        # Adjacency as sets for fast membership check
        adj = [set() for _ in range(n)]
        deg = [0] * n

        left_verts = list(range(n_half))
        right_verts = list(range(n_half, n))
        random.shuffle(left_verts)
        random.shuffle(right_verts)
        
        # We will process pairs. Each pair adds 1 edge.
        # Total edges needed: n * k / 2.
        # Total pairs: n * k / 4.
        # Since n is even and k is usually even (5 is odd, so n must be even for k-regular bipartite? 
        # Wait, sum of degrees must be even. n*k must be even. If k=5, n must be even. 
        # Bipartite regular graphs exist for even n and any k.
        # Number of edges = n*k/2.
        # Number of steps in balanced loop = n*k/4? No, each step adds 1 edge.
        # We need to add n*k/2 edges.
        
        total_edges_needed = n * k // 2
        edges_added = 0
        
        # We maintain a pool of available vertices for each side
        # Instead of shuffling once, we can just iterate through the shuffled lists
        # and pick the first available one.
        
        l_ptr = 0
        r_ptr = 0
        
        # We need a way to pick a random available vertex efficiently.
        # Using a set of available indices is O(1) to check, but random.choice on a set is slow.
        # We can maintain a list and swap-remove.
        
        avail_l = list(range(n_half))
        avail_r = list(range(n_half))
        random.shuffle(avail_l)
        random.shuffle(avail_r)
        
        # To pick a random available vertex, we can just pop from the end? 
        # No, that's deterministic order. 
        # We want random. 
        # We can use a dictionary or just filter. 
        # For n~1000, filtering is fast enough.
        
        success = True
        while edges_added < total_edges_needed:
            if time.time() > deadline - 2:
                success = False
                break
                
            # Find a random available left vertex
            # We can just pick a random index from 0 to len(avail_l)-1 and check if deg < k
            # But avail_l contains all vertices. We need to pick one with deg < k.
            # Let's maintain a list of "active" vertices.
            
            # Optimization: Instead of maintaining complex active lists, 
            # we can just pick a random vertex from the whole side and check if it's full.
            # If full, retry. Since we are balanced, the fraction of full vertices grows slowly.
            
            u = None
            v = None
            
            # Pick u
            attempts = 0
            while u is None and attempts < 100:
                idx = random.randint(0, n_half - 1)
                u = idx
                if deg[u] < k:
                    break
                u = None
                attempts += 1
            
            if u is None:
                success = False
                break
                
            # Pick v
            attempts = 0
            while v is None and attempts < 100:
                idx = random.randint(0, n_half - 1)
                v = n_half + idx
                if deg[v] < k:
                    break
                v = None
                attempts += 1
                
            if v is None:
                success = False
                break
                
            # Check distance
            # We need to ensure dist(u, v) >= min_dist
            # If min_dist is small, most pairs are valid.
            # We need to check if adding (u,v) creates a cycle < g.
            # This is equivalent to checking if dist(u,v) in the current graph is < min_dist.
            # If dist < min_dist, adding edge creates cycle 2*(dist+1) < g.
            
            # Fast check: if min_dist is large, we can just check if they are connected within min_dist-1 steps.
            # If they are NOT connected within min_dist-1 steps, the distance is >= min_dist.
            
            if _bipartite_too_close(adj, u, v, min_dist - 1):
                # They are too close. Try another pair?
                # In a greedy algorithm, we usually just fail or try to find another v.
                # Let's try to find another v.
                found_v = False
                for _ in range(10):
                    idx = random.randint(0, n_half - 1)
                    cand_v = n_half + idx
                    if deg[cand_v] < k and cand_v != v:
                        if not _bipartite_too_close(adj, u, cand_v, min_dist - 1):
                            v = cand_v
                            found_v = True
                            break
                if not found_v:
                    # Maybe try another u?
                    found_u = False
                    for _ in range(10):
                        idx = random.randint(0, n_half - 1)
                        cand_u = idx
                        if deg[cand_u] < k and cand_u != u:
                            if not _bipartite_too_close(adj, cand_u, v, min_dist - 1):
                                u = cand_u
                                found_u = True
                                break
                    if not found_u:
                        success = False
                        break
            else:
                # Valid pair
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges_added += 1
        
        if not success:
            continue

        # Check degrees
        if any(d != k for d in deg):
            continue

        # Build edge list
        edges = []
        for u in range(n_half):
            for v in adj[u]:
                edges.append((u, v))
        
        # Verify
        if _verify_girth_fast(n, edges, k, g):
            return (n, edges)

    return None

def _bipartite_too_close(adj, u, v, limit):
    """
    Returns True if distance between u and v is <= limit.
    We want to avoid adding edge if dist < min_dist.
    So we check if dist <= min_dist - 1.
    """
    if u == v:
        return True
    dist = {u: 0}
    queue = deque([u])
    while queue:
        node = queue.popleft()
        d = dist[node]
        if d >= limit:
            continue
        for nb in adj[node]:
            if nb == v:
                return True
            if nb not in dist:
                dist[nb] = d + 1
                queue.append(nb)
    return False

def _verify_girth_fast(n, edges, k, g):
    """
    Verify k-regular and girth >= g.
    """
    if len(edges) != n * k // 2:
        return False
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != k:
            return False

    adj_np = [np.array(a, dtype=np.int32) for a in adj]
    max_depth = (g - 1) // 2

    for start in range(n):
        dist = np.full(n, -1, dtype=np.int32)
        parent = np.full(n, -1, dtype=np.int32)
        dist[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            d = dist[node]
            if d >= max_depth:
                continue
            for nb in adj_np[node]:
                nb = int(nb)
                if dist[nb] == -1:
                    dist[nb] = d + 1
                    parent[nb] = node
                    queue.append(nb)
                elif nb != parent[node]:
                    cycle_len = int(dist[node]) + int(dist[nb]) + 1
                    if cycle_len < g:
                        return False
    return True

def _lzw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph"""
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
