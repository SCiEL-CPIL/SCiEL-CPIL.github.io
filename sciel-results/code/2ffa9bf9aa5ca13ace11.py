import time
import random
import sys
from collections import deque

def _has_girth_at_least(adj, n, target_girth):
    """Check if girth >= target_girth using BFS. Returns False early if a shorter cycle is found."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            dv = dist[v]
            # If dv >= target_girth // 2, we can stop early for this start?
            # No, we need to check all. But if dv + dist[u] + 1 < target_girth, we fail.
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dv + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    if dv + dist[u] + 1 < target_girth:
                        return False
    return True

def _find_short_cycle(adj, n, max_len):
    """Find a cycle of length < max_len. Returns list of vertices or None."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            dv = dist[v]
            if dv >= max_len // 2:
                continue
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dv + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    cycle_len = dv + dist[u] + 1
                    if cycle_len < max_len:
                        # Reconstruct cycle
                        cycle = [v]
                        curr = v
                        while curr != u:
                            curr = parent[curr]
                            cycle.append(curr)
                        # Now curr is u
                        # The path from start to u is parent chain
                        # The path from start to v is parent chain
                        # We need the full cycle
                        # This reconstruction is tricky. 
                        # For simplicity, we just return the fact that a cycle exists 
                        # and maybe the edges involved.
                        # But for local search, we need to modify edges.
                        # We can just return the vertices v and u and the dist.
                        # The edge (v,u) is the closing edge.
                        # The path v->...->start->...->u is the rest.
                        return cycle_len
    return None

def _construct_cage_3_16():
    """Construct a 3-regular bipartite graph with girth >= 16 using local search on permutations."""
    n_target = 936
    half = n_target // 2
    deadline = time.time() + 100.0
    
    # Try local search for N=936
    # Then N=938, etc.
    for n in range(936, 1458, 2):
        if time.time() > deadline:
            break
        half = n // 2
        
        # Initialize 3 random permutations
        # perm[i][j] is the j-th neighbor of i in matching i
        # Actually, let's store as list of lists: perms[0], perms[1], perms[2]
        # perms[k][i] = j means edge (i, half+j) exists via matching k
        
        perms = [list(range(half)) for _ in range(3)]
        for p in perms:
            random.shuffle(p)
            
        # Check for multiple edges initially
        valid_init = True
        for i in range(half):
            neighbors = [perms[k][i] for k in range(3)]
            if len(set(neighbors)) != 3:
                valid_init = False
                break
        if not valid_init:
            # Retry init
            for _ in range(100):
                perms = [list(range(half)) for _ in range(3)]
                for p in perms:
                    random.shuffle(p)
                valid_init = True
                for i in range(half):
                    neighbors = [perms[k][i] for k in range(3)]
                    if len(set(neighbors)) != 3:
                        valid_init = False
                        break
                if valid_init:
                    break
            if not valid_init:
                continue

        # Local search
        best_girth_found = 0
        for step in range(10000):
            if time.time() > deadline:
                break
                
            # Build adjacency
            adj = [[] for _ in range(n)]
            for i in range(half):
                for k in range(3):
                    j = perms[k][i]
                    adj[i].append(half + j)
                    adj[half + j].append(i)
            
            # Check girth
            if _has_girth_at_least(adj, n, 16):
                edges = []
                for i in range(half):
                    for k in range(3):
                        j = perms[k][i]
                        edges.append((i, half + j))
                return n, edges
            
            # Find a short cycle to break
            # We can just pick a random edge and swap it?
            # Better: find a cycle of length < 16.
            # Since we know girth < 16, there is a cycle.
            # Let's find one quickly.
            # We can reuse the BFS from _has_girth_at_least to return the cycle.
            # Let's modify _has_girth_at_least to return the cycle if found.
            
            # For now, just do a random swap in a random permutation.
            # Pick random matching k
            k = random.randint(0, 2)
            # Pick random node i
            i = random.randint(0, half - 1)
            # Pick random other node w
            w = random.randint(0, half - 1)
            if i == w:
                continue
            
            # Swap perms[k][i] and perms[k][w]
            old_i = perms[k][i]
            old_w = perms[k][w]
            
            # Check if swap creates multiple edges
            # New neighbors for i: perms[0][i], perms[1][i], perms[2][i]
            # One of them is old_w (if k is the one changed)
            # We need to check if old_w is equal to perms[j][i] for j != k
            # And if old_i is equal to perms[j][w] for j != k
            
            conflict = False
            for j in range(3):
                if j == k: continue
                if perms[j][i] == old_w:
                    conflict = True
                    break
                if perms[j][w] == old_i:
                    conflict = True
                    break
            
            if not conflict:
                perms[k][i] = old_w
                perms[k][w] = old_i
            else:
                # Try a different swap or revert
                pass

    return None

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    deadline = time.time() + 38.0
    
    def _build_luw(d):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        rule_list = [rule[c] for c in range(2, d + 1)]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for idx, c in enumerate(range(2, d + 1)):
                    a, b = rule_list[idx]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        i = 0
        while i < len(order):
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            i += 1
        return len(order), edges

    for d in range(5, 20):
        if time.time() > deadline:
            break
        n, edges = _build_luw(d)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        if _has_girth_at_least(adj, n, g):
            return n, edges
        
    d_orig = max(8, g - 4)
    return _build_luw(d_orig)

def construct(k, g):
    start_time = time.time()
    
    if k == 3 and g == 16:
        result = _construct_cage_3_16()
        if result is not None:
            return result
        return _luw_construct(k, g)
    
    return _luw_construct(k, g)
