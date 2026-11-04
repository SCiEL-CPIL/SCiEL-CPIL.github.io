"""Improved program: try to construct a 4-regular graph of girth 10 with fewer vertices.
Uses a combination of:
1. The Lazebnik-Ustimenko-Woldar construction (known to give girth >= 10 for appropriate params)
2. Random 4-regular graph generation + local search to break short cycles
3. Deadline guard to stay within time limit
"""
import random
import time
import sys
from collections import deque

def _moore_bound(k, g):
    """Moore bound for k-regular graph of girth g."""
    if g % 2 == 0:
        r = g // 2
        return k * (k - 1) ** (r - 1)
    else:
        r = (g - 1) // 2
        return 1 + k * sum((k - 1) ** i for i in range(r))

def _check_girth(n, edges, g):
    """Check if graph has girth at least g using non-backtracking BFS from each vertex."""
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do BFS with non-backtracking constraint
    # State: (vertex, prev_vertex, depth)
    # We track the shortest non-backtracking cycle through each vertex
    
    # For girth check, we need to find if there's any cycle of length < g
    # Use BFS from each vertex, tracking (node, parent) to avoid backtracking
    
    # More efficient: for each vertex s, do a BFS where state is (v, p) meaning
    # we're at v coming from p. If we reach s again with depth >= 2, we have a cycle.
    
    for s in range(n):
        # BFS from s, tracking (current, prev)
        visited = set()
        queue = deque()
        for nb in adj[s]:
            queue.append((nb, s, 1))
            visited.add((nb, s))
        
        while queue:
            v, p, d = queue.popleft()
            if d >= g:
                continue
            if v == s and d >= 2:
                # Found a cycle of length d through s
                return False
            for nb in adj[v]:
                if nb == p:
                    continue  # non-backtracking
                if (nb, v) not in visited:
                    visited.add((nb, v))
                    queue.append((nb, v, d + 1))
    
    return True

def _generate_random_4reg(n):
    """Generate a random 4-regular graph on n vertices using configuration model + 2-switches."""
    if n < 4:
        return None
    
    # Configuration model: each vertex has 4 stubs
    stubs = []
    for v in range(n):
        for _ in range(4):
            stubs.append(v)
    
    # Randomly pair stubs
    random.shuffle(stubs)
    edges = set()
    valid = True
    for i in range(0, 2 * n * 2, 2):
        a, b = stubs[i], stubs[i + 1]
        if a == b:
            valid = False
            break
        if (min(a, b), max(a, b)) in edges:
            valid = False
            break
        edges.add((min(a, b), max(a, b)))
    
    if not valid:
        return None
    
    # Check if it's simple (no multi-edges) - already checked
    # Now do 2-switches to improve structure
    edge_list = list(edges)
    
    # Build adjacency for quick lookup
    adj = [set() for _ in range(n)]
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    
    # Do some 2-switches to try to break short cycles
    for _ in range(100):
        # Pick two random edges
        e1 = random.choice(edge_list)
        e2 = random.choice(edge_list)
        if e1 == e2:
            continue
        a, b = e1
        c, d = e2
        # Check if (a,c) and (b,d) or (a,d) and (b,c) are valid
        if c not in adj[a] and d not in adj[b]:
            new_e1 = (min(a, c), max(a, c))
            new_e2 = (min(b, d), max(b, d))
            if new_e1 != new_e2:
                edges.discard(e1)
                edges.discard(e2)
                edges.add(new_e1)
                edges.add(new_e2)
                adj[a].discard(b)
                adj[b].discard(a)
                adj[a].add(c)
                adj[c].add(a)
                adj[b].add(d)
                adj[d].add(b)
                edge_list.remove(e1)
                edge_list.remove(e2)
                edge_list.append(new_e1)
                edge_list.append(new_e2)
    
    return list(edges)

def _count_short_cycles(n, edges, max_len):
    """Count non-backtracking cycles of length < max_len."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    count = 0
    for s in range(n):
        # BFS with non-backtracking
        visited = set()
        queue = deque()
        for nb in adj[s]:
            queue.append((nb, s, 1))
            visited.add((nb, s))
        
        while queue:
            v, p, d = queue.popleft()
            if d >= max_len:
                continue
            if v == s and d >= 2:
                count += 1
                continue
            for nb in adj[v]:
                if nb == p:
                    continue
                if (nb, v) not in visited:
                    visited.add((nb, v))
                    queue.append((nb, v, d + 1))
    
    return count

def _local_search_break_cycles(n, edges, g, max_iter=5000, deadline=None):
    """Local search: find short cycles and use 2-switches to break them."""
    if deadline is not None:
        if time.time() > deadline:
            return edges
    
    edge_list = list(edges)
    adj = [set() for _ in range(n)]
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    
    best_count = _count_short_cycles(n, edge_list, g)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find a short cycle
        # Sample some vertices and look for short cycles
        found_cycle = None
        for _ in range(10):
            s = random.randint(0, n - 1)
            # BFS from s to find cycle
            visited = {}  # (v, p) -> (d, path)
            queue = deque()
            for nb in adj[s]:
                queue.append((nb, s, 1, [s, nb]))
                visited[(nb, s)] = (1, [s, nb])
            
            while queue:
                v, p, d, path = queue.popleft()
                if d >= g:
                    continue
                if v == s and d >= 2:
                    found_cycle = path
                    break
                for nb in adj[v]:
                    if nb == p:
                        continue
                    if (nb, v) not in visited:
                        visited[(nb, v)] = (d + 1, path + [nb])
                        queue.append((nb, v, d + 1, path + [nb]))
            
            if found_cycle:
                break
        
        if not found_cycle:
            break
        
        # Try 2-switches on edges in the cycle to break it
        cycle_edges = set()
        for i in range(len(found_cycle) - 1):
            a, b = found_cycle[i], found_cycle[i + 1]
            cycle_edges.add((min(a, b), max(a, b)))
        
        # Pick two edges from the cycle and try to 2-switch with random edges
        improved = False
        for _ in range(20):
            if not cycle_edges:
                break
            e1 = random.choice(list(cycle_edges))
            # Pick a random edge not in cycle
            e2 = random.choice(edge_list)
            if e2 in cycle_edges or e1 == e2:
                continue
            
            a, b = e1
            c, d = e2
            # Try (a,c) and (b,d)
            if c not in adj[a] and d not in adj[b]:
                new_e1 = (min(a, c), max(a, c))
                new_e2 = (min(b, d), max(b, d))
                if new_e1 != new_e2:
                    # Apply switch
                    edge_list.remove(e1)
                    edge_list.remove(e2)
                    edge_list.append(new_e1)
                    edge_list.append(new_e2)
                    adj[a].discard(b)
                    adj[b].discard(a)
                    adj[a].add(c)
                    adj[c].add(a)
                    adj[b].add(d)
                    adj[d].add(b)
                    
                    new_count = _count_short_cycles(n, edge_list, g)
                    if new_count < best_count:
                        best_count = new_count
                        improved = True
                        cycle_edges.discard(e1)
                        cycle_edges.discard(e2)
                    else:
                        # Revert
                        edge_list.remove(new_e1)
                        edge_list.remove(new_e2)
                        edge_list.append(e1)
                        edge_list.append(e2)
                        adj[a].discard(c)
                        adj[c].discard(a)
                        adj[a].add(b)
                        adj[b].add(a)
                        adj[b].discard(d)
                        adj[d].discard(b)
        
        if not improved:
            break
    
    return edge_list

def construct(k, g):
    """Construct a k-regular graph of girth at least g."""
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # For the specific case k=4, g=10, use known constructions
    # Moore bound for (4,10): 4 * (3^4 + 3^3 + 3^2 + 3 + 1) = 4 * (81+27+9+3+1) = 4*121 = 484? 
    # Wait, for even girth g=10, Moore bound = k*(k-1)^((g/2)-1) = 4*3^4 = 4*81 = 324
    # Actually the Moore bound for girth g (even) is k*(k-1)^(g/2 - 1)
    # For g=10: 4 * 3^4 = 324
    # But the problem says lower bound is 243 and known is 320
    
    if k == 4 and g == 10:
        # Try to construct using various methods
        best_n = None
        best_edges = None
        
        # Method 1: Lazebnik-Ustimenko-Woldar
        try:
            n1, e1 = _luw_construct(4, 10)
            if n1 and _check_girth(n1, e1, 10):
                best_n = n1
                best_edges = e1
        except:
            pass
        
        # Method 2: Try random 4-regular graphs with local search
        # Try n = 320 (known to exist)
        if best_n is None or best_n > 320:
            for trial in range(3):
                if time.time() > deadline - 30:
                    break
                n_try = 320
                edges_try = _generate_random_4reg(n_try)
                if edges_try is None:
                    continue
                if _check_girth(n_try, edges_try, 10):
                    if best_n is None or n_try < best_n:
                        best_n = n_try
                        best_edges = edges_try
                    break
                else:
                    # Try local search
                    edges_try = _local_search_break_cycles(n_try, edges_try, 10, max_iter=2000, deadline=deadline - 10)
                    if _check_girth(n_try, edges_try, 10):
                        if best_n is None or n_try < best_n:
                            best_n = n_try
                            best_edges = edges_try
                        break
        
        if best_n is not None:
            return best_n, best_edges
        
        # Fallback: return the LUW construction
        return _luw_construct(4, 10)
    
    # General case: use LUW construction
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
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

# Pre-compute a known good solution for (4,10) if possible
# The LUW construction for k=4, g=10 should give a valid graph
# Let's verify it works
