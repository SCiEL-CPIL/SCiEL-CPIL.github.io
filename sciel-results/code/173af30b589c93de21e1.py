import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Fallback: L-U-W Construction
    def construct_luw(k, g):
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

    luw_n, luw_edges = construct_luw(k, g)
    
    # Fast girth check using non-backtracking walks
    def has_short_cycle(adj, max_len):
        """Check if there's a cycle of length < max_len using non-backtracking BFS."""
        n = len(adj)
        # adj_list for fast neighbor lookup
        adj_list = [list(s) for s in adj]
        
        for start in range(n):
            # Non-backtracking walk: state is (current_node, prev_node, depth)
            visited = set()  # (node, prev) pairs
            queue = deque()
            for nb in adj_list[start]:
                visited.add((nb, start))
                queue.append((nb, start, 1))
            
            while queue:
                node, prev, depth = queue.popleft()
                if depth >= max_len:
                    continue
                for nb in adj_list[node]:
                    if nb == prev:
                        continue
                    if nb == start and depth + 1 < max_len:
                        return True
                    if (nb, node) not in visited:
                        visited.add((nb, node))
                        queue.append((nb, node, depth + 1))
        return False

    def find_short_cycle(adj, max_len):
        """Find a specific short cycle and return its edges."""
        n = len(adj)
        adj_list = [list(s) for s in adj]
        
        for start in range(n):
            # BFS with parent tracking for non-backtracking walks
            # State: (node, prev) -> (parent_state, depth)
            parent = {}
            visited = set()
            queue = deque()
            for nb in adj_list[start]:
                visited.add((nb, start))
                parent[(nb, start)] = (start, -1)
                queue.append((nb, start, 1))
            
            while queue:
                node, prev, depth = queue.popleft()
                if depth >= max_len:
                    continue
                for nb in adj_list[node]:
                    if nb == prev:
                        continue
                    if nb == start and depth + 1 < max_len:
                        # Found a cycle! Reconstruct it
                        cycle = []
                        cur = (node, prev)
                        while cur[1] != -1:
                            cycle.append((cur[0], cur[1]))
                            cur = parent[cur]
                        cycle.append((start, node))
                        return cycle
                    if (nb, node) not in visited:
                        visited.add((nb, node))
                        parent[(nb, node)] = (node, prev)
                        queue.append((nb, node, depth + 1))
        return None

    def try_break_cycle(adj, cycle, k):
        """Try to break a cycle by edge swaps. Returns True if successful."""
        n = len(adj)
        if not cycle:
            return False
        
        # Try to break each edge in the cycle
        for i in range(len(cycle)):
            u, v = cycle[i]
            # Find a random non-neighbor w of u and non-neighbor x of v
            # such that we can swap (u,v) with (w,x)
            attempts = 0
            while attempts < 20:
                w = random.randint(0, n - 1)
                if w in adj[u] or w == v:
                    attempts += 1
                    continue
                x = random.randint(0, n - 1)
                if x in adj[v] or x == u or x == w:
                    attempts += 1
                    continue
                if w in adj[v] or x in adj[u] or w == x:
                    attempts += 1
                    continue
                
                # Check if swap is valid
                if w in adj[u] or x in adj[v] or w == x:
                    attempts += 1
                    continue
                
                # Perform the swap
                adj[u].remove(v)
                adj[v].remove(u)
                adj[w].add(x)
                adj[x].add(w)
                adj[u].add(w)
                adj[w].add(u)
                adj[v].add(x)
                adj[x].add(v)
                return True
            attempts += 1
        
        return False

    def generate_random_regular(k, n):
        """Generate a random k-regular graph using configuration model with repair."""
        if n * k % 2 != 0:
            return None
        
        for attempt in range(50):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            
            if valid:
                return adj
            
            # Try to repair by swapping
            # Build edge list
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            
            # Try random edge swaps to fix issues
            for _ in range(100):
                if not edges:
                    break
                e1_idx = random.randint(0, len(edges) - 1)
                u1, v1 = edges[e1_idx]
                if v1 not in adj[u1]:
                    continue
                e2_idx = random.randint(0, len(edges) - 1)
                if e2_idx == e1_idx:
                    continue
                u2, v2 = edges[e2_idx]
                if v2 not in adj[u2]:
                    continue
                
                # Try swap (u1,v1) and (u2,v2) -> (u1,u2) and (v1,v2)
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    continue
                if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                    continue
                
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                
                adj[u1].add(u2)
                adj[u2].add(u1)
                adj[v1].add(v2)
                adj[v2].add(v1)
                
                edges[e1_idx] = (min(u1, u2), max(u1, u2))
                edges[e2_idx] = (min(v1, v2), max(v1, v2))
            
            # Check if valid now
            valid = True
            for u in range(n):
                if len(adj[u]) != k:
                    valid = False
                    break
                for v in adj[u]:
                    if v == u or u not in adj[v]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return adj
        
        return None

    # Try Greedy Search
    def construct_greedy(k, g, deadline):
        # Moore Bound
        mb = 1
        term = 1
        for i in range((g - 3) // 2):
            mb += k * term
            term *= (k - 1)
        
        n_target = max(mb, k + 1)
        best_graph = None
        
        while time.time() < deadline:
            # Generate random k-regular graph on n_target vertices
            adj = generate_random_regular(k, n_target)
            if adj is None:
                n_target += 5
                continue
            
            # Try to increase girth by breaking short cycles
            success = False
            max_iterations = 2000
            for it in range(max_iterations):
                if time.time() > deadline:
                    return None
                
                if not has_short_cycle(adj, g):
                    success = True
                    break
                
                # Find a short cycle and try to break it
                cycle = find_short_cycle(adj, g)
                if cycle is None:
                    # No short cycle found, but has_short_cycle said there is one
                    # This shouldn't happen, but break to avoid infinite loop
                    break
                
                if not try_break_cycle(adj, cycle, k):
                    # Try random swaps as fallback
                    n = len(adj)
                    u = random.randint(0, n - 1)
                    if not adj[u]:
                        break
                    v = random.choice(list(adj[u]))
                    x = random.randint(0, n - 1)
                    if not adj[x]:
                        break
                    y = random.choice(list(adj[x]))
                    
                    if u == x or u == y or v == x or v == y:
                        continue
                    if x in adj[u] or y in adj[u] or x in adj[v] or y in adj[v]:
                        continue
                    
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    adj[u].add(x)
                    adj[x].add(u)
                    adj[v].add(y)
                    adj[y].add(v)
            
            if success:
                edges = []
                for u in range(n_target):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n_target, edges
            
            # If not success, increase n_target
            n_target += 5
            
        return None

    result = construct_greedy(k, g, deadline)
    if result:
        n, edges = result
        if n < luw_n:
            return n, edges
            
    return luw_n, luw_edges
