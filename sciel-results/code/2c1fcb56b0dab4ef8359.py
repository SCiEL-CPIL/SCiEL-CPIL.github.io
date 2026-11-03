import time
import random
from collections import deque
import sys

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (u, v) on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if k == 4 and g == 10:
        # 1. Try tensor product + local search
        result = _try_tensor_local_search(deadline)
        if result is not None:
            return result
        
        # 2. Try Cayley graph on Z_16 x Z_20
        result = _try_cayley_320(deadline)
        if result is not None:
            return result

    # Fallback: LUW Construction
    return _luw_construct(k, g)

def _try_cayley_320(deadline):
    m, n = 16, 20
    total = m * n
    
    def check_girth(x, y):
        gens = [
            (1, 0),
            (m-1, 0),
            (x, y),
            ((-x)%m, (-y)%n)
        ]
        
        visited = {}
        visited[(0,0)] = 0
        queue = deque([(0,0)])
        
        while queue:
            u, v = queue.popleft()
            depth = visited[(u,v)]
            
            for g1, g2 in gens:
                nu = (u + g1) % m
                nv = (v + g2) % n
                if (nu, nv) not in visited:
                    visited[(nu, nv)] = depth + 1
                    queue.append((nu, nv))
                else:
                    cycle_len = depth + 1 + visited[(nu, nv)]
                    if cycle_len < 10:
                        return False
        return True

    for x in range(m):
        if time.time() > deadline:
            return None
        for y in range(n):
            if time.time() > deadline:
                return None
            
            if x == 0 and y == 0: continue
            if x == 1 and y == 0: continue
            if x == m-1 and y == 0: continue
            if (2*x) % m == 0 and (2*y) % n == 0: continue
            
            if check_girth(x, y):
                edges = []
                for u in range(m):
                    for v in range(n):
                        nu = (u + 1) % m
                        nv = v
                        if (u, v) < (nu, nv):
                            edges.append(((u*m+v, nu*m+nv)))
                        
                        nu = (u + x) % m
                        nv = (v + y) % n
                        if (u, v) < (nu, nv):
                            edges.append(((u*m+v, nu*m+nv)))
                
                deg = [0]*total
                for e1, e2 in edges:
                    deg[e1] += 1
                    deg[e2] += 1
                if all(d == 4 for d in deg):
                    return total, edges

    return None

def _try_tensor_local_search(deadline):
    """
    Use tensor product of two cycles to create 4-regular bipartite graph,
    then perform edge-switching local search to eliminate short cycles.
    """
    # Tensor product of C_L x C_L gives 4-regular graph on 4*L^2 vertices
    # For 320 vertices: 4*L^2 = 320 => L^2 = 80, not integer
    # Try L=9: 4*81 = 324 vertices (close to 320)
    # Try L=8: 4*64 = 256 vertices (too few, can't have girth 10 with <243 vertices)
    # Actually Moore bound for (4,10) is 243, so 256 might work
    # But we want as few as possible, known optimum is 320
    
    # Let's try different approaches:
    # 1. Start with random 4-regular graph on 320 vertices
    # 2. Use edge switching to eliminate short cycles
    
    n = 320
    rng = random.Random(42)
    
    # Generate random 4-regular graph
    def generate_random_4regular(n, rng):
        """Generate a random 4-regular graph on n vertices."""
        # Use configuration model
        half_edges = [v for v in range(n) for _ in range(4)]
        rng.shuffle(half_edges)
        
        edges = set()
        edge_list = []
        for i in range(0, len(half_edges), 2):
            u, v = half_edges[i], half_edges[i+1]
            if u == v:
                # Self-loop, swap with another
                for j in range(i+2, len(half_edges), 2):
                    if half_edges[j] != u:
                        half_edges[i+1], half_edges[j] = half_edges[j], half_edges[i+1]
                        v = half_edges[i+1]
                        break
            if u < v:
                edge = (u, v)
            else:
                edge = (v, u)
            if edge not in edges:
                edges.add(edge)
                edge_list.append(edge)
        
        # Check if all vertices have degree 4
        deg = [0] * n
        for u, v in edge_list:
            deg[u] += 1
            deg[v] += 1
        if all(d == 4 for d in deg):
            return edge_list
        
        # Fallback: use union of cycles
        # Create 2 disjoint 2-regular graphs (cycles) and union them
        # This gives 4-regular
        cycle1 = list(range(n))
        cycle2 = list(range(n))
        rng.shuffle(cycle1)
        rng.shuffle(cycle2)
        
        edges = []
        for i in range(n):
            u, v = cycle1[i], cycle1[(i+1) % n]
            if u < v:
                edges.append((u, v))
            else:
                edges.append((v, u))
        
        for i in range(n):
            u, v = cycle2[i], cycle2[(i+1) % n]
            if u < v:
                edges.append((u, v))
            else:
                edges.append((v, u))
        
        # Remove duplicates
        seen = set()
        unique_edges = []
        for e in edges:
            if e not in seen:
                seen.add(e)
                unique_edges.append(e)
        
        return unique_edges
    
    # Check girth using BFS from each vertex
    def check_girth(edges, n, g):
        """Check if graph has girth at least g using non-backtracking BFS."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            # BFS tracking (vertex, prev) to avoid backtracking
            visited = {}
            queue = deque()
            for nb in adj[start]:
                visited[(start, nb)] = 1
                queue.append((start, nb, 1))
            
            while queue:
                prev, curr, depth = queue.popleft()
                if depth >= g:
                    continue
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    if (curr, nb) not in visited:
                        visited[(curr, nb)] = depth + 1
                        queue.append((curr, nb, depth + 1))
                    else:
                        cycle_len = depth + 1 + visited[(curr, nb)]
                        if cycle_len < g:
                            return False
        return True
    
    # Count short cycles (for optimization)
    def count_short_cycles(edges, n, max_len):
        """Count cycles of length < max_len."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        count = 0
        for start in range(n):
            # BFS up to depth max_len-1
            visited = {}
            queue = deque()
            for nb in adj[start]:
                visited[(start, nb)] = 1
                queue.append((start, nb, 1))
            
            while queue:
                prev, curr, depth = queue.popleft()
                if depth >= max_len - 1:
                    continue
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    if nb == start:
                        count += 1
                    elif (curr, nb) not in visited:
                        visited[(curr, nb)] = depth + 1
                        queue.append((curr, nb, depth + 1))
        return count // 2  # Each cycle counted twice
    
    # Edge switching: swap edges (u,v) and (w,x) with (u,w) and (v,x)
    def try_edge_switch(edges, n, g, rng):
        """Try a random edge switch and return new edges if it improves girth."""
        if len(edges) < 2:
            return None
        
        # Pick two random edges
        i, j = rng.sample(range(len(edges)), 2)
        u, v = edges[i]
        w, x = edges[j]
        
        # Ensure edges are disjoint
        if len({u, v, w, x}) < 4:
            return None
        
        # Check if new edges exist
        edge_set = set(edges)
        if (min(u,w), max(u,w)) in edge_set or (min(v,x), max(v,x)) in edge_set:
            return None
        
        # Create new edges
        new_edges = list(edges)
        new_edges[i] = (min(u,w), max(u,w))
        new_edges[j] = (min(v,x), max(v,x))
        
        # Check if this improves things
        # Quick check: does it create a short cycle?
        adj = [[] for _ in range(n)]
        for a, b in new_edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # Check degrees
        for node in range(n):
            if len(adj[node]) != 4:
                return None
        
        # Check for very short cycles (4-cycles)
        # A 4-cycle exists if there's a path of length 3 from u to x or v to w
        def has_short_cycle(adj, n, max_len=6):
            for start in range(n):
                visited = set()
                queue = deque([(start, -1, 0)])
                while queue:
                    curr, prev, depth = queue.popleft()
                    if depth >= max_len:
                        continue
                    for nb in adj[curr]:
                        if nb == prev:
                            continue
                        if nb == start and depth + 1 < max_len:
                            return True
                        if (curr, nb) not in visited:
                            visited.add((curr, nb))
                            queue.append((nb, curr, depth + 1))
            return False
        
        if has_short_cycle(adj, n, 8):
            return None
        
        return new_edges
    
    # Main local search
    best_edges = generate_random_4regular(n, rng)
    best_score = count_short_cycles(best_edges, n, 10)
    
    for attempt in range(100):
        if time.time() > deadline - 10:
            break
        
        current_edges = generate_random_4regular(n, rng)
        current_score = count_short_cycles(current_edges, n, 10)
        
        if current_score < best_score:
            best_edges = current_edges
            best_score = current_score
        
        # Local search from this starting point
        for step in range(500):
            if time.time() > deadline - 5:
                break
            
            new_edges = try_edge_switch(current_edges, n, 10, rng)
            if new_edges is None:
                continue
            
            new_score = count_short_cycles(new_edges, n, 10)
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
            
            if current_score == 0:
                break
        
        if current_score < best_score:
            best_edges = current_edges
            best_score = current_score
        
        if best_score == 0:
            break
    
    if best_score == 0:
        # Verify girth
        if check_girth(best_edges, n, 10):
            return n, best_edges
    
    return None

def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)

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
