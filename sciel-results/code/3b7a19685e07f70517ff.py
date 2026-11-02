import numpy as np
import time

def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph from seed."""
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

def _check_girth(n, edges, g):
    """
    Checks if the graph has girth >= g.
    Returns True if valid, False if a cycle shorter than g is found.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # We only need to detect cycles of length < g.
    # Max depth for BFS to detect cycle of length L is floor((L-1)/2).
    # So max depth is floor((g-2)/2).
    max_depth = (g - 2) // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            if dist[u] >= max_depth:
                continue
                
            for w in adj[u]:
                if w == parent[u]:
                    continue
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    queue.append(w)
                else:
                    # Found a cycle
                    # The cycle length is dist[u] + dist[w] + 1
                    # This works because the tree edges form a path from start to u and start to w.
                    # The common ancestor is the LCA, but since we are in a tree (BFS tree),
                    # the path u->...->start->...->w is a cycle.
                    # Wait, is it always a simple cycle?
                    # In a BFS tree, if we encounter a visited node w that is not the parent,
                    # the path u->parent(u)->...->LCA(u,w)->...->w->u is a cycle.
                    # Length is dist[u] + dist[w] - 2*dist[LCA] + 1.
                    # However, a simpler check: if we just want to know if ANY cycle < g exists,
                    # and we stop expanding at max_depth, we might miss the LCA depth.
                    # But note: if a cycle of length L exists, there is a vertex on that cycle
                    # such that the distance from the start to that vertex is <= L/2.
                    # Actually, the standard BFS girth check:
                    # If we find an edge (u, w) where w is visited and w != parent[u],
                    # then we have a cycle of length dist[u] + dist[w] + 1.
                    # This is correct for the cycle formed by the two tree paths.
                    cycle_len = dist[u] + dist[w] + 1
                    if cycle_len < g:
                        return False
                        
    return True

def _generate_random_3_reg(n):
    """
    Generates a random simple 3-regular graph on n vertices.
    Returns list of edges or None if failed to generate simple graph.
    """
    if n < 4 or n % 2 != 0:
        return None
        
    # Create stubs
    stubs = np.repeat(np.arange(n), 3)
    
    for _ in range(100): # Max retries
        np.random.shuffle(stubs)
        edges = []
        valid = True
        edge_set = set()
        
        for i in range(0, len(stubs), 2):
            u = int(stubs[i])
            v = int(stubs[i+1])
            
            if u == v:
                valid = False
                break
                
            if u > v:
                u, v = v, u
                
            if (u, v) in edge_set:
                valid = False
                break
                
            edge_set.add((u, v))
            edges.append((u, v))
            
        if valid:
            return edges
            
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # Fallback for k != 3 or if random search fails
    def fallback():
        return _luw_construct(k, g)
        
    if k != 3:
        return fallback()
        
    # Moore bound for k=3, g=16 is 512 (as per prompt).
    # General lower bound for even g=2r: 3 * (2^r - 1)
    # For g=16, r=8, 3 * 255 = 765.
    # But prompt says 512. I will trust the prompt's "best lower bound".
    # I will start from 512 to be safe and potentially better.
    
    n_min = 512
    best_n = None
    best_edges = None
    
    n = n_min
    # Limit search space to avoid wasting time if 512 is impossible
    # Random graphs should work for n >= ~256 for g=16
    while n < 2000:
        if time.time() > deadline - 5:
            break
            
        # Try to find a valid graph for this n
        # Number of trials per n
        trials = 500
        found = False
        
        for _ in range(trials):
            if time.time() > deadline - 5:
                break
                
            edges = _generate_random_3_reg(n)
            if edges is None:
                continue
                
            if _check_girth(n, edges, g):
                best_n = n
                best_edges = edges
                found = True
                break
                
        if found:
            break
            
        n += 2
        
    if best_n is not None:
        return best_n, best_edges
    else:
        return fallback()

# EVOLVE-BLOCK-END
