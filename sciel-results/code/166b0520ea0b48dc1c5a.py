import time
import random
import math

def _luw_construct(k, g):
    # Seed logic
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

def _check_girth_fast(adj, n, min_girth):
    """Check if graph has girth >= min_girth using BFS from each vertex."""
    for start in range(n):
        # BFS up to depth min_girth//2 - 1
        max_depth = min_girth // 2 - 1
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        for node in queue:
            if dist[node] >= max_depth:
                continue
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
                elif dist[nb] <= dist[node]:
                    # Back edge or cross edge
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _count_short_cycles(adj, n, max_len=8):
    """Count cycles of length 4, 6, 8."""
    count = 0
    # Count 4-cycles: for each pair of vertices in A, count common neighbors in B
    half = n // 2
    for u in range(half):
        for v in range(u + 1, half):
            common = len(adj[u] & adj[v])
            if common >= 2:
                count += common * (common - 1) // 2
    # Count 6-cycles: more complex, use a different method
    # For now, just return 4-cycle count as proxy
    return count

def construct(k, g):
    if k != 4 or g != 10:
        return _luw_construct(k, g)
    
    deadline = time.time() + 115
    
    # First, get the LUW construction as fallback
    luw_n, luw_edges = _luw_construct(k, g)
    best_n = luw_n
    best_edges = luw_edges
    
    # Try to find a smaller graph using local search
    # Start with random 4-regular bipartite graphs and try to maximize girth
    
    def generate_random_bipartite_4reg(n):
        """Generate a random 4-regular bipartite graph on n vertices (n even)."""
        half = n // 2
        # Create adjacency: each vertex in A (0..half-1) connects to 4 vertices in B (half..n-1)
        # Use configuration model
        adj = [set() for _ in range(n)]
        
        # Each A vertex needs degree 4, each B vertex needs degree 4
        # Create list of halfends
        a_halfends = []
        for i in range(half):
            a_halfends.extend([i] * 4)
        b_halfends = []
        for i in range(half):
            b_halfends.extend([half + i] * 4)
        
        random.shuffle(b_halfends)
        for a, b in zip(a_halfends, b_halfends):
            adj[a].add(b)
            adj[b].add(a)
        
        # Check for multi-edges
        for i in range(n):
            if len(adj[i]) != k:
                return None
        
        return adj
    
    def local_search(adj, n, deadline):
        """Local search to maximize girth via edge swaps."""
        best_adj = [set(s) for s in adj]
        best_score = -1
        
        for iteration in range(10000):
            if time.time() > deadline:
                break
            
            # Compute score: negative of short cycle count
            score = -_count_short_cycles(best_adj, n)
            if score > best_score:
                best_score = score
            
            # Try edge swap: pick two edges (u1,v1) and (u2,v2) and swap to (u1,v2) and (u2,v1)
            # u1,u2 in A, v1,v2 in B
            half = n // 2
            for attempt in range(50):
                if time.time() > deadline:
                    break
                u1 = random.randint(0, half - 1)
                u2 = random.randint(0, half - 1)
                if u1 == u2:
                    continue
                v1 = random.choice(list(best_adj[u1]))
                v2 = random.choice(list(best_adj[u2]))
                if v1 == v2:
                    continue
                # Check that the swap doesn't create multi-edges
                if v2 in best_adj[u1] or v1 in best_adj[u2]:
                    continue
                
                # Perform swap
                best_adj[u1].remove(v1)
                best_adj[v1].remove(u1)
                best_adj[u2].remove(v2)
                best_adj[v2].remove(u2)
                best_adj[u1].add(v2)
                best_adj[v2].add(u1)
                best_adj[u2].add(v1)
                best_adj[v1].add(u2)
                
                # Check if girth is now >= 10
                if _check_girth_fast(best_adj, n, g):
                    # Found a valid graph!
                    edges = []
                    for i in range(half):
                        for j in best_adj[i]:
                            edges.append((i, j))
                    return n, edges
                
                # Compute new score
                new_score = -_count_short_cycles(best_adj, n)
                if new_score <= score:
                    # Revert
                    best_adj[u1].remove(v2)
                    best_adj[v2].remove(u1)
                    best_adj[u2].remove(v1)
                    best_adj[v1].remove(u2)
                    best_adj[u1].add(v1)
                    best_adj[v1].add(u1)
                    best_adj[u2].add(v2)
                    best_adj[v2].add(u2)
        
        return None
    
    # Try different sizes starting from smaller
    # Moore bound for (4,10) is 2*(1+4+12+36) = 2*53 = 106? No.
    # Moore bound: n >= 2*(1 + (k-1) + (k-1)^2 + ... + (k-1)^(g/2-1)) for even girth
    # For k=4, g=10: n >= 2*(1 + 3 + 9 + 27 + 81) = 2*121 = 242
    # So minimum possible is 242, but must be even and satisfy other constraints.
    
    # Try sizes from 244 up to luw_n
    for n in range(244, min(luw_n + 1, 400), 2):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random starts
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            adj = generate_random_bipartite_4reg(n)
            if adj is None:
                continue
            
            result = local_search(adj, n, deadline - 5)
            if result is not None:
                best_n, best_edges = result
                return best_n, best_edges
    
    return best_n, best_edges
