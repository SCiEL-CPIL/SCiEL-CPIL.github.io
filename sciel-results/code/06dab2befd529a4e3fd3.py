"""Construct a C4-free graph with no independent set of size 11, maximizing n."""
import time
import itertools
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _has_c4(n, adj):
    """Check if graph has a C4 using bitmasks."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):
                return True
    return False

def _max_independent_set_size(n, adj, target=11):
    """Check if there exists an independent set of size >= target."""
    # Use a branch-and-bound with coloring bound
    # First, compute a greedy coloring of the complement graph to get upper bounds
    
    def greedy_color_bound(vertices, adj):
        """Get an upper bound on independence number using greedy coloring of complement."""
        if not vertices:
            return 0
        # Greedy coloring: assign colors so that adjacent vertices (in original graph)
        # get different colors. The number of colors is an upper bound on alpha.
        color = [0] * n
        color_count = 0
        for v in vertices:
            # Find the smallest color not used by neighbors of v in the subgraph
            used = set()
            for u in vertices:
                if u != v and (adj[v] >> u) & 1 and color[u] > 0:
                    used.add(color[u])
            c = 1
            while c in used:
                c += 1
            color[v] = c
            color_count = max(color_count, c)
        return color_count
    
    best = [0]
    
    def search(vertices, current_size):
        if current_size >= target:
            return True
        if not vertices:
            return False
        if current_size + len(vertices) <= best[0]:
            return False
        
        # Upper bound using greedy coloring
        ub = greedy_color_bound(vertices, adj)
        if current_size + ub <= best[0]:
            return False
        
        # Pick vertex with highest degree in the induced subgraph
        v = max(vertices, key=lambda x: sum(1 for u in vertices if u != x and (adj[x] >> u) & 1))
        rest = [u for u in vertices if u != v]
        
        # Option 1: include v
        new_rest = [u for u in rest if not ((adj[v] >> u) & 1)]
        if search(new_rest, current_size + 1):
            return True
        
        # Option 2: exclude v
        if search(rest, current_size):
            return True
        
        return False
    
    return search(list(range(n)), 0)

def _verify_graph(n, edges):
    """Verify the graph is C4-free and has no independent set of size 11."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    if _has_c4(n, adj):
        return False
    
    if _max_independent_set_size(n, adj, 11):
        return False
    
    return True

def _build_polarity_graph_q5():
    """
    Build the polarity graph of PG(2,5) - a known C4-free graph on 31 vertices.
    PG(2,5) has 5^2 + 5 + 1 = 31 points.
    """
    q = 5
    n = q * q + q + 1  # 31
    
    # Points in PG(2, q) can be represented as (x, y, z) up to scaling
    # We use homogeneous coordinates: (x, y, 1) for affine points, (x, y, 0) for points at infinity
    # Total: q^2 affine + q points at infinity in one direction + 1 point at infinity in another = q^2 + q + 1
    
    points = []
    # Affine points: (x, y, 1) for x, y in F_q
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    # Points at infinity: (x, 1, 0) for x in F_q, and (0, 1, 0) is already covered
    # Actually, points at infinity are (x, y, 0) with (x,y) not both 0, up to scaling
    # So we have (1, 0, 0), (1, 1, 0), ..., (1, q-1, 0) -> q points
    # Plus (0, 1, 0) but that's same as (1, 0, 0) scaled? No, (0,1,0) is different from (1,0,0)
    # Wait, (x, y, 0) with (x,y) != (0,0). Up to scaling, these are (1, 0, 0), (1, 1, 0), ..., (1, q-1, 0)
    # That's q points. But we need q+1 points at infinity for total q^2 + q + 1.
    # Hmm, let me reconsider. PG(2,q) has q^2 + q + 1 points.
    # Affine: q^2 points (x,y,1)
    # Infinity: (x,y,0) with (x,y) != (0,0), up to scaling -> q+1 points: (1,0,0), (1,1,0), ..., (1,q-1,0), (0,1,0)
    # Wait, (0,1,0) is not a multiple of any (1,i,0). So yes, q+1 points at infinity.
    
    for i in range(q):
        points.append((1, i, 0))
    points.append((0, 1, 0))
    
    # Now, two points are adjacent if they are orthogonal w.r.t. a polarity
    # For the standard polarity in PG(2,q), point P=(x,y,z) is orthogonal to Q=(x',y',z')
    # if x*x' + y*y' + z*z' = 0 in F_q (for a specific choice)
    
    # Actually, let's use a different approach. The polarity graph of PG(2,q) for prime q
    # can be constructed as follows: vertices are points of PG(2,q), and two points are
    # adjacent if they are conjugate w.r.t. a fixed polarity.
    
    # For simplicity, let's use the known fact that the polarity graph of PG(2,5) is C4-free
    # and has 31 vertices. We'll construct it using the standard polarity.
    
    adj = [0] * n
    edges = []
    
    def dot(p, q):
        return (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % q
    
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edges.append([i, j])
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    return n, edges, adj

def _greedy_construct_c4free(n_target, seed=42):
    """Greedy construction of a C4-free graph on n_target vertices."""
    random.seed(seed)
    adj = [0] * n_target
    edges = []
    
    all_pairs = list(itertools.combinations(range(n_target), 2))
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        common = adj[u] & adj[v]
        if common & (common - 1):
            continue
        edges.append([u, v])
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    return n_target, edges, adj

def _local_search_alpha(n, adj, edges, deadline):
    """Local search to minimize independence number while maintaining C4-free property."""
    best_alpha_result = _max_independent_set_size(n, adj, 11)
    best_adj = adj[:]
    best_edges = edges[:]
    
    while time.time() < deadline:
        improved = False
        
        for _ in range(200):
            if time.time() > deadline:
                break
            
            u, v = random.sample(range(n), 2)
            if (adj[u] >> v) & 1:
                continue
            
            common = adj[u] & adj[v]
            if common & (common - 1):
                continue
            
            new_adj = adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            new_alpha = _max_independent_set_size(n, new_adj, 11)
            if new_alpha < best_alpha_result:
                best_alpha_result = new_alpha
                best_adj = new_adj
                best_edges = edges + [[u, v]]
                adj = new_adj
                improved = True
                if not new_alpha:
                    return best_adj, best_edges
        
        if not improved:
            # Try removing an edge and re-adding others
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                if not edges:
                    break
                
                idx = random.randint(0, len(edges) - 1)
                u, v = edges[idx]
                
                new_adj = adj[:]
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                
                new_edges = edges[:idx] + edges[idx+1:]
                
                # Try to add back some edges
                all_pairs = list(itertools.combinations(range(n), 2))
                random.shuffle(all_pairs)
                
                for a, b in all_pairs:
                    if time.time() > deadline:
                        break
                    if (new_adj[a] >> b) & 1:
                        continue
                    common = new_adj[a] & new_adj[b]
                    if common & (common - 1):
                        continue
                    new_adj[a] |= (1 << b)
                    new_adj[b] |= (1 << a)
                    new_edges.append([a, b])
                
                new_alpha = _max_independent_set_size(n, new_adj, 11)
                if new_alpha < best_alpha_result:
                    best_alpha_result = new_alpha
                    best_adj = new_adj
                    best_edges = new_edges
                    adj = new_adj
                    edges = new_edges
                    improved = True
                    if not new_alpha:
                        return best_adj, best_edges
    
    return best_adj, best_edges

def _try_construction():
    """Try to find a C4-free graph with no independent set of size 11."""
    start_time = time.time()
    deadline = start_time + 110
    
    best_n = 0
    best_edges = []
    best_adj = None
    
    # Try starting from the polarity graph of PG(2,5) and extending
    n0, edges0, adj0 = _build_polarity_graph_q5()
    
    # Try to extend the polarity graph by adding vertices
    for extra in range(0, 13):
        n_try = n0 + extra
        if n_try > 43:
            break
        
        if time.time() > deadline:
            break
        
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 1000 + n_try)
            
            # Start with the polarity graph and add extra vertices
            adj = adj0[:]
            edges = edges0[:]
            
            # Add new vertices
            for new_v in range(n0, n_try):
                adj.append(0)
                # Connect new vertex to some existing vertices
                candidates = list(range(n0))
                random.shuffle(candidates)
                for v in candidates:
                    if time.time() > deadline:
                        break
                    common = adj[new_v] & adj[v]
                    if common & (common - 1):
                        continue
                    # Also check that adding this edge doesn't create C4
                    # The common check above should suffice for C4-free
                    edges.append([new_v, v])
                    adj[new_v] |= (1 << v)
                    adj[v] |= (1 << new_v)
            
            # Now try to add more edges between existing vertices
            all_pairs = list(itertools.combinations(range(n_try), 2))
            random.shuffle(all_pairs)
            
            for u, v in all_pairs:
                if time.time() > deadline:
                    break
                if (adj[u] >> v) & 1:
                    continue
                common = adj[u] & adj[v]
                if common & (common - 1):
                    continue
                edges.append([u, v])
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            alpha = _max_independent_set_size(n_try, adj, 11)
            
            if not alpha:
                if n_try > best_n:
                    best_n = n_try
                    best_edges = edges
                    best_adj = adj
                    if n_try >= 39:
                        return {'n': best_n, 'edges': best_edges}
            else:
                adj, edges = _local_search_alpha(n_try, adj, edges, deadline)
                alpha = _max_independent_set_size(n_try, adj, 11)
                if not alpha:
                    if n_try > best_n:
                        best_n = n_try
                        best_edges = edges
                        best_adj = adj
                        if n_try >= 39:
                            return {'n': best_n, 'edges': best_edges}
        
        if best_n >= n_try:
            break
    
    # Also try pure random greedy for various n
    for n_try in [43, 42, 41, 40, 39, 38, 37]:
        if time.time() > deadline:
            break
        if best_n >= n_try:
            break
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 2000 + n_try)
            
            n, edges, adj = _greedy_construct_c4free(n_try, seed=attempt * 2000 + n_try)
            
            alpha = _max_independent_set_size(n, adj, 11)
            
            if not alpha:
                if n_try > best_n:
                    best_n = n_try
                    best_edges = edges
                    best_adj = adj
                    if n_try >= 39:
                        return {'n': best_n, 'edges': best_edges}
            else:
                adj, edges = _local_search_alpha(n, adj, edges, deadline)
                alpha = _max_independent_set_size(n, adj, 11)
                if not alpha:
                    if n_try > best_n:
                        best_n = n_try
                        best_edges = edges
                        best_adj = adj
                        if n_try >= 39:
                            return {'n': best_n, 'edges': best_edges}
    
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: return the polarity graph (31 vertices, C4-free, alpha < 11 hopefully)
    return {'n': n0, 'edges': edges0}

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    start_time = time.time()
    
    result = _try_construction()
    
    if _verify_graph(result['n'], result['edges']):
        return result
    
    # Fallback: return a small valid graph
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# EVOLVE-BLOCK-END
