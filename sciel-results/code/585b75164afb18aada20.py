"""C6-free graph construction on 40 vertices via greedy + local search."""
import time
import random
from itertools import combinations

N = 40
DEADLINE = 150.0

def check_c6(adj, u, v):
    """Check if adding edge (u,v) to graph with adjacency matrix adj creates a C6."""
    # We need to find vertices a,b,c,d such that:
    # u-a-b-c-d-v-u is a 6-cycle
    # i.e., a~u, a~b, b~c, c~d, d~v, and u,v not adjacent yet
    # We need a,b,c,d all distinct and not u or v.
    
    Nu = [x for x in range(N) if x != u and x != v and adj[u][x]]
    Nv = [x for x in range(N) if x != u and x != v and adj[v][x]]
    
    for a in Nu:
        if adj[a][v]:  # a-v edge would make a triangle, not C6, but let's be safe
            continue
        for b in range(N):
            if b in (u, v, a):
                continue
            if not adj[a][b]:
                continue
            if adj[b][u] or adj[b][v]:
                continue
            for c in range(N):
                if c in (u, v, a, b):
                    continue
                if not adj[b][c]:
                    continue
                if adj[c][u] or adj[c][v]:
                    continue
                for d in range(N):
                    if d in (u, v, a, b, c):
                        continue
                    if not adj[c][d]:
                        continue
                    if d not in Nv:
                        continue
                    if adj[d][u]:
                        continue
                    # Check: u-a-b-c-d-v-u forms a simple cycle
                    # We need d~v (checked via d in Nv)
                    # All distinct: u,v,a,b,c,d are all different (ensured by checks)
                    # Edges: u-a (a in Nu), a-b (checked), b-c (checked), c-d (checked), d-v (d in Nv)
                    return True
    return False


def check_c6_fast(adj, u, v):
    """Faster C6 check using bitmasks for adjacency."""
    # adj is a list of Python ints (bitmasks)
    Nu = adj[u] & ~(1 << u) & ~(1 << v)
    Nv = adj[v] & ~(1 << u) & ~(1 << v)
    
    # Iterate over possible a in Nu
    a_mask = Nu
    while a_mask:
        a_bit = a_mask & (-a_mask)
        a = a_bit.bit_length() - 1
        a_mask -= a_bit
        
        # a must not be adjacent to v (would create shorter cycle or triangle)
        if adj[a] & (1 << v):
            continue
        
        Na = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
        # b must be in Na, not adjacent to u or v
        b_mask = Na & ~adj[u] & ~adj[v]
        while b_mask:
            b_bit = b_mask & (-b_mask)
            b = b_bit.bit_length() - 1
            b_mask -= b_bit
            
            Nb = adj[b] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
            # c must be in Nb
            c_mask = Nb
            while c_mask:
                c_bit = c_mask & (-c_mask)
                c = c_bit.bit_length() - 1
                c_mask -= c_bit
                
                Nc = adj[c] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                # d must be in Nc and also in Nv
                d_mask = Nc & Nv
                if d_mask:
                    return True
    return False


def greedy_construct(seed, max_edges=None):
    """Greedy construction with random edge ordering."""
    rng = random.Random(seed)
    
    # Initialize empty graph
    adj = [0] * N
    
    # Generate all possible edges
    edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    rng.shuffle(edges)
    
    edge_set = set()
    for u, v in edges:
        if max_edges is not None and len(edge_set) >= max_edges:
            break
        if not check_c6_fast(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add((u, v))
    
    return list(edge_set), adj


def local_search(edges, adj, seed, time_limit=5.0):
    """Try to improve the graph by swapping edges or adding edges."""
    rng = random.Random(seed)
    start_time = time.time()
    best_edges = list(edges)
    best_adj = [x for x in adj]
    
    # Build edge set for quick lookup
    edge_set = set(best_edges)
    
    # Try adding random edges
    all_possible = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    iterations = 0
    while time.time() - start_time < time_limit:
        iterations += 1
        u, v = rng.choice(all_possible)
        if (u, v) in edge_set:
            continue
        if check_c6_fast(best_adj, u, v):
            continue
        # Add the edge
        best_adj[u] |= (1 << v)
        best_adj[v] |= (1 << u)
        best_edges.append((u, v))
        edge_set.add((u, v))
    
    return best_edges, best_adj


def construct():
    start_time = time.time()
    best_edges = []
    best_count = 0
    
    # Start with a good initial structure: multiple K5s
    # K5 has 10 edges, 5 vertices. 8 K5s = 40 vertices, 80 edges.
    # But we can do better by connecting them.
    
    # Strategy: Greedy with multiple restarts, then local search
    for seed in range(200):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
        
        edges, adj = greedy_construct(seed)
        count = len(edges)
        
        if count > best_count:
            best_count = count
            best_edges = list(edges)
        
        # Local search if we have time
        if count >= best_count and time.time() - start_time < DEADLINE - 15:
            edges2, adj2 = local_search(edges, adj, seed, time_limit=2.0)
            count2 = len(edges2)
            if count2 > best_count:
                best_count = count2
                best_edges = list(edges2)
    
    # Final local search on the best solution
    if time.time() - start_time < DEADLINE - 5 and len(best_edges) > 0:
        # Rebuild adjacency
        adj = [0] * N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        best_edges, adj = local_search(best_edges, adj, 999, time_limit=DEADLINE - time.time() - 2.0)
        best_count = len(best_edges)
    
    # Return edges as [u, v] pairs
    return [[u, v] for u, v in best_edges]
