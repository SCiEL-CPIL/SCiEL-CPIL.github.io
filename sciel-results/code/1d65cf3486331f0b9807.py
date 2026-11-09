"""Hexagon-free graph construction via systematic backtracking with bitmask pruning."""
import time
import random
from itertools import combinations

N = 40
DEADLINE = 130  # seconds, safety margin before 150s kill


def _seed_graph_8K5():
    """8 disjoint K5s: 100 edges, guaranteed C6-free."""
    edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([base + a, base + b])
    return edges


def _greedy_fill(adj_bitmasks, n, deadline):
    """
    Greedy: try to add every missing edge in random order if it doesn't create a C6.
    adj_bitmasks[i] is an int bitmask of neighbors of vertex i.
    Returns (adj_bitmasks, edge_count).
    """
    edge_count = sum(bin(adj_bitmasks[i]).count('1') for i in range(n)) // 2
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if time.time() > deadline:
            break
        if (adj_bitmasks[u] >> v) & 1:
            continue
        # Check if adding (u,v) creates a C6
        # A C6 through (u,v): u-v-w-x-y-z-u
        # Need path of length 4 from v to u avoiding u,v
        found = False
        for w in range(n):
            if w == u or w == v:
                continue
            if not ((adj_bitmasks[v] >> w) & 1):
                continue
            for x in range(n):
                if x == u or x == v or x == w:
                    continue
                if not ((adj_bitmasks[w] >> x) & 1):
                    continue
                for y in range(n):
                    if y == u or y == v or y == w or y == x:
                        continue
                    if not ((adj_bitmasks[x] >> y) & 1):
                        continue
                    for z in range(n):
                        if z == u or z == v or z == w or z == x or z == y:
                            continue
                        if not ((adj_bitmasks[y] >> z) & 1):
                            continue
                        if (adj_bitmasks[z] >> u) & 1:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            adj_bitmasks[u] |= (1 << v)
            adj_bitmasks[v] |= (1 << u)
            edge_count += 1
    
    return adj_bitmasks, edge_count


def _backtracking_max_edges(n, deadline):
    """
    Systematic backtracking: build graph vertex by vertex.
    For each vertex i, decide which of vertices 0..i-1 it connects to.
    Prune if adding an edge would create a C6.
    Use greedy upper bound to guide search.
    """
    start_time = time.time()
    best_edges = 0
    best_adj = None
    
    # We'll build adjacency as list of bitmasks
    adj = [0] * n
    
    # Precompute: for each pair (u,v), the set of "forbidden" intermediate vertices
    # Actually, let's just do incremental C6 check with bitmasks
    
    def _c6_check_bitmask(adj, u, v):
        """Check if adding edge (u,v) creates a C6 in graph with bitmask adjacency."""
        # Need path v-w-x-y-z-u where all distinct, w-x-y-z is path of length 4
        # Use bitmask operations for speed
        neighbors_v = adj[v] & ~(1 << u)
        for w in range(n):
            if not ((neighbors_v >> w) & 1):
                continue
            # Path from w to u of length 3: w-x-y-z-u
            # x is neighbor of w, not u,v,w
            neighbors_w = adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << w)
            for x in range(n):
                if not ((neighbors_w >> x) & 1):
                    continue
                # y is neighbor of x, not u,v,w,x
                neighbors_x = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                for y in range(n):
                    if not ((neighbors_x >> y) & 1):
                        continue
                    # z is neighbor of y, not u,v,w,x,y
                    neighbors_y = adj[y] & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x) & ~(1 << y)
                    # z must be neighbor of u
                    if neighbors_y & (adj[u] & ~(1 << v) & ~(1 << w) & ~(1 << x) & ~(1 << y)):
                        return True
        return False
    
    def _count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    # Recursive backtracking: decide edges for vertex i to vertices 0..i-1
    # To make this tractable, we use a heuristic: try to maximize edges greedily
    # but with backtracking when stuck
    
    # Actually, full backtracking on 40 vertices is infeasible.
    # Instead, use a different structural approach:
    # Start from a good seed, then use systematic 2-exchange (remove edge a, add edge b)
    # with a smarter ordering.
    
    # Let's try: start from 8-K5, then systematically try all 2-exchanges
    # in a structured way, keeping the best.
    
    adj = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                adj[base + a] |= (1 << (base + b))
                adj[base + b] |= (1 << (base + a))
    
    current_edges = _count_edges(adj)
    best_edges = current_edges
    best_adj = adj[:]
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    existing = set()
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                existing.add((i, j))
    
    # Phase 1: Greedy add all possible edges
    random.shuffle(all_pairs)
    for u, v in all_pairs:
        if time.time() > deadline:
            break
        if (u, v) in existing:
            continue
        if not _c6_check_bitmask(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            existing.add((u, v))
            current_edges += 1
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
    
    # Phase 2: Systematic 2-exchange search
    # For each existing edge, try removing it and adding a non-edge
    # that allows more total edges
    non_edges = [(u, v) for u, v in all_pairs if (u, v) not in existing]
    
    edge_list = list(existing)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(edge_list)
        random.shuffle(non_edges)
        
        for rem_u, rem_v in edge_list:
            if time.time() > deadline:
                break
            # Remove edge (rem_u, rem_v)
            adj[rem_u] &= ~(1 << rem_v)
            adj[rem_v] &= ~(1 << rem_u)
            existing.discard((rem_u, rem_v))
            current_edges -= 1
            
            # Try to add the best non-edge
            best_add = None
            best_add_count = 0
            for add_u, add_v in non_edges:
                if (add_u, add_v) in existing:
                    continue
                if _c6_check_bitmask(adj, add_u, add_v):
                    continue
                # Check if adding this edge allows further greedy additions
                # For now, just check if it's valid
                best_add = (add_u, add_v)
                break  # Take the first valid one for speed
            
            if best_add:
                add_u, add_v = best_add
                adj[add_u] |= (1 << add_v)
                adj[add_v] |= (1 << add_u)
                existing.add((add_u, add_v))
                non_edges.remove((add_u, add_v))
                current_edges += 1
                
                # Try greedy fill after swap
                for u, v in all_pairs:
                    if time.time() > deadline:
                        break
                    if (u, v) in existing:
                        continue
                    if not _c6_check_bitmask(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        existing.add((u, v))
                        current_edges += 1
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
                    improved = True
            else:
                # Revert
                adj[rem_u] |= (1 << rem_v)
                adj[rem_v] |= (1 << rem_u)
                existing.add((rem_u, rem_v))
                current_edges += 1
    
    # Convert best_adj to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return edges, best_edges


def _verify_no_c6(edges, n):
    """Verify the graph has no C6."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w == u:
                    continue
                for x in adj[w]:
                    if x == u or x == v:
                        continue
                    for y in adj[x]:
                        if y == u or y == v or y == w:
                            continue
                        for z in adj[y]:
                            if z == u or z == v or z == w or z == x:
                                continue
                            if u in adj[z]:
                                return False
    return True


def construct():
    n = N
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Try the backtracking-based approach
    edges, count = _backtracking_max_edges(n, deadline)
    
    # Verify
    if not _verify_no_c6(edges, n):
        # Fallback to 8-K5
        return _seed_graph_8K5()
    
    return edges
