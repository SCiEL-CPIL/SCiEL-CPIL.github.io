import random
import itertools
from collections import defaultdict

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_triangles(n, adj):
    """Find all triangles in the graph."""
    triangles = []
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        triangles.append((u, v, w))
    return triangles

def _max_triangle_degree_sum(n, adj):
    """Compute the maximum degree sum over all triangles."""
    if n <= 2:
        return 0
    triangles = _find_triangles(n, adj)
    if not triangles:
        return 0
    max_sum = 0
    for (u, v, w) in triangles:
        s = len(adj[u]) + len(adj[v]) + len(adj[w])
        if s > max_sum:
            max_sum = s
    return max_sum

def construct():
    n = 16
    target_edges = 65
    
    # Start with K_{8,8} + one edge
    random.seed(42)
    
    def initial_graph():
        edges = set()
        for u in range(8):
            for v in range(8, 16):
                edges.add(frozenset((u, v)))
        # Add one edge within part A
        edges.add(frozenset((0, 1)))
        return [list(e) for e in edges]
    
    edges = initial_graph()
    adj = _build_adjacency(n, edges)
    current_score = _max_triangle_degree_sum(n, adj)
    
    best_score = current_score
    best_edges = edges[:]
    
    # Local search: try edge swaps
    def edge_key(u, v):
        return frozenset((u, v))
    
    def try_swap(edges_set, adj, score):
        """Try one random edge swap and return new score if improved."""
        # Pick a random existing edge to remove
        existing = list(edges_set)
        if not existing:
            return None
        e_remove = random.choice(existing)
        u, v = list(e_remove)
        
        # Try adding a random new edge
        all_pairs = [(a, b) for a in range(n) for b in range(a+1, n)]
        # Filter out existing edges and the one we're removing
        candidates = []
        for a, b in all_pairs:
            if edge_key(a, b) == e_remove:
                continue
            if edge_key(a, b) in edges_set:
                continue
            candidates.append((a, b))
        
        if not candidates:
            return None
        
        a, b = random.choice(candidates)
        
        # Perform swap
        new_edges_set = set(edges_set)
        new_edges_set.remove(e_remove)
        new_edges_set.add(edge_key(a, b))
        
        if len(new_edges_set) != target_edges:
            return None
        
        # Check no self-loops (shouldn't happen)
        new_adj = _build_adjacency(n, [list(e) for e in new_edges_set])
        new_score = _max_triangle_degree_sum(n, new_adj)
        
        return new_score, new_edges_set, new_adj
    
    # Simulated annealing style local search
    import math
    temp = 10.0
    cooling = 0.999
    min_temp = 0.01
    max_iters = 50000
    
    for it in range(max_iters):
        if temp < min_temp:
            break
        
        # Try a swap
        edges_set = set(frozenset(e) for e in edges)
        result = try_swap(edges_set, adj, current_score)
        
        if result is None:
            temp *= cooling
            continue
        
        new_score, new_edges_set, new_adj = result
        
        if new_score <= current_score:
            # Accept improvement
            current_score = new_score
            edges = [list(e) for e in new_edges_set]
            adj = new_adj
            if current_score < best_score:
                best_score = current_score
                best_edges = edges[:]
        else:
            # Accept with probability based on temperature
            delta = new_score - current_score
            prob = math.exp(-delta / temp)
            if random.random() < prob:
                current_score = new_score
                edges = [list(e) for e in new_edges_set]
                adj = new_adj
        
        temp *= cooling
    
    # Return the best found
    return {'n': n, 'edges': best_edges}
