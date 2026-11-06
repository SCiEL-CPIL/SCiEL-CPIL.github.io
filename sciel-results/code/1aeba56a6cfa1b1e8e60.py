import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Best known valid graph
    best_graph = None
    
    # Try n from 43 down to 30
    for n in range(43, 29, -1):
        if time.time() > deadline - 5.0:
            break
        
        # Try to find a valid graph for this n
        graph = try_find_valid_graph(n, deadline)
        if graph is not None:
            best_graph = graph
            break
            
    if best_graph is None:
        # Fallback to seed graph (n=30, 10 triangles)
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
        
    return best_graph

def try_find_valid_graph(n, deadline):
    # Initialize with disjoint triangles
    # Number of triangles
    num_triangles = n // 3
    remainder = n % 3
    
    # Adjacency list
    adj = [set() for _ in range(n)]
    
    # Add triangles
    for t in range(num_triangles):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        adj[a].add(b); adj[b].add(a)
        adj[a].add(c); adj[c].add(a)
        adj[b].add(c); adj[c].add(b)
        
    # If remainder is 1, vertex 3*num_triangles is isolated
    # If remainder is 2, vertices 3*num_triangles, 3*num_triangles+1 are isolated
    # We can also add an edge between the two isolated vertices if remainder is 2?
    # That would make a K2, not a triangle. It's C4-free.
    if remainder == 2:
        a = 3 * num_triangles
        b = 3 * num_triangles + 1
        adj[a].add(b); adj[b].add(a)
        
    # Greedy edge addition
    # We want to reduce alpha.
    # Current alpha estimate: num_triangles + (1 if remainder==1 else 0)
    # Actually, for K2, alpha is 1. So alpha = num_triangles + (1 if remainder==1 else 0)
    
    # Target alpha <= 10
    # We need to add edges to reduce alpha.
    
    # Randomized restarts
    max_restarts = 20
    for r in range(max_restarts):
        if time.time() > deadline - 2.0:
            break
            
        # Reset adj to initial state
        adj = [set() for _ in range(n)]
        for t in range(num_triangles):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            adj[a].add(b); adj[b].add(a)
            adj[a].add(c); adj[c].add(a)
            adj[b].add(c); adj[c].add(b)
            
        if remainder == 2:
            a = 3 * num_triangles
            b = 3 * num_triangles + 1
            adj[a].add(b); adj[b].add(a)
            
        # Greedy loop
        for step in range(100):
            if time.time() > deadline - 2.0:
                break
                
            # Check if alpha <= 10
            if check_alpha(adj, n) <= 10:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
            # Find candidate edges
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj[u]:
                        # Check C4 constraint
                        common = adj[u] & adj[v]
                        if len(common) <= 1:
                            # Estimate alpha reduction
                            # Simple heuristic: count how many independent sets this edge breaks
                            # For now, just pick randomly or by degree
                            candidates.append((u, v))
                            
            if not candidates:
                break
                
            # Pick a candidate
            # To be smarter, we can pick edges that connect vertices in different components
            # or vertices with low degree
            random.shuffle(candidates)
            u, v = candidates[0]
            
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            
    return None

def check_alpha(adj, n):
    # Exact check for alpha <= 10
    # This is equivalent to checking if the complement graph has a clique of size 11
    # For n=43, this might be slow if done naively.
    # We use a fast heuristic first.
    
    # Heuristic: Greedy coloring of complement graph
    # If we can color the complement graph with 10 colors, then alpha <= 10
    # Wait, coloring the complement graph gives an upper bound on clique number of complement,
    # which is alpha of G.
    # So if we can color G_bar with 10 colors, then alpha(G) <= 10.
    
    # Build complement adjacency
    comp_adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u != v and v not in adj[u]:
                comp_adj[u].add(v)
                
    # Greedy coloring
    colors = [-1] * n
    color_classes = []
    
    # Order vertices by degree in comp_adj (descending)
    order = sorted(range(n), key=lambda x: len(comp_adj[x]), reverse=True)
    
    for u in order:
        # Find smallest available color
        used_colors = set()
        for v in comp_adj[u]:
            if colors[v] != -1:
                used_colors.add(colors[v])
                
        c = 0
        while c in used_colors:
            c += 1
        colors[u] = c
        if c >= len(color_classes):
            color_classes.append([])
        color_classes[c].append(u)
        
    if len(color_classes) <= 10:
        return True # alpha <= 10
        
    # If heuristic says alpha > 10, it might be wrong (heuristic is upper bound? No, greedy coloring is not always optimal).
    # Greedy coloring gives an upper bound on chromatic number, which is an upper bound on clique number?
    # No, chromatic number >= clique number.
    # So if greedy coloring uses 10 colors, then chi(G_bar) <= 10, so omega(G_bar) <= 10, so alpha(G) <= 10.
    # If greedy coloring uses 11 colors, it doesn't mean alpha > 10.
    # So we can only confirm alpha <= 10 if greedy coloring uses <= 10 colors.
    # If it uses > 10, we need an exact check or we assume it's > 10 and continue searching.
    
    # For safety, if we are close to deadline, we might want to do an exact check.
    # But for now, we return False if heuristic fails.
    return False
