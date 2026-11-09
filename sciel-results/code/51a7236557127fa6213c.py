import time
import random
import itertools

def _build_adjacency(n, edges):
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def _degrees(adj):
    return [sum(row) for row in adj]

def _triangle_score(adj, n, deg):
    """Compute max degree sum over all triangles."""
    best = 0
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                for k in range(j+1, n):
                    if adj[i][k] and adj[j][k]:
                        s = deg[i] + deg[j] + deg[k]
                        if s > best:
                            best = s
    return best

def _edge_set_to_list(edges_set, n):
    return [[u, v] for (u, v) in edges_set if u < v]

def _score_edges(n, edges):
    adj = _build_adjacency(n, edges)
    deg = _degrees(adj)
    return _triangle_score(adj, n, deg)

def _random_graph(n, m):
    """Generate a random graph with n vertices and m edges."""
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    return set(all_edges[:m])

def _local_search(n, m, max_time=120):
    """Local search to minimize max triangle degree sum."""
    deadline = time.time() + max_time
    
    # Start from K_{8,8} + 1 edge, but try different starting points
    best_edges = None
    best_score = float('inf')
    
    # Multiple restarts with different initial structures
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        # Different starting structures
        if restart % 4 == 0:
            # K_{8,8} + 1 edge
            edges = set((i, j) for i in range(8) for j in range(8, 16))
            # Add one random edge within a part
            part = random.choice([list(range(8)), list(range(8, 16))])
            a, b = random.sample(part, 2)
            if a > b:
                a, b = b, a
            edges.add((a, b))
        elif restart % 4 == 1:
            # K_{8,8} + 2 edges within parts, remove 1 cross edge
            edges = set((i, j) for i in range(8) for j in range(8, 16))
            part_a = list(range(8))
            part_b = list(range(8, 16))
            a1, a2 = random.sample(part_a, 2)
            b1, b2 = random.sample(part_b, 2)
            if a1 > a2: a1, a2 = a2, a1
            if b1 > b2: b1, b2 = b2, b1
            edges.add((a1, a2))
            edges.add((b1, b2))
            # Remove a cross edge
            cross = [(i, j) for i in range(8) for j in range(8, 16)]
            rem = random.choice(cross)
            if rem in edges:
                edges.discard(rem)
        elif restart % 4 == 2:
            # Random graph with 65 edges, biased toward bipartite structure
            edges = set()
            # Start with most of K_{8,8}
            cross_edges = [(i, j) for i in range(8) for j in range(8, 16)]
            random.shuffle(cross_edges)
            edges.update(cross_edges[:63])
            # Add 2 within-part edges
            part_a = list(range(8))
            part_b = list(range(8, 16))
            a1, a2 = random.sample(part_a, 2)
            b1, b2 = random.sample(part_b, 2)
            if a1 > a2: a1, a2 = a2, a1
            if b1 > b2: b1, b2 = b2, b1
            edges.add((a1, a2))
            edges.add((b1, b2))
        else:
            # Fully random
            edges = _random_graph(16, 65)
        
        # Local search from this starting point
        cur_edges = set(edges)
        cur_score = _score_edges(16, list(cur_edges))
        
        if cur_score < best_score:
            best_score = cur_score
            best_edges = set(cur_edges)
        
        # Local search: try edge swaps
        all_possible = [(i, j) for i in range(16) for j in range(i+1, 16)]
        
        for step in range(500):
            if time.time() > deadline - 5:
                break
            
            # Try a random edge swap: remove one edge, add another
            if len(cur_edges) != 65:
                # Fix edge count
                break
            
            # Pick a random edge to remove
            to_remove = random.choice(list(cur_edges))
            # Pick a random edge to add (not already present)
            candidates = [e for e in all_possible if e not in cur_edges]
            if not candidates:
                break
            to_add = random.choice(candidates)
            
            new_edges = set(cur_edges)
            new_edges.discard(to_remove)
            new_edges.add(to_add)
            
            if len(new_edges) != 65:
                continue
            
            new_score = _score_edges(16, list(new_edges))
            
            if new_score <= cur_score:
                cur_edges = new_edges
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_edges = set(cur_edges)
            else:
                # Occasionally accept worse moves (simulated annealing style)
                if random.random() < 0.05:
                    cur_edges = new_edges
                    cur_score = new_score
        
        if best_score <= 23:
            break
    
    return best_edges, best_score

def construct():
    n = 16
    m = 65
    
    # Deadline guard
    start_time = time.time()
    max_time = 120
    
    # First, try a structured approach
    # The key insight: we want to minimize max triangle degree sum
    # Triangles in near-bipartite graphs come from within-part edges
    # If we add k edges within parts, we create triangles involving those edges
    
    # Strategy: Start with K_{8,8}, add within-part edges carefully
    # to balance degrees and minimize triangle sums
    
    best_edges = None
    best_score = float('inf')
    
    # Phase 1: Try structured constructions
    # K_{8,8} + 1 edge gives score 26
    # Can we do better?
    
    # Try: K_{8,8} with some cross edges removed and within-part edges added
    # to balance degrees
    
    # Let's try a different approach: use a (16,65) graph that's not near-bipartite
    # but has balanced degrees
    
    # Actually, let's think about it differently:
    # For a triangle (u,v,w), the degree sum is deg(u)+deg(v)+deg(w)
    # To minimize the max, we want:
    # 1. Few triangles
    # 2. Triangles involving low-degree vertices
    
    # In K_{8,8}+1edge: 8 triangles, each with two degree-9 vertices and one degree-8
    # Sum = 9+9+8 = 26
    
    # What if we use a different structure?
    # E.g., start with a 16-vertex graph where degrees are more balanced
    
    # Let's try: take K_{8,8}, remove t cross edges, add t within-part edges
    # This keeps 65 edges
    
    # For t=1: same as before
    # For t=2: remove 2 cross edges, add 2 within-part edges
    # The within-part edges create triangles, but the removed cross edges
    # reduce degrees of some vertices
    
    # Let's do a smart search
    
    deadline = start_time + max_time
    
    # Try many configurations
    for trial in range(500):
        if time.time() > deadline - 15:
            break
        
        # Configuration: K_{8,8} minus t cross edges, plus t within-part edges
        t = random.randint(1, 4)
        
        cross_edges = [(i, j) for i in range(8) for j in range(8, 16)]
        random.shuffle(cross_edges)
        
        # Remove t cross edges
        removed = cross_edges[:t]
        
        # Add t within-part edges
        part_a = list(range(8))
        part_b = list(range(8, 16))
        
        within_edges = []
        # Try to spread within-part edges across both parts
        for _ in range(t):
            part = random.choice([part_a, part_b])
            a, b = random.sample(part, 2)
            if a > b:
                a, b = b, a
            within_edges.append((a, b))
        
        edges = set(cross_edges[t:])  # K_{8,8} minus removed
        edges.update(within_edges)
        
        # Ensure exactly 65 edges
        if len(edges) != 65:
            continue
        
        score = _score_edges(16, list(edges))
        if score < best_score:
            best_score = score
            best_edges = set(edges)
        
        # Local search from here
        cur_edges = set(edges)
        cur_score = score
        all_possible = [(i, j) for i in range(16) for j in range(i+1, 16)]
        
        for step in range(300):
            if time.time() > deadline - 5:
                break
            
            to_remove = random.choice(list(cur_edges))
            candidates = [e for e in all_possible if e not in cur_edges]
            if not candidates:
                break
            to_add = random.choice(candidates)
            
            new_edges = set(cur_edges)
            new_edges.discard(to_remove)
            new_edges.add(to_add)
            
            if len(new_edges) != 65:
                continue
            
            new_score = _score_edges(16, list(new_edges))
            
            if new_score <= cur_score:
                cur_edges = new_edges
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_edges = set(cur_edges)
        
        if best_score <= 23:
            break
    
    # If we haven't found something good, fall back to seed
    if best_edges is None or len(best_edges) != 65:
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        return {'n': 16, 'edges': edges}
    
    return {'n': 16, 'edges': [[u, v] for (u, v) in best_edges if u < v]}
