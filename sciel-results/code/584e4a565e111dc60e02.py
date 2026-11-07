"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 15}

def _check_valid(edges, n):
    """Check if no simple 3-edge path has endpoints of equal degree."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    
    # For each edge (u,v), check all pairs s in N(u)\{v}, t in N(v)\{u}, s!=t
    for u, v in edges:
        su = adj[u] - {v}
        tv = adj[v] - {u}
        for s in su:
            for t in tv:
                if s != t and deg[s] == deg[t]:
                    return False
    return True

def _edge_count(edges):
    return len(edges)

def _build_graph_from_adj(adj, n):
    """Build edge list from adjacency sets."""
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return edges

def _try_construct(n, seed=42):
    """Try to construct a valid dense graph."""
    rng = random.Random(seed)
    
    # Start with complete graph and try to remove minimal edges
    # Or build up from a structured pattern
    
    # Strategy: Use degree classes. Let's try a design where:
    # - We have groups of vertices with same degree
    # - Edges between groups are structured to avoid the constraint
    
    # Let's try: partition into groups, make it as dense as possible
    # while respecting the constraint
    
    # First, let's try the complete graph minus a carefully chosen set
    
    # Complete graph has 105 edges. Let's see what's minimal to remove.
    
    # For the complete graph K_15, every vertex has degree 14.
    # For any edge (u,v), N(u)\{v} has 13 vertices all with degree 14,
    # N(v)\{u} has 13 vertices all with degree 14.
    # So s and t both have degree 14 -> violation!
    
    # We need to break this. Let's create degree variation.
    
    # Try a design: 
    # Group A: vertices 0-4 (5 vertices), higher degree
    # Group B: vertices 5-9 (5 vertices), medium degree  
    # Group C: vertices 10-14 (5 vertices), lower degree
    
    # Let's try to make a graph where within each group, vertices have
    # the same degree, and between groups, the structure avoids violations.
    
    # Actually, let's use a more systematic approach:
    # Start with a random dense graph and iteratively fix violations
    
    best_edges = None
    best_count = -1
    
    for attempt in range(50):
        # Start with a fairly dense random graph
        adj = [set() for _ in range(n)]
        
        # Add edges with some probability, biased toward dense
        p = 0.7 + 0.2 * (attempt / 50)
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < p:
                    adj[i].add(j)
                    adj[j].add(i)
        
        edges = _build_graph_from_adj(adj, n)
        
        if not _check_valid(edges, n):
            continue
        
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = [list(e) for e in edges]
    
    return best_edges

def _greedy_fix(adj, n, max_iters=10000):
    """Try to fix violations by removing edges greedily."""
    adj = [set(s) for s in adj]
    
    for _ in range(max_iters):
        edges = _build_graph_from_adj(adj, n)
        if _check_valid(edges, n):
            return edges
        
        # Find a violation and remove an edge to fix it
        deg = [len(adj[i]) for i in range(n)]
        found_violation = False
        for u in range(n):
            for v in list(adj[u]):
                if u >= v:
                    continue
                su = adj[u] - {v}
                tv = adj[v] - {u}
                for s in su:
                    for t in tv:
                        if s != t and deg[s] == deg[t]:
                            # Remove edge (u,v) to try to fix
                            adj[u].discard(v)
                            adj[v].discard(u)
                            found_violation = True
                            break
                    if found_violation:
                        break
                if found_violation:
                    break
            if found_violation:
                break
        
        if not found_violation:
            break
    
    edges = _build_graph_from_adj(adj, n)
    if _check_valid(edges, n):
        return edges
    return None

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s
    
    best_edges = None
    best_count = -1
    
    # Try multiple seeds
    for seed in range(200):
        if time.time() > deadline:
            break
        
        # Try random construction
        edges = _try_construct(n, seed=seed)
        if edges is not None:
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = edges
        
        # Also try greedy fix from dense random
        rng = random.Random(seed * 1000 + 7)
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.85:
                    adj[i].add(j)
                    adj[j].add(i)
        
        fixed = _greedy_fix(adj, n, max_iters=5000)
        if fixed is not None:
            count = len(fixed)
            if count > best_count:
                best_count = count
                best_edges = [list(e) for e in fixed]
    
    # Fallback: star graph (always valid)
    if best_edges is None:
        best_edges = [[0, v] for v in range(1, n)]
    else:
        # Ensure proper types
        best_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return best_edges
# EVOLVE-BLOCK-END
