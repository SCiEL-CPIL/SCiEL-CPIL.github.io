import random
import time
import sys

def construct():
    n = 31
    # Initialize graph
    # adjacency matrix for fast lookup
    adj = [[False] * n for _ in range(n)]
    # neighbors sets
    neighbors = [set() for _ in range(n)]
    # common neighbor counts
    common = [[0] * n for _ in range(n)]
    
    edges = []
    
    # Helper to add edge
    def add_edge(u, v):
        adj[u][v] = True
        adj[v][u] = True
        neighbors[u].add(v)
        neighbors[v].add(u)
        edges.append([u, v])
        
        # Update common neighbor counts
        # For any w in neighbors[v], u is a new common neighbor of w and v? No.
        # u is in neighbors[v].
        # For any w in neighbors[v], common[w][v] increases by 1 (because u is in neighbors[w]? No, u is in neighbors[v], and we need u in neighbors[w]).
        # Let's re-derive.
        # common[a][b] = |neighbors[a] & neighbors[b]|
        
        # Adding (u,v):
        # neighbors[u] gains v.
        # neighbors[v] gains u.
        
        # For any w:
        # common[u][w]: neighbors[u] changed. v added.
        # If v in neighbors[w], then common[u][w] += 1.
        # common[v][w]: neighbors[v] changed. u added.
        # If u in neighbors[w], then common[v][w] += 1.
        
        for w in neighbors[v]:
            # w is in neighbors[v].
            # Is v in neighbors[w]? Yes, adjacency is symmetric.
            # So v is in neighbors[u] (new) and neighbors[w] (old).
            # So v is a new common neighbor of u and w.
            common[u][w] += 1
            common[w][u] += 1
            
        for w in neighbors[u]:
            # w is in neighbors[u].
            # Is u in neighbors[w]? Yes.
            # So u is in neighbors[v] (new) and neighbors[w] (old).
            # So u is a new common neighbor of v and w.
            common[v][w] += 1
            common[w][v] += 1

    # Greedy construction
    # We want to add 90 edges.
    # Randomize edge order.
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Time limit guard
    start_time = time.time()
    deadline = 110  # Leave margin
    
    # Try multiple random restarts if time permits?
    # For now, one good greedy pass.
    
    random.seed(42)
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if len(edges) >= 90:
            break
        if time.time() > deadline:
            break
            
        # Check if adding (u,v) creates a C4
        # Condition: common[u][w] == 0 for all w in neighbors[v]
        # and common[v][w] == 0 for all w in neighbors[u]
        
        valid = True
        for w in neighbors[v]:
            if common[u][w] > 0:
                valid = False
                break
        if not valid:
            continue
            
        for w in neighbors[u]:
            if common[v][w] > 0:
                valid = False
                break
        if not valid:
            continue
            
        add_edge(u, v)
        
    # If we have fewer than 90 edges, we might need to add more, but they will create C4s.
    # The task requires exactly 91 edges.
    # If we have < 90, we are in trouble. But 90 is the max, so we should get close.
    # If we have exactly 90, we add one more.
    # If we have < 90, we add enough to reach 91, choosing the ones with min cost.
    
    if len(edges) < 91:
        # Calculate cost for all missing edges
        # Cost of adding (u,v) is number of edges between neighbors[u] and neighbors[v]
        # This is sum(1 for x in neighbors[u] for y in neighbors[v] if adj[x][y])
        # Note: x and y must be distinct from u,v and each other?
        # The path is u-x-y-v.
        # x in N(u), y in N(v).
        # If x==y, it's a path of length 2 (triangle), not a C4.
        # So we count pairs (x,y) with x!=y such that adj[x][y] is True.
        
        missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
        
        # Sort by cost
        def cost(u, v):
            count = 0
            nu = neighbors[u]
            nv = neighbors[v]
            for x in nu:
                for y in nv:
                    if x != y and adj[x][y]:
                        count += 1
            return count

        # We need to add (91 - len(edges)) edges
        # We should pick the ones with lowest cost.
        # However, adding one edge changes the cost of others.
        # Greedy selection of min cost edges.
        
        needed = 91 - len(edges)
        for _ in range(needed):
            if not missing_edges:
                break
            if time.time() > deadline:
                break
                
            # Find min cost edge
            min_cost = float('inf')
            min_edge = None
            for u, v in missing_edges:
                c = cost(u, v)
                if c < min_cost:
                    min_cost = c
                    min_edge = (u, v)
                    
            if min_edge is None:
                break
                
            u, v = min_edge
            add_edge(u, v)
            missing_edges.remove((u, v))
            
    return edges

# EVOLVE-BLOCK-END
