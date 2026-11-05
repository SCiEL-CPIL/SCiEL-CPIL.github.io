import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = 20
    best_edges = []
    best_score = 0
    
    # 1. Run multiple greedy constructions to find a good starting point
    # The greedy algorithm is fast, so we can run many seeds.
    # We will use the best greedy result as the initial state for local search.
    
    best_greedy = []
    for seed in range(100):
        if time.time() > deadline - 5.0:
            break
        candidate = _greedy_construct(n, seed=seed)
        if len(candidate) > len(best_greedy):
            best_greedy = candidate
            
    if not best_greedy:
        return []

    # Initialize state with best greedy result
    adj = [[False] * n for _ in range(n)]
    common = [[0] * n for _ in range(n)]
    edges = list(best_greedy)
    
    for u, v in edges:
        adj[u][v] = adj[v][u] = True
        # Recalculate common counts for initial state
        # This is O(n^3) but n=20 is small.
        # Or we can just compute it from scratch.
        
    # Recompute common counts efficiently
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            cnt = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    cnt += 1
            common[u][v] = cnt

    current_score = len(edges)
    best_score = current_score
    best_edges = [e[:] for e in edges] # Deep copy
    
    # 2. Local Search (Simulated Annealing style / Random Walk)
    # We try to add edges. If stuck, we remove a random edge to escape local optimum.
    
    # Precompute list of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Helper to check if adding (u,v) is valid
    def can_add(u, v):
        if adj[u][v]:
            return False
        # 1. Direct pair
        if common[u][v] >= 3:
            return False
        # 2. Impact on neighbors of u
        for w in range(n):
            if adj[u][w] and w != v:
                if common[w][v] >= 2:
                    return False
        # 3. Impact on neighbors of v
        for w in range(n):
            if adj[v][w] and w != u:
                if common[u][w] >= 2:
                    return False
        return True

    def add_edge(u, v):
        nonlocal edges, current_score
        edges.append([u, v])
        adj[u][v] = adj[v][u] = True
        current_score += 1
        
        # Update common counts
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u][w]:
                common[w][v] += 1
                common[v][w] += 1
            if adj[v][w]:
                common[u][w] += 1
                common[w][u] += 1
        # Note: common[u][v] and common[v][u] do not change directly from adding (u,v)
        # because u is not a neighbor of v in the "common neighbor" sense for the pair (u,v) itself?
        # Wait. Common neighbors of u and v are nodes w such that (u,w) and (v,w) are edges.
        # Adding (u,v) does NOT create new common neighbors for the pair (u,v).
        # It DOES affect pairs (u, w) and (v, w).
        # So the update logic above is correct.

    def remove_edge(u, v):
        nonlocal edges, current_score
        # Remove from list
        edges.remove([u, v])
        adj[u][v] = adj[v][u] = False
        current_score -= 1
        
        # Update common counts
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u][w]:
                common[w][v] -= 1
                common[v][w] -= 1
            if adj[v][w]:
                common[u][w] -= 1
                common[w][u] -= 1

    # Main Local Search Loop
    # Strategy: Try to add random edges. If no edge can be added, remove a random one.
    # We keep a list of "blocked" or just try randomly.
    
    # To make it faster, we can maintain a list of "candidate" edges to try.
    # But with n=20, there are only 190 edges. Random sampling is fine.
    
    # We will run this until deadline.
    # To avoid infinite loops on bad states, we track best.
    
    # Optimization: If we find a new best, we might want to restart from there?
    # No, just continue.
    
    # Let's use a "cooling" factor or just pure random walk with bias to add.
    # Bias: 90% try add, 10% try remove (if stuck).
    
    # Actually, a better heuristic:
    # 1. Try to add ALL possible valid edges (greedy step).
    # 2. If no edges added, remove a random edge.
    # This is a "Greedy with Perturbation".
    
    # Let's implement this loop.
    
    # We need to be careful with `edges` list removal being O(N).
    # For n=20, N=190, it's fine.
    
    # To speed up "try to add all valid edges", we can iterate through all non-edges.
    
    while time.time() < deadline:
        # Phase 1: Greedy Add
        added_any = False
        # Shuffle non-edges to break symmetry
        # We can just iterate all pairs
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u][v]:
                    if can_add(u, v):
                        add_edge(u, v)
                        added_any = True
                        if current_score > best_score:
                            best_score = current_score
                            best_edges = [e[:] for e in edges]
        
        if added_any:
            continue
            
        # Phase 2: Perturbation (Remove a random edge)
        if current_score == 0:
            break
            
        # Pick a random edge to remove
        # To be more intelligent, we could pick an edge that is "critical" (part of many K2,3 violations if we were to add more), 
        # but simple random is often sufficient for small n.
        idx = random.randint(0, len(edges) - 1)
        u, v = edges[idx]
        remove_edge(u, v)
        
        # After removal, we loop back to try adding edges again.
        
    return best_edges

def _greedy_construct(n, seed=42):
    """
    Greedy construction of K2,3-free graph.
    """
    random.seed(seed)
    
    common = [[0] * n for _ in range(n)]
    adj = [[False] * n for _ in range(n)]
    edges = []
    
    possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(possible_edges)
    
    for u, v in possible_edges:
        if adj[u][v]:
            continue
            
        if common[u][v] >= 3:
            continue
            
        valid = True
        
        for a in range(n):
            if adj[u][a] and a != v:
                if common[a][v] >= 2:
                    valid = False
                    break
        
        if not valid:
            continue
            
        for b in range(n):
            if adj[v][b] and b != u:
                if common[u][b] >= 2:
                    valid = False
                    break
        
        if not valid:
            continue
        
        edges.append([u, v])
        adj[u][v] = adj[v][u] = True
        
        for a in range(n):
            if adj[u][a] and a != v:
                common[a][v] += 1
                common[v][a] += 1
                
        for b in range(n):
            if adj[v][b] and b != u:
                common[u][b] += 1
                common[b][u] += 1
                
    return edges
# EVOLVE-BLOCK-END
