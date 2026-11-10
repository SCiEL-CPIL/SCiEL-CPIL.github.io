import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n = 20
    # Adjacency matrix
    A = [[0]*n for _ in range(n)]
    # Common neighbor counts
    # common[u][v] = number of common neighbors of u and v
    common = [[0]*n for _ in range(n)]
    
    edges = []
    edge_set = set()
    
    def add_edge(u, v):
        if u == v or A[u][v]:
            return False
        # Check validity
        # Condition:
        # For all w in N(v) \ {u}, common[u][w] < 2
        # For all w in N(u) \ {v}, common[v][w] < 2
        
        # Note: If graph is currently valid, common[x][y] <= 2 for all x!=y.
        # So common[u][v] <= 2 is already true.
        
        # Check neighbors of v
        for w in range(n):
            if w != u and A[v][w]:
                if common[u][w] >= 2:
                    return False
        # Check neighbors of u
        for w in range(n):
            if w != v and A[u][w]:
                if common[v][w] >= 2:
                    return False
            
        # Valid, perform addition
        A[u][v] = 1
        A[v][u] = 1
        edge_set.add((min(u,v), max(u,v)))
        edges.append([u, v])
        
        # Update common counts
        # v becomes a common neighbor of u and w for all w in N(v) \ {u}
        for w in range(n):
            if w != u and A[v][w]:
                common[u][w] += 1
                common[w][u] += 1
        # u becomes a common neighbor of v and w for all w in N(u) \ {v}
        for w in range(n):
            if w != v and A[u][w]:
                common[v][w] += 1
                common[w][v] += 1
        return True

    def remove_edge(u, v):
        if not A[u][v]:
            return
        A[u][v] = 0
        A[v][u] = 0
        edge_set.discard((min(u,v), max(u,v)))
        edges.remove([u, v])
        
        # Update common counts
        for w in range(n):
            if w != u and A[v][w]:
                common[u][w] -= 1
                common[w][u] -= 1
        for w in range(n):
            if w != v and A[u][w]:
                common[v][w] -= 1
                common[w][v] -= 1

    def get_missing_edges():
        missing = []
        for u in range(n):
            for v in range(u+1, n):
                if not A[u][v]:
                    missing.append((u, v))
        return missing

    def greedy_fill():
        # Try to add all possible edges
        # Shuffle to avoid bias
        missing = get_missing_edges()
        random.shuffle(missing)
        added = 0
        for u, v in missing:
            if add_edge(u, v):
                added += 1
        return added

    def local_search():
        # Try to add edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing = get_missing_edges()
            random.shuffle(missing)
            for u, v in missing:
                if add_edge(u, v):
                    improved = True
                    break # Restart missing list as graph changed
            
        # If stuck, try swaps
        # Remove one, add another
        if time.time() < deadline:
            current_edges = len(edges)
            # Try a few random swaps
            for _ in range(1000):
                if time.time() > deadline:
                    break
                if not edges:
                    break
                # Pick random edge to remove
                idx = random.randint(0, len(edges)-1)
                e = edges[idx]
                u, v = e
                remove_edge(u, v)
                
                # Try to add a random missing edge
                missing = get_missing_edges()
                random.shuffle(missing)
                success = False
                for x, y in missing:
                    if add_edge(x, y):
                        success = True
                        break
                if not success:
                    # Restore original edge
                    add_edge(u, v)
                # If success, we have same number of edges, but different structure
                # Continue search

    # Main loop
    best_edges = 0
    best_graph = []
    best_A = [row[:] for row in A]
    best_common = [row[:] for row in common]
    best_edge_set = set(edge_set)
    
    # We will run multiple restarts or one long search?
    # Given n=20, one long search with randomization might be enough.
    # But greedy fill is deterministic-ish if we don't shuffle.
    # Let's do a few restarts with different random seeds for greedy order.
    
    for restart in range(50): # Max 50 restarts
        if time.time() > deadline:
            break
            
        # Reset
        A = [[0]*n for _ in range(n)]
        common = [[0]*n for _ in range(n)]
        edges = []
        edge_set = set()
        
        # Rebind functions to new state? 
        # Python closures capture variables by reference.
        # Since we reassign A, common, etc., the inner functions will see the new ones 
        # IF they are defined inside construct and use the local variables.
        # Yes, they are local to construct.
        
        # 1. Greedy Fill
        greedy_fill()
        
        # 2. Local Search
        local_search()
        
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_graph = [e[:] for e in edges]
            
        # Optional: If we reached a high number, we can try to polish
        # The local search already does some polishing.
        
    return best_graph

# We need to ensure the inner functions use the CURRENT A, common, etc.
# In Python, if I reassign A = ..., the closure in add_edge refers to the variable A in the enclosing scope.
# Since A is local to construct, and add_edge is defined inside construct, it refers to the local A.
# Reassigning A inside the loop updates the local variable, so add_edge uses the new A.
# This is correct.
