import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 20
    
    # Adjacency matrix for fast access
    # adj[u][v] is 1 if edge exists, 0 otherwise
    # We also maintain a list of neighbors for each vertex for iteration
    
    def is_valid_addition(adj, u, v):
        """
        Checks if adding edge (u, v) violates the K3,3-free condition.
        Condition: For any triple T, |N(T)| <= 2.
        Adding (u, v) only affects triples containing u or v.
        """
        if adj[u][v]:
            return False
            
        # Precompute neighbors of u and v for speed? 
        # adj is a list of lists of 0/1.
        
        # Check triples containing u: {u, a, b}
        # N({u, a, b}) = N(u) ∩ N(a) ∩ N(b)
        # After adding (u, v), N(u) becomes N(u) ∪ {v}.
        # New N({u, a, b}) = (N(u) ∩ N(a) ∩ N(b)) ∪ ({v} ∩ N(a) ∩ N(b))
        # Size increases by 1 if v is in N(a) ∩ N(b).
        # We need New Size <= 2.
        # So if v is in N(a) ∩ N(b), we need Old Size <= 1.
        
        for a in range(n):
            if a == u or a == v:
                continue
            if adj[u][a] == 0: # u and a not connected? No, a is just a vertex.
                # We iterate over all a != u, v
                pass
            
            # We need to check all pairs (a, b) with a < b, a,b != u,v
            # Actually, we can just iterate a, b.
            # To avoid double counting or redundancy, we can just check all a, b != u, v.
            # But (a,b) and (b,a) are the same triple {u,a,b}.
            
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                if a > b:
                    continue
                
                # Check if v is a common neighbor of a and b
                if adj[a][v] == 1 and adj[b][v] == 1:
                    # v is in N(a) ∩ N(b)
                    # So adding u-v adds v to N({u,a,b})
                    # We must ensure current |N({u,a,b})| <= 1
                    # N({u,a,b}) = N(u) ∩ N(a) ∩ N(b)
                    count = 0
                    for w in range(n):
                        if adj[u][w] == 1 and adj[a][w] == 1 and adj[b][w] == 1:
                            count += 1
                    if count > 1:
                        return False
                        
        # Check triples containing v: {v, a, b}
        # Symmetric logic
        for a in range(n):
            if a == u or a == v:
                continue
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                if a > b:
                    continue
                
                # Check if u is a common neighbor of a and b
                if adj[a][u] == 1 and adj[b][u] == 1:
                    # u is in N(a) ∩ N(b)
                    # So adding u-v adds u to N({v,a,b})
                    # We must ensure current |N({v,a,b})| <= 1
                    count = 0
                    for w in range(n):
                        if adj[v][w] == 1 and adj[a][w] == 1 and adj[b][w] == 1:
                            count += 1
                    if count > 1:
                        return False
                        
        return True

    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2

    def greedy_construct(random_seed):
        random.seed(random_seed)
        adj = [[0]*n for _ in range(n)]
        
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, u, v):
                adj[u][v] = 1
                adj[v][u] = 1
                
        return adj

    def local_search(adj, max_steps=2000):
        """
        Hill climbing: try to add edges. If stuck, try removing an edge and re-adding?
        Or just random restarts of greedy might be enough given the relaxed constraint.
        Let's try a simple local search: 
        1. Try to add any valid edge.
        2. If no edge can be added, try removing a random edge and then adding 2 edges?
        Simpler: Just do more greedy restarts with the relaxed constraint.
        The relaxed constraint space is larger, so greedy might find better solutions.
        """
        current_adj = [row[:] for row in adj]
        current_edges = count_edges(current_adj)
        
        # Try to add edges in random order until no more can be added
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i][j] == 0]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(current_adj, u, v):
                current_adj[u][v] = 1
                current_adj[v][u] = 1
                current_edges += 1
                
        return current_adj, current_edges

    # Baseline
    best_adj = None
    best_edges = 0
    
    # Generate initial candidates with relaxed greedy
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        adj = greedy_construct(i)
        edges = count_edges(adj)
        
        # Local search improvement
        adj, edges = local_search(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
            
    # If we have time, try to improve the best solution further
    if best_adj is not None and time.time() < deadline - 10:
        # Try more local search on the best
        adj, edges = local_search(best_adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
            
    # Convert adjacency matrix to list of edges
    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
