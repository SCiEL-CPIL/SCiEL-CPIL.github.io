"""Program for dense odd-order graphs avoiding equal-degree path endpoints."""

# EVOLVE-BLOCK-START
import random

P = {'n': 15}

def construct():
    n = P['n']
    
    def get_degrees(adj):
        return [sum(adj[i]) for i in range(n)]
    
    def check_valid(adj, n):
        """Check if any simple 3-edge path s-u-v-t has deg(s)==deg(t)."""
        deg = get_degrees(adj)
        # Precompute neighbors for speed
        neighbors = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
        
        # For each edge (u,v), check all s in N(u)\{v} and t in N(v)\{u}
        for u in range(n):
            for v in neighbors[u]:
                if u > v: # Only process each edge once? No, path direction matters? 
                    # Path s-u-v-t. u-v is an edge.
                    # We can just iterate all pairs u,v where adj[u][v] is true.
                    pass
                # To avoid duplicate checks, we can just iterate all u,v
                for s in neighbors[u]:
                    if s == v:
                        continue
                    for t in neighbors[v]:
                        if t == s or t == u:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True

    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2

    best_edges = []
    best_count = 0
    
    # Strategy 1: Threshold Graph Heuristic (as per directive)
    # Assign weights 0..14
    weights = list(range(n))
    for C in range(0, n):
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if abs(weights[i] - weights[j]) > C:
                    adj[i][j] = adj[j][i] = True
        if check_valid(adj, n):
            cnt = count_edges(adj)
            if cnt > best_count:
                best_count = cnt
                best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
                
    # Strategy 2: Randomized Greedy (New Design)
    # Try multiple random permutations of edge addition
    for _ in range(20): # 20 restarts
        adj = [[False] * n for _ in range(n)]
        # Start with a random valid seed, e.g., star
        center = random.randint(0, n-1)
        for v in range(n):
            if v != center:
                adj[center][v] = adj[v][center] = True
        
        # List of all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            adj[u][v] = adj[v][u] = True
            if check_valid(adj, n):
                # Keep it
                pass
            else:
                adj[u][v] = adj[v][u] = False
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
            
    # Fallback: Return star if nothing better found
    if not best_edges:
        center = 0
        best_edges = [[0, v] for v in range(1, n)]
        
    # Final validation
    final_adj = [[False] * n for _ in range(n)]
    for u, v in best_edges:
        final_adj[u][v] = final_adj[v][u] = True
    assert check_valid(final_adj, n), "Graph is invalid!"
    
    # Convert to proper Python ints
    edges = [[int(u), int(v)] for u, v in best_edges]
    return edges

# EVOLVE-BLOCK-END
