"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a K4-free graph on 18 vertices minimizing the size of the largest
    triangle-free induced subgraph.
    
    Strategy: Use random greedy construction with local search (edge addition/removal)
    to find a K4-free graph where the largest triangle-free induced subgraph is as small
    as possible.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin before 150s limit
    
    n = 18
    
    def is_k4_free(adj):
        """Check if the graph represented by adjacency matrix is K4-free."""
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            for x in range(w+1, n):
                                if adj[u][x] and adj[v][x] and adj[w][x]:
                                    return False
        return True
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps the graph K4-free."""
        if adj[u][v]:
            return True
        # An edge (u,v) creates a K4 if there exist w,x such that
        # u-v, u-w, u-x, v-w, v-x, w-x all exist
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                for x in range(w+1, n):
                    if x != u and x != v and x != w:
                        if adj[u][x] and adj[v][x] and adj[w][x]:
                            return False
        return True
    
    def max_triangle_free_induced(adj):
        """
        Find the size of the largest triangle-free induced subgraph using
        branch and bound.
        """
        best = 0
        
        # Try all subsets in decreasing order of size
        for size in range(n, 0, -1):
            found = False
            for subset in itertools.combinations(range(n), size):
                # Check if this subset is triangle-free
                is_tf = True
                for i in range(len(subset)):
                    for j in range(i+1, len(subset)):
                        u, v = subset[i], subset[j]
                        if not adj[u][v]:
                            continue
                        for k in range(j+1, len(subset)):
                            w = subset[k]
                            if adj[u][w] and adj[v][w]:
                                is_tf = False
                                break
                        if not is_tf:
                            break
                    if not is_tf:
                        break
                
                if is_tf:
                    best = size
                    found = True
                    break
            
            if found:
                break
        
        return best
    
    def random_greedy_construct():
        """Construct a random K4-free graph using greedy edge addition."""
        adj = [[0]*n for _ in range(n)]
        edges_to_add = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges_to_add)
        
        for u, v in edges_to_add:
            if can_add_edge(adj, u, v):
                adj[u][v] = 1
                adj[v][u] = 1
        
        return adj
    
    def local_search(adj, max_iterations=5000):
        """
        Local search: try to add or remove edges to reduce the largest
        triangle-free induced subgraph while maintaining K4-freeness.
        """
        best_adj = [row[:] for row in adj]
        best_score = max_triangle_free_induced(adj)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try adding a random edge
            u, v = random.randint(0, n-1), random.randint(0, n-1)
            if u == v:
                continue
            u, v = min(u, v), max(u, v)
            
            if not adj[u][v] and can_add_edge(adj, u, v):
                adj[u][v] = 1
                adj[v][u] = 1
                new_score = max_triangle_free_induced(adj)
                if new_score <= best_score:
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = [row[:] for row in adj]
                else:
                    adj[u][v] = 0
                    adj[v][u] = 0
            
            # Try removing a random edge
            if random.random() < 0.3:
                u, v = random.randint(0, n-1), random.randint(0, n-1)
                if u == v:
                    continue
                u, v = min(u, v), max(u, v)
                if adj[u][v]:
                    adj[u][v] = 0
                    adj[v][u] = 0
                    new_score = max_triangle_free_induced(adj)
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = [row[:] for row in adj]
                    else:
                        adj[u][v] = 1
                        adj[v][u] = 1
        
        return best_adj, best_score
    
    # Start with the complete tripartite graph K6,6,6 as initial solution
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj = [[0]*n for _ in range(n)]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a][b] = 1
                    adj[b][a] = 1
    
    # Verify it's K4-free
    if not is_k4_free(adj):
        adj = [[0]*n for _ in range(n)]
    
    # Calculate initial score
    initial_score = max_triangle_free_induced(adj)
    
    best_adj = [row[:] for row in adj]
    best_score = initial_score
    
    # Multiple random restarts with local search
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate a random starting point
        if restart == 0:
            start_adj = [row[:] for row in adj]
        else:
            start_adj = random_greedy_construct()
        
        # Run local search
        search_adj, search_score = local_search(start_adj, max_iterations=3000)
        
        if search_score < best_score:
            best_score = search_score
            best_adj = [row[:] for row in search_adj]
        
        # If we've found a very good solution, we can stop early
        if best_score <= 8:
            break
    
    # Extract edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
