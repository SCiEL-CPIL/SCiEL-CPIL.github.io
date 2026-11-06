import random
import time
from itertools import combinations

def find_c5(n, edges):
    """Find a C5 in the graph. Returns a list of 5 vertices if found, else None."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute adjacency matrix for O(1) lookups? Or just use sets.
    # For n=35, sets are fine.
    
    for verts in combinations(range(n), 5):
        v0, v1, v2, v3, v4 = verts
        # Check if induced subgraph is a C5
        # A C5 has exactly 5 edges and is 2-regular.
        # Since we are checking 5 specific vertices, we just need to check the edges among them.
        edges_in_sub = 0
        degs = [0] * 5
        # Map vertex to index in verts
        # verts is a tuple of ints.
        # We can just check pairs.
        pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
        for i, j in pairs:
            if verts[j] in adj[verts[i]]:
                edges_in_sub += 1
                degs[i] += 1
                degs[j] += 1
        
        if edges_in_sub == 5 and all(d == 2 for d in degs):
            return list(verts)
    return None

def find_independent_set_9(n, edges):
    """Find an independent set of size 9. Returns the set if found, else None."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Backtracking search
    # We want to find a set of 9 vertices with no edges between them.
    # Sort vertices by degree (ascending) to prune faster? 
    # Or just simple recursion.
    
    # State: current independent set, remaining vertices
    # We can use a simple DFS.
    
    def dfs(current_set, remaining):
        if len(current_set) == 9:
            return current_set
        if not remaining:
            return None
        if len(current_set) + len(remaining) < 9:
            return None
            
        # Pick a vertex from remaining
        # Heuristic: pick vertex with lowest degree in the induced subgraph of remaining?
        # For simplicity, just pick the first one, or randomize order.
        # Randomizing order helps avoid local minima.
        v = remaining[0]
        
        # Option 1: Include v
        # New remaining: neighbors of v removed from remaining
        new_remaining = [u for u in remaining[1:] if u not in adj[v]]
        result = dfs(current_set + [v], new_remaining)
        if result:
            return result
        
        # Option 2: Exclude v
        result = dfs(current_set, remaining[1:])
        if result:
            return result
        
        return None
    
    # To speed up, we can use a greedy pre-check or just run the DFS.
    # For n=35, the search space is manageable with pruning.
    # However, worst case is huge.
    # Let's add a simple greedy check first to see if we can easily find one.
    # If greedy finds one, return it.
    # If not, run the exact DFS.
    
    # Actually, the DFS above is exact. Let's just run it.
    # To avoid deep recursion issues or slow performance, we can add a timeout or limit?
    # No, we need exactness.
    
    # Optimization: If the graph is dense, independent sets are small.
    # If the graph is sparse, they are large.
    
    # Let's try a bit of randomization in the vertex selection order for the DFS.
    # But `dfs` is deterministic based on `remaining` order.
    # We can shuffle `remaining` initially? No, that changes the search tree.
    # We can just call the DFS once. If it's too slow, we might need a better algorithm.
    # For n=35, finding an independent set of size 9 is usually fast if one exists.
    # If none exists, it might be slow.
    
    # Let's try a different approach for the check:
    # We just need to know if alpha >= 9.
    # The DFS does this.
    
    return dfs([], list(range(n)))

def construct():
    start_time = time.time()
    timeout = 115  # Leave 5s margin
    
    # Try to construct graphs for n from 35 down to 29
    for n in range(35, 28, -1):
        # Try multiple random restarts
        for attempt in range(50):
            if time.time() - start_time > timeout:
                break
            
            # Initialize with random graph
            # Density heuristic: C5-free graphs are sparse.
            # For n=32, max edges ~ n^2/4 is not right for C5.
            # Turan number for C5 is O(n^1.5).
            # Let's try p = 0.2
            edges = []
            p = 0.2
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        edges.append([i, j])
            
            # Repair loop
            valid = False
            for step in range(1000):
                if time.time() - start_time > timeout:
                    break
                
                # Check C5
                c5 = find_c5(n, edges)
                if c5:
                    # Remove a random edge from the C5
                    # The C5 vertices are c5. The edges are between consecutive vertices in the cycle.
                    # We don't know the order, but we know the set.
                    # Any edge in the induced subgraph of c5 is part of the cycle.
                    # Let's find an edge among c5 vertices.
                    c5_set = set(c5)
                    candidate_edges = []
                    for u, v in edges:
                        if u in c5_set and v in c5_set:
                            candidate_edges.append((u, v))
                    
                    if candidate_edges:
                        u, v = random.choice(candidate_edges)
                        # Remove edge [u, v] or [v, u]
                        edges = [[a, b] for a, b in edges if not ((a == u and b == v) or (a == v and b == u))]
                    else:
                        # Should not happen if c5 is valid
                        pass
                    continue
                
                # Check Independent Set
                ind_set = find_independent_set_9(n, edges)
                if ind_set is None:
                    valid = True
                    break
                else:
                    # Add an edge between two vertices in the independent set
                    # Pick two random vertices from ind_set
                    u, v = random.sample(ind_set, 2)
                    # Check if edge already exists
                    adj_set = set()
                    for a, b in edges:
                        adj_set.add((a, b))
                        adj_set.add((b, a))
                    if (u, v) not in adj_set:
                        edges.append([u, v])
            
            if valid:
                return {'n': n, 'edges': edges}
    
    # Fallback to seed
    return {'n': 28, 'edges': [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]}
