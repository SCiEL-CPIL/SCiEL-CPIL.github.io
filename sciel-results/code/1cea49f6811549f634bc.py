import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # We want to maximize n.
    # Try n from 35 down to 1.
    for n in range(35, 0, -1):
        if time.time() > deadline:
            break
        if find_graph(n, deadline):
            # If found, return it
            # The find_graph function should return the edges
            pass # Placeholder for logic below

    # Fallback: Return the Paley graph on 17 vertices if search fails
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

def find_graph(n, deadline):
    # adj[i] is a bitmask of neighbors of i
    # We will use a local search
    
    def count_k4(adj):
        # Returns number of K4s
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    # Number of edges in the common neighborhood
                    # This counts K4s containing edge (u,v)
                    # Each K4 is counted 6 times (once for each edge)
                    # So we sum and divide by 6 at the end? 
                    # Actually, we just need to know if > 0.
                    # But for optimization, counting is better.
                    # Let's just count edges in common neighborhood.
                    # An edge (w,x) in common means w,x are neighbors of u and v.
                    # So {u,v,w,x} is a K4.
                    # We can iterate w in common, then x in common > w.
                    # This is O(n^3) roughly.
                    # For n=35, 35^3 = 42k operations. Very fast.
                    # Let's do it properly.
                    common_list = [w for w in range(n) if (common >> w) & 1]
                    for i in range(len(common_list)):
                        w = common_list[i]
                        for j in range(i + 1, len(common_list)):
                            x = common_list[j]
                            if (adj[w] >> x) & 1:
                                count += 1
        return count

    def find_k4(adj):
        # Returns a tuple (u,v,w,x) forming a K4, or None
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    common_list = [w for w in range(n) if (common >> w) & 1]
                    for i in range(len(common_list)):
                        w = common_list[i]
                        for j in range(i + 1, len(common_list)):
                            x = common_list[j]
                            if (adj[w] >> x) & 1:
                                return (u, v, w, x)
        return None

    def find_independent_set_6(adj):
        # Returns a list of 6 vertices that are independent, or None
        # Use backtracking
        # Sort vertices by degree (in complement?) or just use simple branching
        # A vertex is non-adjacent to v if not (adj[v] >> u) & 1
        
        candidates = list(range(n))
        
        def backtrack(current_set, available):
            if len(current_set) == 6:
                return current_set
            if not available:
                return None
            
            # Heuristic: pick vertex with fewest connections in available?
            # Or just pick first.
            # Let's try to pick a vertex that is isolated in the induced subgraph?
            # Simple: iterate through available.
            for i, v in enumerate(available):
                # Check if v is independent from current_set
                ok = True
                for u in current_set:
                    if (adj[u] >> v) & 1:
                        ok = False
                        break
                if not ok:
                    continue
                
                new_set = current_set + [v]
                # New available: neighbors of v in available (that are NOT adjacent to v)
                # In the complement graph, these are the neighbors.
                # So we keep vertices in available that are NOT adjacent to v.
                new_available = []
                for w in available[i+1:]:
                    if not ((adj[v] >> w) & 1):
                        new_available.append(w)
                
                res = backtrack(new_set, new_available)
                if res:
                    return res
            return None

        return backtrack([], candidates)

    # Main Search Loop
    # Initialize
    # We want a graph with no K4 and no I6.
    # Random start
    best_adj = None
    
    for attempt in range(20): # 20 restarts
        if time.time() > deadline:
            return False
        
        # Initialize random graph
        # p=0.5 is a good start
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        steps = 0
        max_steps = 5000
        while steps < max_steps:
            if time.time() > deadline:
                return False
            
            # Check K4
            k4 = find_k4(adj)
            if k4:
                # Break a K4 by removing an edge
                u, v, w, x = k4
                # Pick a random edge in the K4
                edges = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                ru, rv = random.choice(edges)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                steps += 1
                continue
            
            # Check I6
            i6 = find_independent_set_6(adj)
            if i6:
                # Break independence by adding an edge
                # Pick two random vertices from the set
                u, v = random.sample(i6, 2)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                steps += 1
                continue
            
            # No violations!
            # Convert to edge list
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
        # If loop finishes without solution, try next restart
    return False

# Note: The above find_graph returns a dict or False. 
# The construct function needs to handle this.

# Let's refine construct to call find_graph and return the result.

def construct():
    start_time = time.time()
    deadline = 110.0
    
    # Try n from 35 down to 1
    for n in range(35, 0, -1):
        if time.time() > deadline:
            break
        result = find_graph(n, deadline)
        if result:
            return result
            
    # Fallback
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
