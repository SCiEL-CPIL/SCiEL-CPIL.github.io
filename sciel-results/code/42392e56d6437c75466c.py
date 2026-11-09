import random
import time

def construct():
    # Fallback to seed if search fails or takes too long
    def seed_graph():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    start_time = time.time()
    max_time = 100  # Leave 20s buffer
    
    # Try to find a C4-free graph with alpha <= 10
    # We will try to build a graph with n=39 or 50
    # Given the complexity, we will try a heuristic search for n=39
    # If it fails, return seed.
    
    # For the sake of reliability and the hint "39", 
    # I will attempt a random greedy construction for n=39.
    
    n_target = 39
    best_n = 30
    best_edges = seed_graph()['edges']
    
    for _ in range(50): # 50 restarts
        if time.time() - start_time > max_time:
            break
        edges = []
        adj = [[0]*n_target for _ in range(n_target)]
        
        # Start with 10 triangles
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges.append([a,b]); edges.append([a,c]); edges.append([b,c])
            adj[a][b]=adj[b][a]=1
            adj[a][c]=adj[c][a]=1
            adj[b][c]=adj[c][b]=1
            
        # Add 9 vertices randomly
        # Ensure C4-free
        valid = True
        for v in range(30, 39):
            # Choose neighbors from 0..29
            # Must be at most 1 per triangle to avoid C4 with triangle edges
            # And must not create C4 with other new vertices
            candidates = []
            for t in range(10):
                # Pick one vertex from triangle t
                u = random.choice([3*t, 3*t+1, 3*t+2])
                candidates.append(u)
            
            # Check if this set of neighbors creates C4
            # C4 can be formed if two neighbors share a common neighbor
            # Or if a new neighbor shares a common neighbor with an existing neighbor of v
            # Since v is new, we only check pairs of neighbors of v
            # If two neighbors u, w of v have a common neighbor x, then v-u-x-w-v is a C4
            # Note: u and w are in different triangles, so they don't share neighbors in the seed.
            # But they might share a neighbor if we add edges between triangles? No, we didn't.
            # So any pair of neighbors from different triangles is safe?
            # Wait, if we connect v to u in T1 and w in T2.
            # Common neighbors of u and w? None in seed.
            # So no C4 formed by v and two seed vertices.
            # What about v and two new vertices?
            # We add vertices one by one.
            # If we add v_i, we must check against previous new vertices v_j.
            # If v_i and v_j share two neighbors, C4.
            
            # To simplify, let's just check the condition:
            # For any two neighbors u, w of v, common_neighbors(u, w) must be 0.
            # In our construction, neighbors are in distinct triangles.
            # So common_neighbors(u, w) is 0 in the seed.
            # So we are safe from seed-induced C4s.
            
            # Now check against other new vertices.
            # If we maintain that any two new vertices share at most 1 neighbor, we are safe.
            # This is the "code" condition.
            
            # For now, let's just add the edges and verify at the end.
            for u in candidates:
                edges.append([v, u])
                adj[v][u] = adj[u][v] = 1
        
        # Verify C4-free and Alpha <= 10
        if is_c4_free(adj, n_target) and alpha_le_10(adj, n_target):
            return {'n': n_target, 'edges': edges}
            
    return {'n': best_n, 'edges': best_edges}

def is_c4_free(adj, n):
    # Check for C4
    # For each pair of vertices u, v, count common neighbors
    # If > 1, C4 exists
    for u in range(n):
        for v in range(u+1, n):
            common = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    common += 1
            if common > 1:
                return False
    return True

def alpha_le_10(adj, n):
    # Greedy check for independent set of size 11
    # This is NP-hard, but for n=39, we can use a simple heuristic
    # or just assume it's true for the constructed structure if it's "dense" enough
    # Actually, we can do a quick DFS for independent set of size 11
    # If found, return False.
    
    def dfs(idx, current_set, adj, n, target_size=11):
        if len(current_set) == target_size:
            return True
        if idx >= n:
            return False
        # Option 1: Include idx
        # Check if idx is independent from current_set
        can_include = True
        for v in current_set:
            if adj[idx][v]:
                can_include = False
                break
        if can_include:
            if dfs(idx+1, current_set + [idx], adj, n, target_size):
                return True
        # Option 2: Exclude idx
        if dfs(idx+1, current_set, adj, n, target_size):
            return True
        return False

    # To speed up, we can prune
    # But for n=39, depth 11, it might be slow.
    # We will use a limited search.
    return not dfs(0, [], adj, n, 11)
