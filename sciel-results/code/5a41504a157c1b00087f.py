import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 150 - 10  # Safety margin
    
    # Try to construct for n=46 first, then 47 if time permits?
    # Or just focus on 46 which is safer.
    # The task says maximize n.
    # Let's try 46.
    
    def is_c4_free(adj):
        n = len(adj)
        for u in range(n):
            for v in range(u+1, n):
                # Count common neighbors
                # Using set intersection
                if len(adj[u] & adj[v]) > 1:
                    return False
        return True

    def min_degree(adj):
        return min(len(adj[i]) for i in range(len(adj)))

    def try_construct(n, p, seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        
        target_min_deg = n - 39
        
        # Repair loop
        # We need to remove edges to make |N(u) & N(v)| <= 1 for all u,v
        # And keep min degree >= target_min_deg
        
        # To speed up, we can precompute intersections?
        # With n=46, O(n^2) check is fast.
        
        max_iterations = 10000
        for _ in range(max_iterations):
            # Find a violating pair
            viol_u, viol_v = -1, -1
            for u in range(n):
                for v in range(u+1, n):
                    common = adj[u] & adj[v]
                    if len(common) > 1:
                        viol_u, viol_v = u, v
                        break
                if viol_u != -1:
                    break
            
            if viol_u == -1:
                # No C4s
                if min_degree(adj) >= target_min_deg:
                    return adj
                else:
                    return None
            
            # Repair: remove edges
            common = adj[viol_u] & adj[viol_v]
            # We need to remove edges to reduce common to 1
            # We have len(common) - 1 edges to remove
            # Strategy: remove edge (u, a) or (v, a) for a in common
            # Prefer removing from higher degree vertex
            # If degrees are equal, random
            
            # Sort common neighbors by degree of the neighbor?
            # Actually, we are removing edge (u, a). This affects deg(u) and deg(a).
            # We want to keep deg(u) and deg(v) high.
            # Removing (u, a) reduces deg(u) and deg(a).
            # It does NOT reduce deg(v).
            # So it's better to remove from the vertex (u or v) that has higher degree?
            # Yes, because we need to preserve the min degree.
            
            # However, removing (u, a) also reduces deg(a).
            # If deg(a) is low, this might hurt.
            # But we only care about the global min degree.
            
            # Let's just pick one common neighbor to disconnect.
            # Which one?
            # If we remove (u, a), 'a' is no longer in N(u).
            # So 'a' is no longer a common neighbor of u and v.
            # This reduces the count by 1.
            
            # We need to do this len(common) - 1 times.
            
            # Let's try to remove edges incident to the vertex with higher degree.
            # If deg(u) >= deg(v), remove (u, a). Else remove (v, a).
            
            # But we should be careful not to drop degree below target.
            
            # Let's just try removing one edge at a time and re-checking.
            # It's a local search.
            
            # Pick a common neighbor a
            # To minimize impact on other constraints, maybe pick a with high degree?
            # No, picking a with high degree means removing (u,a) hurts a's degree more.
            # But a's degree is not the bottleneck usually, u and v are the ones with high intersection.
            
            # Let's just pick a random common neighbor.
            a = random.choice(list(common))
            
            # Decide which edge to remove
            if len(adj[viol_u]) >= len(adj[viol_v]):
                # Remove (viol_u, a)
                adj[viol_u].remove(a)
                adj[a].remove(viol_u)
            else:
                # Remove (viol_v, a)
                adj[viol_v].remove(a)
                adj[a].remove(viol_v)
                
        return None

    best_n = 39
    best_edges = []
    
    # Try n=46
    # We need p such that we have enough slack.
    # Min deg 7.
    # Try p=0.3, 0.4, etc.
    
    for p in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for seed in range(100):
            if time.time() > deadline:
                break
            adj = try_construct(46, p, seed)
            if adj is not None:
                # Verify
                if is_c4_free(adj) and min_degree(adj) >= 46 - 39:
                    best_n = 46
                    edges = []
                    for u in range(46):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    best_edges = edges
                    # Try to improve to 47?
                    # If we have 46, we can try 47 in remaining time
                    break
        if best_n == 46:
            break
        if time.time() > deadline:
            break
            
    if best_n < 47 and time.time() < deadline - 5:
        # Try 47
        for p in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for seed in range(50):
                if time.time() > deadline:
                    break
                adj = try_construct(47, p, seed)
                if adj is not None:
                    if is_c4_free(adj) and min_degree(adj) >= 47 - 39:
                        best_n = 47
                        edges = []
                        for u in range(47):
                            for v in adj[u]:
                                if u < v:
                                    edges.append([u, v])
                        best_edges = edges
                        break
            if best_n == 47:
                break
            if time.time() > deadline:
                break

    if best_n > 39:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
