import random
import time
import sys

def has_c4(adj, u, v):
    """Check if adding edge (u,v) creates a C4."""
    # A C4 is u-x-v-y-u.
    # If we add (u,v), a C4 is formed if there exists a path of length 3 between u and v in the current graph.
    # Path u-x-y-v.
    # This is equivalent to: Is there an edge (x,y) such that x in N(u) and y in N(v)?
    # Wait, x is neighbor of u, y is neighbor of v.
    # And x-y is an edge.
    # Also x != y.
    
    nu = adj[u]
    nv = adj[v]
    
    # We check if there is an edge between any node in nv and any node in nu.
    # w in nv, x in nu. Edge (w,x) exists?
    # Note: w cannot be u, x cannot be v.
    # Also w != x.
    
    # Optimization: iterate over smaller set
    if len(nv) < len(nu):
        for w in nv:
            if w == u: continue
            # Check if any neighbor of w is in nu
            # adj[w] is a set
            # intersection = adj[w].intersection(nu)
            # if intersection is not empty and doesn't contain v?
            # Wait, if x in nu and x in adj[w], then x is neighbor of u and w.
            # w is neighbor of v.
            # So u-x-w-v is a path. Adding u-v closes C4.
            # We must ensure x != v.
            # v is not in nu (simple graph, no self loops, u-v not in adj yet).
            # So x != v is naturally true if v not in nu.
            # But what if x == u? u is not in adj[u].
            # So we just check intersection.
            if adj[w] & nu:
                return True
    else:
        for x in nu:
            if x == v: continue
            if adj[x] & nv:
                return True
    return False

def construct():
    target_n = 46
    target_deg = target_n - 39 # 7
    
    start_time = time.time()
    time_limit = 105 # Leave 15s for safety
    
    best_n = 39
    best_edges = []
    
    # We will try to find a valid graph for n=46.
    # If we fail, we return the best we found (which might be 45, 44, etc. if we implement fallbacks, 
    # but for now we focus on 46 and fallback to 39).
    
    # To improve success rate, we use Local Search.
    # 1. Random initialization
    # 2. Repair/Improve loop
    
    for attempt in range(50): # Fewer, longer attempts
        if time.time() - start_time > time_limit:
            break
            
        n = target_n
        adj = [set() for _ in range(n)]
        
        # Initialize with random edges
        # Target edges ~ n * target_deg / 2 = 46 * 7 / 2 = 161
        # Start with slightly fewer to allow growth
        target_edges = 160
        candidates = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(candidates)
        
        added = 0
        for u, v in candidates:
            if added >= target_edges:
                break
            if not has_c4(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                added += 1
        
        # Local Search Phase
        # We want to increase min degree to target_deg.
        # Strategy:
        # While min_degree < target_deg and time < limit:
        #   Find vertex u with min degree.
        #   Find a vertex v not in N(u) such that adding (u,v) doesn't create C4.
        #   If found, add it.
        #   If not found, try a swap:
        #      Remove an edge (u, w) and add (u, v) ? No, that keeps degree same.
        #      We need to increase degree.
        #      If we can't add an edge to u, maybe we can add an edge to some other low-degree vertex?
        #      Or, we can try to "free up" a slot by removing an edge that is "redundant" for C4-freeness?
        #      Actually, if we are stuck, we can restart or try a different random seed.
        
        # A more powerful local search:
        # Allow temporary violations? No, must remain C4-free.
        # 
        # Let's try "Edge Swapping" to improve degree distribution.
        # If min_deg is 6, we have vertices with deg 6.
        # We want to get them to 7.
        # We can try to add an edge (u, v) where deg(u) < target and deg(v) < target.
        # If no such pair exists (all non-edges between low-deg vertices create C4),
        # we might need to remove an edge (x, y) where deg(x) > target and deg(y) > target,
        # to "unblock" a C4? No, removing an edge doesn't unblock a C4 for *other* edges, 
        # it only reduces degrees.
        
        # Actually, the constraint is hard.
        # If we can't add an edge to u, it's because every non-neighbor v of u forms a C4 with u.
        # This means for every v not in N(u), there is x in N(u) and y in N(v) such that x-y is edge?
        # No, C4 is u-x-y-v-u.
        # So if we remove an edge (x, y), we might break the C4 that was preventing (u, v).
        
        # So the move is:
        # 1. Identify a "blocking" edge (x, y) for a desired edge (u, v).
        # 2. Remove (x, y).
        # 3. Add (u, v).
        # This keeps edge count constant.
        # Does it improve min degree?
        # deg(u) +1, deg(v) +1.
        # deg(x) -1, deg(y) -1.
        # If x, y have high degree, this balances the graph.
        
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= target_deg:
                break
                
            # Find vertex u with min degree
            u = min(range(n), key=lambda i: len(adj[i]))
            if len(adj[u]) >= target_deg:
                break
            
            # Try to find a swap that increases degree of u
            # Candidates for u: v not in N(u)
            # For each v, check if (u,v) creates C4.
            # If it does, identify the "witness" edge (x,y) that completes the C4.
            # If we remove (x,y), does (u,v) become safe?
            # And do deg(x), deg(y) stay >= target_deg?
            
            found_swap = False
            # Get non-neighbors of u
            non_neighbors = [v for v in range(n) if v != u and v not in adj[u]]
            random.shuffle(non_neighbors)
            
            for v in non_neighbors:
                # Check if adding (u,v) creates C4
                # We need to find if there is x in N(u), y in N(v) such that (x,y) in E
                # And we want to remove (x,y).
                
                # Find all "blocking" edges (x,y)
                # x in N(u), y in N(v), (x,y) in E
                blockers = []
                for x in adj[u]:
                    for y in adj[v]:
                        if y in adj[x]: # x-y is edge
                            # Ensure x != y, x != v, y != u
                            if x != y and x != v and y != u:
                                blockers.append((x, y))
                
                if not blockers:
                    # No C4, just add it
                    adj[u].add(v)
                    adj[v].add(u)
                    improved = True
                    found_swap = True
                    break
                else:
                    # Try to remove one blocker
                    # Prefer blockers where x, y have high degree
                    # Sort blockers by degree sum descending
                    blockers.sort(key=lambda e: len(adj[e[0]]) + len(adj[e[1]]), reverse=True)
                    
                    for x, y in blockers:
                        # Check if removing (x,y) and adding (u,v) is valid
                        # 1. deg(x)-1 >= target_deg?
                        # 2. deg(y)-1 >= target_deg?
                        # 3. After removing (x,y), does (u,v) still create a C4?
                        #    It might create a C4 via a different path!
                        #    We must check has_c4 on the graph with (x,y) removed and (u,v) added.
                        
                        if len(adj[x]) <= target_deg or len(adj[y]) <= target_deg:
                            continue
                            
                        # Temporarily remove x-y
                        adj[x].remove(y)
                        adj[y].remove(x)
                        
                        # Check if u-v is now safe
                        if not has_c4(adj, u, v):
                            # Add u-v
                            adj[u].add(v)
                            adj[v].add(u)
                            # Check if this new edge creates a C4 with the rest?
                            # has_c4(u,v) checks if adding u-v creates C4 given current adj.
                            # Since we removed x-y, the specific C4 u-x-y-v is broken.
                            # But are there others? has_c4 checks ALL.
                            # So if has_c4 returns False, it's safe.
                            
                            # Update degrees are implicit in sets
                            improved = True
                            found_swap = True
                            break
                        else:
                            # Revert
                            adj[x].add(y)
                            adj[y].add(x)
                    
                    if found_swap:
                        break
            
            if not found_swap:
                # Stuck? Try a different u?
                # Or break and restart?
                # Let's try a few random swaps or break.
                # For efficiency, break and restart.
                break
        
        # Check final result
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg >= target_deg:
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
            
    # Fallback
    return {'n': 39, 'edges': []}
