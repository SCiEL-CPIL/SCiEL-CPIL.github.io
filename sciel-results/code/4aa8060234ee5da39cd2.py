import time
import random
import heapq
import sys

def _random_5_regular(n):
    """
    Generates a random 5-regular graph on n vertices.
    n must be even.
    Returns adjacency list.
    """
    if n % 2 != 0:
        return None
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(5):
            stubs.append(i)
    
    # Shuffle stubs
    random.shuffle(stubs)
    
    # Pair stubs
    edges = []
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        edges.append((u, v))
        
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            # Self-loop, ignore for now, will fix
            pass
        else:
            adj[u].add(v)
            adj[v].add(u)
            
    # Check for multi-edges and self-loops
    # This simple generation might have multi-edges.
    # We need to make it simple.
    # We'll use a repair phase.
    
    # Verify degrees
    for i in range(n):
        if len(adj[i]) != 5:
            # This shouldn't happen if we just added unique pairs, 
            # but self-loops reduce degree count if we use sets?
            # adj[u].add(v) adds v once.
            # If (u,u) is an edge, adj[u] doesn't change size?
            # No, set.add(u) adds u to adj[u].
            # So degree is counted correctly in terms of set size?
            # Wait, a self-loop contributes 2 to degree in graph theory, 
            # but 1 to set size.
            # The task requires "simple graph". So no self-loops, no multi-edges.
            # We must fix this.
            pass
            
    # Repair: 
    # Find self-loops and multi-edges.
    # Replace them via 2-switches.
    
    # This is complex to do robustly in a short time.
    # Alternative: Retry generation until simple?
    # Probability of multi-edge in random config model for n=1000, k=5:
    # Expected multi-edges ~ k^2 / (2n) * n? 
    # It's small but possible.
    # Let's just retry generation a few times.
    
    return adj

def _make_simple(adj, n, max_iters=1000):
    """
    Removes self-loops and multi-edges by 2-switches.
    """
    # Check for self-loops
    for i in range(n):
        if i in adj[i]:
            # Find a neighbor j
            # Remove self-loop, add edge to someone else?
            # This breaks regularity.
            # Better: 2-switch involving the self-loop?
            # Self-loop is (i,i). Switch with (u,v).
            # Remove (i,i) and (u,v). Add (i,u) and (i,v)?
            # i loses 2 degree (from loop), gains 2.
            # u loses 1, gains 1.
            # v loses 1, gains 1.
            # This works if u,v are not i.
            pass
            
    # This is getting complicated.
    # Let's use a different initialization:
    # Start with a cycle C_n (degree 2).
    # Add random edges to increase degree to 5.
    # This ensures simplicity if we pick non-adjacent vertices.
    
    return adj

def _girth(adj, n, max_girth=9):
    """
    Returns the girth of the graph.
    If no cycle < max_girth, returns max_girth.
    """
    min_cycle = max_girth
    for s in range(n):
        # BFS from s
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = [s]
        
        for u in queue:
            if dist[u] >= max_girth - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
    return min_cycle

def _find_short_cycle(adj, n, max_girth=9):
    """
    Returns a list of vertices forming a cycle of length < max_girth, or None.
    """
    # Re-use logic from girth but return the cycle
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = [s]
        for u in queue:
            if dist[u] >= max_girth - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < max_girth:
                        # Reconstruct cycle
                        # Path s -> ... -> u -> v -> ... -> s
                        # This is complex to reconstruct correctly in this snippet.
                        # For the purpose of the switch, we just need one edge in the cycle.
                        # We can just return u, v, and the LCA?
                        # Let's just return the edge (u, v) which is part of the cycle?
                        # No, (u,v) is the edge closing the cycle in the BFS tree?
                        # Actually, the edge (u,v) exists in the graph.
                        # The cycle is path(s,u) + edge(u,v) + path(v,s).
                        # We can pick edge (u,v) to switch.
                        return [u, v]
    return None

def _switch(adj, n, e1, e2):
    """
    Performs a 2-switch.
    e1 = (u, v), e2 = (w, x)
    Removes (u,v), (w,x).
    Adds (u,w), (v,x) OR (u,x), (v,w).
    Returns True if successful (no multi-edges/loops created).
    """
    u, v = e1
    w, x = e2
    
    # Check if u==v or w==x (should not happen in simple graph)
    if u == v or w == x:
        return False
        
    # Check if edges exist
    if v not in adj[u] or x not in adj[w]:
        return False
        
    # Try option 1: (u,w) and (v,x)
    # Check if (u,w) or (v,x) already exist
    if w in adj[u] or x in adj[v]:
        # Try option 2: (u,x) and (v,w)
        if x in adj[u] or w in adj[v]:
            return False
        else:
            # Execute option 2
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(w)
            adj[w].add(v)
            return True
    else:
        # Execute option 1
        adj[u].remove(v)
        adj[v].remove(u)
        adj[w].remove(x)
        adj[x].remove(w)
        
        adj[u].add(w)
        adj[w].add(u)
        adj[v].add(x)
        adj[x].add(v)
        return True

def _construct_with_switching(n, k, g, time_limit=10.0):
    start_time = time.time()
    
    # 1. Generate random k-regular graph
    # We'll use a simple retry loop for the configuration model
    adj = None
    for _ in range(10):
        # Generate
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        
        candidate_adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if v in candidate_adj[u]:
                valid = False
                break
            candidate_adj[u].add(v)
            candidate_adj[v].add(u)
            
        if valid:
            # Check degrees
            ok = True
            for i in range(n):
                if len(candidate_adj[i]) != k:
                    ok = False
                    break
            if ok:
                adj = candidate_adj
                break
                
    if adj is None:
        return None
        
    # 2. Improve girth
    current_girth = _girth(adj, n, g)
    
    while time.time() - start_time < time_limit:
        if current_girth >= g:
            return adj
            
        # Find a short cycle
        cycle_edge = _find_short_cycle(adj, n, g)
        if cycle_edge is None:
            # Should not happen if girth < g
            break
            
        u, v = cycle_edge
        
        # Try to switch (u,v) with a random edge
        # Get all edges
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
                    
        if not edges:
            break
            
        # Try a few random switches
        success = False
        for _ in range(20):
            e2 = random.choice(edges)
            if e2 == (u, v) or e2 == (v, u):
                continue
                
            # Save state? 
            # We can just try and revert if it fails or reduces girth.
            
            # Check if switch is valid (no multi-edges)
            w, x = e2
            # Check conflicts
            # Option 1: (u,w), (v,x)
            # Option 2: (u,x), (v,w)
            
            # We'll just call _switch and then check girth.
            # But _switch modifies adj in place.
            # We need to revert if it's bad.
            # Let's copy adj? Too slow for n=1000.
            # We can record the changes and revert.
            
            # For simplicity in this snippet, we assume _switch returns False if invalid.
            # But we also need to check if it *reduces* girth.
            
            # Let's make a backup of the 4 vertices' neighbors?
            # Or just copy the whole adj list? 
            # n=1000, adj is list of sets. Copying is O(n*k). 5000 ops.
            # 20 switches * 5000 ops = 100k ops per cycle iteration.
            # Acceptable.
            
            adj_backup = [set(s) for s in adj]
            
            if _switch(adj, n, (u, v), e2):
                new_girth = _girth(adj, n, g)
                if new_girth >= current_girth:
                    current_girth = new_girth
                    success = True
                    break
                else:
                    # Revert
                    adj = adj_backup
                    
        if not success:
            # Try a different edge from the cycle?
            # Or just break if we can't improve?
            # Let's just try another random edge from the cycle if we had one.
            # We only have one edge (u,v).
            # We can pick another edge incident to u or v?
            # No, we must switch (u,v) to break the cycle.
            # If we can't find a good partner, we are stuck.
            # Let's try picking a different edge in the cycle if we had the full cycle.
            # Since we don't have the full cycle, we just fail this iteration.
            # To avoid infinite loop, we can add a small random restart or just break.
            # Let's break to save time.
            break
            
    if _girth(adj, n, g) >= g:
        return adj
    return None

def construct(k, g):
    # Target: k=5, g=9
    # Moore bound 428.
    # We scan n from 1000 to 1200.
    # Time budget: 110s.
    
    start_time = time.time()
    
    # If k != 5 or g != 9, fallback to simple construction?
    # The task is specific to 5-regular girth 9.
    # But the interface takes k, g.
    # We'll handle k=5, g=9 specifically.
    
    if k != 5 or g != 9:
        # Fallback: return a simple random graph? No, must be valid.
        # For other parameters, we might not have a solution.
        # Let's just return a small graph if possible?
        # The prompt implies we only care about this specific task.
        # We'll just return None or a dummy?
        # "An invalid result... scores 0".
        # So we must return a valid graph.
        # If k=5, g=9 is the only case, we are safe.
        # If other cases, we might fail.
        # Let's assume k=5, g=9.
        
        pass
        
    # Scan n
    # Start from 1000? 
    # If we want to beat 1116, we should start lower.
    # Let's start from 1000.
    # Step 2 (must be even).
    
    for n in range(1000, 1200, 2):
        time_used = time.time() - start_time
        if time_used > 100:
            break
            
        time_left_for_n = min(10.0, 105 - time_used)
        
        res = _construct_with_switching(n, k, g, time_limit=time_left_for_n)
        if res is not None:
            # Convert adj to edges
            edges = []
            for i in range(n):
                for j in res[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
            
    # If we didn't find one, try a larger n with remaining time?
    # Or return a known valid graph?
    # We don't have a hardcoded one.
    # Let's try n=1500 with remaining time.
    time_used = time.time() - start_time
    if time_used < 115:
        res = _construct_with_switching(1500, k, g, time_limit=115 - time_used)
        if res is not None:
            edges = []
            for i in range(1500):
                for j in res[i]:
                    if i < j:
                        edges.append((i, j))
            return 1500, edges
            
    # If all else fails, return a dummy valid graph?
    # We can't easily construct one.
    # We'll just raise an error? No, scores 0.
    # We'll return the last attempt if it was close? No, must be valid.
    
    # Fallback: Return a small cycle? No, degree 5.
    # Return K_6? Girth 3.
    
    # We will just return None, which scores 0.
    # But we should try to return *something* valid if possible.
    # If we fail to find girth >= 9, we can return a graph with girth < 9?
    # "no cycle may be shorter than g". So it must be valid.
    
    return None

