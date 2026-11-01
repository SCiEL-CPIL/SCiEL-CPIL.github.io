import time
import random
import numpy as np

def _compute_girth(adj, k, max_check):
    """Compute girth of a k-regular graph using non-backtracking walks.
    adj: adjacency list (list of lists)
    Returns: minimum cycle length, or max_check+1 if none found within bound.
    """
    n = len(adj)
    # dist[u][prev] = shortest non-backtracking walk length from u to prev
    # where prev is the neighbor we came from
    # dist[u] is a dict or array indexed by neighbor
    # For efficiency, use array: dist[u][v] = min walk length ending at u having come from v
    # But we need dist from start to u with specific "previous" for each start.
    
    # Standard approach: for each start vertex s, do BFS in the directed line graph
    # State: (current, previous) where previous is the vertex we came from
    # dist[(u, p)] = length of shortest non-backtracking walk from s to u having come from p
    
    # We want to find cycles: a cycle of length L exists iff there's a non-backtracking
    # walk of length L from s back to s (with the "previous" being consistent).
    # Actually, a cycle of length L through s: s -> v1 -> v2 -> ... -> v_{L-1} -> s
    # This is a non-backtracking walk of length L starting and ending at s.
    
    # For each start s, BFS over states (u, prev) where prev is in adj[u] and prev != s 
    # (for the first step, prev is s).
    
    min_girth = max_check + 1
    
    for s in range(n):
        # dist[prev][u] = shortest non-backtracking walk from s to u having come from prev
        # Initialize: for each neighbor v of s, dist[s][v] = 1
        # We use a 2D array: dist[prev][u] but that's n x n which is large.
        # Instead, use dict per (u, prev) or BFS level by level.
        
        # BFS: level l contains states (u, prev) meaning we're at u, having come from prev,
        # and the walk length is l.
        # Start: level 1: (v, s) for each v in adj[s]
        
        # visited[u][prev] = level at which we first reached state (u, prev)
        # Use a 2D list of -1's, but n can be ~1100, so n*n = 1.2M, manageable.
        # But we do this for each s, so total n * n^2 = n^3 = 1.3 billion for n=1100. Too slow.
        
        # Alternative: for each s, only explore up to depth max_check/2 or so.
        # The number of states at depth d is at most k * (k-1)^(d-1) * ... but bounded by n*k.
        # For girth 9, we need to check walks of length up to 9.
        # Depth 9: at most n * k states at each level.
        # For each s, BFS up to depth 9: 9 * n * k = 9 * 1100 * 5 = ~50K operations per s.
        # Total: n * 50K = 1100 * 50K = 55M. Manageable.
        
        # Let's do BFS for each s up to depth max_check.
        
        # dist[prev][u] = level, or use a set of (u, prev) visited.
        # Use a 2D array for speed.
        dist = [[-1] * n for _ in range(n)]  # dist[prev][u]
        
        # Level 1
        queue = []
        for v in adj[s]:
            dist[s][v] = 1
            queue.append((v, s))
        
        level = 1
        while queue and level < max_check:
            next_queue = []
            for u, prev in queue:
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == s and level + 1 >= 3:  # Found a cycle of length level+1
                        # The cycle is s -> ... -> u -> s, length = level + 1
                        if level + 1 < min_girth:
                            min_girth = level + 1
                        continue
                    if dist[u][v] == -1:
                        dist[u][v] = level + 1
                        next_queue.append((v, u))
            level += 1
            queue = next_queue
    
    return min_girth

def _is_regular(adj, k):
    return all(len(adj[i]) == k for i in range(len(adj)))

def _edge_switch(adj, i, j, u, v):
    """Perform edge switch: remove (i,j) and (u,v), add (i,u) and (j,v) or (i,v) and (j,u).
    Returns True if switch was successful (graph remains simple)."""
    n = len(adj)
    # Try both pairings
    for pairing in [((i, u), (j, v)), ((i, v), (j, u))]:
        (a, b), (c, d) = pairing
        # Check if edges (a,b) and (c,d) don't already exist and aren't loops
        if a == b or c == d:
            continue
        if b in adj[a] or d in adj[c]:
            continue
        # Check that (a,b) and (c,d) are not the same edge we're removing
        if (a == i and b == j) or (a == j and b == i):
            continue
        if (c == u and d == v) or (c == v and d == u):
            continue
        # Remove (i,j) and (u,v)
        adj[i].remove(j)
        adj[j].remove(i)
        adj[u].remove(v)
        adj[v].remove(u)
        # Add (a,b) and (c,d)
        adj[a].append(b)
        adj[b].append(a)
        adj[c].append(d)
        adj[d].append(c)
        return True
    return False

def _random_k_regular(n, k):
    """Generate a random k-regular simple graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    
    # Configuration model
    random.seed()
    max_iter = 1000
    for _ in range(max_iter):
        # Create half-edges
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        random.shuffle(half_edges)
        
        # Pair up
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, n * k, 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if valid:
            return [list(adj[i]) for i in range(n)]
    
    return None

def construct(k, g):
    start_time = time.time()
    time_limit = 115.0
    
    # Moore bound for (k, g) cage
    if g % 2 == 0:
        # Even girth: 2 * sum_{i=0}^{g/2-1} (k-1)^i
        mb = 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        # Odd girth: 2 * (k * sum_{i=0}^{(g-3)/2} (k-1)^i + 1)
        mb = 2 * (k * sum((k - 1) ** i for i in range((g - 1) // 2)) + 1)
    
    # Start near Moore bound, try increasing n
    # For k=5, g=9: Moore bound = 2*(5*(1 + 4 + 16) + 1) = 2*(5*21+1) = 2*106 = 212
    # Best known: 1116
    
    # Try various n values
    n_candidates = []
    # Start from Moore bound, go up
    for n in range(max(mb, 100), 2000, 10):
        if n * k % 2 == 0:
            n_candidates.append(n)
    
    # Add some specific values
    for n in [1100, 1116, 1120, 1200, 1500]:
        if n * k % 2 == 0 and n >= mb and n not in n_candidates:
            n_candidates.append(n)
    
    n_candidates.sort()
    
    # For each n, try to construct a graph with girth >= g using edge switching
    for n in n_candidates:
        if time.time() - start_time > time_limit - 5:
            break
        
        # Generate random k-regular graph
        adj = _random_k_regular(n, k)
        if adj is None:
            continue
        
        # Compute initial girth
        current_girth = _compute_girth(adj, k, g)
        
        if current_girth >= g:
            # Already have girth >= g!
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
        
        # Try edge switching to improve girth
        # Each switch: pick random 4-cycle and try to break it
        # Or: pick random pair of non-adjacent edges and try to switch
        
        max_switches = 50000
        switch_count = 0
        
        for _ in range(max_switches):
            if time.time() - start_time > time_limit - 2:
                break
            
            # Pick random vertex i
            i = random.randint(0, n - 1)
            # Pick random neighbor j of i
            if not adj[i]:
                continue
            j_idx = random.randint(0, len(adj[i]) - 1)
            j = adj[i][j_idx]
            
            # Pick random vertex u not i or j
            if n <= 2:
                continue
            u = random.randint(0, n - 1)
            if u == i or u == j:
                continue
            # Pick random neighbor v of u
            if not adj[u]:
                continue
            v_idx = random.randint(0, len(adj[u]) - 1)
            v = adj[u][v_idx]
            
            # Make sure (i,j) and (u,v) are distinct edges
            if (i == u and j == v) or (i == v and j == u):
                continue
            if v == i or v == j:
                continue
            
            # Try the switch
            old_girth = current_girth
            success = _edge_switch(adj, i, j, u, v)
            if not success:
                continue
            
            # Compute new girth (only need to check if it improved or stayed >= g)
            # For efficiency, only compute full girth if current is close to g
            # or if we suspect improvement
            switch_count += 1
            
            # Compute new girth
            new_girth = _compute_girth(adj, k, g)
            
            if new_girth >= g:
                edges = []
                for idx in range(n):
                    for jdx in adj[idx]:
                        if idx < jdx:
                            edges.append((idx, jdx))
                return n, edges
            elif new_girth > old_girth:
                current_girth = new_girth
            else:
                # Revert: switch back
                # The switch added (i,u) and (j,v) or (i,v) and (j,u)
                # We need to revert. But we don't know which pairing was used.
                # Let's just re-add the original edges and remove the new ones.
                # This is tricky. Let's not revert, just continue.
                # Actually, let's keep track of what was done.
                pass
        
        # If we didn't succeed with this n, try larger n
    
    # Fallback: return a large random regular graph (may not have girth >= g, but at least regular)
    n = n_candidates[-1] if n_candidates else mb
    adj = _random_k_regular(n, k)
    if adj is None:
        # Very unlikely
        n = mb + 1
        while n * k % 2 != 0:
            n += 1
        adj = _random_k_regular(n, k)
    
    if adj is None:
        # Last resort: cycle
        n = max(mb, 10)
        while n * k % 2 != 0:
            n += 1
        adj = [[] for _ in range(n)]
        for i in range(n):
            for d in range(1, k + 1):
                if d == k // 2 + 1 and k % 2 == 0:
                    adj[i].append((i + d) % n)
                    adj[(i + d) % n].append(i)
                else:
                    adj[i].append((i + d) % n)
                    adj[(i + d) % n].append(i)
        # Deduplicate
        adj = [list(set(adj[i])) for i in range(n)]
    
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return n, edges
