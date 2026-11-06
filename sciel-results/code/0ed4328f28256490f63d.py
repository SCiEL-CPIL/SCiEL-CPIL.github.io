import random
import time

def construct():
    # Target n values
    for n in [47, 46]:
        min_deg_required = n - 39
        
        # Try multiple random orders/instances
        for trial in range(2000):
            random.seed(trial)
            
            # Shuffle vertex order to break symmetries
            order = list(range(n))
            random.shuffle(order)
            
            adj = [set() for _ in range(n)]
            success = True
            
            # Process vertices in shuffled order
            for u in order:
                # We want u to have degree at least min_deg_required
                # But we don't know final degrees yet.
                # Strategy: Try to add edges to u.
                # How many edges should u have?
                # Ideally min_deg_required.
                # Let's aim for exactly min_deg_required for simplicity, 
                # or slightly more if possible?
                # If we aim for min_deg_required, we must ensure we can add that many.
                
                target_deg = min_deg_required
                
                # Candidates for neighbors of u
                # We can connect to any v != u
                # But we should prioritize v's that are "hard" to connect later?
                # Or just random.
                
                # Get all potential neighbors
                candidates = [v for v in range(n) if v != u and v not in adj[u]]
                
                # Shuffle candidates
                random.shuffle(candidates)
                
                added = 0
                for v in candidates:
                    if added >= target_deg:
                        break
                    
                    # Check if adding (u, v) is safe
                    if is_safe_add(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        added += 1
                
                if added < target_deg:
                    # We couldn't add enough edges to u
                    # This trial fails
                    success = False
                    break
            
            if not success:
                continue
            
            # Verify min degree
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg < min_deg_required:
                continue
                
            # Verify C4-free
            if is_c4_free(adj, n):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
    # Fallback
    # Try n=45
    for n in [45, 44, 43, 42, 41, 40, 39]:
        min_deg_required = n - 39
        if min_deg_required <= 0:
            return {'n': n, 'edges': []}
            
        for trial in range(1000):
            random.seed(10000 + trial)
            order = list(range(n))
            random.shuffle(order)
            adj = [set() for _ in range(n)]
            success = True
            for u in order:
                target_deg = min_deg_required
                candidates = [v for v in range(n) if v != u and v not in adj[u]]
                random.shuffle(candidates)
                added = 0
                for v in candidates:
                    if added >= target_deg:
                        break
                    if is_safe_add(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        added += 1
                if added < target_deg:
                    success = False
                    break
            if not success:
                continue
            if min(len(adj[i]) for i in range(n)) >= min_deg_required:
                if is_c4_free(adj, n):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
                    
    return {'n': 39, 'edges': []}

def is_safe_add(adj, u, v):
    # Check if adding edge (u,v) creates a C4
    # 1. |N(u) \cap N(v)| <= 1
    # Note: adj sets do NOT contain u or v themselves.
    # If u and v are not currently connected.
    
    # Condition 1: Common neighbors of u and v
    common_uv = adj[u].intersection(adj[v])
    if len(common_uv) >= 2:
        return False
        
    # Condition 2: For all w in N(v), N(u) \cap N(w) must be empty
    # Why? Because if x in N(u) \cap N(w), then u-x-w-v-u is a C4?
    # Let's trace:
    # Edge (u,v) is added.
    # Path u-w-v exists? No, w is in N(v). So (w,v) is edge.
    # If x is in N(u) and N(w), then (u,x) and (x,w) are edges.
    # Cycle: u - x - w - v - u.
    # Vertices: u, x, w, v.
    # Are they distinct?
    # u != v (distinct).
    # w != u (w in N(v), u not in N(v) initially? No, u not in adj[v] because we are checking adding it. So u not in N(v). Thus w != u).
    # w != v (simple graph).
    # x != u (x in N(u)).
    # x != v (if x=v, v in N(u)? No, v not in adj[u]).
    # x != w (if x=w, w in N(u)? Then u-w edge. And w in N(w)? No).
    # So if x exists, it forms a C4.
    # So we require N(u) \cap N(w) is EMPTY for all w in N(v).
    
    for w in adj[v]:
        if adj[u].intersection(adj[w]):
            return False
            
    # Condition 3: Symmetric to Condition 2
    # For all w in N(u), N(v) \cap N(w) must be empty
    for w in adj[u]:
        if adj[v].intersection(adj[w]):
            return False
            
    return True

def is_c4_free(adj, n):
    for u in range(n):
        for v in range(u+1, n):
            if len(adj[u].intersection(adj[v])) >= 2:
                return False
    return True
