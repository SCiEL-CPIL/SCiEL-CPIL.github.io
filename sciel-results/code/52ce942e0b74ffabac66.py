import random
import time

def construct():
    # Try n=47 first, then 46
    for n in [47, 46]:
        min_deg_required = n - 39
        
        # Try multiple random instances
        for trial in range(500):
            random.seed(trial) # Deterministic for reproducibility in debugging, but trial changes seed
            
            # Initialize G as empty
            adj = [set() for _ in range(n)]
            
            # 1. Start with a very sparse random graph to avoid C4s
            # Probability p should be low enough that C4s are unlikely
            # Expected C4s ~ n^4 * p^4. For n=47, p=0.05, 47^4 * 0.05^4 ~ 4.8M * 6.25e-6 ~ 30 C4s. 
            # Maybe p=0.02 is safer.
            p = 0.02
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < p:
                        adj[u].add(v)
                        adj[v].add(u)
                        
            # Check if initial graph has C4s. If yes, we can try to remove them, 
            # but since it's sparse, it might be easier to just restart if C4s exist.
            # Or, we can just proceed to add edges and check C4s then.
            # Actually, if it starts with C4s, it's invalid. 
            # Let's check and skip if C4s exist.
            has_c4 = False
            for a in range(n):
                for c in range(a+1, n):
                    if c not in adj[a]:
                        common = adj[a].intersection(adj[c])
                        if len(common) >= 2:
                            has_c4 = True
                            break
                if has_c4:
                    break
            if has_c4:
                continue

            # 2. Add edges to increase degrees
            # We want min degree >= min_deg_required
            success = True
            max_iter = 1000
            for _ in range(max_iter):
                # Check current min degree
                min_deg = min(len(adj[i]) for i in range(n))
                if min_deg >= min_deg_required:
                    break
                
                # Find vertices with degree < min_deg_required
                low_deg_vertices = [i for i in range(n) if len(adj[i]) < min_deg_required]
                
                # Try to add an edge incident to one of these vertices
                # Pick a random low degree vertex
                u = random.choice(low_deg_vertices)
                
                # Find candidates v such that (u,v) not in adj and adding it doesn't create C4
                # Condition for C4: |N(u) \ N(v)| ... no, |N(u) \cap N(v)| < 2
                # If |N(u) \cap N(v)| >= 2, adding (u,v) creates a C4.
                
                candidates = []
                for v in range(n):
                    if v == u or v in adj[u]:
                        continue
                    # Check C4 condition
                    # Intersection of neighbors
                    # Optimization: if len(adj[u]) + len(adj[v]) - len(adj[u].intersection(adj[v])) ... 
                    # Just check intersection size
                    inter = adj[u].intersection(adj[v])
                    if len(inter) < 2:
                        candidates.append(v)
                        
                if not candidates:
                    # No valid edge to add for u. 
                    # This might mean we are stuck. 
                    # We can try to "swap" or just fail this trial.
                    # For now, fail this trial.
                    success = False
                    break
                
                # Choose a candidate
                # Prefer candidates that also have low degree?
                # Or random?
                # Let's pick one that maximizes the degree of the other vertex? 
                # Or just random to maintain diversity.
                v = random.choice(candidates)
                
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                
            if not success:
                continue
                
            # Final Verification
            is_c4_free = True
            for a in range(n):
                for c in range(a+1, n):
                    if c not in adj[a]:
                        common = adj[a].intersection(adj[c])
                        if len(common) >= 2:
                            is_c4_free = False
                            break
                if not is_c4_free:
                    break
            
            if is_c4_free:
                min_deg_final = min(len(adj[i]) for i in range(n))
                if min_deg_final >= min_deg_required:
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
                    
    # Fallback: Return a valid small graph
    # For n=39, min deg 0 is required. Empty graph works.
    # But let's try to return something slightly better if 46/47 fails?
    # The task says maximize n. If 46 fails, we should return the best we found?
    # No, the loop returns immediately if it finds 47 or 46.
    # If it fails, it falls through.
    # We should probably try n=45, 44... if 46 fails?
    # The prompt says "Maximize n".
    # Let's add a fallback loop for smaller n.
    
    for n in [45, 44, 43, 42, 41, 40, 39]:
        min_deg_required = n - 39
        # Quick check: empty graph has min deg 0.
        # If min_deg_required <= 0, empty graph is valid.
        if min_deg_required <= 0:
            return {'n': n, 'edges': []}
        
        # Try to construct for this n
        for trial in range(100):
            random.seed(1000 + trial)
            adj = [set() for _ in range(n)]
            p = 0.02
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < p:
                        adj[u].add(v)
                        adj[v].add(u)
                        
            has_c4 = False
            for a in range(n):
                for c in range(a+1, n):
                    if c not in adj[a]:
                        if len(adj[a].intersection(adj[c])) >= 2:
                            has_c4 = True
                            break
                if has_c4: break
            if has_c4: continue

            success = True
            for _ in range(500):
                min_deg = min(len(adj[i]) for i in range(n))
                if min_deg >= min_deg_required:
                    break
                low_deg_vertices = [i for i in range(n) if len(adj[i]) < min_deg_required]
                u = random.choice(low_deg_vertices)
                candidates = []
                for v in range(n):
                    if v == u or v in adj[u]: continue
                    if len(adj[u].intersection(adj[v])) < 2:
                        candidates.append(v)
                if not candidates:
                    success = False
                    break
                v = random.choice(candidates)
                adj[u].add(v)
                adj[v].add(u)
                
            if not success: continue
            
            is_c4_free = True
            for a in range(n):
                for c in range(a+1, n):
                    if c not in adj[a]:
                        if len(adj[a].intersection(adj[c])) >= 2:
                            is_c4_free = False
                            break
                if not is_c4_free: break
            
            if is_c4_free:
                if min(len(adj[i]) for i in range(n)) >= min_deg_required:
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}

    # Absolute fallback
    return {'n': 39, 'edges': []}
