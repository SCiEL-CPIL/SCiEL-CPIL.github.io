import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def _has_c4(adj, n):
        """Check if graph has a C4 using common neighbor count. Returns (bool, (u,v) if found)."""
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if len(common) > 1:
                    return True, (u, v, list(common)[:2])
        return False, None
    
    def _is_c4_free(adj, n):
        has, _ = _has_c4(adj, n)
        return not has
    
    def _try_greedy(n, target_deg, seed, deadline):
        """Try to build C4-free graph on n vertices with min_degree >= target_deg."""
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        
        # Precompute all edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        def is_safe(u, v):
            """Check if adding edge (u,v) keeps graph C4-free."""
            if len(adj[u] & adj[v]) > 1:
                return False
            for z in adj[v]:
                if z == u:
                    continue
                if len(adj[u] & adj[z]) >= 1:
                    return False
            for z in adj[u]:
                if z == v:
                    continue
                if len(adj[v] & adj[z]) >= 1:
                    return False
            return True
        
        # Multi-pass greedy with randomized ordering
        max_passes = n * 4
        for pass_num in range(max_passes):
            if time.time() > deadline:
                break
            
            # Check if target already met
            if all(len(adj[i]) >= target_deg for i in range(n)):
                break
            
            # Shuffle edges for diversity
            shuffled = all_edges[:]
            rng.shuffle(shuffled)
            
            best_edge = None
            best_key = None
            
            for u, v in shuffled:
                if v in adj[u]:
                    continue
                
                deg_u = len(adj[u])
                deg_v = len(adj[v])
                
                if deg_u >= target_deg and deg_v >= target_deg:
                    continue
                
                if not is_safe(u, v):
                    continue
                
                key = (max(deg_u, deg_v), min(deg_u, deg_v))
                
                if best_key is None or key < best_key:
                    best_key = key
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
        
        # Final verification
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg >= target_deg:
            if _is_c4_free(adj, n):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return edges
        return None
    
    def _try_random_greedy(n, target_deg, seed, deadline):
        """Randomized greedy: pick random safe edges, prioritize low-degree vertices."""
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        def is_safe(u, v):
            if len(adj[u] & adj[v]) > 1:
                return False
            for z in adj[v]:
                if z == u:
                    continue
                if len(adj[u] & adj[z]) >= 1:
                    return False
            for z in adj[u]:
                if z == v:
                    continue
                if len(adj[v] & adj[z]) >= 1:
                    return False
            return True
        
        # Phase 1: Add edges to boost low-degree vertices
        for _ in range(n * 5):
            if time.time() > deadline:
                break
            
            if all(len(adj[i]) >= target_deg for i in range(n)):
                break
            
            # Find vertices below target
            under = [i for i in range(n) if len(adj[i]) < target_deg]
            if not under:
                break
            
            # Pick a random under-degree vertex and try to add an edge to it
            attempts = 0
            added = False
            for _ in range(50):
                if attempts >= 50:
                    break
                attempts += 1
                u = rng.choice(under)
                candidates = [v for v in range(n) if v != u and v not in adj[u]]
                if not candidates:
                    break
                v = rng.choice(candidates)
                if is_safe(u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    added = True
                    break
            
            if not added:
                # Try all pairs involving under-degree vertices
                found = False
                for u in under:
                    for v in range(n):
                        if v == u or v in adj[u]:
                            continue
                        if len(adj[v]) >= target_deg and len(adj[u]) >= target_deg:
                            continue
                        if is_safe(u, v):
                            adj[u].add(v)
                            adj[v].add(u)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    break
        
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg >= target_deg:
            if _is_c4_free(adj, n):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return edges
        return None
    
    def _try_repair(n, target_deg, seed, deadline):
        """Build a dense graph, then repair C4s by removing edges."""
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        # Start with a random dense graph
        for u, v in all_edges:
            if rng.random() < 0.7:  # ~70% edge probability
                adj[u].add(v)
                adj[v].add(u)
        
        # Repair: remove edges that create C4s
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            has_c4, info = _has_c4(adj, n)
            if not has_c4:
                break
            
            if info is None:
                break
            
            u, v, commons = info
            # commons has at least 2 vertices; pick one to remove an edge
            # Remove edge (u, c) or (v, c) for c in commons
            removed = False
            for c in commons:
                # Check if removing (u,c) or (v,c) is safe for degrees
                deg_u = len(adj[u])
                deg_v = len(adj[v])
                deg_c = len(adj[c])
                
                # Prefer removing from higher-degree vertex
                if deg_u > deg_v and deg_u > target_deg:
                    adj[u].discard(c)
                    adj[c].discard(u)
                    removed = True
                    break
                elif deg_v > deg_u and deg_v > target_deg:
                    adj[v].discard(c)
                    adj[c].discard(v)
                    removed = True
                    break
                elif deg_c > target_deg:
                    # Remove (u,c) if u has more degree, else (v,c)
                    if deg_u >= deg_v:
                        adj[u].discard(c)
                        adj[c].discard(u)
                    else:
                        adj[v].discard(c)
                        adj[c].discard(v)
                    removed = True
                    break
            
            if not removed:
                break
        
        # Now try to add back edges to meet degree requirements
        for _ in range(n * 3):
            if time.time() > deadline:
                break
            
            if all(len(adj[i]) >= target_deg for i in range(n)):
                break
            
            # Find most under-degree vertex
            min_deg_vertex = min(range(n), key=lambda i: len(adj[i]))
            if len(adj[min_deg_vertex]) >= target_deg:
                break
            
            u = min_deg_vertex
            added = False
            for v in range(n):
                if v == u or v in adj[u]:
                    continue
                if len(adj[v]) >= target_deg and len(adj[u]) >= target_deg:
                    continue
                # Check if safe
                if len(adj[u] & adj[v]) > 1:
                    continue
                safe = True
                for z in adj[v]:
                    if z == u:
                        continue
                    if len(adj[u] & adj[z]) >= 1:
                        safe = False
                        break
                if safe:
                    for z in adj[u]:
                        if z == v:
                            continue
                        if len(adj[v] & adj[z]) >= 1:
                            safe = False
                            break
                if safe:
                    adj[u].add(v)
                    adj[v].add(u)
                    added = True
                    break
            
            if not added:
                break
        
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg >= target_deg:
            if _is_c4_free(adj, n):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return edges
        return None
    
    best_n = 39
    best_edges = []
    
    # Try from highest n down
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline:
            break
        
        target_deg = max(1, n - 39)
        
        # Try multiple strategies with multiple seeds
        max_seeds = 30 if n >= 46 else 50
        found = False
        
        for seed in range(max_seeds):
            if time.time() > deadline:
                break
            
            # Alternate between strategies
            if seed % 3 == 0:
                edges = _try_greedy(n, target_deg, seed, deadline)
            elif seed % 3 == 1:
                edges = _try_random_greedy(n, target_deg, seed, deadline)
            else:
                edges = _try_repair(n, target_deg, seed, deadline)
            
            if edges is not None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                found = True
                break
        
        if found:
            continue
        else:
            # Didn't find for this n, continue to lower n
            continue
    
    if best_n <= 39:
        return {'n': 39, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
