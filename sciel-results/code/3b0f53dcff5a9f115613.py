import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 150 - 10  # Safety margin
    
    def is_c4_free(adj):
        n = len(adj)
        for u in range(n):
            for v in range(u+1, n):
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
        
        # We want to ensure min degree >= target_min_deg and C4-free.
        # Strategy: 
        # 1. Remove edges to break C4s.
        # 2. If degree drops below target, try to add edges that don't create C4s.
        
        max_iterations = 20000
        for _ in range(max_iterations):
            if time.time() > deadline:
                return None
            
            # Check for C4
            viol_u, viol_v = -1, -1
            for u in range(n):
                for v in range(u+1, n):
                    if len(adj[u] & adj[v]) > 1:
                        viol_u, viol_v = u, v
                        break
                if viol_u != -1:
                    break
            
            if viol_u == -1:
                # No C4s
                if min_degree(adj) >= target_min_deg:
                    return adj
                else:
                    # Try to add edges to boost degree
                    # Find a vertex with low degree
                    low_v = min(range(n), key=lambda x: len(adj[x]))
                    if len(adj[low_v]) >= target_min_deg:
                        return adj # Should have caught above, but safe
                    
                    # Try to add an edge from low_v to a non-neighbor w
                    # such that w has no common neighbors with low_v (to avoid C4)
                    # Actually, adding (low_v, w) creates C4 if N(low_v) & N(w) is not empty?
                    # No, adding (low_v, w) creates a C4 if there exists x such that 
                    # low_v-x-w-low_v? No.
                    # A C4 is u-v-w-x-u.
                    # If we add (u, v), it creates a C4 if there is a path of length 2 between u and v.
                    # i.e., if N(u) & N(v) is not empty.
                    # So we can only add (u, v) if N(u) & N(v) is empty.
                    
                    candidates = []
                    for w in range(n):
                        if w != low_v and w not in adj[low_v]:
                            if not (adj[low_v] & adj[w]):
                                candidates.append(w)
                    
                    if candidates:
                        w = random.choice(candidates)
                        adj[low_v].add(w)
                        adj[w].add(low_v)
                        continue
                    else:
                        # Cannot add any edge to low_v without creating C4
                        # If we are stuck here and min degree < target, we fail
                        return None
                        # Or we can try to remove edges elsewhere to free up space? 
                        # That's complex. Let's just fail this seed.

            else:
                # Remove an edge to break C4
                common = adj[viol_u] & adj[viol_v]
                # We need to reduce common size to <= 1
                # Pick a common neighbor a
                # To minimize degree loss, pick the edge (u, a) or (v, a) 
                # where the other endpoint has higher degree?
                # Or just pick one.
                a = random.choice(list(common))
                
                # Decide which edge to remove
                # If we remove (u, a), deg(u) and deg(a) decrease.
                # If we remove (v, a), deg(v) and deg(a) decrease.
                # We want to preserve min degree.
                # So remove from the vertex with higher degree?
                # If deg(u) > deg(v), remove (u, a).
                # This keeps deg(v) intact.
                
                if len(adj[viol_u]) >= len(adj[viol_v]):
                    adj[viol_u].remove(a)
                    adj[a].remove(viol_u)
                else:
                    adj[viol_v].remove(a)
                    adj[a].remove(viol_v)
                
        return None

    best_n = 39
    best_edges = []
    
    # Try n=46
    # Target min degree is 7.
    # Let's start with p=0.25 (exp deg ~11) to have slack.
    for p in [0.2, 0.25, 0.3, 0.35, 0.4]:
        for seed in range(50):
            if time.time() > deadline:
                break
            adj = try_construct(46, p, seed)
            if adj is not None:
                if is_c4_free(adj) and min_degree(adj) >= 46 - 39:
                    best_n = 46
                    edges = []
                    for u in range(46):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    best_edges = edges
                    break
        if best_n == 46:
            break
        if time.time() > deadline:
            break
            
    if best_n < 47 and time.time() < deadline - 10:
        # Try 47
        # Target min degree is 8.
        for p in [0.25, 0.3, 0.35, 0.4]:
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
