import time
import random
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin
    
    n = 15
    all_edges = list(itertools.combinations(range(n), 2))
    
    def get_degrees(edge_set):
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg

    def is_valid_full(edge_set):
        deg = get_degrees(edge_set)
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    def is_valid_addition(deg, adj, new_edge):
        u, v = new_edge
        # Check paths s-u-v-t
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    return False
        # Check paths s-v-u-t
        for s in adj[v]:
            if s == u:
                continue
            for t in adj[u]:
                if t == v or t == s:
                    continue
                if deg[s] == deg[t]:
                    return False
        return True

    def local_search(edge_set, max_passes=5):
        deg = get_degrees(edge_set)
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        for _ in range(max_passes):
            improved = False
            # Shuffle to avoid bias? Or just iterate.
            for e in all_edges:
                if e in edge_set:
                    continue
                if is_valid_addition(deg, adj, e):
                    u, v = e
                    edge_set.add(e)
                    deg[u] += 1
                    deg[v] += 1
                    adj[u].add(v)
                    adj[v].add(u)
                    improved = True
            if not improved:
                break
        return edge_set

    best_edges = set()
    best_score = 0
    
    rng = random.Random(42)
    
    # Try multiple initializations
    for init_seed in range(20):
        if time.time() > DEADLINE:
            break
            
        rng_seed = 1000 + init_seed
        rng = random.Random(rng_seed)
        
        # Random initial sparse graph
        current_edges = set()
        # Maybe start with a random matching or just empty?
        # Or start with a random dense graph and repair?
        # Greedy from empty is safe.
        
        # Greedy fill
        edges_to_try = list(all_edges)
        rng.shuffle(edges_to_try)
        for e in edges_to_try:
            # Check validity
            # We need deg and adj for current_edges
            # This is inefficient if we compute deg/adj every time.
            # Let's maintain them.
            pass
        
        # Better: Maintain deg/adj during greedy fill
        deg = [0]*n
        adj = [set() for _ in range(n)]
        current_edges = set()
        
        for e in edges_to_try:
            if is_valid_addition(deg, adj, e):
                u, v = e
                current_edges.add(e)
                deg[u] += 1
                deg[v] += 1
                adj[u].add(v)
                adj[v].add(u)
                
        current_edges = local_search(current_edges)
        
        if len(current_edges) > best_score:
            best_score = len(current_edges)
            best_edges = set(current_edges)
            
    # ILS on best found
    while time.time() < DEADLINE:
        current_edges = set(best_edges)
        deg = get_degrees(current_edges)
        adj = [set() for _ in range(n)]
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Perturb: Remove k random edges
        k = random.randint(3, 15)
        edges_list = list(current_edges)
        if len(edges_list) > k:
            to_remove = random.sample(edges_list, k)
            for e in to_remove:
                u, v = e
                current_edges.remove(e)
                deg[u] -= 1
                deg[v] -= 1
                adj[u].discard(v)
                adj[v].discard(u)
        
        # Local search
        current_edges = local_search(current_edges)
        
        if len(current_edges) > best_score:
            best_score = len(current_edges)
            best_edges = set(current_edges)
            
    result = [[u, v] for u, v in sorted(best_edges)]
    return result
