import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def time_remaining():
        return deadline - time.time()
    
    def check_c4(n, adj):
        """Return True if graph has a C4."""
        for u in range(n):
            for v in range(u + 1, n):
                common_count = 0
                for w in adj[u]:
                    if w in adj[v]:
                        common_count += 1
                        if common_count >= 2:
                            return True
        return False
    
    def is_valid(n, edges, min_deg_target):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        if check_c4(n, adj):
            return False
        min_deg = min(len(adj[i]) for i in range(n))
        return min_deg >= min_deg_target
    
    def greedy_build(n, seed_val):
        random.seed(seed_val)
        adj = [set() for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time_remaining() < 3:
                break
            common = adj[u] & adj[v]
            if len(common) < 2:
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        return n, edges, adj
    
    def local_improve(n, edges, adj, min_deg_target, seed_val):
        """Try to improve the graph by swapping edges."""
        random.seed(seed_val)
        current_edges = list(edges)
        current_adj = [set(s) for s in adj]
        
        for _ in range(50):
            if time_remaining() < 3:
                break
            # Try removing a random edge and adding a different one
            if not current_edges:
                break
            idx = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[idx]
            # Remove edge
            current_adj[u].discard(v)
            current_adj[v].discard(u)
            current_edges.pop(idx)
            
            # Try to add a new edge
            candidates = [(i, j) for i in range(n) for j in range(i + 1, n) 
                         if i not in current_adj[j]]
            random.shuffle(candidates)
            added = False
            for a, b in candidates:
                common = current_adj[a] & current_adj[b]
                if len(common) < 2:
                    current_adj[a].add(b)
                    current_adj[b].add(a)
                    current_edges.append([a, b])
                    added = True
                    break
            if not added:
                # Put edge back
                current_adj[u].add(v)
                current_adj[v].add(u)
                current_edges.append([u, v])
        
        return current_edges, current_adj
    
    best_result = {'n': 39, 'edges': []}
    
    for n in range(47, 38, -1):
        if time_remaining() < 15:
            break
        
        min_deg_target = n - 39
        if min_deg_target < 0:
            min_deg_target = 0
        
        found = False
        
        for attempt in range(300):
            if time_remaining() < 15:
                break
            
            seed_val = attempt * 7919 + n * 104729 + 12345
            n_found, edges_found, adj_found = greedy_build(n, seed_val)
            
            min_deg = min(len(adj_found[i]) for i in range(n_found))
            if min_deg >= min_deg_target and not check_c4(n_found, adj_found):
                best_result = {'n': n_found, 'edges': edges_found}
                found = True
                break
        
        if found:
            break
    
    return best_result
