import time
import random

def has_independent_set_size_k(adj, n, k):
    """
    Returns True if the graph has an independent set of size k.
    Uses greedy to find a large set first, then backtracking if needed.
    """
    # Greedy attempt
    remaining = set(range(n))
    ind_set = []
    while remaining:
        # Pick vertex with min degree in remaining?
        # Or just random
        v = random.choice(list(remaining))
        ind_set.append(v)
        remaining.remove(v)
        remaining -= adj[v]
    
    if len(ind_set) >= k:
        return True
        
    # If greedy found < k, it might still be >= k.
    # Run backtracking
    # To speed up, use the greedy set as a guide?
    # Simple backtracking with pruning
    
    def dfs(current_set, candidates):
        if len(current_set) == k:
            return True
        if not candidates:
            return False
        if len(candidates) < k - len(current_set):
            return False
            
        # Heuristic: pick vertex with highest degree in candidates to prune neighbors?
        # Or lowest?
        # If we pick a vertex with high degree, we remove many candidates.
        # This helps if we are trying to prove NO set exists (prune branches).
        # If we are trying to find a set, we want to keep options open?
        # Actually, for "does there exist", picking high degree is often better for pruning "no" branches?
        # Let's just pick the first one for simplicity, or random.
        
        # Let's try to pick a vertex that is not in the greedy set?
        # No, let's just pick random.
        idx = random.randrange(len(candidates))
        v = candidates[idx]
        
        # Branch 1: Include v
        # New candidates: candidates without v and without neighbors of v
        new_candidates = []
        for u in candidates:
            if u != v and u not in adj[v]:
                new_candidates.append(u)
        
        if dfs(current_set + [v], new_candidates):
            return True
            
        # Branch 2: Exclude v
        # New candidates: candidates without v
        new_candidates2 = candidates[:idx] + candidates[idx+1:]
        if dfs(current_set, new_candidates2):
            return True
            
        return False

    return dfs([], list(range(n)))

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Fallback
    fallback_n = 26
    fallback_edges = []
    for i in range(fallback_n):
        for j in range(i+1, fallback_n):
            if (j - i) % fallback_n % 3 == 1:
                fallback_edges.append([i, j])
    fallback = {'n': fallback_n, 'edges': fallback_edges}
    
    for n in range(40, 25, -1):
        # Time budget for this n
        # If n=40, give it 30s. If n=39, 20s, etc.
        # Or just use remaining time.
        # Let's just use the main deadline.
        
        attempts = 0
        while time.time() < deadline:
            attempts += 1
            if time.time() > deadline:
                break
            
            # 1. Generate random triangle-free graph
            adj = [set() for _ in range(n)]
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if not (adj[u] & adj[v]):
                    adj[u].add(v)
                    adj[v].add(u)
            
            # 2. Repair
            for _ in range(200): # Max repair iterations
                if time.time() > deadline:
                    break
                
                # Find independent set
                remaining = set(range(n))
                ind_set = []
                while remaining:
                    v = random.choice(list(remaining))
                    ind_set.append(v)
                    remaining.remove(v)
                    remaining -= adj[v]
                
                if len(ind_set) < 10:
                    # Verify with exact check
                    if not has_independent_set_size_k(adj, n, 10):
                        edge_list = []
                        for u in range(n):
                            for v in adj[u]:
                                if u < v:
                                    edge_list.append([u, v])
                        return {'n': n, 'edges': edge_list}
                    # If verification fails, it means alpha >= 10, so continue
                    break # Stuck?
                
                # Try to add edges
                pairs_in_ind = [(i, j) for i in range(len(ind_set)) for j in range(i+1, len(ind_set))]
                random.shuffle(pairs_in_ind)
                
                added = False
                for idx_u, idx_v in pairs_in_ind:
                    u = ind_set[idx_u]
                    v = ind_set[idx_v]
                    if u not in adj[v]:
                        if not (adj[u] & adj[v]):
                            adj[u].add(v)
                            adj[v].add(u)
                            added = True
                            break
                if not added:
                    break # Stuck
            
            # If we didn't return, try next restart
    
    return fallback
