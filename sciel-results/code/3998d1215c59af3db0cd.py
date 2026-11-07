# EVOLVE-BLOCK-START
import random
import itertools
import time

def construct():
    n = 15
    all_edges = list(itertools.combinations(range(n), 2))
    
    def bit(i):
        return 1 << i

    def is_valid(adj, degs):
        """
        Returns True if no forbidden 3-path exists.
        """
        deg_groups = [0] * (n + 1)
        for i in range(n):
            deg_groups[degs[i]] |= bit(i)
            
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & bit(v)):
                    continue
                
                S_mask = adj[u] & ~bit(v)
                T_mask = adj[v] & ~bit(u)
                
                for d in range(n + 1):
                    A = S_mask & deg_groups[d]
                    B = T_mask & deg_groups[d]
                    
                    if A and B:
                        # Violation if there exists s in A, t in B with s != t
                        if not (A == B and bin(A).count('1') == 1):
                            return False
        return True

    def get_adj_degs(edge_set):
        adj = [0] * n
        degs = [0] * n
        for u, v in edge_set:
            adj[u] |= bit(v)
            adj[v] |= bit(u)
            degs[u] += 1
            degs[v] += 1
        return adj, degs

    best_edges = set()
    best_count = 0
    
    # Greedy initialization
    def greedy_init():
        current = set()
        adj, degs = get_adj_degs(current)
        missing = list(all_edges)
        random.shuffle(missing)
        
        improved = True
        while improved:
            improved = False
            random.shuffle(missing)
            for e in missing:
                if e in current:
                    continue
                new_edges = current | {e}
                new_adj, new_degs = get_adj_degs(new_edges)
                if is_valid(new_adj, new_degs):
                    current = new_edges
                    adj, degs = new_adj, new_degs
                    missing.remove(e)
                    improved = True
                    break
        return current

    # B&B Search
    def search(current_edges, adj, degs, candidates, depth, node_limit):
        nonlocal best_edges, best_count
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
            
        if not candidates:
            return
            
        # Bound
        if len(current_edges) + len(candidates) <= best_count:
            return
            
        # Check validity
        if not is_valid(adj, degs):
            return
            
        # Node limit check
        if node_limit[0] <= 0:
            return
        node_limit[0] -= 1
        
        # Heuristic ordering: 
        # Calculate a score for each candidate
        # Score: number of 3-paths involving this edge where deg(s) == deg(t) is likely?
        # Simple: Random pick
        idx = random.randint(0, len(candidates) - 1)
        edge = candidates[idx]
        remaining = candidates[:idx] + candidates[idx+1:]
        
        u, v = edge
        new_adj = adj[:]
        new_degs = degs[:]
        new_adj[u] |= bit(v)
        new_adj[v] |= bit(u)
        new_degs[u] += 1
        new_degs[v] += 1
        new_current = current_edges | {edge}
        
        # Branch 1: Include
        search(new_current, new_adj, new_degs, remaining, depth + 1, node_limit)
        
        # Branch 2: Exclude
        search(current_edges, adj, degs, remaining, depth + 1, node_limit)

    start_time = time.time()
    time_limit = 55 # seconds
    
    for trial in range(50):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(trial)
        
        # Get a good starting point
        init_edges = greedy_init()
        if len(init_edges) > best_count:
            best_count = len(init_edges)
            best_edges = set(init_edges)
            
        # Run B&B from this point or from scratch?
        # From scratch is safer for B&B logic, but slow.
        # Let's run B&B from an empty set but with a high node limit
        # Or run B&B starting from init_edges?
        # If we start from init_edges, we only add edges. We don't remove.
        # This is "Greedy + B&B Add".
        
        adj, degs = get_adj_degs(init_edges)
        candidates = [e for e in all_edges if e not in init_edges]
        
        node_limit = [5000] # Low limit per trial
        search(set(init_edges), adj, degs, candidates, 0, node_limit)

    result = [[u, v] for (u, v) in best_edges]
    final_result = []
    for edge in result:
        u = int(edge[0])
        v = int(edge[1])
        final_result.append([u, v])
    return final_result
# EVOLVE-BLOCK-END
