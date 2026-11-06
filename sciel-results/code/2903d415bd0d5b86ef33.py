import time
import random
import math

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 115

    def get_state_stats(adj, edges):
        max_b = 0
        sum_b = 0
        for u, v in edges:
            common = adj[u] & adj[v]
            b = common.bit_count()
            if b > max_b:
                max_b = b
            sum_b += b
        return max_b, sum_b

    def solve_single_seed(seed):
        random.seed(seed)
        adj = [0] * n
        edges = set()
        
        # Initialization Strategy 1: Random
        # Strategy 2: Bipartite + 1 edge
        # Let's mix.
        
        if seed % 2 == 0:
            # Random
            while len(edges) < target_edges:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v:
                    u, v = min(u, v), max(u, v)
                    if (u, v) not in edges:
                        edges.add((u, v))
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        else:
            # Bipartite K20,20 + 1 edge
            # Partition 0: 0-19, Partition 1: 20-39
            for u in range(20):
                for v in range(20, 40):
                    edges.add((u, v))
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            # Add 1 edge in Partition 0
            u1, v1 = 0, 1
            edges.add((u1, v1))
            adj[u1] |= (1 << v1)
            adj[v1] |= (1 << u1)
            
        max_b, sum_b = get_state_stats(adj, edges)
        
        best_max = max_b
        best_sum = sum_b
        best_adj = adj[:]
        best_edges = edges.copy()
        
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() < deadline - 10 and T > min_T:
            # 1. Remove random edge
            # Bias: pick from top 20% highest book edges?
            # To avoid sorting, just pick random.
            # Or pick random, if book < avg, pick another?
            # Let's just pick random for simplicity and speed.
            e_rem = random.choice(tuple(edges))
            u1, v1 = e_rem
            
            # Save state for revert
            # adj is list of ints, easy to copy
            adj_backup = adj[:]
            edges_backup = edges.copy()
            
            # Apply removal
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            edges.discard(e_rem)
            
            # 2. Find best add
            # Sample 30 random non-edges
            best_add = None
            best_potential = 999
            for _ in range(30):
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2: continue
                u2, v2 = min(u2, v2), max(u2, v2)
                if (u2, v2) in edges: continue
                potential = (adj[u2] & adj[v2]).bit_count()
                if potential < best_potential:
                    best_potential = potential
                    best_add = (u2, v2)
                    
            if best_add is None:
                # Revert
                adj = adj_backup
                edges = edges_backup
                continue
                
            # Apply addition
            u2, v2 = best_add
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            edges.add(best_add)
            
            # 3. Evaluate
            new_max_b, new_sum_b = get_state_stats(adj, edges)
            
            # 4. Accept/Reject
            accept = False
            if new_max_b < max_b:
                accept = True
            elif new_max_b == max_b and new_sum_b < sum_b:
                accept = True
            else:
                delta = (new_max_b - max_b) * 10 + (new_sum_b - sum_b)
                if delta < 0:
                    accept = True
                else:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        accept = True
                        
            if accept:
                max_b = new_max_b
                sum_b = new_sum_b
                if max_b < best_max or (max_b == best_max and sum_b < best_sum):
                    best_max = max_b
                    best_sum = sum_b
                    best_adj = adj[:]
                    best_edges = edges.copy()
            else:
                # Revert
                adj = adj_backup
                edges = edges_backup
                
            T *= cooling
            
            if best_max <= 7:
                break
                
        return best_max, best_sum, best_edges

    global_best_max = 999
    global_best_sum = 999
    global_best_edges = None
    
    # Run multiple restarts
    # Estimate time per restart.
    # If each restart takes ~10s, we can do ~10 restarts.
    # Let's just run until deadline.
    
    for seed in range(100):
        if time.time() > deadline - 5:
            break
            
        max_b, sum_b, edges = solve_single_seed(seed)
        
        if max_b < global_best_max or (max_b == global_best_max and sum_b < global_best_sum):
            global_best_max = max_b
            global_best_sum = sum_b
            global_best_edges = edges
            
        if global_best_max <= 7:
            break
            
    if global_best_edges is None:
        # Fallback
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return [[u, v] for (u, v) in sorted(global_best_edges)]
