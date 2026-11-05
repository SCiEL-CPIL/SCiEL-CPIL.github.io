"""C5-free graphs without independent 9-sets: Greedy edge addition from 8xK4 seed."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill

    def make_8k4():
        n = 32
        edges = []
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return n, edges

    def check_c5_free(adj, n):
        """Check if graph is C5-free. adj is list of sets."""
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    # Check for path of length 3 from u to v: u-a-b-v
                    for a in adj[u]:
                        if a == v:
                            continue
                        for b in adj[a]:
                            if b == u or b == a:
                                continue
                            if v in adj[b]:
                                if len({u, a, b, v}) == 4:
                                    return False
        return True

    def has_alpha_ge_9(adj, n, time_limit=5.0):
        """Check if graph has an independent set of size >= 9."""
        t0 = time.time()
        
        # Convert adj to a set-of-sets for fast lookup
        adj_set = [adj[i] for i in range(n)]
        
        def backtrack(candidates, current_size):
            if time.time() - t0 > time_limit:
                return False
            if current_size >= 9:
                return True
            if not candidates:
                return False
            if len(candidates) + current_size < 9:
                return False
            
            # Pick vertex with minimum degree in the candidate subgraph
            # to maximize pruning
            min_deg = len(candidates)
            min_idx = 0
            for i, v in enumerate(candidates):
                deg = sum(1 for x in candidates if x in adj_set[v])
                if deg < min_deg:
                    min_deg = deg
                    min_idx = i
            
            v = candidates.pop(min_idx)
            
            # Include v
            new_candidates = [x for x in candidates if x not in adj_set[v]]
            if backtrack(new_candidates, current_size + 1):
                return True
            
            # Exclude v
            if backtrack(candidates, current_size):
                return True
            
            return False

        candidates = list(range(n))
        return backtrack(candidates, 0)

    def try_greedy_edge_addition(n, time_budget):
        """
        Start from 8xK4 (if n>=32) or a random C5-free graph,
        then greedily add edges that don't create C5.
        Periodically check alpha < 9.
        """
        random.seed(42 + n)
        
        # Build initial graph: 8xK4 for the first 32 vertices
        adj = [set() for _ in range(n)]
        edges = []
        
        if n >= 32:
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i + 1, 4):
                        adj[base + i].add(base + j)
                        adj[base + j].add(base + i)
                        edges.append([base + i, base + j])
        
        # Generate all possible additional edges
        all_possible = []
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    all_possible.append((u, v))
        
        random.shuffle(all_possible)
        
        t_start_search = time.time()
        
        for u, v in all_possible:
            if time.time() > t_start_search + time_budget:
                break
            
            # Check if adding (u, v) creates a C5
            creates_c5 = False
            for a in adj[u]:
                if a == v:
                    continue
                for b in adj[a]:
                    if b == u or b == a:
                        continue
                    if v in adj[b]:
                        if len({u, a, b, v}) == 4:
                            creates_c5 = True
                            break
                if creates_c5:
                    break
            
            if not creates_c5:
                # Add the edge
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
            
            # Periodically check alpha
            if len(edges) % 5 == 0:
                if not has_alpha_ge_9(adj, n, time_limit=3.0):
                    return n, edges
        
        # Final check
        if not has_alpha_ge_9(adj, n, time_limit=5.0):
            return n, edges
        
        return None

    # Try n = 33, 34, 35
    best_n, best_edges = make_8k4()
    
    for n_try in [33, 34, 35]:
        if time.time() > deadline - 25:
            break
        
        remaining = deadline - time.time()
        time_budget = remaining * 0.7
        if time_budget < 10:
            break
        
        result = try_greedy_edge_addition(n_try, time_budget)
        if result:
            best_n, best_edges = result
            continue
    
    # Verify the best solution
    if time.time() < deadline - 5:
        adj = [set() for _ in range(best_n)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        if not check_c5_free(adj, best_n):
            best_n, best_edges = make_8k4()
        elif has_alpha_ge_9(adj, best_n, time_limit=10.0):
            best_n, best_edges = make_8k4()

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
