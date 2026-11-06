import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: The Andrásfai graph And(9) with n=26
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    best_result = {'n': n_seed, 'edges': edges_seed}
    best_n = n_seed

    # Try to find a graph for n=39, then 38, etc.
    for target_n in [39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27]:
        if time.time() > deadline:
            break
        if target_n <= best_n:
            continue
            
        # Time budget per n
        time_budget = 30 if target_n == 39 else (10 if target_n == 38 else 5)
        search_deadline = time.time() + time_budget
        
        found = search_for_graph(target_n, search_deadline)
        if found:
            best_result = found
            best_n = target_n
            if target_n == 39:
                break

    return best_result

def search_for_graph(n, deadline):
    full_mask = (1 << n) - 1
    
    def exact_check(adj):
        # 1. Triangle free
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    if adj[u] & adj[v]:
                        return False
        
        # 2. Alpha < 10
        # Check if complement has clique of size 10
        # Complement edge (u,v) if NOT in adj
        
        def dfs(clique, candidates):
            if len(clique) == 10:
                return True
            if len(candidates) < 10 - len(clique):
                return False
            
            # Order candidates by degree in complement subgraph
            best_v = -1
            max_deg = -1
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                non_neigh = candidates & (full_mask ^ adj[v])
                deg = non_neigh.bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                c ^= lsb
                
            if best_v == -1:
                return False
                
            # Branch 1: Include best_v
            new_candidates = candidates & (full_mask ^ adj[best_v]) & ~(1 << best_v)
            if dfs(clique + [best_v], new_candidates):
                return True
                
            # Branch 2: Exclude best_v
            candidates = candidates & ~(1 << best_v)
            if dfs(clique, candidates):
                return True
                
            return False

        return not dfs([], full_mask)

    # Search Loop
    while time.time() < deadline:
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Phase 1: Remove triangles
        for _ in range(100):
            found_triangle = False
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        common = adj[u] & adj[v]
                        if common:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            found_triangle = True
                            break
                if found_triangle:
                    break
            if not found_triangle:
                break
        
        # Phase 2: Reduce alpha
        for _ in range(100):
            # Find large independent set
            remaining = full_mask
            indep_set = []
            while remaining:
                best_v = -1
                min_deg = n
                r = remaining
                while r:
                    lsb = r & -r
                    v = lsb.bit_length() - 1
                    deg = (adj[v] & remaining).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                    r ^= lsb
                
                if best_v == -1:
                    break
                
                indep_set.append(best_v)
                remaining &= ~(adj[best_v] | (1 << best_v))
                
                if len(indep_set) >= 10:
                    break
            
            if len(indep_set) < 10:
                if exact_check(adj):
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
            
            if len(indep_set) >= 10:
                fixed = False
                for k in range(len(indep_set)):
                    for l in range(k+1, len(indep_set)):
                        u = indep_set[k]
                        v = indep_set[l]
                        if not (adj[u] & adj[v]):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            fixed = True
                            break
                    if fixed:
                        break
                if not fixed:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v and (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v and not ((adj[u] >> v) & 1) and not (adj[u] & adj[v]):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

    return None
