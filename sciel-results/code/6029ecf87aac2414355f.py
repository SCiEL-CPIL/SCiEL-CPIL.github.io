"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    deadline = time.time() + 140  # 140s safety margin (evaluator kills at 150s)
    
    def has_c4(adj, n):
        """Check if graph has a C4 using matrix multiplication."""
        if n < 4:
            return False
        common = adj @ adj.T
        np.fill_diagonal(common, 0)
        return np.any(common >= 2)
    
    def find_c4s(adj, n):
        """Return list of C4s as (a,b,c,d) where a-b-c-d-a is a 4-cycle."""
        common = adj @ adj.T
        np.fill_diagonal(common, 0)
        c4s = []
        for a in range(n):
            for b in range(a + 1, n):
                if common[a, b] >= 2:
                    # Find common neighbors of a and b
                    cn_a = np.where(adj[a] == 1)[0]
                    cn_b = np.where(adj[b] == 1)[0]
                    common_neigh = np.intersect1d(cn_a, cn_b)
                    for i in range(len(common_neigh)):
                        for j in range(i + 1, len(common_neigh)):
                            c = common_neigh[i]
                            d = common_neigh[j]
                            # a-b-c-d-a: need edges a-b, b-c, c-d, d-a
                            if adj[a, b] and adj[b, c] and adj[c, d] and adj[d, a]:
                                c4s.append((a, b, c, d))
        return c4s
    
    def min_degree(adj, n):
        return int(np.min(adj.sum(axis=1)))
    
    def degrees(adj, n):
        return adj.sum(axis=1)
    
    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges
    
    def greedy_construct(n, target_deg, max_time=10):
        """Greedy: start with random edges, add more while avoiding C4."""
        start_time = time.time()
        # Start with a random graph with moderate density
        p_init = max(0.1, target_deg / n * 0.8)
        adj = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p_init:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        # Try to add edges to boost min degree while avoiding C4
        # Get all non-edges
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    non_edges.append((i, j))
        random.shuffle(non_edges)
        
        for (u, v) in non_edges:
            if time.time() - start_time > max_time:
                break
            if min_degree(adj, n) >= target_deg and not has_c4(adj, n):
                break
            # Check if adding edge (u,v) creates a C4
            # A C4 involving edge (u,v) would be u-v-x-y-u where x,y are common neighbors
            # Actually, adding edge (u,v) creates a C4 if there exist x,y such that
            # u-x, x-v, v-y, y-u are all edges (forming u-x-v-y-u)
            # Or u-x, x-y, y-v, v-u... wait, v-u is the new edge.
            # C4: u-v-x-y-u. Need: adj[u,v] (new), adj[v,x], adj[x,y], adj[y,u]
            # So we need x in N(v) and y in N(u) such that adj[x,y]=1
            # Also C4: u-x-v-y-u: need adj[u,x], adj[x,v], adj[v,y](new? no, v-y must exist), adj[y,u]
            # Actually simpler: adding edge (u,v) creates a C4 iff there exist distinct x,y 
            # such that u-x, x-v, v-y, y-u are all present (x,y common neighbors of u and v)
            # OR u-x, x-y, y-v with v-u new... 
            # The standard check: adding (u,v) creates a C4 iff |N(u) ∩ N(v)| >= 2
            # because if x,y are both in N(u)∩N(v), then u-x-v-y-u is a C4.
            common = np.intersect1d(np.where(adj[u] == 1)[0], np.where(adj[v] == 1)[0])
            if len(common) >= 2:
                continue  # would create C4
            adj[u, v] = 1
            adj[v, u] = 1
        
        if has_c4(adj, n):
            return None
        if min_degree(adj, n) >= target_deg:
            return adj
        return None
    
    def random_with_repair(n, target_deg, max_attempts=2000, max_time=20):
        """Random sampling with C4 repair."""
        start_time = time.time()
        for attempt in range(max_attempts):
            if time.time() - start_time > max_time:
                break
            
            # Vary the probability
            p = random.uniform(0.3, 0.7)
            adj = np.zeros((n, n), dtype=np.uint8)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i, j] = 1
                        adj[j, i] = 1
            
            # Check min degree
            if min_degree(adj, n) < target_deg:
                continue
            
            # Check C4
            if not has_c4(adj, n):
                return adj
            
            # Try to repair: remove edges to break C4s
            # Find all C4s and remove one edge from each
            repaired = adj.copy()
            for repair_iter in range(100):
                if time.time() - start_time > max_time:
                    break
                if not has_c4(repaired, n):
                    break
                c4s = find_c4s(repaired, n)
                if not c4s:
                    break
                # Remove a random edge from a random C4
                c4 = random.choice(c4s)
                a, b, c, d = c4
                # Edges in C4: (a,b), (b,c), (c,d), (d,a)
                candidate_edges = [(a, b), (b, c), (c, d), (d, a)]
                # Only consider edges that exist
                valid_edges = [(u, v) for (u, v) in candidate_edges if repaired[u, v] == 1]
                if not valid_edges:
                    break
                u, v = random.choice(valid_edges)
                repaired[u, v] = 0
                repaired[v, u] = 0
            
            if has_c4(repaired, n):
                continue
            if min_degree(repaired, n) >= target_deg:
                return repaired
        
        return None
    
    def local_search(n, target_deg, adj, max_iters=5000, max_time=15):
        """Local search: perturb a valid graph to try to maintain/improve."""
        start_time = time.time()
        best_adj = adj.copy()
        best_min_deg = min_degree(best_adj, n)
        
        for it in range(max_iters):
            if time.time() - start_time > max_time:
                break
            
            # Randomly flip an edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            new_adj = best_adj.copy()
            if new_adj[u, v] == 1:
                new_adj[u, v] = 0
                new_adj[v, u] = 0
            else:
                new_adj[u, v] = 1
                new_adj[v, u] = 1
            
            if has_c4(new_adj, n):
                continue
            md = min_degree(new_adj, n)
            if md >= target_deg and md >= best_min_deg:
                best_adj = new_adj
                best_min_deg = md
                if md >= target_deg:
                    # Could try to increase further, but we're satisfied
                    pass
        
        return best_adj
    
    # Search for largest n
    best_result = {'n': 39, 'edges': []}
    
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline - 10:
            break
        target_deg = n - 39
        if target_deg < 1:
            continue
        
        # Phase 1: Greedy construction (multiple attempts with different seeds)
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            random.seed(attempt * 1000 + n)
            adj = greedy_construct(n, target_deg, max_time=8)
            if adj is not None:
                edges = edges_from_adj(adj, n)
                return {'n': n, 'edges': edges}
        
        # Phase 2: Random with repair
        random.seed(n * 777 + 42)
        adj = random_with_repair(n, target_deg, max_attempts=3000, max_time=25)
        if adj is not None:
            edges = edges_from_adj(adj, n)
            return {'n': n, 'edges': edges}
        
        # Phase 3: Try local search from random starting points
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            random.seed(attempt * 500 + n * 13)
            p = random.uniform(0.4, 0.6)
            adj = np.zeros((n, n), dtype=np.uint8)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i, j] = 1
                        adj[j, i] = 1
            # Only do local search if min degree is close
            if min_degree(adj, n) >= target_deg - 2:
                adj = local_search(n, target_deg, adj, max_iters=3000, max_time=12)
                if not has_c4(adj, n) and min_degree(adj, n) >= target_deg:
                    edges = edges_from_adj(adj, n)
                    return {'n': n, 'edges': edges}
    
    # Fallback
    return best_result
# EVOLVE-BLOCK-END
