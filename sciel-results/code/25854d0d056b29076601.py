import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def popcount(x):
        return x.bit_count()

    def is_clique(vertices, adj):
        if len(vertices) <= 1:
            return True
        for u in vertices:
            for v in vertices:
                if u == v:
                    continue
                if not ((adj[u] >> v) & 1):
                    return False
        return True

    def find_max_clique(adj, n, rng):
        best_clique = []
        
        def bronk(R, P, X):
            nonlocal best_clique
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = R[:]
                return
            
            if len(R) + popcount(P) <= len(best_clique):
                return

            pivot_mask = P | X
            if pivot_mask:
                max_deg = -1
                candidates_for_pivot = []
                temp = pivot_mask
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    deg = popcount(adj[v] & P)
                    if deg > max_deg:
                        max_deg = deg
                        candidates_for_pivot = [v]
                    elif deg == max_deg:
                        candidates_for_pivot.append(v)
                
                if candidates_for_pivot:
                    u = candidates_for_pivot[rng.randint(0, len(candidates_for_pivot) - 1)]
                else:
                    u = -1
            else:
                u = -1

            if u != -1:
                candidates = P & ~adj[u]
                while candidates:
                    lsb = candidates & (-candidates)
                    v = lsb.bit_length() - 1
                    candidates ^= lsb
                    
                    R.append(v)
                    bronk(R, P & adj[v], X & adj[v])
                    R.pop()
                    
                    P &= ~lsb
                    X |= lsb
            else:
                while P:
                    lsb = P & (-P)
                    v = lsb.bit_length() - 1
                    P ^= lsb
                    
                    R.append(v)
                    bronk(R, P & adj[v], X & adj[v])
                    R.pop()
                    
                    X |= lsb

        bronk([], (1 << n) - 1, 0)
        return best_clique

    def greedy_partition_max_clique(n, edges, adj, rng, deadline):
        rem_adj = adj[:]
        remaining_edges_count = len(edges)
        cliques = []
        
        while remaining_edges_count > 0:
            if time.time() > deadline:
                for u in range(n):
                    mask = rem_adj[u] & ((1 << u) - 1)
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        cliques.append([u, v])
                        rem_adj[u] &= ~(1 << v)
                        rem_adj[v] &= ~(1 << u)
                        mask ^= lsb
                return cliques

            clique = find_max_clique(rem_adj, n, rng)
            
            if not clique:
                break
            
            edges_removed = 0
            for i in range(len(clique)):
                u_c = clique[i]
                for j in range(i + 1, len(clique)):
                    v_c = clique[j]
                    if (rem_adj[u_c] >> v_c) & 1:
                        rem_adj[u_c] &= ~(1 << v_c)
                        rem_adj[v_c] &= ~(1 << u_c)
                        edges_removed += 1
            
            if edges_removed == 0:
                break
                
            cliques.append(clique)
            remaining_edges_count -= edges_removed

        return cliques

    def local_search(cliques, n, adj):
        # Simple local search: try to merge or move
        # We limit iterations to keep it fast
        max_iters = 50
        improved = True
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            
            # Try merging
            for i in range(len(cliques)):
                for j in range(i + 1, len(cliques)):
                    union = set(cliques[i]) | set(cliques[j])
                    if is_clique(list(union), adj):
                        cliques[i] = list(union)
                        cliques.pop(j)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue

            # Try moving
            for i in range(len(cliques)):
                if len(cliques[i]) <= 1:
                    continue
                for v in cliques[i][:]:
                    new_i = [x for x in cliques[i] if x != v]
                    if not new_i or is_clique(new_i, adj):
                        for j in range(len(cliques)):
                            if i == j:
                                continue
                            new_j = cliques[j] + [v]
                            if is_clique(new_j, adj):
                                cliques[i] = new_i
                                cliques[j] = new_j
                                if not cliques[i]:
                                    cliques.pop(i)
                                    if j > i:
                                        j -= 1
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
        return cliques

    def solve_instance(n, p, seed, time_limit):
        edges = gen_graph(n, p, seed)
        if not edges:
            return []
            
        adj = build_adj(n, edges)
        
        baseline_cliques = [[u, v] for u, v in edges]
        best_cliques = baseline_cliques
        best_count = len(baseline_cliques)
        
        start = time.time()
        attempt = 0
        end_time = start + time_limit
        
        while time.time() < end_time:
            rng = random.Random(seed * 1000 + attempt)
            
            result = greedy_partition_max_clique(n, edges, adj, rng, end_time)
            
            if result is not None:
                # Apply local search
                polished = local_search(result, n, adj)
                count = len(polished)
                if count < best_count:
                    best_count = count
                    best_cliques = polished
            
            attempt += 1
            
        return best_cliques

    time_budget_1 = 40.0
    time_budget_2 = 70.0
    
    current_time = time.time()
    remaining = DEADLINE - current_time
    if time_budget_1 + time_budget_2 > remaining:
        scale = remaining / (time_budget_1 + time_budget_2)
        time_budget_1 *= scale
        time_budget_2 *= scale

    result1 = solve_instance(32, 0.6, 601, time_budget_1)
    
    if time.time() > DEADLINE:
        edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        actual_budget_2 = DEADLINE - time.time()
        result2 = solve_instance(64, 0.55, 602, actual_budget_2)
    
    return [result1, result2]
