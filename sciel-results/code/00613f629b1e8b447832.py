import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 325.0  # 300s limit with buffer
    
    def time_left():
        return deadline - time.time()
    
    def count_triangles_in_set(candidates, adj):
        """Count triangles in the induced subgraph on 'candidates'."""
        n_cand = len(candidates)
        if n_cand < 3:
            return 0
        count = 0
        for i in range(n_cand):
            a = candidates[i]
            for j in range(i + 1, n_cand):
                b = candidates[j]
                if adj[a] & (1 << b):
                    common = adj[a] & adj[b]
                    for k in range(j + 1, n_cand):
                        c = candidates[k]
                        if common & (1 << c):
                            count += 1
        return count

    def count_ind_triangles_in_set(candidates, adj):
        """Count independent triangles (triangles in complement) in 'candidates'."""
        n_cand = len(candidates)
        if n_cand < 3:
            return 0
        count = 0
        for i in range(n_cand):
            a = candidates[i]
            for j in range(i + 1, n_cand):
                b = candidates[j]
                if not (adj[a] & (1 << b)):
                    for k in range(j + 1, n_cand):
                        c = candidates[k]
                        if not (adj[a] & (1 << c)) and not (adj[b] & (1 << c)):
                            count += 1
        return count

    def total_cost(adj, n):
        """Calculate total number of K5 and Independent 5-sets."""
        k5_count = 0
        ind5_count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    common_list = []
                    for k in range(n):
                        if k != i and k != j and (common & (1 << k)):
                            common_list.append(k)
                    k5_count += count_triangles_in_set(common_list, adj)
                else:
                    common_non_list = []
                    for k in range(n):
                        if k != i and k != j:
                            if not (adj[i] & (1 << k)) and not (adj[j] & (1 << k)):
                                common_non_list.append(k)
                    ind5_count += count_ind_triangles_in_set(common_non_list, adj)
        
        return k5_count // 10 + ind5_count // 10

    def delta_cost(i, j, adj, n, is_edge):
        """Calculate change in cost if we flip edge (i,j)."""
        if is_edge:
            common = adj[i] & adj[j]
            common_list = []
            for k in range(n):
                if k != i and k != j and (common & (1 << k)):
                    common_list.append(k)
            delta_k5 = -count_triangles_in_set(common_list, adj)
            
            common_non_list = []
            for k in range(n):
                if k != i and k != j:
                    if not (adj[i] & (1 << k)) and not (adj[j] & (1 << k)):
                        common_non_list.append(k)
            delta_ind5 = count_ind_triangles_in_set(common_non_list, adj)
            
            return delta_k5 + delta_ind5
        else:
            common = adj[i] & adj[j]
            common_list = []
            for k in range(n):
                if k != i and k != j and (common & (1 << k)):
                    common_list.append(k)
            delta_k5 = count_triangles_in_set(common_list, adj)
            
            common_non_list = []
            for k in range(n):
                if k != i and k != j:
                    if not (adj[i] & (1 << k)) and not (adj[j] & (1 << k)):
                        common_non_list.append(k)
            delta_ind5 = -count_ind_triangles_in_set(common_non_list, adj)
            
            return delta_k5 + delta_ind5

    # Try to find a valid graph for N = 42, 41, ... down to 38
    for N in range(42, 37, -1):
        if time_left() < 30:
            break
        
        # Multiple restarts for this N
        max_restarts = 50
        for restart in range(max_restarts):
            if time_left() < 10:
                break
            
            # Initialize random graph with p=0.5
            adj = [0] * N
            for i in range(N):
                for j in range(i + 1, N):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            current_cost = total_cost(adj, N)
            
            # Simulated Annealing
            T = 5.0
            cooling = 0.995
            min_T = 0.01
            steps = 0
            max_steps = 20000
            
            while T > min_T and steps < max_steps:
                if time_left() < 5:
                    break
                
                # Pick random edge
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                
                is_edge = bool(adj[i] & (1 << j))
                
                d_cost = delta_cost(i, j, adj, N, is_edge)
                
                # Accept if better or with probability exp(-d/T)
                if d_cost < 0 or (d_cost == 0) or random.random() < np.exp(-d_cost / T):
                    if is_edge:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    else:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    
                    current_cost += d_cost
                    if current_cost == 0:
                        edges = []
                        for u in range(N):
                            for v in range(u + 1, N):
                                if adj[u] & (1 << v):
                                    edges.append((u, v))
                        return N, edges
                
                T *= cooling
                steps += 1
                
                # If cost is stuck, perturb
                if steps % 500 == 0 and current_cost > 0:
                    for _ in range(5):
                        u = random.randint(0, N - 1)
                        v = random.randint(0, N - 1)
                        if u == v:
                            continue
                        if u > v:
                            u, v = v, u
                        if adj[u] & (1 << v):
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                    current_cost = total_cost(adj, N)
                    
    # Return the seed Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges
