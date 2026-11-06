import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 290

    # 1. Generate Paley 37
    def get_paley_37():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = []
        adj = [0] * q
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    edges.append((i, j))
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return q, adj, edges

    n, adj, edges = get_paley_37()
    
    # Helper to get list of cliques and independent sets of size k
    def get_k_sets(n, adj, k, is_clique):
        sets = []
        for comb in combinations(range(n), k):
            valid = True
            for i in range(k):
                for j in range(i + 1, k):
                    u, v = comb[i], comb[j]
                    connected = bool(adj[u] & (1 << v))
                    if is_clique and not connected:
                        valid = False
                        break
                    if not is_clique and connected:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                sets.append(comb)
        return sets

    # Initialize K3, K4, bar_K3, bar_K4
    k3s = get_k_sets(n, adj, 3, True)
    bar_k3s = get_k_sets(n, adj, 3, False)
    k4s = get_k_sets(n, adj, 4, True)
    bar_k4s = get_k_sets(n, adj, 4, False)

    best_n = n
    best_adj = adj
    best_edges = edges

    # Try to extend
    for target_n in range(n + 1, 43):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Try to add vertex target_n
        success = False
        # Try multiple random restarts for the new vertex's connections
        for attempt in range(50):
            # Random initial state for new vertex connections
            new_adj = [0] * target_n
            for i in range(n):
                new_adj[i] = adj[i]
            
            # Determine connections for vertex 'n' (index n)
            # Randomly initialize
            conn_mask = 0
            for i in range(n):
                if random.random() < 0.5:
                    conn_mask |= (1 << i)
            
            # Local search to fix violations
            # Violation: 
            # 1. Any K4 in old graph where new vertex connects to all 4.
            # 2. Any bar_K4 in old graph where new vertex connects to none.
            
            def count_violations(conn_mask):
                viol = 0
                # Check K4s
                for k4 in k4s:
                    # If new vertex connects to all 4, it's a K5
                    if all(conn_mask & (1 << v) for v in k4):
                        viol += 1
                # Check bar_K4s
                for bk4 in bar_k4s:
                    # If new vertex connects to none, it's a bar_K5
                    if not any(conn_mask & (1 << v) for v in bk4):
                        viol += 1
                return viol

            current_viol = count_violations(conn_mask)
            
            # SA to reduce violations
            T = 1.0
            for step in range(200):
                if time.time() - start_time > time_limit - 5:
                    break
                # Flip a random bit
                idx = random.randint(0, n - 1)
                new_mask = conn_mask ^ (1 << idx)
                new_viol = count_violations(new_mask)
                
                delta = new_viol - current_viol
                if delta < 0 or random.random() < pow(2.71828, -delta / T):
                    conn_mask = new_mask
                    current_viol = new_viol
                    
                    if current_viol == 0:
                        break
                T *= 0.95
                if T < 0.01:
                    T = 0.01

            if current_viol == 0:
                # Success! Update graph
                new_adj[n] = conn_mask
                for i in range(n):
                    if conn_mask & (1 << i):
                        new_adj[i] |= (1 << n)
                        edges.append((i, n))
                
                # Update K3, K4, bar_K3, bar_K4 lists
                # New K3s involving n: Pairs (i,j) in K3s_old such that i,j connected to n
                # Actually, we need to find new K3s:
                # A K3 involving n is {n, i, j} where i,j are neighbors of n and (i,j) is an edge.
                new_k3s = []
                neighbors_of_n = [i for i in range(n) if conn_mask & (1 << i)]
                for i_idx in range(len(neighbors_of_n)):
                    for j_idx in range(i_idx + 1, len(neighbors_of_n)):
                        i, j = neighbors_of_n[i_idx], neighbors_of_n[j_idx]
                        if new_adj[i] & (1 << j):
                            new_k3s.append((i, j, n))
                
                # New bar_K3s involving n: Pairs (i,j) in bar_K3s_old such that i,j not connected to n
                new_bar_k3s = []
                non_neighbors_of_n = [i for i in range(n) if not (conn_mask & (1 << i))]
                for i_idx in range(len(non_neighbors_of_n)):
                    for j_idx in range(i_idx + 1, len(non_neighbors_of_n)):
                        i, j = non_neighbors_of_n[i_idx], non_neighbors_of_n[j_idx]
                        if not (new_adj[i] & (1 << j)):
                            new_bar_k3s.append((i, j, n))
                
                # New K4s involving n: Triples (i,j,k) in K3s_old such that i,j,k connected to n
                # We can find them from new_k3s? No, new_k3s are K3s.
                # A K4 involving n is {n, i, j, k} where {i,j,k} is a K3 in old graph and n connects to i,j,k.
                new_k4s = []
                for k3 in k3s:
                    i, j, k = k3
                    if (conn_mask & (1 << i)) and (conn_mask & (1 << j)) and (conn_mask & (1 << k)):
                        new_k4s.append((i, j, k, n))
                        
                # New bar_K4s involving n: Triples (i,j,k) in bar_K3s_old such that i,j,k not connected to n
                new_bar_k4s = []
                for bk3 in bar_k3s:
                    i, j, k = bk3
                    if not (conn_mask & (1 << i)) and not (conn_mask & (1 << j)) and not (conn_mask & (1 << k)):
                        new_bar_k4s.append((i, j, k, n))
                        
                # Update lists
                k3s.extend(new_k3s)
                bar_k3s.extend(new_bar_k3s)
                k4s.extend(new_k4s)
                bar_k4s.extend(new_bar_k4s)
                
                # Update state
                n = target_n
                adj = new_adj
                success = True
                break

        if not success:
            # Failed to extend
            break

    return best_n if best_n > n else n, best_edges if best_n > n else edges
