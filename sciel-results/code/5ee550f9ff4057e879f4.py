import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def count_triangles(mask, adj):
        """Count triangles in the induced subgraph defined by mask."""
        count = 0
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            m ^= lsb
            neigh = adj[u] & mask & ~(1 << u)
            deg_sum = 0
            n_neigh = neigh
            while n_neigh:
                lsb2 = n_neigh & -n_neigh
                v = lsb2.bit_length() - 1
                n_neigh ^= lsb2
                deg_sum += (adj[v] & neigh).bit_count()
            count += deg_sum // 2
        return count // 3

    def count_indep_5(mask, adj):
        """Count independent sets of size 5 in the induced subgraph defined by mask."""
        vertices = []
        m = mask
        while m:
            lsb = m & -m
            vertices.append(lsb.bit_length() - 1)
            m ^= lsb
        
        count = 0
        n_v = len(vertices)
        if n_v < 5:
            return 0
        
        for i in range(n_v):
            for j in range(i+1, n_v):
                if (adj[vertices[i]] >> vertices[j]) & 1:
                    continue
                for k in range(j+1, n_v):
                    if (adj[vertices[i]] >> vertices[k]) & 1 or (adj[vertices[j]] >> vertices[k]) & 1:
                        continue
                    for l in range(k+1, n_v):
                        if (adj[vertices[i]] >> vertices[l]) & 1 or (adj[vertices[j]] >> vertices[l]) & 1 or (adj[vertices[k]] >> vertices[l]) & 1:
                            continue
                        for m_idx in range(l+1, n_v):
                            if (adj[vertices[i]] >> vertices[m_idx]) & 1 or (adj[vertices[j]] >> vertices[m_idx]) & 1 or (adj[vertices[k]] >> vertices[m_idx]) & 1 or (adj[vertices[l]] >> vertices[m_idx]) & 1:
                                continue
                            count += 1
        return count

    def calculate_cost(n, adj):
        k4_count = 0
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    k4_count += count_triangles(common, adj)
        
        total_i5_in_nonn = 0
        all_mask = (1 << n) - 1
        for u in range(n):
            non_n = all_mask & ~(adj[u] | (1 << u))
            total_i5_in_nonn += count_indep_5(non_n, adj)
            
        i6_count = total_i5_in_nonn // 6
        
        return k4_count + i6_count

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def is_valid(n, adj):
        return calculate_cost(n, adj) == 0

    def spectral_construction(n, iterations=10000, learning_rate=0.1):
        """Use spectral optimization to find a graph with low spectral radius."""
        # Initialize with random symmetric matrix with entries in {-1, 1}
        M = np.random.choice([-1, 1], size=(n, n)).astype(float)
        np.fill_diagonal(M, 0)
        # Make symmetric
        M = (M + M.T) / 2
        # Ensure diagonal is 0
        np.fill_diagonal(M, 0)
        
        best_M = M.copy()
        best_lambda = np.inf
        
        for it in range(iterations):
            if time.time() > deadline - 10:
                break
                
            # Compute eigenvalues
            try:
                eigenvalues = np.linalg.eigvalsh(M)
                current_lambda = max(abs(eigenvalues[0]), abs(eigenvalues[-1]))
            except:
                break
            
            if current_lambda < best_lambda:
                best_lambda = current_lambda
                best_M = M.copy()
            
            # Pick random entry to flip
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i == j:
                continue
            
            old_val = M[i, j]
            # Try flipping sign
            M[i, j] = -old_val
            M[j, i] = -old_val
            
            # Check if this improved spectral radius
            try:
                new_eigenvalues = np.linalg.eigvalsh(M)
                new_lambda = max(abs(new_eigenvalues[0]), abs(new_eigenvalues[-1]))
            except:
                M[i, j] = old_val
                M[j, i] = old_val
                continue
            
            if new_lambda > current_lambda:
                # Revert
                M[i, j] = old_val
                M[j, i] = old_val
            else:
                # Accept and update
                current_lambda = new_lambda
                if current_lambda < best_lambda:
                    best_lambda = current_lambda
                    best_M = M.copy()
        
        # Threshold to create graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if best_M[i, j] > 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        return adj

    best_graph = None
    best_n = 0

    # Try spectral construction for various n
    for target_n in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 15:
            break
        
        # Multiple spectral restarts
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            
            adj = spectral_construction(target_n, iterations=5000)
            current_cost = calculate_cost(target_n, adj)
            
            if current_cost == 0:
                best_graph = adj
                best_n = target_n
                break
            
            # If close to valid, try local search to fix violations
            if current_cost <= 5:
                # Local search to reduce violations
                T = 5.0
                for s in range(3000):
                    if time.time() > deadline - 10:
                        break
                    
                    u = random.randint(0, target_n - 1)
                    v = random.randint(0, target_n - 1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    
                    is_edge = bool((adj[u] >> v) & 1)
                    
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    
                    all_mask = (1 << target_n) - 1
                    non_common = all_mask & ~(adj[u] | adj[v]) & ~(1 << u) & ~(1 << v)
                    
                    tri_count = count_triangles(common, adj)
                    i5_count = count_indep_5(non_common, adj)
                    
                    if is_edge:
                        delta = -tri_count + i5_count
                    else:
                        delta = tri_count - i5_count
                    
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        current_cost += delta
                        
                        if current_cost == 0:
                            best_graph = adj
                            best_n = target_n
                            break
                    
                    T *= 0.995
                    if T < 0.5:
                        T = 0.5
                
                if best_n == target_n:
                    break
        
        if best_n == target_n:
            break
    
    if best_graph is None:
        # Fallback: try standard simulated annealing
        for target_n in [35, 34, 33, 32]:
            if time.time() > deadline - 15:
                break
            
            for restart in range(30):
                if time.time() > deadline - 10:
                    break
                
                adj = [0] * target_n
                for u in range(target_n):
                    for v in range(u+1, target_n):
                        if random.random() < 0.5:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                
                current_cost = calculate_cost(target_n, adj)
                if current_cost == 0:
                    best_graph = adj
                    best_n = target_n
                    break
                
                T = 10.0
                for s in range(5000):
                    if time.time() > deadline - 10:
                        break
                    
                    u = random.randint(0, target_n - 1)
                    v = random.randint(0, target_n - 1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    
                    is_edge = bool((adj[u] >> v) & 1)
                    
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    
                    all_mask = (1 << target_n) - 1
                    non_common = all_mask & ~(adj[u] | adj[v]) & ~(1 << u) & ~(1 << v)
                    
                    tri_count = count_triangles(common, adj)
                    i5_count = count_indep_5(non_common, adj)
                    
                    if is_edge:
                        delta = -tri_count + i5_count
                    else:
                        delta = tri_count - i5_count
                    
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        current_cost += delta
                        
                        if current_cost == 0:
                            best_graph = adj
                            best_n = target_n
                            break
                    
                    T *= 0.99
                    if T < 0.1:
                        T = 0.1
                
                if best_n == target_n:
                    break
            
            if best_n == target_n:
                break
    
    if best_graph is None:
        # Final fallback: Paley graph of order 17
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [0] * 17
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return {'n': 17, 'edges': get_edges(adj, 17)}
    else:
        return {'n': best_n, 'edges': get_edges(best_graph, best_n)}
