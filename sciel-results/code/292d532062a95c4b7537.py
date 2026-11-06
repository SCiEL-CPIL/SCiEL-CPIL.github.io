import time
import random
import numpy as np

def construct():
    """
    Constructs a graph with no K5 and no independent 5-set.
    Attempts to find a 42-vertex graph by extending the 37-vertex Paley graph.
    Falls back to the best valid graph found (at least 37) if 42 is not found in time.
    """
    start_time = time.time()
    timeout = 300  # Safety margin before 330s kill

    # Fallback: Paley graph on 37 vertices
    def paley_graph(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    # Get initial 37-vertex graph as bitsets
    N_START = 37
    q = N_START
    residues = {(x * x) % q for x in range(1, q)}
    
    # Helper to convert adj list to edge list
    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    # Helper to find one bad set (K5 or independent 5-set)
    # Returns (type, vertices) where type=1 for K5, 0 for Independent Set
    # Returns None if no bad set found (graph is valid)
    def find_one_bad_set(adj, n):
        mask = (1 << n) - 1
        
        # Find K5
        for i in range(n):
            neighbors_i = adj[i] & ~((1 << (i + 1)) - 1)
            while neighbors_i:
                j_bit = neighbors_i & -neighbors_i
                j = j_bit.bit_length() - 1
                neighbors_i ^= j_bit
                
                common_ij = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
                
                temp_common = common_ij
                while temp_common:
                    k_bit = temp_common & -temp_common
                    k = k_bit.bit_length() - 1
                    temp_common ^= k_bit
                    
                    common_ijk = common_ij & adj[k] & ~((1 << (k + 1)) - 1)
                    
                    temp_common_ijk = common_ijk
                    while temp_common_ijk:
                        l_bit = temp_common_ijk & -temp_common_ijk
                        l = l_bit.bit_length() - 1
                        temp_common_ijk ^= l_bit
                        
                        if adj[k] & (1 << l):
                            common_ijkl = common_ijk & adj[l]
                            common_ijkl &= ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
                            if common_ijkl:
                                m_bit = common_ijkl & -common_ijkl
                                m = m_bit.bit_length() - 1
                                return (1, (i, j, k, l, m))

        # Find Independent 5-set
        comp_adj = [~adj[i] & mask for i in range(n)]
        for i in range(n):
            comp_adj[i] &= ~(1 << i)
            
        for i in range(n):
            neighbors_i = comp_adj[i] & ~((1 << (i + 1)) - 1)
            while neighbors_i:
                j_bit = neighbors_i & -neighbors_i
                j = j_bit.bit_length() - 1
                neighbors_i ^= j_bit
                
                common_ij = comp_adj[i] & comp_adj[j] & ~((1 << (j + 1)) - 1)
                
                temp_common = common_ij
                while temp_common:
                    k_bit = temp_common & -temp_common
                    k = k_bit.bit_length() - 1
                    temp_common ^= k_bit
                    
                    common_ijk = common_ij & comp_adj[k] & ~((1 << (k + 1)) - 1)
                    
                    temp_common_ijk = common_ijk
                    while temp_common_ijk:
                        l_bit = temp_common_ijk & -temp_common_ijk
                        l = l_bit.bit_length() - 1
                        temp_common_ijk ^= l_bit
                        
                        if comp_adj[k] & (1 << l):
                            common_ijkl = common_ijk & comp_adj[l]
                            common_ijkl &= ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
                            if common_ijkl:
                                m_bit = common_ijkl & -common_ijkl
                                m = m_bit.bit_length() - 1
                                return (0, (i, j, k, l, m))
                                
        return None

    best_n = N_START
    best_adj = [0] * N_START
    # Initialize best_adj with Paley graph
    for i in range(N_START):
        for j in range(i + 1, N_START):
            if (j - i) % q in residues:
                best_adj[i] |= (1 << j)
                best_adj[j] |= (1 << i)

    TARGET_N = 42
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout - 10:
            break
            
        # Start from 37-vertex Paley graph
        adj = [0] * TARGET_N
        for i in range(N_START):
            adj[i] = best_adj[i] # Copy the valid base
        
        current_n = N_START
        success = True
        
        for new_v in range(N_START, TARGET_N):
            if time.time() - start_time > timeout - 5:
                success = False
                break
                
            # Initialize edges for new_v randomly
            for u in range(new_v):
                if random.random() < 0.5:
                    adj[u] |= (1 << new_v)
                    adj[new_v] |= (1 << u)
            
            # Local search to repair
            repair_steps = 0
            max_repair_steps = 2000
            
            while repair_steps < max_repair_steps:
                if time.time() - start_time > timeout - 2:
                    success = False
                    break
                    
                bad = find_one_bad_set(adj, new_v + 1)
                if bad is None:
                    # Valid!
                    repair_steps = max_repair_steps + 1
                    break
                    
                is_clique, v_set = bad
                i, j, k, l, m = v_set
                vertices = [i, j, k, l, m]
                random.shuffle(vertices)
                u, v = vertices[0], vertices[1]
                
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                repair_steps += 1
                
            if not success:
                break
                
            if repair_steps <= max_repair_steps:
                # Failed to repair within max steps
                success = False
                break
                
            current_n = new_v + 1
            
        if success and current_n > best_n:
            best_n = current_n
            best_adj = adj[:]
            if best_n == TARGET_N:
                break

    # Return best found
    edges = adj_to_edges(best_adj, best_n)
    return best_n, edges
