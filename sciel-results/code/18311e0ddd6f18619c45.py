import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Build Hadamard Graph of order 32
    N = 32
    # Sylvester construction for Hadamard matrix of order 2^k
    # H_1 = [[1]]
    # H_{2n} = [[H_n, H_n], [H_n, -H_n]]
    # We can generate the graph directly without building the full matrix.
    # Vertex i is connected to j if H[i][j] == -1.
    # H[i][j] = (-1)^(popcount(i & j)) ?
    # For Sylvester Hadamard, H[i][j] = (-1)^(i dot j) over GF(2).
    # So H[i][j] = -1 if popcount(i & j) is odd.
    
    n_base = N
    adj = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            # Check if H[i][j] == -1
            # i dot j = popcount(i & j)
            if bin(i & j).count('1') % 2 == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Verify base graph
    def is_c4_free(n, adj):
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if bin(common).count('1') > 1:
                    return False
        return True

    def max_independent_set_size(n, adj):
        best = [0]
        
        def popcount(x):
            return bin(x).count('1')

        def branch(candidates_mask, current_size):
            if candidates_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            if current_size + popcount(candidates_mask) <= best[0]:
                return
            
            # Pick vertex with highest degree in candidate subgraph
            best_v = -1
            best_deg = -1
            m = candidates_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = popcount(candidates_mask & adj[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m -= lsb
            
            if best_v == -1:
                return

            v_bit = 1 << best_v
            
            # Include best_v
            new_candidates = candidates_mask & ~adj[best_v] & ~v_bit
            branch(new_candidates, current_size + 1)
            
            # Exclude best_v
            branch(candidates_mask & ~v_bit, current_size)

        branch((1 << n) - 1, 0)
        return best[0]

    if not is_c4_free(n_base, adj):
        return {'n': n_base, 'edges': []} # Should not happen

    alpha_base = max_independent_set_size(n_base, adj)
    if alpha_base >= 11:
        # Base is invalid, return it anyway? No, task requires alpha < 11.
        # But 32-vertex Hadamard graph has alpha ~ 8.
        pass

    best_n = n_base
    best_adj = adj[:]

    # 2. Iteratively add vertices
    for step in range(15): # Try to add up to 15 vertices
        if time.time() > deadline:
            break
        
        n = len(adj)
        
        # Build conflict graph H
        # H[i] has neighbors j such that N(i) & N(j) != 0 in G
        h_adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] & adj[j]) != 0:
                    h_adj[i] |= (1 << j)
                    h_adj[j] |= (1 << i)
        
        # Find Max Independent Set in H
        # We need a function that returns the mask of the MIS
        def find_max_independent_set_mask(n, adj_h):
            best_mask = [0]
            best_size = [0]
            
            def popcount(x):
                return bin(x).count('1')

            def branch(candidates_mask, current_mask, current_size):
                if candidates_mask == 0:
                    if current_size > best_size[0]:
                        best_size[0] = current_size
                        best_mask[0] = current_mask
                    return
                
                if current_size + popcount(candidates_mask) <= best_size[0]:
                    return
                
                # Pick vertex with highest degree in candidate subgraph
                best_v = -1
                best_deg = -1
                m = candidates_mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    deg = popcount(candidates_mask & adj_h[v])
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    m -= lsb
                
                if best_v == -1:
                    return

                v_bit = 1 << best_v
                
                # Include best_v
                new_candidates = candidates_mask & ~adj_h[best_v] & ~v_bit
                branch(new_candidates, current_mask | v_bit, current_size + 1)
                
                # Exclude best_v
                branch(candidates_mask & ~v_bit, current_mask, current_size)

            branch((1 << n) - 1, 0, 0)
            return best_mask[0]

        mis_mask = find_max_independent_set_mask(n, h_adj)
        
        # Try adding a vertex connected to MIS
        # If MIS is empty, we can't add a vertex (isolated vertex in H means isolated in G? No.)
        if mis_mask == 0:
            break

        new_v = n
        adj_new = adj + [0]
        # Connect new_v to all vertices in mis_mask
        for i in range(n):
            if mis_mask & (1 << i):
                adj_new[new_v] |= (1 << i)
                adj_new[i] |= (1 << new_v)
        
        # Check C4
        if not is_c4_free(n + 1, adj_new):
            # Should not happen if MIS is correct, but just in case
            continue
            
        # Check Alpha
        alpha_new = max_independent_set_size(n + 1, adj_new)
        
        if alpha_new < 11:
            adj = adj_new
            best_n = n + 1
            best_adj = adj_new[:]
            if best_n >= 39:
                break
        else:
            # Try to find a smaller independent set?
            # For now, if the largest MIS doesn't work, we might be stuck.
            # We can try to remove one vertex from MIS and retry?
            # Or just break?
            # Let's try to find a valid set by backtracking on the MIS choice.
            # This is complex. Let's just break if the best candidate fails.
            # But maybe a smaller set works?
            # Let's try a few random subsets of MIS.
            found = False
            for _ in range(10):
                # Random subset of MIS
                subset_mask = 0
                m = mis_mask
                while m:
                    lsb = m & -m
                    if random.random() < 0.8: # Keep 80%
                        subset_mask |= lsb
                    m -= lsb
                if subset_mask == 0:
                    continue
                
                adj_try = adj + [0]
                for i in range(n):
                    if subset_mask & (1 << i):
                        adj_try[new_v] |= (1 << i)
                        adj_try[i] |= (1 << new_v)
                
                if is_c4_free(n + 1, adj_try):
                    alpha_try = max_independent_set_size(n + 1, adj_try)
                    if alpha_try < 11:
                        adj = adj_try
                        best_n = n + 1
                        best_adj = adj_try[:]
                        found = True
                        if best_n >= 39:
                            break
                        break
            if not found:
                break

    edge_list = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])
                
    return {'n': best_n, 'edges': edge_list}
