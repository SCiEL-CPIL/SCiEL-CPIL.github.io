import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    n = 18
    
    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        """Check if graph contains a K4 using bitmask operations."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    d_candidates = common & adj[c]
                    d_candidates &= ~((1 << (c + 1)) - 1)
                    if d_candidates:
                        return True
        return False

    def has_triangle_free_set_of_size(adj, target_size):
        """Check if there is a triangle-free induced subgraph of size at least target_size."""
        t0 = time.time()
        
        # Order vertices by degree descending
        deg = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -deg[v])
        
        def backtrack(idx, current_mask, current_size):
            if time.time() - t0 > 5.0:  # Timeout per check
                return "timeout"
            if idx == n:
                return current_size >= target_size
            
            # Prune: if even taking all remaining vertices doesn't reach target
            if current_size + (n - idx) < target_size:
                return False
            
            v = order[idx]
            
            # Option 1: Exclude v
            res = backtrack(idx + 1, current_mask, current_size)
            if res == "timeout":
                return "timeout"
            if res:
                return True
            
            # Option 2: Include v
            N_v_mask = adj[v]
            C = current_mask & N_v_mask
            
            # Check if C is independent (no edges in C)
            tmp_c = C
            valid = True
            while tmp_c:
                bit = tmp_c & (-tmp_c)
                u = bit.bit_length() - 1
                tmp_c ^= bit
                if adj[u] & C:
                    valid = False
                    break
            
            if valid:
                res = backtrack(idx + 1, current_mask | (1 << v), current_size + 1)
                if res == "timeout":
                    return "timeout"
                if res:
                    return True
            
            return False
        
        result = backtrack(0, 0, 0)
        if result == "timeout":
            return "timeout"
        return result

    def max_triangle_free_size(adj, time_limit=2.0):
        """Find largest induced triangle-free subgraph using branch and bound."""
        best = [0]
        t0 = time.time()
        timeout = [False]
        
        deg = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -deg[v])
        
        def backtrack(idx, current_mask, current_size):
            if timeout[0]:
                return
            if time.time() - t0 > time_limit:
                timeout[0] = True
                return
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            if current_size + (n - idx) <= best[0]:
                return
            
            v = order[idx]
            
            # Option 1: Exclude v
            backtrack(idx + 1, current_mask, current_size)
            
            # Option 2: Include v
            if not timeout[0]:
                N_v_mask = adj[v]
                C = current_mask & N_v_mask
                
                tmp_c = C
                valid = True
                while tmp_c:
                    bit = tmp_c & (-tmp_c)
                    u = bit.bit_length() - 1
                    tmp_c ^= bit
                    if adj[u] & C:
                        valid = False
                        break
                
                if valid:
                    backtrack(idx + 1, current_mask | (1 << v), current_size + 1)
        
        backtrack(0, 0, 0)
        return best[0]

    def construct_paley_17():
        """Construct Paley(17) graph on vertices 0..16."""
        adj = [0] * 17
        residues = {1, 4, 9, 16, 8, 2, 15, 13}
        for i in range(17):
            for j in range(i + 1, 17):
                diff = (j - i) % 17
                if diff in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    # Step 1: Get Paley(17)
    adj_17 = construct_paley_17()
    
    # Step 2: Determine T_17
    T_17 = max_triangle_free_size(adj_17 + [0], 5.0) # adj_17 + [0] is size 17? No, adj_17 is size 17.
    # Wait, max_triangle_free_size expects size n=18?
    # My helper functions use global n=18.
    # I need to be careful.
    
    # Let's create a dummy graph of size 18 for T_17 calculation?
    # Or just calculate T for Paley(17) directly.
    # The current helpers use n=18.
    # I will create a helper that works on a given adj list.
    
    # Re-implementing max_triangle_free_size to take adj as argument and n from len(adj)
    # This is safer.
    
    def max_tf_size_general(adj, time_limit=2.0):
        n_local = len(adj)
        best = [0]
        t0 = time.time()
        timeout = [False]
        
        deg = [bin(adj[v]).count('1') for v in range(n_local)]
        order = sorted(range(n_local), key=lambda v: -deg[v])
        
        def backtrack(idx, current_mask, current_size):
            if timeout[0]:
                return
            if time.time() - t0 > time_limit:
                timeout[0] = True
                return
            if idx == n_local:
                if current_size > best[0]:
                    best[0] = current_size
                return
            if current_size + (n_local - idx) <= best[0]:
                return
            
            v = order[idx]
            
            backtrack(idx + 1, current_mask, current_size)
            
            if not timeout[0]:
                N_v_mask = adj[v]
                C = current_mask & N_v_mask
                
                tmp_c = C
                valid = True
                while tmp_c:
                    bit = tmp_c & (-tmp_c)
                    u = bit.bit_length() - 1
                    tmp_c ^= bit
                    if adj[u] & C:
                        valid = False
                        break
                
                if valid:
                    backtrack(idx + 1, current_mask | (1 << v), current_size + 1)
        
        backtrack(0, 0, 0)
        return best[0]

    def has_tf_set_general(adj, target_size, time_limit=2.0):
        n_local = len(adj)
        t0 = time.time()
        
        deg = [bin(adj[v]).count('1') for v in range(n_local)]
        order = sorted(range(n_local), key=lambda v: -deg[v])
        
        def backtrack(idx, current_mask, current_size):
            if time.time() - t0 > time_limit:
                return "timeout"
            if idx == n_local:
                return current_size >= target_size
            
            if current_size + (n_local - idx) < target_size:
                return False
            
            v = order[idx]
            
            res = backtrack(idx + 1, current_mask, current_size)
            if res == "timeout":
                return "timeout"
            if res:
                return True
            
            N_v_mask = adj[v]
            C = current_mask & N_v_mask
            
            tmp_c = C
            valid = True
            while tmp_c:
                bit = tmp_c & (-tmp_c)
                u = bit.bit_length() - 1
                tmp_c ^= bit
                if adj[u] & C:
                    valid = False
                    break
            
            if valid:
                res = backtrack(idx + 1, current_mask | (1 << v), current_size + 1)
                if res == "timeout":
                    return "timeout"
                if res:
                    return True
            
            return False
        
        result = backtrack(0, 0, 0)
        if result == "timeout":
            return "timeout"
        return result

    T_17 = max_tf_size_general(adj_17, 5.0)
    
    # Step 3: Search for N
    best_N = 0
    best_score = T_17 + 1 # Isolated vertex score
    best_adj = adj_17 + [0]
    
    random.seed(42)
    
    # Try to find a graph with score T_17
    for trial in range(10000):
        if time.time() > deadline - 10:
            break
            
        # Generate random triangle-free N in Paley(17)
        N_mask = 0
        vertices = list(range(17))
        random.shuffle(vertices)
        for v in vertices:
            neigh_in_N = N_mask & adj_17[v]
            is_indep = True
            tmp = neigh_in_N
            while tmp:
                bit = tmp & (-tmp)
                u = bit.bit_length() - 1
                tmp ^= bit
                if adj_17[u] & neigh_in_N:
                    is_indep = False
                    break
            if is_indep:
                N_mask |= (1 << v)
        
        # Construct full graph
        adj_full = adj_17[:] + [0]
        for v in range(17):
            if (N_mask >> v) & 1:
                adj_full[17] |= (1 << v)
                adj_full[v] |= (1 << 17)
        
        # Check K4-free
        if has_k4(adj_full):
            continue
            
        # Check if score is T_17
        # If there is NO TF set of size T_17 + 1, then max size is <= T_17.
        # Since subgraph 0..16 has size T_17, max size is exactly T_17.
        check_res = has_tf_set_general(adj_full, T_17 + 1, 1.0)
        
        if check_res == "timeout":
            # If timeout, assume it might be bad, calculate exact score?
            # Or skip?
            # Better to calculate exact score to be sure.
            score = max_tf_size_general(adj_full, 2.0)
            if score < best_score:
                best_score = score
                best_N = N_mask
                best_adj = adj_full
        else:
            if not check_res:
                # Score is T_17. Optimal!
                return edges_from_adj(adj_full)
            else:
                # Score > T_17. Calculate exact score to keep best bad candidate.
                score = max_tf_size_general(adj_full, 2.0)
                if score < best_score:
                    best_score = score
                    best_N = N_mask
                    best_adj = adj_full

    # Fallback
    return edges_from_adj(best_adj)
