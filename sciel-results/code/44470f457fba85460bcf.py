import time

def construct():
    start_time = time.time()
    
    def get_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        # adj_mask[i] is an integer bitmask of neighbors of i
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    def solve_instance(n, p, seed):
        adj_mask = get_graph(n, p, seed)
        
        # Check if graph is empty
        if not any(adj_mask):
            return []
            
        cliques = []
        
        # Helper to find max bit index
        def max_bit(x):
            if x == 0:
                return -1
            return x.bit_length() - 1

        # Helper to iterate through set bits
        def iter_bits(x):
            while x:
                lsb = x & -x
                yield lsb.bit_length() - 1
                x ^= lsb

        while True:
            if time.time() - start_time > 115:
                break
                
            # Find vertex with max degree in remaining graph
            max_deg = -1
            v = -1
            for i in range(n):
                deg = adj_mask[i].bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = i
            
            if v == -1 or max_deg == 0:
                break
                
            # Build clique greedily
            C = [v]
            # Candidates are neighbors of v
            U = adj_mask[v]
            
            # We want to add vertices to C.
            # At each step, pick u in U that maximizes the number of connections to other candidates in U.
            # This helps form a large clique.
            
            while U:
                best_u = -1
                best_score = -1
                
                # Iterate through candidates in U
                for u in iter_bits(U):
                    # Score: how many neighbors does u have within U?
                    # adj_mask[u] & U gives neighbors of u that are also in U
                    score = (adj_mask[u] & U).bit_count()
                    if score > best_score:
                        best_score = score
                        best_u = u
                
                if best_u == -1:
                    break
                    
                C.append(best_u)
                # Update candidates: must be neighbors of best_u AND in previous U
                U &= adj_mask[best_u]
                
            # Add clique to result
            cliques.append(C)
            
            # Remove edges within C from adj_mask
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v2 = C[i], C[j]
                    adj_mask[u] &= ~(1 << v2)
                    adj_mask[v2] &= ~(1 << u)
                    
            # Optimization: If we have a lot of time, we could try to improve, 
            # but the greedy approach is usually sufficient for these densities.
            
        return cliques

    results = []
    # Instance 1
    results.append(solve_instance(32, 0.6, 601))
    
    # Instance 2
    results.append(solve_instance(64, 0.55, 602))
    
    return results
