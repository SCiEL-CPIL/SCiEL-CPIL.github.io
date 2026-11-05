import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s out of 120s limit
    
    def bitmask_to_edges(n, adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def has_clique(adj, n, size):
        """
        Returns True if there is a clique of given size in the graph defined by adj bitmasks.
        """
        # Pre-filter
        for i in range(n):
            if adj[i].bit_count() < size - 1:
                return False
        
        def find_clique(candidates, current_size):
            if current_size == size:
                return True
            if candidates.bit_count() < size - current_size:
                return False
            
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                c ^= lsb
                
                # Neighbors of v in candidates
                new_cands = candidates & adj[v]
                
                # Recurse
                if find_clique(new_cands, current_size - 1):
                    return True
                    
            return False

        return find_clique((1 << n) - 1, size)

    def find_k4(adj, n):
        """
        Returns a tuple (u, v, w, x) forming a K4, or None.
        """
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        # Check for x
                        common_3 = adj[u] & adj[v] & adj[w]
                        if common_3:
                            lsb_x = common_3 & -common_3
                            x = lsb_x.bit_length() - 1
                            return (u, v, w, x)
        return None

    def has_k4(adj, n):
        return find_k4(adj, n) is not None

    def has_ind6(adj, n):
        mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = mask & ~(adj[i] | (1 << i))
        return has_clique(comp_adj, n, 6)

    def get_penalty(adj, n):
        k4 = has_k4(adj, n)
        ind6 = has_ind6(adj, n)
        if k4 and ind6:
            return 200
        if k4:
            return 100
        if ind6:
            return 1000
        return 0

    def local_search(n, adj, time_limit):
        current_adj = adj[:]
        current_penalty = get_penalty(current_adj, n)
        
        if current_penalty == 0:
            return current_adj

        temp = 50.0
        cooling = 0.999
        best_adj = current_adj[:]
        best_penalty = current_penalty

        while time.time() < time_limit:
            move_u, move_v = -1, -1
            
            # Targeted repair with 50% probability
            if random.random() < 0.5:
                k4 = find_k4(current_adj, n)
                if k4:
                    u, v, w, x = k4
                    edges_in_k4 = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                    move_u, move_v = random.choice(edges_in_k4)
                else:
                    # Fallback to random if no K4 found (maybe only Ind6 issue)
                    move_u = random.randint(0, n - 1)
                    move_v = random.randint(0, n - 1)
            else:
                move_u = random.randint(0, n - 1)
                move_v = random.randint(0, n - 1)
                
            if move_u == move_v:
                continue

            # Apply move
            if (current_adj[move_u] >> move_v) & 1:
                current_adj[move_u] &= ~(1 << move_v)
                current_adj[move_v] &= ~(1 << move_u)
            else:
                current_adj[move_u] |= (1 << move_v)
                current_adj[move_v] |= (1 << move_u)
            
            new_penalty = get_penalty(current_adj, n)
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                    best_adj = current_adj[:]
                if current_penalty == 0:
                    return current_adj
            else:
                delta = new_penalty - current_penalty
                if delta == 0:
                    current_penalty = new_penalty
                else:
                    if random.random() < pow(2.71828, -delta / temp):
                        current_penalty = new_penalty
                    else:
                        # Revert
                        if (current_adj[move_u] >> move_v) & 1:
                            current_adj[move_u] &= ~(1 << move_v)
                            current_adj[move_v] &= ~(1 << move_u)
                        else:
                            current_adj[move_u] |= (1 << move_v)
                            current_adj[move_v] |= (1 << move_u)
            
            temp *= cooling

        return best_adj

    best_n = 17
    best_adj = None
    
    # Seed: Paley graph on 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj17 = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj17[i] |= (1 << j)
                adj17[j] |= (1 << i)
    best_adj = adj17

    # Try to find larger graphs
    targets = [35, 34, 33, 32, 30, 28, 25, 22, 20]
    
    for target_n in targets:
        if time.time() > deadline - 10:
            break
            
        # Initialize random graph
        adj = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Run local search
        slice_end = time.time() + 10
        if slice_end > deadline - 5:
            slice_end = deadline - 5
            
        result_adj = local_search(target_n, adj, slice_end)
        
        # Verify
        if not has_k4(result_adj, target_n) and not has_ind6(result_adj, target_n):
            best_n = target_n
            best_adj = result_adj
            break

    if best_n == 17:
        return {'n': 17, 'edges': bitmask_to_edges(17, best_adj)}
    
    return {'n': best_n, 'edges': bitmask_to_edges(best_n, best_adj)}
