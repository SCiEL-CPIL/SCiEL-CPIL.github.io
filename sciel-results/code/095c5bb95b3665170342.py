import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin
    
    # 1. Construct Paley(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    n = q
    
    # Helper to get all K4 and Ind4 in current graph
    def get_k4s_and_ind4s(adj, n):
        k4s = []
        ind4s = []
        # Precompute masks for speed?
        # Iterate over all 4-sets
        for combo in combinations(range(n), 4):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            is_clique = True
            is_indep = True
            for v in combo:
                others = mask ^ (1 << v)
                if (adj[v] & others) != others:
                    is_clique = False
                if ((~adj[v]) & others) != others:
                    is_indep = False
            if is_clique:
                k4s.append(mask)
            if is_indep:
                ind4s.append(mask)
        return k4s, ind4s

    # Helper to find a valid connection vector for a new vertex
    def find_valid_connection(adj, n, k4s, ind4s):
        # Cost function: number of K4s in S + number of Ind4s in S^c
        # S is represented by an integer mask
        
        def cost(mask):
            c = 0
            for k4 in k4s:
                if (mask & k4) == k4:
                    c += 1
            full_mask = (1 << n) - 1
            comp = mask ^ full_mask
            for i4 in ind4s:
                if (comp & i4) == i4:
                    c += 1
            return c
        
        best_mask = None
        best_cost = float('inf')
        
        # Try a few random starts and hill climb
        for _ in range(50):
            if time.time() > deadline:
                break
            mask = random.getrandbits(n)
            c = cost(mask)
            if c == 0:
                return mask
            # Hill climbing
            for _ in range(100):
                if time.time() > deadline:
                    break
                # Flip a random bit
                bit = random.randint(0, n - 1)
                new_mask = mask ^ (1 << bit)
                new_c = cost(new_mask)
                if new_c < c:
                    mask = new_mask
                    c = new_c
                    if c == 0:
                        return mask
                elif new_c == c:
                    # Accept with some probability to escape local minima? 
                    # For small n, simple hill climb might suffice
                    if random.random() < 0.1:
                        mask = new_mask
                        c = new_c
            if c < best_cost:
                best_cost = c
                best_mask = mask
                if c == 0:
                    return mask
        
        return best_mask if best_cost == 0 else None

    # Grow the graph
    for target_n in range(q + 1, 43):
        if time.time() > deadline - 10:
            break
        
        # Get K4s and Ind4s for current graph
        k4s, ind4s = get_k4s_and_ind4s(adj, n)
        
        # Try to find a valid connection for the new vertex
        mask = find_valid_connection(adj, n, k4s, ind4s)
        
        if mask is None:
            # If we can't add the vertex, we might need to backtrack or just stop
            # For this task, if we reach 42, we are done. If we stop early, we return what we have.
            break
        else:
            # Add the new vertex
            new_idx = n
            adj.append(mask)
            # Update adjacency of existing vertices
            for i in range(n):
                if (mask >> i) & 1:
                    adj[i] |= (1 << new_idx)
            n += 1
            
    # Convert adj to edge list
    edges = []
    for i in range(n):
        m = adj[i]
        for j in range(i + 1, n):
            if (m >> j) & 1:
                edges.append((i, j))
                
    return n, edges
