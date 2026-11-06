import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 290  # safety margin before 300s kill

    def time_left():
        return deadline - time.time()

    # 1. Build Paley Graph P(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
                edges.append((i, j))
    
    n = q

    # Precompute all K4s and Ind4s for the base graph
    # We need to check if adding a vertex creates a K5 or Ind5.
    # A new K5 is formed if the new vertex connects to a K4 in the old graph.
    # A new Ind5 is formed if the new vertex is non-adjacent to an Ind4 in the old graph.
    
    k4s = []
    ind4s = []
    
    # Optimization: Use bitsets for adjacency to speed up K4/Ind4 enumeration?
    # n=37 is small enough for O(n^4) = 37^4 ~ 1.8M checks, which is fine in Python if optimized or even brute force with pruning.
    
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] & (1 << j)): continue
            for k in range(j+1, n):
                if not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)): continue
                for l in range(k+1, n):
                    if (adj[i] & (1 << l)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                        k4s.append((i, j, k, l))
                    if not (adj[i] & (1 << l)) and not (adj[j] & (1 << l)) and not (adj[k] & (1 << l)):
                        ind4s.append((i, j, k, l))

    # To check if a candidate neighbor mask `mask` is valid:
    # 1. For every K4 in the graph, `mask` must NOT contain all 4 vertices. (Otherwise K5)
    # 2. For every Ind4 in the graph, `mask` must NOT contain NONE of the 4 vertices (i.e., it must contain at least one). (Otherwise Ind5)
    
    def is_valid_mask(mask):
        # Check K4 constraints: mask must not be a superset of any K4
        for k4 in k4s:
            if (mask & (1 << k4[0])) and (mask & (1 << k4[1])) and (mask & (1 << k4[2])) and (mask & (1 << k4[3])):
                return False
        # Check Ind4 constraints: mask must intersect every Ind4
        # i.e., mask must NOT be a subset of the complement of any Ind4
        # Equivalently: for every ind4, at least one vertex is in mask.
        for i4 in ind4s:
            if not (mask & (1 << i4[0])) and not (mask & (1 << i4[1])) and not (mask & (1 << i4[2])) and not (mask & (1 << i4[3])):
                return False
        return True

    def count_violations(mask):
        viol = 0
        for k4 in k4s:
            if (mask & (1 << k4[0])) and (mask & (1 << k4[1])) and (mask & (1 << k4[2])) and (mask & (1 << k4[3])):
                viol += 1
        for i4 in ind4s:
            if not (mask & (1 << i4[0])) and not (mask & (1 << i4[1])) and not (mask & (1 << i4[2])) and not (mask & (1 << i4[3])):
                viol += 1
        return viol

    def try_add_vertex():
        # Local search to find a valid mask
        # Start with a random mask
        best_mask = random.getrandbits(n)
        best_viol = count_violations(best_mask)
        
        if best_viol == 0:
            return best_mask

        # Simulated Annealing / Random Walk
        # We want to minimize violations.
        # Step: flip a random bit.
        
        # To speed up, we can precompute for each bit v, which K4s and Ind4s it affects.
        # But n is small, let's just do the direct check first. If too slow, optimize.
        
        for step in range(5000):
            if time_left() < 5:
                return None
            
            # Pick random bit
            bit = random.randint(0, n - 1)
            new_mask = best_mask ^ (1 << bit)
            new_viol = count_violations(new_mask)
            
            if new_viol < best_viol:
                best_mask = new_mask
                best_viol = new_viol
                if best_viol == 0:
                    return best_mask
            elif new_viol == best_viol and random.random() < 0.1:
                # Plateau move
                best_mask = new_mask
            elif new_viol > best_viol and random.random() < 0.01:
                # Rare uphill move to escape local minima
                best_mask = new_mask
                best_viol = new_viol
                
            # Random restart
            if step % 200 == 0:
                best_mask = random.getrandbits(n)
                best_viol = count_violations(best_mask)
                if best_viol == 0:
                    return best_mask

        return None

    # Loop to add vertices
    # We can add multiple vertices. After adding one, we must update k4s and ind4s.
    
    added = 0
    while added < 10:  # Cap at 10 extra vertices (total 47)
        if time_left() < 30:
            break
            
        mask = try_add_vertex()
        if mask is None:
            break
        
        # Add vertex n
        new_adj = [0] * (n + 1)
        for i in range(n):
            new_adj[i] = adj[i]
            if mask & (1 << i):
                new_adj[i] |= 1 << n
                new_adj[n] |= 1 << i
                edges.append((i, n))
        
        adj = new_adj
        n += 1
        added += 1
        
        # Update K4s and Ind4s
        # New K4s must involve the new vertex (n-1)
        new_k4s = []
        new_ind4s = []
        
        # Iterate over triples (i, j, k) in the old graph
        for i in range(n-1):
            for j in range(i+1, n-1):
                for k in range(j+1, n-1):
                    # Check if (i, j, k) form a triangle (part of K4)
                    # We need to check if adding n-1 creates a K4.
                    # This requires (i, j, k) to be a triangle AND n-1 connected to all three.
                    # Or, we can just check all triples and see if they form a K4 with n-1.
                    
                    # Actually, it's easier to just check if the set {i, j, k, n-1} is a K4.
                    # This requires:
                    # 1. i-j, i-k, j-k are edges (triangle in old graph)
                    # 2. i-(n-1), j-(n-1), k-(n-1) are edges (n-1 connected to all)
                    
                    if (adj[i] & (1 << j)) and (adj[i] & (1 << k)) and (adj[j] & (1 << k)):
                        if (adj[i] & (1 << (n-1))) and (adj[j] & (1 << (n-1))) and (adj[k] & (1 << (n-1))):
                            new_k4s.append((i, j, k, n-1))
                            
                    # Check for Ind4 with n-1
                    # 1. i-j, i-k, j-k are non-edges
                    # 2. i-(n-1), j-(n-1), k-(n-1) are non-edges
                    if not (adj[i] & (1 << j)) and not (adj[i] & (1 << k)) and not (adj[j] & (1 << k)):
                        if not (adj[i] & (1 << (n-1))) and not (adj[j] & (1 << (n-1))) and not (adj[k] & (1 << (n-1))):
                            new_ind4s.append((i, j, k, n-1))
        
        k4s.extend(new_k4s)
        ind4s.extend(new_ind4s)

    return n, edges
