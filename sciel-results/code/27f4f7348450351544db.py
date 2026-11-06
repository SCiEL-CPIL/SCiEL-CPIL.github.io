import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_graph = {'n': 5, 'edges': []}
    
    for target_n in [21, 20]:
        if time.time() > deadline - 5:
            break
            
        print(f"Searching for n={target_n}...")
        
        # Precompute indices for subsets of size 6
        # We only use size 6 for search cost
        import itertools
        subsets_6 = list(itertools.combinations(range(target_n), 6))
        
        # Local Search
        # Adjacency matrix (0/1)
        adj = np.zeros((target_n, target_n), dtype=np.uint8)
        
        def init_random():
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1
        
        init_random()
        
        def cost_size_6():
            count = 0
            for s in subsets_6:
                # Extract submatrix
                # s is a tuple of 6 indices
                # adj[np.ix_(s, s)]
                # This creates a 6x6 matrix
                # Sum along axis 1
                # Check if all equal
                # Optimization: 
                # degs = np.sum(adj[np.ix_(s, s)], axis=1)
                # if np.all(degs == degs[0]): count += 1
                # This is slow in Python loop.
                
                # Faster:
                # Use bitsets in Python integers is actually quite fast for small n
                # Let's go back to bitsets but optimize the loop
                pass
            return count

        # Let's stick to Python bitsets but optimize the inner loop
        # Precompute masks
        masks_6 = []
        for s in subsets_6:
            m = 0
            for i in s:
                m |= (1 << i)
            masks_6.append(m)
            
        adj_bits = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if adj[i, j]:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)

        def fast_cost_size_6():
            c = 0
            # Local variable for speed
            ab = adj_bits
            for m in masks_6:
                # Check regularity
                # Get verts
                # Instead of getting verts, we can just check degrees
                # d_v = popcount(ab[v] & m)
                # We need to check if d_v is constant for all v in m.
                
                # Extract verts
                verts = []
                tmp = m
                while tmp:
                    lsb = tmp & -tmp
                    i = lsb.bit_length() - 1
                    verts.append(i)
                    tmp ^= lsb
                
                d0 = (ab[verts[0]] & m).bit_count()
                is_reg = True
                for v in verts[1:]:
                    if (ab[v] & m).bit_count() != d0:
                        is_reg = False
                        break
                if is_reg:
                    c += 1
            return c

        current_cost = fast_cost_size_6()
        best_cost = current_cost
        best_adj_bits = adj_bits[:]
        
        iterations = 0
        max_iters = 20000 # Adjust based on speed
        
        while time.time() < deadline - 10 and iterations < max_iters:
            iterations += 1
            
            if current_cost == 0:
                # Check full validity
                # Generate masks for k=7..n
                full_valid = True
                for k in range(7, target_n + 1):
                    masks_k = []
                    for s in itertools.combinations(range(target_n), k):
                        m = 0
                        for i in s:
                            m |= (1 << i)
                        masks_k.append(m)
                        
                    for m in masks_k:
                        verts = []
                        tmp = m
                        while tmp:
                            lsb = tmp & -tmp
                            i = lsb.bit_length() - 1
                            verts.append(i)
                            tmp ^= lsb
                            
                        d0 = (adj_bits[verts[0]] & m).bit_count()
                        is_reg = True
                        for v in verts[1:]:
                            if (adj_bits[v] & m).bit_count() != d0:
                                is_reg = False
                                break
                        if is_reg:
                            full_valid = False
                            break
                    if not full_valid:
                        break
                
                if full_valid:
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if adj_bits[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': target_n, 'edges': edges}
            
            # Flip edge
            u = random.randint(0, target_n - 1)
            v = random.randint(0, target_n - 1)
            if u == v: continue
            
            adj_bits[u] ^= (1 << v)
            adj_bits[v] ^= (1 << u)
            
            new_cost = fast_cost_size_6()
            
            if new_cost < current_cost or (new_cost == current_cost and random.random() < 0.1):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj_bits = adj_bits[:]
            else:
                adj_bits[u] ^= (1 << v)
                adj_bits[v] ^= (1 << u)
                
            if iterations % 5000 == 0 and best_cost == current_cost:
                # Restart
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < 0.5:
                            adj_bits[i] |= (1 << j)
                            adj_bits[j] |= (1 << i)
                        else:
                            adj_bits[i] &= ~(1 << j)
                            adj_bits[j] &= ~(1 << i)
                current_cost = fast_cost_size_6()

    return best_graph
