import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def try_find_valid_graph(n, max_restarts=1000, max_steps=1000):
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
                
            A = [0] * n
            
            # Optional: Start with a random triangle-free graph
            # For now, start empty
            
            stuck = False
            for step in range(max_steps):
                if time.time() > deadline:
                    stuck = True
                    break
                    
                # Check for independent set of size 10
                # Use randomized greedy search
                found_iset = None
                for _ in range(5): # 5 random restarts for IS search
                    # Shuffle vertices to randomize the "max neighbors" choice?
                    # Or just pick random vertices in the DFS
                    # Let's modify dfs to be randomized
                    pass
                
                # Re-implement IS finder with randomization
                def find_iset_randomized():
                    # Randomized DFS
                    # We pick a random vertex from candidates, not the one with max neighbors
                    # This helps escape local minima
                    def dfs(candidates_mask, current_set):
                        if len(current_set) == 10:
                            return current_set
                        if candidates_mask.bit_count() < (10 - len(current_set)):
                            return None
                        
                        # Pick a random vertex from candidates
                        c = candidates_mask
                        # Get list of set bits
                        verts = []
                        while c:
                            lsb = c & -c
                            u = lsb.bit_length() - 1
                            verts.append(u)
                            c ^= lsb
                        
                        if not verts:
                            return None
                            
                        # Random shuffle
                        random.shuffle(verts)
                        
                        for v in verts:
                            new_set = current_set + [v]
                            new_candidates = candidates_mask & ~(A[v] | (1 << v))
                            res = dfs(new_candidates, new_set)
                            if res:
                                return res
                        return None
                    
                    return dfs((1 << n) - 1, [])

                iset = find_iset_randomized()
                
                if iset is None:
                    # Likely valid
                    edges = []
                    for u in range(n):
                        for v in range(u+1, n):
                            if A[u] & (1 << v):
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
                
                # Try to add edge
                added = False
                pairs = [(u, v) for i, u in enumerate(iset) for v in iset[i+1:]]
                random.shuffle(pairs)
                
                for u, v in pairs:
                    if (A[u] & A[v]) == 0:
                        A[u] |= (1 << v)
                        A[v] |= (1 << u)
                        added = True
                        break
                
                if not added:
                    stuck = True
                    break
        
        return None

    # Try n=39
    res = try_find_valid_graph(39)
    if res:
        return res
        
    # Try n=35
    res = try_find_valid_graph(35)
    if res:
        return res
        
    # Try n=30
    res = try_find_valid_graph(30)
    if res:
        return res
        
    # Fallback
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
