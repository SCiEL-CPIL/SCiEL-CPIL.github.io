import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s margin
    
    # Target number of vertices
    # Exoo's graph has 42 vertices.
    # We will try to find a valid graph on 42 vertices.
    # If we fail, we can fall back to 37 (Paley).
    
    n = 42
    
    # Precompute all 5-sets
    # To speed up, we can just iterate
    # But 850k is manageable.
    
    best_graph = None
    best_score = float('inf') # Number of bad 5-sets
    
    # Initialize with Paley 37 + 5 random?
    # Or just random?
    # Random is usually fine for this size with good local search.
    
    # Let's try a few restarts with random initialization.
    
    # Helper to count bad sets and edge blame
    def evaluate(adj):
        """
        adj: list of int bitmasks
        Returns: (num_k5, num_indep5, edge_blame_k5, edge_blame_indep)
        edge_blame_k5[e]: number of K5s containing edge e
        edge_blame_indep[e]: number of independent sets containing edge e (i.e., non-edges)
        """
        num_k5 = 0
        num_indep = 0
        # We need to map edges to indices or just use (u,v) pairs
        # For speed, let's just count total bad sets first.
        # If total is 0, return.
        
        # To optimize, we can just count.
        # If we want to guide the search, we need blame.
        
        # Let's just count first.
        for v1 in range(n):
            for v2 in range(v1+1, n):
                for v3 in range(v2+1, n):
                    for v4 in range(v3+1, n):
                        for v5 in range(v4+1, n):
                            # Check K5
                            is_k5 = True
                            is_indep = True
                            # Pairs: (1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)
                            # Using bitmasks
                            # adj[u] & (1 << v) != 0 means edge exists
                            
                            # Check all 10 pairs
                            if (adj[v1] >> v2) & 1:
                                is_indep = False
                            else:
                                is_k5 = False
                            
                            if is_k5 and is_indep: # Impossible unless n<5 or specific logic error, but let's be safe
                                pass # Actually, if one pair is edge, indep fails. If one pair is non-edge, k5 fails.
                                # Wait.
                                # If ANY pair is non-edge, it's NOT a K5.
                                # If ANY pair is edge, it's NOT an Independent Set.
                                
                            # Let's do it properly
                            is_k5 = True
                            is_indep = True
                            pairs = [(v1,v2), (v1,v3), (v1,v4), (v1,v5), (v2,v3), (v2,v4), (v2,v5), (v3,v4), (v3,v5), (v4,v5)]
                            for u, v in pairs:
                                if (adj[u] >> v) & 1:
                                    is_indep = False
                                else:
                                    is_k5 = False
                                if not is_k5 and not is_indep:
                                    break
                            
                            if is_k5:
                                num_k5 += 1
                            if is_indep:
                                num_indep += 1
        return num_k5 + num_indep

    # This evaluation is too slow in pure Python nested loops for 300s if we do many iterations.
    # 850k * 10 ops = 8.5M ops per eval.
    # 300s / 0.1s (approx time for 8.5M ops in Py) = 3000 evals?
    # Maybe. But we need to update the graph.
    
    # Let's try a different approach:
    # Use the known fact that Exoo's graph exists.
    # We can try to construct it using a greedy algorithm with backtracking?
    # Or use a library? No, only stdlib/NumPy/SciPy.
    
    # Let's try to use NumPy for faster counting?
    # Or optimize the Python loop.
    
    # Actually, 42 vertices is small enough that we can use a more clever search.
    # Let's try to start with a graph that is ALMOST valid.
    # The Paley graph on 37 vertices is valid.
    # Can we add 5 vertices to it?
    
    # Let's try a simple Hill Climbing on the edge set of a 42-vertex graph.
    # We will use a fast checker.
    
    # Optimization:
    # Precompute the list of 5-sets? No, memory is fine but generation takes time.
    # Just iterate.
    
    # Let's try to speed up the inner loop.
    
    def fast_eval(adj):
        bad_count = 0
        # Use local variables for speed
        adj_local = adj
        n_local = n
        for v1 in range(n_local):
            a1 = adj_local[v1]
            for v2 in range(v1+1, n_local):
                a2 = adj_local[v2]
                # If v1-v2 is not edge, it can't be K5. If it is edge, it can't be Indep (for that pair).
                # Actually, we just check all 5 sets.
                
                # Let's just use the bitmask check.
                # We can prune early.
                
                for v3 in range(v2+1, n_local):
                    a3 = adj_local[v3]
                    for v4 in range(v3+1, n_local):
                        a4 = adj_local[v4]
                        for v5 in range(v4+1, n_local):
                            a5 = adj_local[v5]
                            
                            # Check K5
                            # Edges: 12, 13, 14, 15, 23, 24, 25, 34, 35, 45
                            # If all present -> K5
                            # If all absent -> Indep
                            
                            # We can check a few critical ones first.
                            # If (1,2) is edge, Indep is impossible.
                            # If (1,2) is non-edge, K5 is impossible.
                            
                            e12 = (a1 >> v2) & 1
                            if e12:
                                # Can be K5, cannot be Indep
                                # Check remaining 9 edges for K5
                                # If any is 0, not K5.
                                if (a1 >> v3) & 1 and (a1 >> v4) & 1 and (a1 >> v5) & 1 and \
                                   (a2 >> v3) & 1 and (a2 >> v4) & 1 and (a2 >> v5) & 1 and \
                                   (a3 >> v4) & 1 and (a3 >> v5) & 1 and \
                                   (a4 >> v5) & 1:
                                    bad_count += 1
                            else:
                                # Can be Indep, cannot be K5
                                # Check remaining 9 edges for Indep (all 0)
                                if not ((a1 >> v3) & 1 or (a1 >> v4) & 1 or (a1 >> v5) & 1 or \
                                       (a2 >> v3) & 1 or (a2 >> v4) & 1 or (a2 >> v5) & 1 or \
                                       (a3 >> v4) & 1 or (a3 >> v5) & 1 or \
                                       (a4 >> v5) & 1):
                                    bad_count += 1
        return bad_count

    # This is faster.
    
    # Search Loop
    best_bad = float('inf')
    best_adj = None
    
    for restart in range(100):
        if time.time() > deadline - 10:
            break
            
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        curr_bad = fast_eval(adj)
        if curr_bad == 0:
            best_adj = adj
            break
        if curr_bad < best_bad:
            best_bad = curr_bad
            best_adj = adj[:]
            
        # Hill Climbing
        T = 1.0
        for step in range(5000):
            if time.time() > deadline - 5:
                break
                
            # Pick random edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            # Flip
            adj[i] ^= (1 << j)
            adj[j] ^= (1 << i)
            
            new_bad = fast_eval(adj)
            
            # Accept if better, or with prob exp(-delta/T)
            if new_bad <= curr_bad:
                curr_bad = new_bad
                if new_bad < best_bad:
                    best_bad = new_bad
                    best_adj = adj[:]
                    if new_bad == 0:
                        break
            else:
                # Revert
                adj[i] ^= (1 << j)
                adj[j] ^= (1 << i)
                # Optionally accept with SA probability
                # delta = new_bad - curr_bad
                # if random.random() < math.exp(-delta / T):
                #     pass # keep flipped
                # else:
                #     pass # already reverted
                # Let's stick to simple hill climbing for now, or add SA if needed.
                # Given the size, simple HC might get stuck.
                # Let's add a small chance to accept worse moves to escape local minima.
                if random.random() < 0.01:
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)
                    curr_bad = new_bad
                # else: keep reverted

        # Cooldown
        # T = T * 0.9

    # If we found a valid graph on 42, return it.
    if best_adj is not None and best_bad == 0:
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (best_adj[i] >> j) & 1:
                    edges.append((i, j))
        return n, edges

    # Fallback: Return Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
