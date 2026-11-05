import time
import math
import itertools

def construct():
    start_time = time.time()
    deadline = 55.0 # Safety margin
    
    # 255 = 3 * 5 * 17
    # Candidates must be neighbors of 255
    # i.e., gcd(x, 255) > 1
    candidates = [x for x in range(1, 256) if math.gcd(x, 255) > 1]
    
    # Build adjacency matrix/list
    n = len(candidates)
    # Map value to index
    val_to_idx = {val: i for i, val in enumerate(candidates)}
    
    # Adjacency list
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].append(j)
                adj[j].append(i)
                
    # We need a clique containing 255.
    # 255 is in candidates. Find its index.
    idx_255 = val_to_idx[255]
    
    # The clique must be a subset of {255} U (Neighbors of 255 in the graph).
    # Actually, the clique is in the subgraph induced by candidates.
    # And it must contain idx_255.
    # This is equivalent to finding a max clique in the subgraph induced by 
    # the common neighbors of idx_255? No.
    # If S is a clique containing v, then S\{v} is a clique in the subgraph 
    # induced by N(v).
    
    # N(idx_255) are the candidates connected to 255.
    # Since all candidates are neighbors of 255 (by definition of candidates list),
    # the subgraph is just the graph on `candidates` minus 255?
    # No, we just need to find a max clique in the graph `G` that contains `idx_255`.
    
    # Bron-Kerbosch with pivoting
    best_clique = [idx_255]
    
    # Pivoting strategy: choose pivot u to minimize branching.
    # R: current clique, P: candidates, X: excluded (already processed)
    
    def bronk(R, P, X):
        nonlocal best_clique
        if time.time() - start_time > deadline:
            return
        
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = R.copy()
            return
            
        # Choose pivot u from P U X with max degree in P
        u = -1
        max_deg = -1
        for v in P:
            # deg in P
            deg = len([w for w in P if w in adj[v]])
            if deg > max_deg:
                max_deg = deg
                u = v
        # Also check X? Standard pivoting uses P U X.
        # If P is empty, we are done.
        # If u is -1 (P empty), handled above.
        
        # Branch on P \ N(u)
        # If u is in X, N(u) in P is subset of P.
        # We iterate over v in P such that v is NOT adjacent to u.
        # If P is empty, return.
        
        # Optimization: if P is large, this list comprehension is slow.
        # Precompute adj sets?
        
        # For small n (119), list comprehension is okay.
        
        P_not_Nu = [v for v in P if v != u and v not in adj[u]]
        
        for v in P_not_Nu:
            # New R = R + v
            # New P = P intersect N(v)
            # New X = X intersect N(v)
            
            new_R = R + [v]
            new_P = [p for p in P if p in adj[v]]
            new_X = [x for x in X if x in adj[v]]
            
            bronk(new_R, new_P, new_X)
            
            P.remove(v)
            X.append(v)
            
            if time.time() - start_time > deadline:
                return

    # Initial call
    # R = [idx_255]
    # P = neighbors of idx_255 in candidates
    # X = []
    
    P_initial = [p for p in range(n) if p != idx_255 and p in adj[idx_255]]
    # Note: All candidates are neighbors of 255, so P_initial is all candidates except 255.
    
    bronk([idx_255], P_initial, [])
    
    # Convert best_clique indices back to values
    result_vals = [candidates[i] for i in best_clique]
    
    return result_vals
