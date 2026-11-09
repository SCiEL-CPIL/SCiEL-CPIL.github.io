import math
import time

def construct():
    """
    Finds the largest subset of [1, 255] containing 255 such that 
    gcd(a, b) > 1 for all distinct pairs a, b.
    """
    start_time = time.time()
    deadline = 50.0  # Safety margin for 60s limit
    
    # 1. Generate candidates: numbers in [1, 255] that share a factor with 255
    # 255 = 3 * 5 * 17
    # A number x must have gcd(x, 255) > 1, so x must be divisible by 3, 5, or 17.
    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    
    # 2. Build adjacency matrix for the graph where edges connect numbers with gcd > 1
    n = len(candidates)
    # Map value to index for quick lookup if needed, but we can just use indices
    # adj[i][j] is True if gcd(candidates[i], candidates[j]) > 1
    adj = [[False] * n for _ in range(n)]
    
    for i in range(n):
        if time.time() > deadline:
            # Fallback to multiples of 3
            return [i for i in range(1, 256) if i % 3 == 0]
        
        ci = candidates[i]
        for j in range(i + 1, n):
            cj = candidates[j]
            if math.gcd(ci, cj) > 1:
                adj[i][j] = True
                adj[j][i] = True

    # 3. Find Maximum Clique using Bron-Kerbosch with Pivot
    # We want the largest clique.
    max_clique_size = 0
    max_clique_set = set()
    
    # Bron-Kerbosch with pivot to find max clique
    # R: current clique
    # P: candidates that can be added to R
    # X: candidates that have been removed (to avoid duplicates)
    
    def bron_kerbosch(R, P, X):
        nonlocal max_clique_size, max_clique_set
        
        if time.time() > deadline:
            return
            
        if len(P) == 0 and len(X) == 0:
            if len(R) > max_clique_size:
                max_clique_size = len(R)
                max_clique_set = set(R)
            return
        
        # Choose pivot u from P union X to minimize branching
        # u is the vertex with the most neighbors in P
        P_union_X = list(P) + list(X)
        if not P_union_X:
            return
            
        # Find pivot with max degree in P
        pivot = -1
        max_deg = -1
        for u in P_union_X:
            deg = sum(1 for v in P if adj[u][v])
            if deg > max_deg:
                max_deg = deg
                pivot = u
                
        # Iterate over candidates in P that are NOT neighbors of pivot
        # If P is empty, loop doesn't run.
        # We must iterate over a copy of P because we modify P
        candidates_to_iterate = [v for v in list(P) if not adj[pivot][v]]
        
        for v in candidates_to_iterate:
            # Recurse with v in R
            R_new = R | {v}
            P_new = P & {u for u in P if adj[v][u]}
            X_new = X & {u for u in X if adj[v][u]}
            
            bron_kerbosch(R_new, P_new, X_new)
            
            # Remove v from P, add to X
            P.discard(v)
            X.add(v)

    # Initialize
    # We can start with empty R, P = all candidates, X = empty
    P_init = set(range(n))
    X_init = set()
    
    # Optimization: Since we know multiples of 3 form a clique of size 85, 
    # we can start with that to prune search? 
    # Bron-Kerbosch finds *all* maximal cliques. 
    # With 127 nodes, it might be slow if the graph is dense.
    # However, the graph is quite dense.
    # Let's check if 85 is the max.
    # A simpler heuristic: Just return multiples of 3.
    # But the prompt asks to "Maximize cardinality". 
    # If 85 is not the max, we miss points.
    # Given the structure (3, 5, 17), 85 is highly likely to be the max.
    # Let's run the BK algorithm but with a time limit.
    
    # To speed up BK, we can use a greedy lower bound to prune.
    # We know size 85 is possible.
    
    # Let's try to run BK. If it's too slow, we fallback.
    # Actually, for n=127, BK with pivot is usually fast enough for a dense graph.
    
    bron_kerbosch(set(), P_init, X_init)
    
    if max_clique_set:
        # Convert indices back to values
        result = [candidates[i] for i in max_clique_set]
        # Ensure 255 is in result (it should be)
        if 255 not in result:
            # If 255 wasn't picked (shouldn't happen if it's a max clique in connected graph), 
            # we force it? No, if it's not in the max clique, the max clique doesn't contain 255.
            # But the problem REQUIRES 255.
            # So we must find the max clique CONTAINING 255.
            # My BK finds global max.
            # Let's check if 255 is in the found clique.
            if 255 not in result:
                # Find max clique containing 255
                # 255 is index of 255 in candidates
                idx_255 = candidates.index(255)
                # Neighbors of 255 are all candidates (since 255 is div by 3,5,17 and all candidates are div by 3,5,17? 
                # No, candidate x is div by 3, 5, or 17. 255 is div by 3,5,17.
                # gcd(255, x) > 1 is always true for x in candidates.
                # So 255 is connected to ALL candidates.
                # Thus, any maximal clique in the graph is a candidate for the max clique containing 255?
                # No. If the global max clique does not contain 255, we need the next best one that does.
                # But since 255 is a "hub" connected to everyone, 
                # if there is a clique C not containing 255, then C U {255} is also a clique (larger by 1).
                # So the global max clique MUST contain 255.
                pass
        
        return sorted(result)
    else:
        # Fallback
        return [i for i in range(1, 256) if i % 3 == 0]

# Run construct to verify
# construct()
