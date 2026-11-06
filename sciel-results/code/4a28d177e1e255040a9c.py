import math
import itertools

def construct():
    # Generate candidates: numbers in [1, 255] divisible by 3, 5, or 17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    # Map values to indices for easier graph handling
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Build adjacency matrix
    # adj[i][j] is True if gcd(candidates[i], candidates[j]) > 1
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True
    
    # Bron-Kerbosch algorithm with pivot to find maximum clique
    # R: current clique
    # P: candidates to add to R
    # X: candidates already processed (to avoid duplicates)
    best_clique = []
    
    def bronk(R, P, X):
        nonlocal best_clique
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = R.copy()
            return
        
        # Choose pivot u from P union X
        # We want u that maximizes the number of neighbors in P
        # This minimizes the number of recursive calls
        if not P:
            return
            
        # Simple pivot selection: pick the one with max degree in P
        # For small N, picking any is okay, but let's try to be smart
        # If P is empty, we return.
        # Pivot u should be in P union X.
        # To simplify, we can just pick the first element in P if we don't want to optimize pivot selection heavily.
        # But let's do a basic optimization: pick u in P that has the most neighbors in P.
        max_neighbors = -1
        pivot = P[0]
        for u in P:
            count = sum(1 for v in P if adj[u][v])
            if count > max_neighbors:
                max_neighbors = count
                pivot = u
        
        # Iterate over P \ N(pivot)
        # These are vertices in P that are NOT adjacent to pivot
        # We must check all of them because the max clique might not contain pivot
        for v in list(P):
            if not adj[v][pivot]:
                new_R = R + [v]
                new_P = [x for x in P if adj[v][x]]
                new_X = [x for x in X if adj[v][x]]
                bronk(new_R, new_P, new_X)
                P.remove(v)
                X.append(v)
        
        # Also need to handle the case where we might have missed something?
        # No, Bron-Kerbosch with pivot iterates over P \ N(u).
        # The cliques containing u are found in the recursive calls where u is in R.
        # The cliques NOT containing u are found in the recursive calls for v in P \ N(u).
        # Wait, the standard algorithm:
        # for v in P \ N(u):
        #    bronk(R+{v}, P \cap N(v), X \cap N(v))
        #    P.remove(v); X.add(v)
        # This covers all maximal cliques.
        
    # Initial call
    # P = all vertices, X = empty
    bronk([], candidates, [])
    
    # Convert indices back to values
    result = [candidates[i] for i in best_clique]
    
    # Ensure 255 is in the result? 
    # 255 is in candidates. 
    # Does the max clique contain 255?
    # 255 is div by 3, 5, 17.
    # It is connected to all multiples of 3, 5, 17.
    # In S_3, 255 is connected to all other elements in S_3 (gcd >= 3).
    # So S_3 is a clique containing 255.
    # So the max clique will contain 255 if it's the max.
    
    # If for some reason best_clique is empty (shouldn't happen), return [255]
    if not result:
        return [255]
        
    return result

