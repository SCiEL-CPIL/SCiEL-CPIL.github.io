import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # 1. Generate candidates: x in [1, 255] such that gcd(x, 255) > 1
    # 255 = 3 * 5 * 17
    candidates = []
    for x in range(1, 256):
        if gcd(x, 255) > 1:
            candidates.append(x)
    
    # Ensure 255 is included
    if 255 not in candidates:
        candidates.append(255)
        
    n = len(candidates)
    
    # 2. Build adjacency list
    # adj[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # 3. Find Maximum Clique
    # Using a simple recursive backtracking with pivot heuristic (Bron-Kerbosch)
    # We want the largest clique.
    max_clique = []
    max_size = 0
    
    # Sort candidates by degree to improve pruning? 
    # For simplicity and speed on small n, we can just run BK.
    # We need to ensure 255 is in the final set.
    # Since 255 is connected to all candidates (gcd(x, 255)>1 is the condition for being a candidate),
    # 255 is a universal vertex in this graph.
    # Any maximum clique must contain 255?
    # Not necessarily. A clique not containing 255 could be larger?
    # But if a clique C does not contain 255, we can add 255 to it because 255 is connected to everyone.
    # So C U {255} is a larger clique.
    # Thus, the maximum clique MUST contain 255.
    
    # We can fix 255 in the clique and find the max clique in the neighborhood of 255.
    # The neighborhood of 255 is all other candidates.
    # So we just need to find the max clique in the graph induced by candidates.
    # Since 255 is universal, max_clique = {255} U max_clique_rest.
    
    # Let's find max clique in the subgraph of candidates excluding 255?
    # No, just run on all candidates. The algorithm will find the max clique.
    # We will verify at the end that 255 is in it. If not, we add it (valid as it's universal).
    
    def bron_kerbosch(R, P, X):
        nonlocal max_clique, max_size
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > max_size:
                max_size = len(R)
                max_clique = R.copy()
            return
        
        # Choose pivot u from P U X to minimize |P \ N(u)|
        # For simplicity, we can just pick any u or the one with max degree in P
        # A common heuristic is to pick u from P U X that maximizes |P ∩ N(u)|
        pivot_u = None
        max_intersect = -1
        
        # To speed up, we can limit the search if we already have a good size
        # But 127 nodes is small.
        
        candidates_for_pivot = list(P) + list(X)
        for u in candidates_for_pivot:
            inter = len(P.intersection(adj[u]))
            if inter > max_intersect:
                max_intersect = inter
                pivot_u = u
                
        if pivot_u is None:
            # P and X are empty, handled above
            return
            
        # Iterate over P \ N(pivot_u)
        P_not_N = P.difference(adj[pivot_u])
        
        for v in list(P_not_N):
            new_R = R.copy()
            new_R.add(v)
            
            new_P = P.intersection(adj[v])
            new_X = X.intersection(adj[v])
            
            # Pruning: if len(new_R) + len(new_P) <= max_size, skip
            if len(new_R) + len(new_P) <= max_size:
                continue
                
            bron_kerbosch(new_R, new_P, new_X)
            
            P.remove(v)
            X.add(v)
            
            if time.time() > deadline:
                return

    # Initial call
    # R = empty, P = all vertices, X = empty
    initial_P = set(range(n))
    initial_R = set()
    initial_X = set()
    
    # Optimization: Start with a greedy heuristic to get a good lower bound for max_size
    # Greedy: pick vertex with max degree, add to clique, remove non-neighbors.
    greedy_clique = set()
    remaining = set(range(n))
    while remaining:
        # Pick vertex with max degree in remaining
        v = max(remaining, key=lambda x: len(adj[x].intersection(remaining)))
        greedy_clique.add(v)
        remaining = remaining.intersection(adj[v])
        if time.time() > deadline:
            break
            
    max_size = len(greedy_clique)
    max_clique = greedy_clique
    
    # Now run Bron-Kerbosch to find if there's a larger one
    # To make it faster, we can restrict the search or just run it.
    # With 127 nodes, it should be fast.
    
    # We will run BK.
    bron_kerbosch(initial_R, initial_P, initial_X)
    
    # Ensure 255 is in the result
    # Find index of 255
    idx_255 = candidates.index(255)
    
    if idx_255 not in max_clique:
        # Add 255. It is connected to all, so it's a valid extension.
        max_clique.add(idx_255)
        max_size += 1
        
    # Convert indices back to values
    result = [candidates[i] for i in max_clique]
    
    # Final verification
    if len(result) == 0:
        return [255]
        
    # Verify 255 is in result
    if 255 not in result:
        result.append(255)
        
    # Verify pairwise gcd
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if gcd(result[i], result[j]) == 1:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to multiples of 3
        return list(range(3, 256, 3))
        
    return result
