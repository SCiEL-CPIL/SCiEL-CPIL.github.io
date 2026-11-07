import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    best_distinct = 0
    best_subset = list(range(8)) # Fallback
    
    # We want to maximize distinct degrees.
    # Max possible is k.
    # Iterate k from 17 down to 8? Or 8 to 17?
    # If we find a subset of size k with k distinct degrees, that's the max for that size.
    # But a larger size could have more distinct degrees.
    # So we must check all.
    
    # Optimization: 
    # If we find 17 distinct degrees in size 17, we can stop? No, max is 17.
    # If we find 8 distinct degrees in size 8, can size 9 have 9? Yes.
    # So we just check all.
    
    start_time = time.time()
    
    for k in range(8, n + 1):
        for combo in combinations(range(n), k):
            # Check time limit
            if time.time() - start_time > 50:
                break
                
            # Calculate degrees in induced subgraph
            deg_set = set()
            # combo is a tuple of vertices
            # For each v in combo, count neighbors in combo
            # Using set intersection for speed?
            # adj[v] & set(combo)
            # Creating set(combo) for each v is expensive.
            # Create set_combo once per combo.
            set_combo = set(combo)
            
            degrees = []
            for v in combo:
                # Count neighbors in set_combo
                # adj[v] is a set
                # len(adj[v] & set_combo)
                deg = len(adj[v] & set_combo)
                degrees.append(deg)
                
            distinct_count = len(set(degrees))
            
            if distinct_count > best_distinct:
                best_distinct = distinct_count
                best_subset = list(combo)
                
            # Early termination for this k?
            # If distinct_count == k, it's the max possible for this k.
            # But we still need to check larger k.
            
        if time.time() - start_time > 50:
            break
            
    return best_subset
