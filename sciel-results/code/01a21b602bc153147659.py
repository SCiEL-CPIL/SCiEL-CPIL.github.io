import random
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 110
    
    def popcount(x):
        return x.bit_count()

    def is_valid_graph_bitmask(masks, n):
        if n < 6:
            return True
        
        # Precompute subset masks? No, just iterate.
        # Optimization: Only need to check up to size n.
        # But if a subset of size 6 is regular, it's invalid.
        # We can stop early if we find one.
        
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                # Create mask for this subset
                sub_mask = 0
                for v in subset:
                    sub_mask |= (1 << v)
                
                # Calculate degrees for vertices in subset
                degrees = []
                for v in subset:
                    # Count neighbors in subset
                    deg = popcount(masks[v] & sub_mask)
                    degrees.append(deg)
                
                # Check if all degrees are equal
                if len(set(degrees)) == 1:
                    return False
        return True

    def find_graph(n, time_budget):
        # Local search
        # 1. Random start
        # 2. Check validity
        # 3. If invalid, find a regular subset, flip an edge to break it
        
        best_valid = None
        
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Generate random graph
            masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        masks[i] |= (1 << j)
                        masks[j] |= (1 << i)
            
            # Local search loop
            for step in range(1000):
                if time.time() - start_time > time_limit:
                    return best_valid
                
                # Check validity
                if is_valid_graph_bitmask(masks, n):
                    # Found valid graph
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if masks[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                
                # Find a regular subset to break
                # We need to modify is_valid to return the offending subset
                # Or just pick a random edge and flip it?
                # Picking a random edge might be slow to converge.
                # Let's find a regular subset.
                
                offending_subset = None
                for k in range(6, n + 1):
                    for subset in combinations(range(n), k):
                        sub_mask = 0
                        for v in subset:
                            sub_mask |= (1 << v)
                        degrees = [popcount(masks[v] & sub_mask) for v in subset]
                        if len(set(degrees)) == 1:
                            offending_subset = subset
                            break
                    if offending_subset:
                        break
                
                if not offending_subset:
                    # Should not happen if is_valid returned False
                    break
                
                # Flip an edge involving a vertex in offending_subset
                # Pick a random vertex in the subset
                u = random.choice(offending_subset)
                # Pick a random vertex v (could be in or out of subset)
                # Flipping edge (u, v) changes degree of u and v.
                # If v is in subset, it changes degrees of u and v in the induced subgraph.
                # If v is not in subset, it changes degree of u in the induced subgraph.
                # Either way, it's likely to break regularity.
                
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                # Flip edge (u, v)
                masks[u] ^= (1 << v)
                masks[v] ^= (1 << u)
            
        return best_valid

    for n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() - start_time > time_limit:
            break
        
        result = find_graph(n, time_limit - (time.time() - start_time))
        if result:
            return result
            
    # Fallback
    n = 6
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (i == 0 and j == 1):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
