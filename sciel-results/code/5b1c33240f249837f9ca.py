import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Build adjacency for quick lookup
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # All edges as a set for tracking
    all_edges = set(tuple(sorted(e)) for e in P['edges'])
    
    def greedy_biclique(remaining_edges, randomize=False):
        """Find a good biclique covering many remaining edges."""
        if not remaining_edges:
            return None, set()
        
        # Find candidate bicliques
        # For each pair (u,v) that is an edge, try to build a biclique with u on left, v on right
        # Or more generally, try all pairs
        
        best_cover = set()
        best_biclique = None
        
        # Try each edge as a seed
        edge_list = list(remaining_edges)
        if randomize:
            random.shuffle(edge_list)
        
        for u, v in edge_list:
            if (u, v) not in remaining_edges and (v, u) not in remaining_edges:
                continue
                
            # Start with left={u}, right={v}
            # Expand: add vertices to left that are connected to all in right
            # Add vertices to right that are connected to all in left
            
            left = {u}
            right = {v}
            
            # Expand left
            changed = True
            while changed:
                changed = False
                for candidate in range(P['n']):
                    if candidate in left or candidate in right:
                        continue
                    # Check if candidate is connected to all in right
                    if all(candidate in adj[x] for x in right):
                        # Also check that adding this doesn't violate anything
                        # It's fine because we only care about cross edges being in the graph
                        left.add(candidate)
                        changed = True
            
            # Expand right
            changed = True
            while changed:
                changed = False
                for candidate in range(P['n']):
                    if candidate in left or candidate in right:
                        continue
                    # Check if candidate is connected to all in left
                    if all(candidate in adj[x] for x in left):
                        right.add(candidate)
                        changed = True
            
            # Count how many remaining edges are covered
            covered = set()
            for l in left:
                for r in right:
                    e = tuple(sorted((l, r)))
                    if e in remaining_edges:
                        covered.add(e)
            
            if len(covered) > len(best_cover):
                best_cover = covered
                best_biclique = (left, right)
                
                if randomize and len(covered) > len(remaining_edges) * 0.5:
                    break  # Good enough for randomization
        
        return best_biclique, best_cover
    
    best_result = None
    best_count = float('inf')
    
    # Run multiple iterations with different random seeds
    for seed in range(100):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        remaining = set(all_edges)
        bicliques = []
        
        while remaining:
            if time.time() > deadline:
                break
                
            biclique, covered = greedy_biclique(remaining, randomize=(seed > 0))
            
            if biclique is None or not covered:
                # Fallback: cover one edge at a time
                for e in remaining:
                    u, v = e
                    bicliques.append(([u], [v]))
                    remaining.remove(e)
                break
            
            left, right = biclique
            bicliques.append((list(left), list(right)))
            remaining -= covered
        
        if time.time() > deadline:
            continue
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_result = bicliques
        
        if best_count == 1:
            break
    
    if best_result is None:
        # Fallback: cover each edge individually
        result = []
        for u, v in P['edges']:
            result.append(([u], [v]))
        return result
    
    # Convert to required format
    final_result = []
    for left, right in best_result:
        final_result.append((list(left), list(right)))
    
    return final_result
