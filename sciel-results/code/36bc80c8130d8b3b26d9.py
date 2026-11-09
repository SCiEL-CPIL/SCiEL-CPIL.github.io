"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _maximal_bicliques(adj, n, limit=5000):
    """Find maximal bicliques using a greedy approach."""
    bicliques = []
    seen = set()
    
    # Try all pairs of vertices as seeds
    for u in range(n):
        for v in range(u+1, n):
            if v not in adj[u]:
                continue
            # Start with biclique {u} x {v}
            left = {u}
            right = {v}
            
            # Expand greedily
            changed = True
            while changed:
                changed = False
                # Try adding vertices to left
                candidates_left = set()
                for w in range(n):
                    if w not in left and w not in right:
                        # w can join left if connected to all in right
                        if all(w in adj[x] for x in right):
                            candidates_left.add(w)
                if candidates_left:
                    # Add the one with highest degree in right
                    best = max(candidates_left, key=lambda w: len(adj[w] & right))
                    left.add(best)
                    changed = True
                
                # Try adding vertices to right
                candidates_right = set()
                for w in range(n):
                    if w not in left and w not in right:
                        # w can join right if connected to all in left
                        if all(w in adj[x] for x in left):
                            candidates_right.add(w)
                if candidates_right:
                    best = max(candidates_right, key=lambda w: len(adj[w] & left))
                    right.add(best)
                    changed = True
            
            # Check if this is maximal
            # A biclique is maximal if no vertex can be added to either side
            is_maximal = True
            for w in range(n):
                if w not in left and w not in right:
                    if all(w in adj[x] for x in right):
                        is_maximal = False
                        break
                    if all(w in adj[x] for x in left):
                        is_maximal = False
                        break
            
            if is_maximal:
                key = (frozenset(left), frozenset(right))
                if key not in seen:
                    seen.add(key)
                    bicliques.append((left, right))
                    if len(bicliques) >= limit:
                        return bicliques
    
    return bicliques

def _greedy_partition(adj, n, deadline):
    """Greedy edge partition into bicliques."""
    remaining = set()
    for u, v in P['edges']:
        remaining.add((u, v))
    
    bicliques = []
    
    while remaining and time.time() < deadline:
        # Find the largest possible biclique that covers at least one remaining edge
        best_biclique = None
        best_count = 0
        
        # Try each remaining edge as a seed
        for edge in list(remaining):
            u, v = edge
            left = {u}
            right = {v}
            
            # Expand
            changed = True
            while changed:
                changed = False
                for w in range(n):
                    if w not in left and w not in right:
                        if all(w in adj[x] for x in right):
                            left.add(w)
                            changed = True
                            break
                for w in range(n):
                    if w not in left and w not in right:
                        if all(w in adj[x] for x in left):
                            right.add(w)
                            changed = True
                            break
            
            # Count how many remaining edges this biclique covers
            count = 0
            for lu in left:
                for rv in right:
                    if (lu, rv) in remaining:
                        count += 1
                    elif (rv, lu) in remaining:
                        count += 1
            
            if count > best_count:
                best_count = count
                best_biclique = (left, right)
        
        if best_biclique is None:
            break
        
        left, right = best_biclique
        # Remove covered edges
        for lu in left:
            for rv in right:
                remaining.discard((lu, rv))
                remaining.discard((rv, lu))
        
        bicliques.append((list(left), list(right)))
    
    # Handle any remaining edges
    for edge in remaining:
        u, v = edge
        bicliques.append(([u], [v]))
    
    return bicliques

def _local_search(bicliques, adj, n, deadline):
    """Try to merge bicliques to reduce count."""
    bicliques = [list(b) for b in bicliques]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(bicliques)):
            for j in range(i+1, len(bicliques)):
                if time.time() >= deadline:
                    break
                
                li, ri = bicliques[i]
                lj, rj = bicliques[j]
                
                # Try merging: (li ∪ lj) x (ri ∪ rj) must be a valid biclique
                # i.e., all cross pairs must be edges
                valid = True
                for u in li:
                    for v in rj:
                        if v not in adj[u]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    for u in lj:
                        for v in ri:
                            if v not in adj[u]:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    # Check that the merged biclique doesn't introduce non-edges
                    # Already checked above
                    new_left = list(set(li) | set(lj))
                    new_right = list(set(ri) | set(rj))
                    
                    # Verify all cross pairs are edges
                    all_valid = True
                    for u in new_left:
                        for v in new_right:
                            if v not in adj[u]:
                                all_valid = False
                                break
                        if not all_valid:
                            break
                    
                    if all_valid:
                        bicliques[i] = [new_left, new_right]
                        bicliques.pop(j)
                        improved = True
                        break
            if improved:
                break
    
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    adj = _build_adj()
    n = P['n']
    
    # Start with greedy partition
    bicliques = _greedy_partition(adj, n, deadline)
    
    # Local search to merge
    if time.time() < deadline:
        bicliques = _local_search(bicliques, adj, n, deadline)
    
    # Try multiple random restarts
    best = bicliques
    best_count = len(bicliques)
    
    for restart in range(20):
        if time.time() >= deadline:
            break
        
        # Randomize the order of edge processing
        random.seed(restart)
        edges_shuffled = list(P['edges'])
        random.shuffle(edges_shuffled)
        
        remaining = set()
        for u, v in edges_shuffled:
            remaining.add((u, v))
        
        bicliques_restart = []
        
        while remaining and time.time() < deadline:
            best_biclique = None
            best_count = 0
            
            for edge in list(remaining):
                u, v = edge
                left = {u}
                right = {v}
                
                changed = True
                while changed:
                    changed = False
                    for w in range(n):
                        if w not in left and w not in right:
                            if all(w in adj[x] for x in right):
                                left.add(w)
                                changed = True
                                break
                    for w in range(n):
                        if w not in left and w not in right:
                            if all(w in adj[x] for x in left):
                                right.add(w)
                                changed = True
                                break
                
                count = 0
                for lu in left:
                    for rv in right:
                        if (lu, rv) in remaining:
                            count += 1
                        elif (rv, lu) in remaining:
                            count += 1
                
                if count > best_count:
                    best_count = count
                    best_biclique = (left, right)
            
            if best_biclique is None:
                break
            
            left, right = best_biclique
            for lu in left:
                for rv in right:
                    remaining.discard((lu, rv))
                    remaining.discard((rv, lu))
            
            bicliques_restart.append((list(left), list(right)))
        
        for edge in remaining:
            u, v = edge
            bicliques_restart.append(([u], [v]))
        
        if time.time() < deadline:
            bicliques_restart = _local_search(bicliques_restart, adj, n, deadline)
        
        if len(bicliques_restart) < best_count:
            best = bicliques_restart
            best_count = len(bicliques_restart)
    
    # Convert to required format
    result = []
    for left, right in best:
        result.append([list(left), list(right)])
    
    return result
# EVOLVE-BLOCK-END
