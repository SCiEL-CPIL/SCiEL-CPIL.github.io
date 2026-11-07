import time
import random
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _build_adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_maximal_biclique(adj, start_u, start_v, n, deadline):
    """
    Find a maximal biclique containing edge (start_u, start_v).
    Returns (L, R) as frozensets.
    """
    L = {start_u}
    R = {start_v}
    
    # Precompute intersection of neighbors for speed
    # We want to add w to L if N(w) superset R
    # We want to add w to R if N(w) superset L
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        # Try to add to L
        # Candidates for L: vertices not in L or R
        # Condition: all vertices in R are neighbors of w
        for w in range(n):
            if w not in L and w not in R:
                # Check if R is subset of adj[w]
                # Optimization: if len(R) > len(adj[w]), skip
                if len(R) <= len(adj[w]):
                    # Efficient check
                    if all(r in adj[w] for r in R):
                        L.add(w)
                        changed = True
        
        # Try to add to R
        for w in range(n):
            if w not in L and w not in R:
                if len(L) <= len(adj[w]):
                    if all(l in adj[w] for l in L):
                        R.add(w)
                        changed = True
                        
    return frozenset(L), frozenset(R)

def _greedy_partition_with_refinement(adj, edge_set, n, deadline, seed):
    random.seed(seed)
    
    remaining = set(edge_set)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
            
        # Generate candidates
        # Pick top edges by degree in remaining graph
        deg = [0] * n
        for u, v in remaining:
            deg[u] += 1
            deg[v] += 1
            
        # Sort remaining edges by sum of degrees
        sorted_edges = sorted(remaining, key=lambda e: deg[e[0]] + deg[e[1]], reverse=True)
        
        # Take top K edges as seeds for biclique expansion
        # K depends on remaining size
        k = min(20, len(remaining))
        seed_edges = sorted_edges[:k]
        
        # Also add some random edges if remaining is large
        if len(remaining) > 50:
            random_edges = random.sample(list(remaining), min(10, len(remaining)))
            seed_edges.extend(random_edges)
        
        candidates = []
        for u, v in seed_edges:
            # Only expand if u, v are still in remaining (they are, by definition)
            L, R = _find_maximal_biclique(adj, u, v, n, deadline)
            
            # Calculate actual coverage in 'remaining'
            covered = set()
            for l in L:
                for r in R:
                    e = _edge_key(l, r)
                    if e in remaining:
                        covered.add(e)
            
            if covered:
                # Store (coverage_count, L, R, covered_set)
                candidates.append((len(covered), L, R, covered))
        
        if not candidates:
            # Fallback: single edge
            u, v = next(iter(remaining))
            bicliques.append((frozenset([u]), frozenset([v])))
            remaining.remove(_edge_key(u, v))
            continue
            
        # Sort candidates by coverage size
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Select the best candidate
        # To add some diversity, we could pick from top 3 randomly, 
        # but for now let's stick to best to ensure progress, 
        # and rely on multi-seed for diversity.
        best_count, best_L, best_R, best_covered = candidates[0]
        
        bicliques.append((best_L, best_R))
        
        # Remove covered edges
        for e in best_covered:
            remaining.discard(e)
            
    # Handle leftovers
    for u, v in remaining:
        bicliques.append((frozenset([u]), frozenset([v])))
        
    # Refinement: Merge pass
    # Try to merge any two bicliques if their union is a valid biclique
    # and covers exactly the union of their edges.
    merged = True
    while merged:
        if time.time() > deadline:
            break
        merged = False
        new_bicliques = []
        n_b = len(bicliques)
        used = [False] * n_b
        
        for i in range(n_b):
            if used[i]:
                continue
            L1, R1 = bicliques[i]
            edges1 = set()
            for l in L1:
                for r in R1:
                    edges1.add(_edge_key(l, r))
            
            for j in range(i+1, n_b):
                if used[j]:
                    continue
                L2, R2 = bicliques[j]
                
                # Potential merged biclique
                L_union = L1 | L2
                R_union = R1 | R2
                
                # Check if (L_union, R_union) is a valid biclique in the ORIGINAL graph
                # i.e., all cross edges exist in adj
                valid = True
                for l in L_union:
                    for r in R_union:
                        if r not in adj[l]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Calculate edges covered by this merged biclique
                edges_merged = set()
                for l in L_union:
                    for r in R_union:
                        edges_merged.add(_edge_key(l, r))
                
                # Check if edges_merged is exactly edges1 U edges2
                # Note: edges1 and edges2 are disjoint by construction of partition
                if edges_merged == (edges1 | set(_edge_key(l, r) for l in L2 for r in R2)):
                    # We can merge!
                    new_bicliques.append((L_union, R_union))
                    used[i] = True
                    used[j] = True
                    merged = True
                    break # Restart loop to handle new state cleanly
                    
            if used[i]:
                continue
            if not merged: # If we didn't merge i with anything
                 new_bicliques.append(bicliques[i])
                 used[i] = True
        
        if merged:
            bicliques = new_bicliques
            # Repeat to catch cascading merges
            
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 75.0 # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    edge_set = set()
    for u, v in edges:
        edge_set.add(_edge_key(u, v))
        
    adj = _build_adjacency(edges, n)
    
    best_solution = None
    best_count = float('inf')
    
    # Try multiple seeds
    # 30 seeds should be fast enough given the small graph size
    for seed in range(30):
        if time.time() > deadline:
            break
            
        bicliques = _greedy_partition_with_refinement(adj, edge_set, n, deadline, seed)
        
        count = len(bicliques)
        if count < best_count:
            best_count = count
            best_solution = bicliques
            
        if count <= 10:
            break
            
    if best_solution is None:
        # Fallback
        best_solution = [(frozenset([u]), frozenset([v])) for u, v in edge_set]
        
    # Convert to required format
    result = []
    for L, R in best_solution:
        # Ensure lists are sorted for consistency, though not strictly required
        # The problem says "sequences must be list or tuple"
        result.append([list(L), list(R)])
        
    return result
