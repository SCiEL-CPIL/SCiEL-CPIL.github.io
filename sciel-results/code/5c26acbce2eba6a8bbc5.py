import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _seeded_random(seed):
    return random.Random(seed)

def _build_adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _star_partition(edges, n):
    """Star partition: each vertex with degree > 0 becomes a star center."""
    adj = _build_adjacency(edges, n)
    bicliques = []
    for u in range(n):
        if adj[u]:
            bicliques.append(([u], sorted(adj[u])))
    return bicliques

def _greedy_biclique_partition(edges, n, seed=42):
    """
    Greedy approach: repeatedly find the largest biclique in remaining edges,
    remove those edges, repeat until no edges remain.
    """
    rng = _seeded_random(seed)
    remaining = set((min(u,v), max(u,v)) for u, v in edges)
    bicliques = []
    
    while remaining:
        # Find a good biclique in the remaining graph
        # Strategy: pick a random edge, try to expand it to a larger biclique
        best_biclique = None
        best_size = 0
        
        # Sample some edges to try
        sample_edges = list(remaining)
        rng.shuffle(sample_edges)
        sample_edges = sample_edges[:min(50, len(sample_edges))]
        
        for e in sample_edges:
            u, v = e
            # Try to expand: L = {u}, R = {v}, try to add more vertices
            L = {u}
            R = {v}
            # Try adding vertices to L: a vertex x can be added to L if it connects to all of R
            # Try adding vertices to R: a vertex y can be added to R if it connects to all of L
            
            changed = True
            while changed:
                changed = False
                # Try adding to L
                for x in range(n):
                    if x not in L and x not in R:
                        if all(x in adj[x] for x in R) if False else all(y in adj[x] for y in R):
                            L.add(x)
                            changed = True
                # Try adding to R
                for y in range(n):
                    if y not in L and y not in R:
                        if all(y in adj[x] for x in L):
                            R.add(y)
                            changed = True
            
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best_biclique = (L, R)
        
        if best_biclique is None or best_size == 0:
            # Fallback: just take one edge
            u, v = next(iter(remaining))
            best_biclique = ({u}, {v})
        
        L, R = best_biclique
        bicliques.append((sorted(L), sorted(R)))
        
        # Remove all edges between L and R from remaining
        for u in L:
            for v in R:
                e = (min(u,v), max(u,v))
                remaining.discard(e)
    
    return bicliques

def _greedy_with_local_search(edges, n, seed=42, max_iters=10000):
    """
    Improved greedy: start with star partition, then try to merge bicliques
    if the merged set forms a valid biclique.
    """
    rng = _seeded_random(seed)
    adj = _build_adjacency(edges, n)
    
    # Start with star partition
    bicliques = _star_partition(edges, n)
    
    # Track which edges are covered by which biclique
    # For each biclique, compute its edge set
    def get_edges(bi):
        L, R = bi
        es = set()
        for u in L:
            for v in R:
                e = (min(u,v), max(u,v))
                es.add(e)
        return es
    
    # Try to merge bicliques
    improved = True
    while improved:
        improved = False
        bicliques = [list(bi) for bi in bicliques]
        
        for i in range(len(bicliques)):
            for j in range(i+1, len(bicliques)):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Try merging: new_L = L1 ∪ L2, new_R = R1 ∪ R2
                # But we need all cross pairs to be edges
                new_L = set(L1) | set(L2)
                new_R = set(R1) | set(R2)
                
                # Check if all cross pairs are edges
                valid = True
                for u in new_L:
                    for v in new_R:
                        if v not in adj[u]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    bicliques[i] = [sorted(new_L), sorted(new_R)]
                    bicliques.pop(j)
                    improved = True
                    break
            if improved:
                break
    
    return bicliques

def _local_search_partition(edges, n, seed=42, time_limit=45):
    """
    Local search: start with a greedy partition, then try to improve by
    moving vertices between bicliques or merging/splitting.
    """
    rng = _seeded_random(seed)
    adj = _build_adjacency(edges, n)
    start_time = time.time()
    
    # Start with greedy
    bicliques = _greedy_biclique_partition(edges, n, seed)
    
    def count_bicliques(bis):
        return len(bis)
    
    best_count = count_bicliques(bicliques)
    best_bicliques = [list(bi) for bi in bicliques]
    
    # Try to improve by merging
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(len(bicliques)):
            for j in range(i+1, len(bicliques)):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                new_L = set(L1) | set(L2)
                new_R = set(R1) | set(R2)
                
                valid = True
                for u in new_L:
                    for v in new_R:
                        if v not in adj[u]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    bicliques[i] = [sorted(new_L), sorted(new_R)]
                    bicliques.pop(j)
                    improved = True
                    if count_bicliques(bicliques) < best_count:
                        best_count = count_bicliques(bicliques)
                        best_bicliques = [list(bi) for bi in bicliques]
                    break
            if improved:
                break
    
    return best_bicliques

def _verify_partition(bicliques, edges, n):
    """Verify that the partition is valid."""
    edge_set = set((min(u,v), max(u,v)) for u, v in edges)
    covered = set()
    
    for L, R in bicliques:
        if not L or not R:
            return False
        # Check no vertex appears in both sides
        if set(L) & set(R):
            return False
        # Check all cross pairs are edges
        for u in L:
            for v in R:
                e = (min(u,v), max(u,v))
                if e not in edge_set:
                    return False
                covered.add(e)
    
    # Check all edges are covered exactly once
    if covered != edge_set:
        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = 75  # Safety margin before 90s limit
    
    n = P['n']
    edges = P['edges']
    
    # First, get a valid baseline: star partition
    best_bicliques = _star_partition(edges, n)
    best_count = len(best_bicliques)
    
    # Try greedy biclique partition
    if time.time() - start_time < deadline:
        greedy = _greedy_biclique_partition(edges, n, seed=42)
        if _verify_partition(greedy, edges, n) and len(greedy) < best_count:
            best_bicliques = greedy
            best_count = len(greedy)
    
    # Try greedy with local search
    if time.time() - start_time < deadline:
        gls = _greedy_with_local_search(edges, n, seed=42)
        if _verify_partition(gls, edges, n) and len(gls) < best_count:
            best_bicliques = gls
            best_count = len(gls)
    
    # Try local search from different seeds
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if time.time() - start_time >= deadline:
            break
        ls = _local_search_partition(edges, n, seed=seed, time_limit=deadline - (time.time() - start_time))
        if _verify_partition(ls, edges, n) and len(ls) < best_count:
            best_bicliques = ls
            best_count = len(ls)
    
    # Convert to proper format: list of [left, right] with Python ints
    result = []
    for L, R in best_bicliques:
        left = [int(v) for v in L]
        right = [int(v) for v in R]
        result.append([left, right])
    
    return result
# EVOLVE-BLOCK-END
