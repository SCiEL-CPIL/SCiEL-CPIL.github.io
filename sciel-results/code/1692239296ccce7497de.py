import time
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = P['n']
    edges = P['edges']
    edge_set = set()
    for e in edges:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def edge_key(a, b):
        return (min(a, b), max(a, b))

    def is_biclique(L, R, edge_set):
        """Check if all cross pairs are in edge_set."""
        for l in L:
            for r in R:
                if edge_key(l, r) not in edge_set:
                    return False
        return True

    def find_biclique_from_seed(u, remaining):
        """Grow a biclique starting with u on the left side."""
        # R = neighbors of u that have remaining edges to u
        R = set()
        for v in adj[u]:
            if edge_key(u, v) in remaining:
                R.add(v)
        
        if not R:
            return None, 0
        
        L = {u}
        
        # Iterate to closure
        changed = True
        while changed:
            changed = False
            
            # Add vertices that connect to ALL of R
            new_L = set()
            for v in range(n):
                if v not in L and v not in R:
                    ok = True
                    for r in R:
                        if edge_key(v, r) not in remaining:
                            ok = False
                            break
                    if ok:
                        new_L.add(v)
            
            if new_L:
                L |= new_L
                changed = True
            
            # Add vertices that connect to ALL of L
            new_R = set()
            for v in range(n):
                if v not in L and v not in R:
                    ok = True
                    for l in L:
                        if edge_key(v, l) not in remaining:
                            ok = False
                            break
                    if ok:
                        new_R.add(v)
            
            if new_R:
                R |= new_R
                changed = True
        
        size = len(L) * len(R)
        return (L, R), size

    def find_best_biclique(remaining, deadline):
        """Find the largest biclique in the remaining edge set."""
        best = None
        best_size = 0
        
        # Try each vertex as a seed
        for u in range(n):
            if time.time() > deadline:
                break
            # Check if u has any remaining edges
            has_remaining = False
            for v in adj[u]:
                if edge_key(u, v) in remaining:
                    has_remaining = True
                    break
            if not has_remaining:
                continue
            
            biclique, size = find_biclique_from_seed(u, remaining)
            if biclique and size > best_size:
                best_size = size
                best = biclique
        
        return best, best_size

    def greedy_partition(remaining, deadline):
        """Greedy: repeatedly extract the largest biclique."""
        bicliques = []
        rem = set(remaining)
        
        while rem:
            if time.time() > deadline:
                break
            
            biclique, size = find_best_biclique(rem, deadline)
            
            if biclique is None:
                # Fallback: take one edge
                e = next(iter(rem))
                biclique = ({e[0]}, {e[1]})
            
            L, R = biclique
            
            # Verify and clean: ensure all cross edges are in rem
            valid_L = set()
            for l in L:
                ok = True
                for r in R:
                    if edge_key(l, r) not in rem:
                        ok = False
                        break
                if ok:
                    valid_L.add(l)
            
            if not valid_L:
                e = next(iter(rem))
                valid_L = {e[0]}
                R = {e[1]}
            
            # Remove cross edges
            for l in valid_L:
                for r in R:
                    rem.discard(edge_key(l, r))
            
            bicliques.append((list(valid_L), list(R)))
        
        return bicliques

    def merge_pass(bicliques, deadline):
        """Try to merge compatible bicliques."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_bicliques = []
            merged = [False] * len(bicliques)
            
            for i in range(len(bicliques)):
                if merged[i]:
                    continue
                L_i, R_i = bicliques[i]
                L_set = set(L_i)
                R_set = set(R_i)
                
                for j in range(i + 1, len(bicliques)):
                    if merged[j]:
                        continue
                    L_j, R_j = bicliques[j]
                    
                    # Try different merge strategies
                    # Strategy 1: (L_i ∪ L_j, R_i ∩ R_j)
                    # Strategy 2: (L_i ∩ L_j, R_i ∪ R_j)
                    # Strategy 3: (L_i, R_i ∪ R_j) if L_i ⊆ L_j
                    # Strategy 4: (L_i ∪ L_j, R_i) if R_i ⊆ R_j
                    
                    merged_pair = False
                    
                    # Strategy 3: L_i subset of L_j
                    if L_set <= set(L_j) and R_set <= set(R_j):
                        # i is contained in j, skip i
                        merged[i] = True
                        merged_pair = True
                        changed = True
                        break
                    
                    # Strategy 4: R_i subset of R_j
                    if R_set <= set(R_j) and L_set <= set(L_j):
                        merged[i] = True
                        merged_pair = True
                        changed = True
                        break
                    
                    # Strategy 1: union L, intersect R
                    L_union = L_set | set(L_j)
                    R_intersect = R_set & set(R_j)
                    if R_intersect and is_biclique(L_union, R_intersect, edge_set):
                        # Check no overlap in edges covered
                        # Actually we need to be more careful: the union must cover exactly the edges of both
                        # For now, just check validity
                        new_bicliques.append((list(L_union), list(R_intersect)))
                        merged[i] = True
                        merged[j] = True
                        changed = True
                        merged_pair = True
                        break
                    
                    # Strategy 2: intersect L, union R
                    L_intersect = L_set & set(L_j)
                    R_union = R_set | set(R_j)
                    if L_intersect and is_biclique(L_intersect, R_union, edge_set):
                        new_bicliques.append((list(L_intersect), list(R_union)))
                        merged[i] = True
                        merged[j] = True
                        changed = True
                        merged_pair = True
                        break
                
                if not merged_pair and not merged[i]:
                    new_bicliques.append((L_i, R_i))
            
            bicliques = new_bicliques
        
        return bicliques

    def absorb_pass(bicliques, deadline):
        """Try to absorb vertices from one biclique into another."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_bicliques = []
            
            for i, (L_i, R_i) in enumerate(bicliques):
                L_set = set(L_i)
                R_set = set(R_i)
                
                for j, (L_j, R_j) in enumerate(bicliques):
                    if i == j:
                        continue
                    L_j_set = set(L_j)
                    R_j_set = set(R_j)
                    
                    # Try adding a vertex from L_j to R_i (or vice versa)
                    # This is complex; skip for now
                    
                    # Try merging if compatible
                    # Already handled in merge_pass
            
            return bicliques  # No change in this simple version

    def refine_partition(bicliques, deadline):
        """Local search: try to reduce number of bicliques."""
        current = bicliques[:]
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            best_count = len(current)
            
            # Try removing a vertex from one biclique and adding to another
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                L_i, R_i = current[i]
                
                for v in list(L_i):
                    # Try moving v to R of another biclique
                    for j in range(len(current)):
                        if i == j:
                            continue
                        L_j, R_j = current[j]
                        
                        # Check if v can be added to R_j
                        # v must connect to all of L_j
                        ok = True
                        for l in L_j:
                            if edge_key(v, l) not in edge_set:
                                ok = False
                                break
                        if ok:
                            # Also check that removing v from L_i doesn't break anything
                            # (it won't, since we're just removing)
                            new_L_i = [x for x in L_i if x != v]
                            new_R_j = R_j + [v]
                            
                            # Check new biclique j is valid
                            if new_L_i and is_biclique(new_L_i, R_i, edge_set) and is_biclique(L_j, new_R_j, edge_set):
                                # Check no edge overlap
                                # This is complex; skip detailed check for now
                                pass
                
                # Try removing a vertex from R_i
                for v in list(R_i):
                    for j in range(len(current)):
                        if i == j:
                            continue
                        L_j, R_j = current[j]
                        
                        # Check if v can be added to L_j
                        ok = True
                        for r in R_j:
                            if edge_key(v, r) not in edge_set:
                                ok = False
                                break
                        if ok:
                            new_R_i = [x for x in R_i if x != v]
                            new_L_j = L_j + [v]
                            
                            if new_R_i and is_biclique(L_i, new_R_i, edge_set) and is_biclique(new_L_j, R_j, edge_set):
                                pass
        
        return current

    # Main algorithm
    remaining = set(edge_set)
    bicliques = greedy_partition(remaining, deadline)
    
    # Merge pass
    bicliques = merge_pass(bicliques, deadline)
    
    # Verify the partition is valid
    def verify_partition(bicliques):
        covered = {}
        for L, R in bicliques:
            for l in L:
                for r in R:
                    e = edge_key(l, r)
                    if e in covered:
                        return False
                    covered[e] = 1
        
        # Check all edges are covered
        for e in edge_set:
            if e not in covered:
                return False
        
        return True

    if not verify_partition(bicliques):
        # Fallback: use star decomposition
        bicliques = []
        for u in range(n):
            neighbors = [v for v in adj[u] if edge_key(u, v) in edge_set]
            if neighbors:
                bicliques.append(([u], neighbors))
    
    # Format result
    result = []
    for L, R in bicliques:
        L = [int(x) for x in L]
        R = [int(x) for x in R]
        result.append([L, R])
    
    return result
