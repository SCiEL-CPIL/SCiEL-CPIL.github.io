"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _build_bitsets(n):
    """Build adjacency bitsets for fast intersection operations."""
    adj_bits = [0] * n
    for u in range(n):
        for v in P['edges']:
            if v[0] == u:
                adj_bits[u] |= (1 << v[1])
            elif v[1] == u:
                adj_bits[u] |= (1 << v[0])
    return adj_bits

def _find_biclique_bitset(seed_left, seed_right, adj_bits, deadline):
    """Expand a biclique using bitset intersection. Returns (left_set, right_set)."""
    left = set(seed_left) if isinstance(seed_left, set) else set([seed_left])
    right = set(seed_right) if isinstance(seed_right, set) else set([seed_right])
    
    # Compute intersection of neighborhoods
    if not left or not right:
        return left, right
    
    # For each vertex in left, intersect their adjacency bitmasks
    common = None
    for u in left:
        if common is None:
            common = adj_bits[u]
        else:
            common &= adj_bits[u]
        if common == 0:
            break
    
    if common == 0:
        return left, set()
    
    # Filter right to only those in common neighborhood
    new_right = set()
    for v in right:
        if (common >> v) & 1:
            new_right.add(v)
    
    # Now expand left: find all vertices connected to all of new_right
    if not new_right:
        return left, set()
    
    common_left = None
    for v in new_right:
        if common_left is None:
            common_left = adj_bits[v]
        else:
            common_left &= adj_bits[v]
        if common_left == 0:
            break
    
    if common_left is not None and common_left != 0:
        new_left = set()
        for u in range(P['n']):
            if (common_left >> u) & 1:
                new_left.add(u)
        # Keep only vertices that were in original left or connect to all right
        # Actually we want to maximize, so take all valid left vertices
        # But we need to ensure they connect to all right
        left = new_left
    
    return left, new_right

def _find_max_biclique(uncovered, adj_bits, deadline):
    """Find a large biclique using bitset-based greedy approach on uncovered edges."""
    if not uncovered:
        return None
    
    # Build adjacency restricted to uncovered edges
    unc_adj = [set() for _ in range(P['n'])]
    for e in uncovered:
        u, v = e
        unc_adj[u].add(v)
        unc_adj[v].add(u)
    
    # Build bitsets for uncovered adjacency
    unc_bits = [0] * P['n']
    for u in range(P['n']):
        for v in unc_adj[u]:
            unc_bits[u] |= (1 << v)
    
    best_left = set()
    best_right = set()
    best_count = 0
    
    # Try each vertex as seed
    involved = set()
    for e in uncovered:
        involved.add(e[0])
        involved.add(e[1])
    
    for seed in sorted(involved):
        if time.time() > deadline:
            break
        
        # Try seed on left
        left = {seed}
        right = set(unc_adj[seed])
        
        # Expand right: keep only those connected to all left
        while True:
            common = None
            for u in left:
                if common is None:
                    common = unc_bits[u]
                else:
                    common &= unc_bits[u]
                if common == 0:
                    break
            
            if common == 0:
                right = set()
                break
            
            new_right = set()
            for v in right:
                if (common >> v) & 1:
                    new_right.add(v)
            
            if len(new_right) == len(right):
                break
            right = new_right
        
        if right:
            # Expand left: find all vertices connected to all right
            common_left = None
            for v in right:
                if common_left is None:
                    common_left = unc_bits[v]
                else:
                    common_left &= unc_bits[v]
                if common_left == 0:
                    break
            
            if common_left is not None and common_left != 0:
                new_left = set()
                for u in range(P['n']):
                    if (common_left >> u) & 1:
                        new_left.add(u)
                # Only keep vertices that have edges in uncovered to all right
                # Actually common_left already ensures this
                left = new_left
        
        if left and right and len(left) * len(right) > best_count:
            best_count = len(left) * len(right)
            best_left = set(left)
            best_right = set(right)
        
        # Try seed on right
        right2 = {seed}
        left2 = set(unc_adj[seed])
        
        while True:
            common = None
            for v in right2:
                if common is None:
                    common = unc_bits[v]
                else:
                    common &= unc_bits[v]
                if common == 0:
                    break
            
            if common == 0:
                left2 = set()
                break
            
            new_left = set()
            for u in left2:
                if (common >> u) & 1:
                    new_left.add(u)
            
            if len(new_left) == len(left2):
                break
            left2 = new_left
        
        if left2:
            common_right = None
            for u in left2:
                if common_right is None:
                    common_right = unc_bits[u]
                else:
                    common_right &= unc_bits[u]
                if common_right == 0:
                    break
            
            if common_right is not None and common_right != 0:
                new_right = set()
                for v in range(P['n']):
                    if (common_right >> v) & 1:
                        new_right.add(v)
                right2 = new_right
        
        if left2 and right2 and len(left2) * len(right2) > best_count:
            best_count = len(left2) * len(right2)
            best_left = set(left2)
            best_right = set(right2)
    
    if best_count == 0:
        e = next(iter(uncovered))
        return [e[0]], [e[1]]
    
    return list(best_left), list(best_right)

def _greedy_partition(adj_bits, deadline):
    """Greedy: repeatedly extract max biclique from uncovered edges."""
    all_edges = set((min(u,v), max(u,v)) for u,v in P['edges'])
    uncovered = set(all_edges)
    bicliques = []
    
    while uncovered and time.time() < deadline:
        left, right = _find_max_biclique(uncovered, adj_bits, deadline)
        if left is None:
            for e in sorted(uncovered):
                bicliques.append([e[0], e[1]])
            break
        
        covered = set()
        for u in left:
            for v in right:
                e = (min(u,v), max(u,v))
                if e in uncovered:
                    covered.add(e)
        
        if not covered:
            e = next(iter(uncovered))
            bicliques.append([e[0], e[1]])
            uncovered.discard(e)
            continue
        
        bicliques.append([list(left), list(right)])
        uncovered -= covered
    
    for e in sorted(uncovered):
        bicliques.append([e[0], e[1]])
    
    return bicliques

def _try_merge(bicliques, adj, deadline):
    """Try to merge pairs of bicliques if their union is still valid."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        i = 0
        while i < len(bicliques) and time.time() < deadline:
            j = i + 1
            while j < len(bicliques) and time.time() < deadline:
                li, ri = bicliques[i][0], bicliques[i][1]
                lj, rj = bicliques[j][0], bicliques[j][1]
                
                # Try: can all of left_j connect to all of right_i?
                # If yes, merge left_j into left_i
                valid = True
                for u in lj:
                    for v in ri:
                        if v not in adj[u]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid and set(lj) - set(li):
                    # Check that edges from lj to ri are actually in this biclique's edge set
                    # For now, assume they are covered
                    bicliques[i][0] = list(set(li) | set(lj))
                    del bicliques[j]
                    changed = True
                    break
                
                # Try: can all of right_j connect to all of left_i?
                valid = True
                for u in li:
                    for v in rj:
                        if v not in adj[u]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid and set(rj) - set(ri):
                    bicliques[i][1] = list(set(ri) | set(rj))
                    del bicliques[j]
                    changed = True
                    break
                
                j += 1
            i += 1
    
    return bicliques

def _validate(bicliques, adj):
    """Validate the partition: check all constraints."""
    all_edges = set((min(u,v), max(u,v)) for u,v in P['edges'])
    covered = set()
    
    for bic in bicliques:
        left, right = bic
        if not left or not right:
            return False
        if set(left) & set(right):
            return False
        for u in left:
            for v in right:
                if v not in adj[u]:
                    return False
                e = (min(u,v), max(u,v))
                if e in covered:
                    return False
                covered.add(e)
    
    if covered != all_edges:
        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin before 90s kill
    
    adj = _build_adj()
    adj_bits = _build_bitsets(P['n'])
    
    # Greedy partition
    bicliques = _greedy_partition(adj_bits, deadline)
    
    # Try to merge
    bicliques = _try_merge(bicliques, adj, deadline)
    
    # Validate
    if not _validate(bicliques, adj):
        # Fallback: one biclique per edge
        bicliques = [[u, v] for u, v in P['edges']]
    
    # Ensure proper types
    result = []
    for bic in bicliques:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        result.append([left, right])
    
    return result
# EVOLVE-BLOCK-END
