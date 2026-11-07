import time
import random

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as bitmasks and sets
    adj_mask = [0] * n
    adj_set = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        adj_set[u].add(v)
        adj_set[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    all_edges = list(edge_set)
    total_edges = len(all_edges)
    
    def mask_to_list(mask):
        """Convert bitmask to list of vertex indices."""
        result = []
        m = mask
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            result.append(v)
            m ^= low
        return result

    def list_to_mask(lst):
        """Convert list of vertex indices to bitmask."""
        mask = 0
        for v in lst:
            mask |= (1 << v)
        return mask

    def get_cross_edges_mask(L_mask, R_mask):
        """Return set of edges between L and R (only those that exist in graph)."""
        result = set()
        lm = L_mask
        while lm:
            low = lm & (-lm)
            u = low.bit_length() - 1
            # Only consider neighbors of u that are in R
            rm = R_mask & adj_mask[u]
            while rm:
                low_r = rm & (-rm)
                v = low_r.bit_length() - 1
                result.add((min(u, v), max(u, v)))
                rm ^= low_r
            lm ^= low
        return result
    
    def is_valid_biclique(L_mask, R_mask):
        """Check if (L_mask, R_mask) forms a valid biclique in the host graph."""
        if L_mask == 0 or R_mask == 0:
            return False
        lm = L_mask
        while lm:
            low = lm & (-lm)
            v = low.bit_length() - 1
            if (adj_mask[v] & R_mask) != R_mask:
                return False
            lm ^= low
        return True
    
    def max_biclique_containing(edge, remaining):
        """Find a maximal biclique containing the given edge, restricted to remaining edges."""
        u, v = edge
        L_mask = 1 << u
        R_mask = 1 << v
        
        changed = True
        while changed:
            changed = False
            # Try to add vertex x to L
            for x in range(n):
                if x == u or x == v:
                    continue
                if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                    continue
                # x can join L if all r in R are neighbors of x AND all (x,r) edges are in remaining
                ok = True
                rm = R_mask
                while rm:
                    low_r = rm & (-rm)
                    r = low_r.bit_length() - 1
                    if not (adj_mask[x] >> r) & 1:
                        ok = False
                        break
                    e = (min(x, r), max(x, r))
                    if e not in remaining:
                        ok = False
                        break
                    rm ^= low_r
                if ok:
                    L_mask |= (1 << x)
                    changed = True
            
            # Try to add vertex y to R
            for y in range(n):
                if y == u or y == v:
                    continue
                if (L_mask >> y) & 1 or (R_mask >> y) & 1:
                    continue
                ok = True
                lm = L_mask
                while lm:
                    low_l = lm & (-lm)
                    l = low_l.bit_length() - 1
                    if not (adj_mask[y] >> l) & 1:
                        ok = False
                        break
                    e = (min(l, y), max(l, y))
                    if e not in remaining:
                        ok = False
                        break
                    lm ^= low_l
                if ok:
                    R_mask |= (1 << y)
                    changed = True
        
        return L_mask, R_mask

    def greedy_init():
        remaining = set(all_edges)
        bicliques = []
        while remaining:
            best_L = 0
            best_R = 0
            best_count = 0
            
            sample = list(remaining)
            random.shuffle(sample)
            # Increase sample size for better initial solutions
            for e in sample[:30]:
                L_mask, R_mask = max_biclique_containing(e, remaining)
                cross = get_cross_edges_mask(L_mask, R_mask)
                count = len(cross & remaining)
                if count > best_count:
                    best_count = count
                    best_L = L_mask
                    best_R = R_mask
            
            if best_L == 0:
                e = next(iter(remaining))
                best_L = 1 << e[0]
                best_R = 1 << e[1]
            
            L_list = mask_to_list(best_L)
            R_list = mask_to_list(best_R)
            bicliques.append([L_list, R_list])
            
            cross = get_cross_edges_mask(best_L, best_R)
            remaining -= cross
        
        return bicliques
    
    def verify_partition(bicliques):
        """Verify that the bicliques form a valid partition of all edges."""
        covered = set()
        for left, right in bicliques:
            L_mask = list_to_mask(left)
            R_mask = list_to_mask(right)
            # Check biclique validity
            if not is_valid_biclique(L_mask, R_mask):
                return False
            cross = get_cross_edges_mask(L_mask, R_mask)
            # Check no overlap
            if covered & cross:
                return False
            covered |= cross
        
        return covered == edge_set
    
    def try_merge_pair(bi, bj, bicliques):
        """Try to merge two bicliques into one. Returns new list or None."""
        Li, Ri = bicliques[bi]
        Lj, Rj = bicliques[bj]
        
        ei = get_cross_edges_mask(list_to_mask(Li), list_to_mask(Ri))
        ej = get_cross_edges_mask(list_to_mask(Lj), list_to_mask(Rj))
        
        union = ei | ej
        
        # Find all vertices involved
        vertices = set()
        for u, v in union:
            vertices.add(u)
            vertices.add(v)
        
        # Check if union forms a complete bipartite graph
        # Build subgraph
        sub_adj = {v: set() for v in vertices}
        for u, v in union:
            sub_adj[u].add(v)
            sub_adj[v].add(u)
        
        # Bipartite check
        color = {}
        for start in vertices:
            if start not in color:
                color[start] = 0
                stack = [start]
                while stack:
                    curr = stack.pop()
                    for nb in sub_adj[curr]:
                        if nb not in color:
                            color[nb] = 1 - color[curr]
                            stack.append(nb)
                        elif color[nb] == color[curr]:
                            return None
        
        L = [v for v in vertices if color[v] == 0]
        R = [v for v in vertices if color[v] == 1]
        
        # Verify completeness: all pairs in L×R must be in union
        L_set = set(L)
        R_set = set(R)
        for u in L:
            for v in R:
                if (min(u, v), max(u, v)) not in union:
                    return None
        
        # Verify no within-side edges in union
        for u in L:
            for w in L:
                if u < w and (u, w) in union:
                    return None
        for v in R:
            for w in R:
                if v < w and (v, w) in union:
                    return None
        
        new_bicliques = [b for k, b in enumerate(bicliques) if k != bi and k != bj]
        new_bicliques.append([L, R])
        return new_bicliques

    def local_search(bicliques):
        """Apply merge operations to reduce biclique count."""
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            nb = len(bicliques)
            for i in range(nb):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, nb):
                    result = try_merge_pair(i, j, bicliques)
                    if result is not None:
                        bicliques = result
                        improved = True
                        break
                if improved:
                    break
        return bicliques
    
    def split_reassign(bicliques):
        """Try to improve by splitting a biclique and redistributing edges."""
        if time.time() > deadline - 5:
            return bicliques
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            nb = len(bicliques)
            
            for i in range(nb):
                if time.time() > deadline - 3:
                    break
                Li, Ri = bicliques[i]
                
                # Try moving a vertex from L to R (or R to L) if it helps
                for v in Li:
                    if len(Li) <= 1:
                        break
                    new_L = [x for x in Li if x != v]
                    new_R = Ri + [v]
                    
                    # Check if new_L, new_R is still a valid biclique
                    if not new_L or not new_R:
                        continue
                    L_mask = list_to_mask(new_L)
                    R_mask = list_to_mask(new_R)
                    if not is_valid_biclique(L_mask, R_mask):
                        continue
                    
                    # The moved vertex v now has edges to new_R that were previously
                    # internal to L. These edges need to be covered elsewhere.
                    # Check if those edges are already covered by other bicliques
                    # or if we can absorb them.
                    
                    # For simplicity, only accept if the new biclique covers fewer edges
                    # and the "lost" edges are covered by other bicliques
                    old_cross = get_cross_edges_mask(list_to_mask(Li), list_to_mask(Ri))
                    new_cross = get_cross_edges_mask(L_mask, R_mask)
                    
                    # Edges lost: old_cross - new_cross
                    lost = old_cross - new_cross
                    if not lost:
                        continue
                    
                    # Check if lost edges are covered by other bicliques
                    other_covered = set()
                    for k in range(nb):
                        if k == i:
                            continue
                        Lk, Rk = bicliques[k]
                        other_covered |= get_cross_edges_mask(list_to_mask(Lk), list_to_mask(Rk))
                    
                    if lost <= other_covered:
                        # Accept the move
                        bicliques[i] = [new_L, new_R]
                        improved = True
                        break
                
                if improved:
                    break
                
                # Also try moving from R to L
                for v in Ri:
                    if len(Ri) <= 1:
                        break
                    new_R = [x for x in Ri if x != v]
                    new_L = Li + [v]
                    
                    if not new_L or not new_R:
                        continue
                    L_mask = list_to_mask(new_L)
                    R_mask = list_to_mask(new_R)
                    if not is_valid_biclique(L_mask, R_mask):
                        continue
                    
                    old_cross = get_cross_edges_mask(list_to_mask(Li), list_to_mask(Ri))
                    new_cross = get_cross_edges_mask(L_mask, R_mask)
                    lost = old_cross - new_cross
                    if not lost:
                        continue
                    
                    other_covered = set()
                    for k in range(nb):
                        if k == i:
                            continue
                        Lk, Rk = bicliques[k]
                        other_covered |= get_cross_edges_mask(list_to_mask(Lk), list_to_mask(Rk))
                    
                    if lost <= other_covered:
                        bicliques[i] = [new_L, new_R]
                        improved = True
                        break
                
                if improved:
                    break
        
        return bicliques

    # Main search
    best_bicliques = None
    best_count = float('inf')
    
    # Increase restarts to find better solutions
    for restart in range(100):
        if time.time() > deadline - 8:
            break
        
        bicliques = greedy_init()
        bicliques = local_search(bicliques)
        bicliques = split_reassign(bicliques)
        
        # Final verification
        if verify_partition(bicliques):
            if len(bicliques) < best_count:
                best_count = len(bicliques)
                best_bicliques = bicliques
        else:
            # Fallback: use single-edge bicliques (always valid)
            pass
    
    if best_bicliques is None:
        # Absolute fallback: each edge is its own biclique
        best_bicliques = [[[u], [v]] for u, v in all_edges]
    
    result = []
    for left, right in best_bicliques:
        result.append([list(left), list(right)])
    
    return result

# EVOLVE-BLOCK-END
