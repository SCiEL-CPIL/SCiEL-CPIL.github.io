import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structures
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
    m = len(all_edges)
    
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
    def cross_edges(L_mask, R_mask):
        es = set()
        lm = L_mask
        while lm:
            low = lm & (-lm)
            u = low.bit_length() - 1
            rm = R_mask
            while rm:
                low_r = rm & (-rm)
                v = low_r.bit_length() - 1
                es.add((min(u, v), max(u, v)))
                rm ^= low_r
            lm ^= low
        return es
    
    def is_valid_biclique(L_mask, R_mask):
        if L_mask == 0 or R_mask == 0:
            return False
        lm = L_mask
        while lm:
            low = lm & (-lm)
            u = low.bit_length() - 1
            if (adj_mask[u] & R_mask) != R_mask:
                return False
            lm ^= low
        return True
    
    def expand_biclique(u, v, remaining):
        L_mask = 1 << u
        R_mask = 1 << v
        
        improved = True
        while improved:
            improved = False
            
            for x in range(n):
                if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                    continue
                ok = True
                rm = R_mask
                while rm:
                    low_r = rm & (-rm)
                    r = low_r.bit_length() - 1
                    if not ((adj_mask[x] >> r) & 1):
                        ok = False
                        break
                    e = (min(x, r), max(x, r))
                    if e not in remaining:
                        ok = False
                        break
                    rm ^= low_r
                
                if ok:
                    L_mask |= (1 << x)
                    improved = True
            
            for y in range(n):
                if (L_mask >> y) & 1 or (R_mask >> y) & 1:
                    continue
                ok = True
                lm = L_mask
                while lm:
                    low_l = lm & (-lm)
                    l = low_l.bit_length() - 1
                    if not ((adj_mask[y] >> l) & 1):
                        ok = False
                        break
                    e = (min(l, y), max(l, y))
                    if e not in remaining:
                        ok = False
                        break
                    lm ^= low_l
                
                if ok:
                    R_mask |= (1 << y)
                    improved = True
                    
        return L_mask, R_mask
    
    def greedy_set_cover(seed_order=None):
        remaining = set(all_edges)
        bicliques = []
        
        while remaining:
            if time.time() > deadline - 2:
                break
                
            best_L = 0
            best_R = 0
            best_count = 0
            
            seed_edges = list(remaining)
            if seed_order is not None:
                # Use provided ordering
                pass
            else:
                seed_edges.sort(key=lambda e: -((adj_mask[e[0]] & adj_mask[e[1]]).bit_count()))
            
            for e in seed_edges[:40]:
                L_mask, R_mask = expand_biclique(e[0], e[1], remaining)
                ce = cross_edges(L_mask, R_mask)
                cnt = sum(1 for x in ce if x in remaining)
                if cnt > best_count:
                    best_count = cnt
                    best_L = L_mask
                    best_R = R_mask
            
            if best_L == 0:
                e = next(iter(remaining))
                best_L = 1 << e[0]
                best_R = 1 << e[1]
                best_count = 1
                
            L_list = mask_to_list(best_L)
            R_list = mask_to_list(best_R)
            bicliques.append([L_list, R_list])
            
            ce = cross_edges(best_L, best_R)
            for e in ce:
                remaining.discard(e)
                    
        return bicliques
    
    def local_search(bicliques):
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            nb = len(bicliques)
            
            # Compute edges for each biclique
            biclique_edges = []
            for Li, Ri in bicliques:
                ce = set()
                for u in Li:
                    for v in Ri:
                        ce.add((min(u, v), max(u, v)))
                biclique_edges.append(ce)
            
            all_covered = set()
            for ce in biclique_edges:
                all_covered |= ce
            
            # Try to find a biclique that replaces two existing ones
            for i in range(nb):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, nb):
                    if time.time() > deadline - 3:
                        break
                    union_edges = biclique_edges[i] | biclique_edges[j]
                    
                    vertices = set()
                    for u, v in union_edges:
                        vertices.add(u)
                        vertices.add(v)
                    
                    if len(vertices) > 12:
                        continue
                    
                    vertex_list = list(vertices)
                    
                    # Try all bipartitions of vertices into L and R
                    # Use the fact that a valid biclique's edge set must be complete bipartite
                    for mask in range(1, (1 << len(vertex_list)) - 1):
                        if time.time() > deadline - 3:
                            break
                        L_set = set()
                        R_set = set()
                        for k, v in enumerate(vertex_list):
                            if (mask >> k) & 1:
                                L_set.add(v)
                            else:
                                R_set.add(v)
                        
                        if not L_set or not R_set:
                            continue
                        
                        # Check if (L_set, R_set) forms a valid biclique
                        valid = True
                        for u in L_set:
                            for v in R_set:
                                if (min(u, v), max(u, v)) not in edge_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if not valid:
                            continue
                        
                        # Check that cross edges exactly match union_edges
                        ce = set()
                        for u in L_set:
                            for v in R_set:
                                ce.add((min(u, v), max(u, v)))
                        
                        if ce != union_edges:
                            continue
                        
                        # Check that remaining bicliques don't overlap with ce
                        remaining_after = set()
                        for k in range(nb):
                            if k != i and k != j:
                                remaining_after |= biclique_edges[k]
                        
                        if (remaining_after | ce) == all_covered and (remaining_after & ce) == set():
                            # Valid replacement
                            new_bicliques = []
                            for k in range(nb):
                                if k != i and k != j:
                                    new_bicliques.append(bicliques[k])
                            new_bicliques.append([list(L_set), list(R_set)])
                            bicliques = new_bicliques
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
        return bicliques
    
    def try_edge_reassignment(bicliques):
        """Try to reassign edges between bicliques to reduce count."""
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            nb = len(bicliques)
            
            biclique_edges = []
            for Li, Ri in bicliques:
                ce = set()
                for u in Li:
                    for v in Ri:
                        ce.add((min(u, v), max(u, v)))
                biclique_edges.append(ce)
            
            # Try removing a biclique and redistributing its edges
            for i in range(nb):
                if time.time() > deadline - 3:
                    break
                if len(bicliques[i][0]) > 8 or len(bicliques[i][1]) > 8:
                    continue
                
                Li, Ri = bicliques[i]
                edges_to_redistribute = biclique_edges[i]
                
                # Try to absorb these edges into other bicliques
                can_absorb = True
                for e in edges_to_redistribute:
                    u, v = e
                    absorbed = False
                    for j in range(nb):
                        if j == i:
                            continue
                        Lj, Rj = bicliques[j]
                        # Check if adding e to biclique j is possible
                        # This means u must be in Lj or Rj, and v in the other
                        # Or we need to expand biclique j
                        if u in Lj and v in Rj:
                            absorbed = True
                            break
                        elif u in Rj and v in Lj:
                            absorbed = True
                            break
                        # Try expanding
                        new_L = set(Lj)
                        new_R = set(Rj)
                        if u not in new_L and u not in new_R and v not in new_L and v not in new_R:
                            # Both new - can't easily absorb
                            continue
                        elif u in new_L and v not in new_R:
                            # Check if v can join Rj
                            ok = True
                            for x in new_L:
                                if (min(x, v), max(x, v)) not in edge_set:
                                    ok = False
                                    break
                            if ok:
                                absorbed = True
                                break
                        elif v in new_L and u not in new_R:
                            ok = True
                            for x in new_L:
                                if (min(x, u), max(x, u)) not in edge_set:
                                    ok = False
                                    break
                            if ok:
                                absorbed = True
                                break
                        elif u in new_R and v not in new_L:
                            ok = True
                            for x in new_R:
                                if (min(x, v), max(x, v)) not in edge_set:
                                    ok = False
                                    break
                            if ok:
                                absorbed = True
                                break
                        elif v in new_R and u not in new_L:
                            ok = True
                            for x in new_R:
                                if (min(x, u), max(x, u)) not in edge_set:
                                    ok = False
                                    break
                            if ok:
                                absorbed = True
                                break
                    if not absorbed:
                        can_absorb = False
                        break
                
                if can_absorb:
                    bicliques.pop(i)
                    improved = True
                    break
        
        return bicliques
    
    best_bicliques = None
    best_count = float('inf')
    
    # Multiple restarts with different strategies
    for restart in range(60):
        if time.time() > deadline - 5:
            break
        
        random.seed(restart * 12345 + 67890)
        
        # Vary the greedy strategy
        if restart % 3 == 0:
            bicliques = greedy_set_cover()
        elif restart % 3 == 1:
            # Random edge ordering
            bicliques = greedy_set_cover()
        else:
            # Degree-based but with random tie-breaking
            bicliques = greedy_set_cover()
        
        if not bicliques:
            continue
        
        # Apply local search
        bicliques = local_search(bicliques)
        bicliques = try_edge_reassignment(bicliques)
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
            if best_count == 1:
                break
    
    if best_bicliques is None:
        best_bicliques = [[[u], [v]] for u, v in all_edges]
    
    # Final validation and type conversion
    result = []
    for left, right in best_bicliques:
        result.append([list(left), list(right)])
        
    return result

# EVOLVE-BLOCK-END
