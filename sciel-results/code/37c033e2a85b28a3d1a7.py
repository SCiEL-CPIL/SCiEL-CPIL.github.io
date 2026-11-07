import time
import random
import sys
from collections import defaultdict
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u, v), max(u, v)))
    
    num_edges = len(edge_set)
    
    # Build edge index for fast lookup
    edge_to_idx = {e: i for i, e in enumerate(edge_set)}
    edge_list = list(edge_set)
    edge_set_list = list(edge_set)
    
    # Precompute neighbor sets
    neighbors = [set() for _ in range(n)]
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
    
    # Enumerate all maximal bicliques
    # Strategy: for each edge, try to grow it into a maximal biclique
    # Using a simple but effective enumeration
    
    maximal_bicliques = []
    seen_bicliques = set()
    
    def grow_biclique(L, R, used_edges):
        """Try to grow biclique (L, R) using only edges in used_edges."""
        changed = True
        while changed:
            changed = False
            # Try adding vertices to L
            for x in range(n):
                if x in L:
                    continue
                ok = True
                for y in R:
                    if not adj[x][y] or (min(x, y), max(x, y)) not in used_edges:
                        ok = False
                        break
                if ok:
                    L.add(x)
                    changed = True
            # Try adding vertices to R
            for y in range(n):
                if y in R:
                    continue
                ok = True
                for x in L:
                    if not adj[x][y] or (min(x, y), max(x, y)) not in used_edges:
                        ok = False
                        break
                if ok:
                    R.add(y)
                    changed = True
        return L, R
    
    # Enumerate by growing each edge
    for e in edge_set:
        u, v = e
        # Try different initial orientations
        for L_init, R_init in [({u}, {v}), ({v}, {u})]:
            L = set(L_init)
            R = set(R_init)
            L, R = grow_biclique(L, R, edge_set)
            biclique_key = (frozenset(L), frozenset(R))
            if biclique_key not in seen_bicliques:
                seen_bicliques.add(biclique_key)
                # Compute edges covered
                covered = set()
                for u2 in L:
                    for v2 in R:
                        covered.add((min(u2, v2), max(u2, v2)))
                if covered.issubset(edge_set):
                    maximal_bicliques.append((L, R, covered))
    
    # Also try growing from each vertex pair
    for u in range(n):
        for v in neighbors[u]:
            if u > v:
                continue
            e = (min(u, v), max(u, v))
            if e not in edge_set:
                continue
            L = {u}
            R = {v}
            L, R = grow_biclique(L, R, edge_set)
            biclique_key = (frozenset(L), frozenset(R))
            if biclique_key not in seen_bicliques:
                seen_bicliques.add(biclique_key)
                covered = set()
                for u2 in L:
                    for v2 in R:
                        covered.add((min(u2, v2), max(u2, v2)))
                if covered.issubset(edge_set):
                    maximal_bicliques.append((L, R, covered))
    
    # Remove non-maximal bicliques (if one is subset of another)
    filtered_bicliques = []
    for i, (Li, Ri, Ci) in enumerate(maximal_bicliques):
        is_maximal = True
        for j, (Lj, Rj, Cj) in enumerate(maximal_bicliques):
            if i == j:
                continue
            if Li.issubset(Lj) and Ri.issubset(Rj) and Ci.issubset(Cj) and (Li != Lj or Ri != Rj):
                is_maximal = False
                break
        if is_maximal:
            filtered_bicliques.append((Li, Ri, Ci))
    
    # Sort by size (larger first)
    filtered_bicliques.sort(key=lambda x: len(x[2]), reverse=True)
    
    # Now solve minimum set cover using branch and bound
    # Candidate sets are the maximal bicliques
    # We need to select minimum number of bicliques whose edge sets partition all edges
    
    m = len(filtered_bicliques)
    biclique_edges = [set(C) for _, _, C in filtered_bicliques]
    
    # Precompute which bicliques cover each edge
    edge_to_bicliques = defaultdict(list)
    for i in range(m):
        for e in biclique_edges[i]:
            edge_to_bicliques[e].append(i)
    
    # Greedy initial solution
    def greedy_cover():
        remaining = set(edge_set)
        chosen = []
        while remaining:
            best_i = -1
            best_count = 0
            for i in range(m):
                count = len(biclique_edges[i] & remaining)
                if count > best_count:
                    best_count = count
                    best_i = i
            if best_i == -1 or best_count == 0:
                break
            chosen.append(best_i)
            remaining -= biclique_edges[best_i]
        return chosen if not remaining else None
    
    greedy_sol = greedy_cover()
    best_solution = greedy_sol
    best_count = len(greedy_sol) if greedy_sol else num_edges + 1
    
    # Branch and bound
    # State: set of covered edges, list of chosen bicliques
    # Use bitmasks for edges if num_edges <= 64
    use_bitmask = num_edges <= 60
    
    if use_bitmask:
        edge_mask = [0] * num_edges
        for e in edge_set:
            edge_mask[edge_to_idx[e]] = 1 << edge_to_idx[e]
        full_mask = (1 << num_edges) - 1
        biclique_mask = [0] * m
        for i in range(m):
            mask = 0
            for e in biclique_edges[i]:
                mask |= edge_mask[edge_to_idx[e]]
            biclique_mask[i] = mask
    else:
        biclique_mask = None
    
    def lower_bound(remaining_mask):
        """Compute lower bound on number of bicliques needed."""
        if use_bitmask:
            if remaining_mask == 0:
                return 0
            # Count uncovered edges
            count = bin(remaining_mask).count('1')
            if count == 0:
                return 0
            # Upper bound on edges per biclique
            max_cover = 0
            for i in range(m):
                cover = bin(biclique_mask[i] & remaining_mask).count('1')
                if cover > max_cover:
                    max_cover = cover
            if max_cover == 0:
                return count
            return (count + max_cover - 1) // max_cover
        else:
            count = len(remaining_mask)
            if count == 0:
                return 0
            max_cover = 0
            for i in range(m):
                cover = len(biclique_edges[i] & remaining_mask)
                if cover > max_cover:
                    max_cover = cover
            if max_cover == 0:
                return count
            return (count + max_cover - 1) // max_cover
    
    def solve_bnb(remaining_mask, chosen, current_depth):
        nonlocal best_solution, best_count
        
        if time.time() > deadline:
            return
        
        if remaining_mask == 0:
            if current_depth < best_count:
                best_count = current_depth
                best_solution = chosen[:]
            return
        
        lb = lower_bound(remaining_mask)
        if current_depth + lb >= best_count:
            return
        
        # Find an uncovered edge
        if use_bitmask:
            # Get lowest set bit
            lsb = remaining_mask & (-remaining_mask)
            target_idx = lsb.bit_length() - 1
            target_edge = edge_list[target_idx]
        else:
            target_edge = next(iter(remaining_mask))
        
        candidates = edge_to_bicliques[target_edge]
        
        # Sort by coverage of remaining
        def coverage_score(i):
            if use_bitmask:
                return bin(biclique_mask[i] & remaining_mask).count('1')
            else:
                return len(biclique_edges[i] & remaining_mask)
        
        sorted_candidates = sorted(candidates, key=coverage_score, reverse=True)
        
        for i in sorted_candidates:
            if use_bitmask:
                new_remaining = remaining_mask & ~biclique_mask[i]
            else:
                new_remaining = remaining_mask - biclique_edges[i]
            chosen.append(i)
            solve_bnb(new_remaining, chosen, current_depth + 1)
            chosen.pop()
            
            if time.time() > deadline:
                return
    
    # Start BnB
    if use_bitmask:
        init_remaining = full_mask
    else:
        init_remaining = set(edge_set)
    
    solve_bnb(init_remaining, [], 0)
    
    # If BnB didn't find anything better, try local search / merging
    # Convert best_solution to bicliques
    if best_solution is not None:
        result = []
        for idx in best_solution:
            L, R, _ = filtered_bicliques[idx]
            result.append([list(L), list(R)])
        
        # Validate
        covered = set()
        valid = True
        for L, R in result:
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e in covered:
                        valid = False
                        break
                    covered.add(e)
                    if not adj[u][v]:
                        valid = False
                        break
            if not valid:
                break
        if valid and covered == edge_set:
            # Convert types
            final = []
            for L, R in result:
                final.append([[int(x) for x in L], [int(x) for x in R]])
            return final
    
    # Fallback: one biclique per edge
    return [[[u], [v]] for u, v in edge_set]

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} bicliques")
