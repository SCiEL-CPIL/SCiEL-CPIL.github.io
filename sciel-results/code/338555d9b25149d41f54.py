import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)


def _build_adj(n, edge_set):
    """Build adjacency as list of sets."""
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _expand_biclique_maximal(start_edge, adj, n, deadline):
    """
    Expand a biclique starting from start_edge to a maximal biclique.
    Uses iterative common-neighbor intersection for determinism.
    Returns (L, R) as sets.
    """
    u, v = start_edge
    L = {u}
    R = {v}
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        # Find all vertices connected to ALL in R (candidates for L)
        common_R = set.intersection(*(adj[r] for r in R)) if R else set(range(n))
        # Candidates for L: in common_R, not in R
        cand_L = common_R - R
        # Add any that aren't already in L
        to_add_L = cand_L - L
        if to_add_L:
            L |= to_add_L
            changed = True
        
        # Find all vertices connected to ALL in L (candidates for R)
        common_L = set.intersection(*(adj[l] for l in L)) if L else set(range(n))
        # Candidates for R: in common_L, not in L
        cand_R = common_L - L
        to_add_R = cand_R - R
        if to_add_R:
            R |= to_add_R
            changed = True
    
    return L, R


def _biclique_edge_set(L, R, edge_set):
    """Get the set of edges covered by biclique (L,R) that are in edge_set."""
    covered = set()
    for l in L:
        for r in R:
            ek = _edge_key(l, r)
            if ek in edge_set:
                covered.add(ek)
    return covered


def _greedy_partition(edge_set, n, adj, deadline, seed=None):
    """
    Greedy set cover: repeatedly pick the biclique covering most uncovered edges.
    """
    if seed is not None:
        random.seed(seed)
    
    bicliques = []
    remaining = set(edge_set)
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Build adjacency for remaining subgraph
        rem_adj = [set() for _ in range(n)]
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Compute degrees in remaining subgraph
        deg = [0] * n
        for u, v in remaining:
            deg[u] += 1
            deg[v] += 1
        
        # Score each edge by potential biclique size (common neighbors in remaining)
        scored_edges = []
        for u, v in remaining:
            # Common neighbors of u and v in remaining graph
            common = rem_adj[u] & rem_adj[v]
            # Potential biclique size: |L|*|R| where L includes u+common, R includes v+common
            # Approximate: (1 + len(common)) * (1 + len(common)) as upper bound
            potential = (1 + len(common)) ** 2
            scored_edges.append((potential, u, v))
        
        # Sort by potential descending, take top candidates
        scored_edges.sort(reverse=True)
        top_k = min(25, len(scored_edges))
        candidates = [scored_edges[i][1:] for i in range(top_k)]
        
        # Add some random candidates for diversity
        if len(remaining) > 30:
            rand_edges = random.sample(list(remaining), min(10, len(remaining)))
            candidates.extend(rand_edges)
        
        best_biclique = None
        best_cover = 0
        
        for u, v in candidates:
            if time.time() > deadline:
                break
            L, R = _expand_biclique_maximal((u, v), rem_adj, n, deadline)
            
            # Count actual edges covered
            cover = 0
            for l in L:
                for r in R:
                    if _edge_key(l, r) in remaining:
                        cover += 1
            
            if cover > best_cover:
                best_cover = cover
                best_biclique = (L, R)
        
        if best_biclique is None or best_cover == 0:
            edge = next(iter(remaining))
            u, v = edge
            best_biclique = ({u}, {v})
        
        L, R = best_biclique
        bicliques.append((L, R))
        
        # Remove covered edges
        covered = _biclique_edge_set(L, R, remaining)
        remaining -= covered
    
    # Leftover edges as single-edge bicliques
    for u, v in remaining:
        bicliques.append(({u}, {v}))
    
    return bicliques


def _try_merge_pass(bicliques, edge_set, n, deadline):
    """
    Try to merge pairs of bicliques if their union forms a valid biclique
    that covers exactly the union of their edges.
    """
    current = [(set(L), set(R)) for L, R in bicliques]
    changed = True
    
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        for i in range(len(current)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(current)):
                if time.time() > deadline:
                    break
                
                Li, Ri = current[i]
                Lj, Rj = current[j]
                
                # Try different merge orientations
                merges = [
                    (Li | Lj, Ri | Rj),
                    (Li | Rj, Ri | Lj),
                    (Ri | Lj, Li | Rj),
                    (Ri | Rj, Li | Lj),
                ]
                
                for newL, newR in merges:
                    if not newL or not newR:
                        continue
                    if newL & newR:
                        continue
                    
                    # Check: all cross pairs must be edges in original graph
                    valid = True
                    covered_edges = set()
                    for l in newL:
                        for r in newR:
                            ek = _edge_key(l, r)
                            if ek not in edge_set:
                                valid = False
                                break
                            covered_edges.add(ek)
                        if not valid:
                            break
                    
                    if not valid:
                        continue
                    
                    # Check: covered_edges must equal union of edges from biclique i and j
                    edges_i = set()
                    for l in Li:
                        for r in Ri:
                            edges_i.add(_edge_key(l, r))
                    edges_j = set()
                    for l in Lj:
                        for r in Rj:
                            edges_j.add(_edge_key(l, r))
                    
                    if covered_edges == (edges_i | edges_j) and not (edges_i & edges_j):
                        # Valid merge
                        current.pop(j)
                        current[i] = (newL, newR)
                        changed = True
                        break
                
                if changed:
                    break
            
            if changed:
                break
    
    return current


def _local_search_move(bicliques, edge_set, n, deadline):
    """
    Try to reduce biclique count by moving vertices or splitting/recombining.
    For each biclique, try removing a vertex and see if the remaining edges
    can be absorbed by other bicliques.
    """
    current = [(set(L), set(R)) for L, R in bicliques]
    changed = True
    
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        for i in range(len(current)):
            if time.time() > deadline:
                break
            
            Li, Ri = current[i]
            
            # Try removing each vertex from L or R
            for side, vert in [('L', v) for v in Li] + [('R', v) for v in Ri]:
                if time.time() > deadline:
                    break
                
                if side == 'L':
                    new_Li = Li - {vert}
                    new_Ri = Ri
                else:
                    new_Li = Li
                    new_Ri = Ri - {vert}
                
                if not new_Li or not new_Ri:
                    continue
                
                # The edges lost by removing vert
                lost_edges = set()
                if side == 'L':
                    for r in Ri:
                        ek = _edge_key(vert, r)
                        if ek in edge_set:
                            lost_edges.add(ek)
                else:
                    for l in Li:
                        ek = _edge_key(l, vert)
                        if ek in edge_set:
                            lost_edges.add(ek)
                
                # Check if lost_edges can be covered by other bicliques
                # (i.e., they're already covered by other bicliques or are within-side)
                other_covered = set()
                for j in range(len(current)):
                    if j == i:
                        continue
                    Lj, Rj = current[j]
                    for l in Lj:
                        for r in Rj:
                            ek = _edge_key(l, r)
                            if ek in edge_set:
                                other_covered.add(ek)
                
                # lost_edges must either be covered by other bicliques or be within-side edges
                uncovered = lost_edges - other_covered
                if uncovered:
                    continue
                
                # Check that new biclique i is still valid
                valid = True
                for l in new_Li:
                    for r in new_Ri:
                        ek = _edge_key(l, r)
                        if ek not in edge_set:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check no overlap in edges with other bicliques
                new_edges_i = set()
                for l in new_Li:
                    for r in new_Ri:
                        ek = _edge_key(l, r)
                        if ek in edge_set:
                            new_edges_i.add(ek)
                
                overlap = False
                for j in range(len(current)):
                    if j == i:
                        continue
                    Lj, Rj = current[j]
                    for l in Lj:
                        for r in Rj:
                            ek = _edge_key(l, r)
                            if ek in edge_set and ek in new_edges_i:
                                overlap = True
                                break
                        if overlap:
                            break
                    if overlap:
                        break
                
                if overlap:
                    continue
                
                # Valid move: update biclique i
                if side == 'L':
                    current[i] = (new_Li, new_Ri)
                else:
                    current[i] = (new_Li, new_Ri)
                changed = True
                break
            
            if changed:
                break
    
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    n = P['n']
    edges = P['edges']
    
    # Build edge set and adjacency
    edge_set = set()
    for u, v in edges:
        edge_set.add(_edge_key(u, v))
    
    adj = _build_adj(n, edge_set)
    
    best_solution = None
    best_count = float('inf')
    
    # Phase 1: Multiple greedy restarts
    num_restarts = 30
    for seed in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        bicliques = _greedy_partition(edge_set, n, adj, deadline - 10.0, seed=seed)
        count = len(bicliques)
        
        if count < best_count:
            best_count = count
            best_solution = bicliques[:]
        
        if best_count <= 8:
            break
    
    if best_solution is None:
        # Fallback
        best_solution = [({u}, {v}) for u, v in edge_set]
    
    # Phase 2: Merge passes to reduce count
    merged = _try_merge_pass(best_solution, edge_set, n, deadline - 8.0)
    
    if len(merged) < best_count:
        best_count = len(merged)
        best_solution = merged
    
    # Phase 3: Local search - try moving vertices
    local = _local_search_move(best_solution, edge_set, n, deadline - 5.0)
    
    if len(local) < best_count:
        best_count = len(local)
        best_solution = local
    
    # Phase 4: Try a few more merge passes with different initial orderings
    for attempt in range(8):
        if time.time() > deadline - 3.0:
            break
        
        # Shuffle bicliques to find different merge opportunities
        shuffled = best_solution[:]
        random.shuffle(shuffled)
        merged2 = _try_merge_pass(shuffled, edge_set, n, deadline - 2.0)
        
        if len(merged2) < best_count:
            best_count = len(merged2)
            best_solution = merged2
        
        # Also try local search after shuffle
        local2 = _local_search_move(merged2, edge_set, n, deadline - 1.5)
        if len(local2) < best_count:
            best_count = len(local2)
            best_solution = local2
    
    # Final merge pass
    final_merged = _try_merge_pass(best_solution, edge_set, n, deadline - 0.5)
    if len(final_merged) < best_count:
        best_count = len(final_merged)
        best_solution = final_merged
    
    # Convert to required format: list of [left_list, right_list]
    result = []
    for L, R in best_solution:
        result.append([list(L), list(R)])
    
    return result
