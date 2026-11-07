"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_biclique_from_seed(seed_left, seed_right, remaining_set, adj, n, deadline):
    """
    Grow a biclique from initial left/right sets, constrained to remaining edges.
    Returns (left, right, covered_edges_set).
    """
    left = set(seed_left)
    right = set(seed_right)
    
    if not left or not right:
        return None
    
    # Ensure no overlap
    if left & right:
        return None
    
    # Check initial validity
    for u in left:
        for v in right:
            k = _edge_key(u, v)
            if k not in remaining_set:
                return None
    
    changed = True
    while changed:
        changed = False
        
        # Try to expand left
        candidates = []
        for w in range(n):
            if w in left or w in right:
                continue
            # w must connect to all of right via remaining edges
            ok = True
            for r in right:
                if _edge_key(w, r) not in remaining_set:
                    ok = False
                    break
            if ok:
                candidates.append(w)
        
        # Add as many as possible (order doesn't matter for validity, but affects future)
        # To maximize size, add all valid candidates
        if candidates:
            left.update(candidates)
            changed = True
        
        # Try to expand right
        candidates = []
        for w in range(n):
            if w in left or w in right:
                continue
            ok = True
            for l in left:
                if _edge_key(l, w) not in remaining_set:
                    ok = False
                    break
            if ok:
                candidates.append(w)
        
        if candidates:
            right.update(candidates)
            changed = True
        
        if time.time() > deadline:
            break
    
    # Calculate covered edges
    covered = set()
    for u in left:
        for v in right:
            k = _edge_key(u, v)
            if k in remaining_set:
                covered.add(k)
    
    return (left, right, covered)

def _try_merge(b1, b2, remaining_set, adj, n, deadline):
    """
    Try to merge two bicliques.
    b1, b2 are (left, right, covered) tuples.
    Returns merged (left, right, covered) or None.
    """
    l1, r1, c1 = b1
    l2, r2, c2 = b2
    
    # Try 4 combinations
    # 1. L = l1 U l2, R = r1 U r2
    # 2. L = l1 U r2, R = r1 U l2
    # 3. L = r1 U l2, R = l1 U r2  (symmetric to 2)
    # 4. L = r1 U r2, R = l1 U l2
    
    candidates = [
        (l1 | l2, r1 | r2),
        (l1 | r2, r1 | l2),
        (r1 | l2, l1 | r2),
        (r1 | r2, l1 | l2)
    ]
    
    best_merged = None
    best_size = 0
    
    for nl, nr in candidates:
        if not nl or not nr:
            continue
        if nl & nr:
            continue
        
        # Check validity: all cross edges must be in remaining_set
        valid = True
        covered = set()
        for u in nl:
            for v in nr:
                k = _edge_key(u, v)
                if k in remaining_set:
                    covered.add(k)
                else:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # The union of covered edges must be a subset of the new covered edges?
        # Actually, the new biclique covers 'covered'.
        # The original bicliques covered c1 and c2.
        # For a valid partition merge, we need c1 U c2 == covered?
        # No, we just need the new biclique to be valid in the remaining graph.
        # But we also need to ensure we don't "lose" edges or "add" edges that weren't there?
        # Wait, if we merge, we replace b1 and b2 with the new biclique.
        # The edges covered by the new biclique are 'covered'.
        # We need 'covered' to be a subset of (c1 U c2)? No, that's not right.
        # The edges in the graph are fixed.
        # If we merge, we are forming a larger biclique.
        # The cross edges of this larger biclique MUST be a subset of the original graph edges.
        # Also, since we are partitioning, the edges covered by the new biclique should ideally
        # be exactly c1 U c2 if we want a clean partition step?
        # Actually, if we merge b1 and b2, we remove c1 and c2 from 'remaining'.
        # Then we add 'covered' to the partition.
        # If 'covered' contains edges NOT in c1 U c2, that's fine as long as they were in remaining.
        # But wait, if 'covered' contains edges NOT in c1 U c2, it means we are covering MORE edges
        # than the two bicliques did. This is valid! It's a better cover.
        # However, we must ensure that all edges in 'covered' were actually in 'remaining_set'
        # (which we checked) and that they are valid graph edges.
        
        # One constraint: The new biclique's sides must be disjoint (checked).
        # Another: The new biclique must not introduce "internal" edges that are NOT in the graph?
        # The definition says "Within-side edges are allowed". So that's fine.
        
        # So we just check if 'covered' is a valid set of edges.
        # We should prefer merges that cover more edges or result in simpler structure?
        # For now, any valid merge is good.
        
        size = len(covered)
        if size > best_size:
            best_size = size
            best_merged = (nl, nr, covered)
        
        if time.time() > deadline:
            break
    
    return best_merged

def _greedy_partition(remaining_edges, adj, n, deadline):
    """
    Greedily partition remaining edges into bicliques.
    """
    remaining_set = set(remaining_edges)
    partition = []
    
    while remaining_set:
        if time.time() > deadline:
            break
        
        # Find a large biclique
        # Strategy: Pick a random edge, or high-degree vertex?
        # Let's pick an edge (u, v) and grow.
        # To be deterministic or semi-deterministic, sort edges?
        # Let's try picking the edge with the highest sum of degrees in remaining graph.
        
        # Calculate degrees in remaining graph
        deg = defaultdict(int)
        for u, v in remaining_set:
            deg[u] += 1
            deg[v] += 1
        
        # Find max degree edge
        best_edge = None
        max_score = -1
        for u, v in remaining_set:
            score = deg[u] + deg[v]
            if score > max_score:
                max_score = score
                best_edge = (u, v)
        
        if best_edge is None:
            break
            
        u, v = best_edge
        
        # Try to grow biclique from this seed
        # Try both orientations: {u} left, {v} right AND {v} left, {u} right
        b1 = _find_biclique_from_seed({u}, {v}, remaining_set, adj, n, deadline)
        b2 = _find_biclique_from_seed({v}, {u}, remaining_set, adj, n, deadline)
        
        if b1 is None and b2 is None:
            # Fallback: single edge
            partition.append([u, [v]])
            remaining_set.remove(best_edge)
            continue
        
        # Choose the one that covers more edges
        if b1 is None:
            best_b = b2
        elif b2 is None:
            best_b = b1
        else:
            if len(b1[2]) >= len(b2[2]):
                best_b = b1
            else:
                best_b = b2
        
        l, r, covered = best_b
        partition.append([sorted(l), sorted(r)])
        remaining_set -= covered
    
    return partition

def _local_search_improve(partition, remaining_edges, adj, n, deadline):
    """
    Try to improve the partition by merging bicliques.
    """
    # Convert partition to internal format
    bicliques = []
    for l, r in partition:
        l_set = set(l)
        r_set = set(r)
        covered = set()
        for u in l_set:
            for v in r_set:
                k = _edge_key(u, v)
                if k in remaining_edges:
                    covered.add(k)
        bicliques.append((l_set, r_set, covered))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                if time.time() > deadline:
                    break
                
                # Try to merge bicliques i and j
                # The remaining edges for the merge check should be the union of their covered edges?
                # No, the merge check is: can we form a valid biclique from the union of vertices
                # such that all cross edges exist in the ORIGINAL graph?
                # And does this new biclique cover a superset of the edges covered by i and j?
                # Actually, if we merge, we replace i and j with the new biclique.
                # The new biclique must be valid in the original graph.
                # And it must cover at least the edges covered by i and j?
                # If it covers MORE, that's fine, as long as those extra edges are in the graph
                # and were not covered by OTHER bicliques.
                # This is getting complex.
                
                # Simpler check:
                # Let S_i be edges covered by biclique i.
                # Let S_j be edges covered by biclique j.
                # We want to find a biclique B such that S_i U S_j subset of Edges(B).
                # And Edges(B) must be a subset of the original graph edges.
                # Also, Edges(B) \ (S_i U S_j) must be disjoint from edges covered by other bicliques.
                
                # Let's just try the merge function from inspiration, which checks validity in 'remaining_set'.
                # But 'remaining_set' here is tricky.
                # Let's pass the union of covered edges as the "remaining" context? No.
                
                # Let's use the original graph adjacency.
                merged = _try_merge(bicliques[i], bicliques[j], set(), adj, n, deadline)
                # Wait, _try_merge checks against 'remaining_set'.
                # If we pass empty set, it will fail unless covered is empty.
                # We need a version of _try_merge that checks against the FULL graph.
                
                # Let's write a specific merge check for the full graph.
                l1, r1, c1 = bicliques[i]
                l2, r2, c2 = bicliques[j]
                
                candidates = [
                    (l1 | l2, r1 | r2),
                    (l1 | r2, r1 | l2),
                    (r1 | l2, l1 | r2),
                    (r1 | r2, l1 | l2)
                ]
                
                found_merge = False
                for nl, nr in candidates:
                    if not nl or not nr or (nl & nr):
                        continue
                    
                    # Check if all cross edges exist in graph
                    valid = True
                    new_covered = set()
                    for u in nl:
                        for v in nr:
                            k = _edge_key(u, v)
                            if k in adj[u]: # adj[u] contains v if edge exists
                                new_covered.add(k)
                            else:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if not valid:
                        continue
                    
                    # Check if new_covered contains c1 U c2
                    if not (c1 | c2).issubset(new_covered):
                        continue
                    
                    # Check if new_covered is disjoint from edges covered by OTHER bicliques
                    other_edges = set()
                    for idx, (ll, rr, cc) in enumerate(bicliques):
                        if idx != i and idx != j:
                            other_edges |= cc
                    
                    if new_covered & other_edges:
                        continue
                    
                    # If valid, perform merge
                    new_biclique = (nl, nr, new_covered)
                    # Remove i and j, add new_biclique
                    # Order matters, let's remove j first then i (since j > i)
                    bicliques.pop(j)
                    bicliques.pop(i)
                    bicliques.append(new_biclique)
                    improved = True
                    found_merge = True
                    break
                
                if found_merge:
                    break
            
            if found_merge:
                break
    
    # Convert back to list of lists
    result = []
    for l, r, c in bicliques:
        result.append([sorted(l), sorted(r)])
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    edges = P['edges']
    
    adj = _build_adj(n, edges)
    edge_set = set(_edge_key(u, v) for u, v in edges)
    
    # Initial greedy partition
    partition = _greedy_partition(edge_set, adj, n, deadline)
    
    # Local search improvement
    if time.time() < deadline:
        partition = _local_search_improve(partition, edge_set, adj, n, deadline)
    
    # Final validation and type conversion
    # Ensure types are int and list
    final_partition = []
    for biclique in partition:
        left = [int(v) for v in biclique[0]]
        right = [int(v) for v in biclique[1]]
        if left and right:
            final_partition.append([left, right])
    
    return final_partition
# EVOLVE-BLOCK-END
