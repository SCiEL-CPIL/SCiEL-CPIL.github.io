import time
import random
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = P['n']
    edges = P['edges']
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    # Adjacency bitmasks for fast intersection
    # adj_mask[i] is an integer where bit j is 1 if (i, j) is an edge
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Helper to check if a set of edges forms a biclique
    # This is the core of the "Bitset Optimization" strategy
    def is_biclique(edge_list):
        if not edge_list:
            return True
        # Find vertices
        vertices = set()
        for u, v in edge_list:
            vertices.add(u)
            vertices.add(v)
        
        # Check bipartiteness and completeness
        # We try to partition vertices into L and R such that all edges are between L and R
        # and no edges exist within L or within R (in the subgraph induced by vertices)
        
        # First, check if the induced subgraph is bipartite
        color = {}
        is_bip = True
        for start in vertices:
            if start in color:
                continue
            color[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                # Iterate neighbors in the induced subgraph
                # We can use adj_mask[u] & vertex_mask
                for v in vertices:
                    if v == u: continue
                    if (adj_mask[u] >> v) & 1:
                        if v in color:
                            if color[v] == color[u]:
                                return False
                        else:
                            color[v] = 1 - color[u]
                            queue.append(v)
                            
        # If bipartite, check if complete
        L = [v for v in vertices if color[v] == 0]
        R = [v for v in vertices if color[v] == 1]
        
        # Check all pairs in L x R are edges
        for u in L:
            for v in R:
                if not ((adj_mask[u] >> v) & 1):
                    return False
        return True

    # Heuristic to find a large biclique containing a specific edge (u, v)
    # Returns (L, R)
    def find_biclique(u, v, remaining_edges):
        # We want to maximize |L|*|R|
        # Start with L={u}, R={v}
        L = {u}
        R = {v}
        
        # Iterate until no changes
        changed = True
        while changed:
            changed = False
            # Try to add nodes to L
            # A node x can be added to L if it is connected to all nodes in R
            # and is in remaining_edges with some node in R?
            # Actually, for the biclique to be valid in the final partition,
            # we just need the edges to exist in the original graph.
            # The "remaining_edges" constraint is for the partition coverage,
            # but the biclique itself must be a valid biclique in G.
            
            # Candidates for L: neighbors of u that are connected to all R
            # We scan all vertices
            for x in range(n):
                if x in L or x in R:
                    continue
                # Check if x is connected to all in R
                can_add = True
                for y in R:
                    if not ((adj_mask[x] >> y) & 1):
                        can_add = False
                        break
                if can_add:
                    # Check if adding x creates a conflict?
                    # No, if x is connected to all R, and L is connected to all R,
                    # then L U {x} is connected to all R.
                    # We don't care about edges within L for the biclique definition
                    # (within-side edges are allowed).
                    L.add(x)
                    changed = True
                    
            # Try to add nodes to R
            for y in range(n):
                if y in L or y in R:
                    continue
                can_add = True
                for x in L:
                    if not ((adj_mask[y] >> x) & 1):
                        can_add = False
                        break
                if can_add:
                    R.add(y)
                    changed = True
                    
        return list(L), list(R)

    # Trivial solution: one biclique per vertex (star cover)
    # This is always valid.
    trivial_partition = []
    for u in range(n):
        neighbors = [v for a, v in P['edges'] if a == u]
        if neighbors:
            trivial_partition.append([ [u], neighbors ])
            
    best_partition = trivial_partition
    best_count = len(trivial_partition)
    
    if time.time() > deadline:
        return best_partition

    # Greedy Set Cover with Local Search
    # We maintain a list of bicliques.
    # We start with a greedy cover and then try to improve it by merging or swapping.
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        current_partition = []
        uncovered = set(edge_set)
        
        # 1. Greedy Phase: Pick largest biclique covering most uncovered edges
        while uncovered:
            if time.time() > deadline:
                break
                
            # Pick a random edge from uncovered to seed the biclique
            # Or pick the edge that is part of the largest potential biclique?
            # To speed up, pick random edge, find maximal biclique, see how many uncovered edges it covers.
            # If it covers 0, skip? No, it must cover at least the seed edge.
            
            # Optimization: Try a few random edges and pick the one that yields the best coverage
            best_biclique = None
            best_coverage = -1
            
            sample_size = min(10, len(uncovered))
            candidates = random.sample(list(uncovered), sample_size)
            
            for u, v in candidates:
                L, R = find_biclique(u, v, uncovered)
                # Count how many uncovered edges are in this biclique
                count = 0
                for lu in L:
                    for rv in R:
                        e = tuple(sorted((lu, rv)))
                        if e in uncovered:
                            count += 1
                            
                if count > best_coverage:
                    best_coverage = count
                    best_biclique = (L, R)
                    
            if best_biclique is None:
                # Fallback: single edge
                u, v = random.choice(list(uncovered))
                best_biclique = ([u], [v])
                best_coverage = 1
                
            L, R = best_biclique
            current_partition.append([L, R])
            
            # Remove covered edges
            for lu in L:
                for rv in R:
                    e = tuple(sorted((lu, rv)))
                    uncovered.discard(e)
                    
        # 2. Local Search Phase: Try to merge bicliques
        # If two bicliques can be merged into one valid biclique, do it.
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            new_partition = []
            used = [False] * len(current_partition)
            
            for i in range(len(current_partition)):
                if used[i]:
                    continue
                L1, R1 = current_partition[i][0], current_partition[i][1]
                merged = False
                for j in range(i + 1, len(current_partition)):
                    if used[j]:
                        continue
                    L2, R2 = current_partition[j][0], current_partition[j][1]
                    
                    # Check if union is a biclique
                    # Union edges = E(L1, R1) U E(L2, R2)
                    # We can check is_biclique on the combined edge list
                    # But constructing the list is O(|L1||R1| + |L2||R2|)
                    # For small n, this is fine.
                    
                    edges1 = [tuple(sorted((u, v))) for u in L1 for v in R1]
                    edges2 = [tuple(sorted((u, v))) for u in L2 for v in R2]
                    all_edges = edges1 + edges2
                    
                    if is_biclique(all_edges):
                        # Merge!
                        # Find the bipartition of the union
                        # We can just take the union of vertices and run is_biclique logic to get L, R
                        # Or simpler: The union is a biclique, so it has a unique bipartition (up to swap)
                        # We can find it by BFS on the union graph.
                        
                        vertices = set(L1).union(R1).union(L2).union(R2)
                        color = {}
                        start_node = L1[0]
                        color[start_node] = 0
                        queue = [start_node]
                        while queue:
                            u = queue.pop(0)
                            for v in vertices:
                                if v == u: continue
                                if (adj_mask[u] >> v) & 1:
                                    if v in color:
                                        if color[v] == color[u]:
                                             # Should not happen if is_biclique passed
                                             pass
                                    else:
                                        color[v] = 1 - color[u]
                                        queue.append(v)
                                        
                        new_L = [v for v in vertices if color.get(v, -1) == 0]
                        new_R = [v for v in vertices if color.get(v, -1) == 1]
                        
                        new_partition.append([new_L, new_R])
                        used[i] = True
                        used[j] = True
                        improved = True
                        merged = True
                        break
                        
                if not merged:
                    new_partition.append(current_partition[i])
                    used[i] = True
                    
            current_partition = new_partition
            
        if len(current_partition) < best_count:
            best_count = len(current_partition)
            best_partition = current_partition
            if best_count == 1:
                break

    # Final format check
    final_result = []
    for b in best_partition:
        L = b[0]
        R = b[1]
        # Ensure lists of ints
        L = [int(x) for x in L]
        R = [int(x) for x in R]
        final_result.append([L, R])
        
    return final_result

# EVOLVE-BLOCK-END
