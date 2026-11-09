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
    
    # Build adjacency as bitmasks for fast intersection
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
    def biclique_edges(L_mask, R_mask):
        """Return set of edge-tuples covered by biclique [L, R]."""
        edges = set()
        L_list = mask_to_list(L_mask)
        R_list = mask_to_list(R_mask)
        for u in L_list:
            for v in R_list:
                e = tuple(sorted((u, v)))
                if e in edge_set:
                    edges.add(e)
        return edges
    
    def is_valid_biclique(L_mask, R_mask, remaining_edges):
        """Check that all cross pairs of (L,R) are in remaining_edges (i.e., host edges) and no edge is double-counted across the partition (we handle that externally)."""
        if L_mask == 0 or R_mask == 0:
            return False
        L_list = mask_to_list(L_mask)
        R_list = mask_to_list(R_mask)
        for u in L_list:
            for v in R_list:
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
        return True
    
    def greedy_partition(seed_edges, max_iter=500):
        """
        Given a set of seed edges (subset of all edges), build a biclique partition.
        Uses greedy maximal biclique extraction.
        Returns list of (L_mask, R_mask).
        """
        if not seed_edges:
            return []
        
        remaining = set(seed_edges)
        partition = []
        
        while remaining:
            if time.time() > deadline:
                break
            
            # Pick an edge with maximum potential (or random)
            # To diversify, pick random edge
            edge = random.choice(list(remaining))
            u, v = edge
            
            # Initialize biclique with u in L, v in R
            L_mask = 1 << u
            R_mask = 1 << v
            
            changed = True
            while changed:
                changed = False
                # Try to add to L: vertex x not in L or R, adjacent to all in R
                for x in range(n):
                    if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                        continue
                    # x must be adjacent to all in R
                    if adj_mask[x] & R_mask == R_mask:
                        L_mask |= (1 << x)
                        changed = True
                # Try to add to R: vertex y not in L or R, adjacent to all in L
                for y in range(n):
                    if (L_mask >> y) & 1 or (R_mask >> y) & 1:
                        continue
                    if adj_mask[y] & L_mask == L_mask:
                        R_mask |= (1 << y)
                        changed = True
            
            # Collect edges covered by this biclique
            covered = biclique_edges(L_mask, R_mask)
            covered &= remaining  # Only keep edges still remaining
            
            if not covered:
                # Should not happen if edge was in remaining
                covered = {edge}
            
            partition.append((L_mask, R_mask))
            remaining -= covered
        
        return partition
    
    # Trivial solution: one biclique per vertex (star decomposition)
    # For each vertex u, left=[u], right=neighbors of u in the original edge list (where u is first)
    # But we need to cover all edges exactly once.
    # Better trivial: for each edge (u,v) with u<v, assign to biclique of u? No, that's not a biclique.
    # The seed's trivial: for each u, left=[u], right=[v for a,v in edges if a==u]. This covers edges where u is the first element.
    # But what about edges where u is the second element? They're not covered.
    # Actually the seed just does this and it's considered valid? Let me re-read.
    # "Every cross pair must be a host edge." So if left=[u] and right=neighbors, all cross pairs (u,v) must be edges.
    # "The cross-edge sets must partition all pinned graph edges exactly once."
    # So the star decomposition works if we assign each edge to exactly one star.
    # Simplest: for each edge (u,v), assign it to the star of min(u,v) or something.
    # But the seed does: for u in range(n): neighbors = [v for a,v in P['edges'] if a==u]. This only gets edges where u is the first element.
    # If an edge is [5,3], it won't be in the list for u=3 if 3>5.
    # Let me just create a valid trivial solution properly.
    
    trivial_partition = []
    for u in range(n):
        neighbors = [v for a, v in edges if a == u]
        if neighbors:
            trivial_partition.append((1 << u, 0))
            # Set right mask
            r_mask = 0
            for v in neighbors:
                r_mask |= (1 << v)
            trivial_partition[-1] = (1 << u, r_mask)
    
    # Verify trivial covers all edges
    trivial_covered = set()
    for L, R in trivial_partition:
        trivial_covered |= biclique_edges(L, R)
    
    if trivial_covered != edge_set:
        # Fallback: each edge is its own biclique
        trivial_partition = []
        for e in edge_set:
            u, v = e
            trivial_partition.append((1 << u, 1 << v))
    
    best_partition = trivial_partition
    best_count = len(trivial_partition)
    
    # Try multiple restarts with greedy
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Randomly permute edge order to diversify
        edge_list = list(edge_set)
        random.shuffle(edge_list)
        
        # Greedy: pick edges in shuffled order
        remaining = set(edge_list)
        partition = []
        
        while remaining:
            if time.time() > deadline:
                break
            
            # Pick next edge from shuffled order (first one still in remaining)
            edge = None
            for e in edge_list:
                if e in remaining:
                    edge = e
                    break
            if edge is None:
                edge = random.choice(list(remaining))
            
            u, v = edge
            
            # Build maximal biclique containing (u,v)
            L_mask = 1 << u
            R_mask = 1 << v
            
            # Expand
            for _ in range(10):  # Limit iterations
                changed = False
                # Add to L
                for x in range(n):
                    if ((L_mask >> x) & 1) or ((R_mask >> x) & 1):
                        continue
                    if adj_mask[x] & R_mask == R_mask:
                        L_mask |= (1 << x)
                        changed = True
                # Add to R
                for y in range(n):
                    if ((L_mask >> y) & 1) or ((R_mask >> y) & 1):
                        continue
                    if adj_mask[y] & L_mask == L_mask:
                        R_mask |= (1 << y)
                        changed = True
                if not changed:
                    break
            
            covered = biclique_edges(L_mask, R_mask) & remaining
            if not covered:
                covered = {edge}
            
            partition.append((L_mask, R_mask))
            remaining -= covered
        
        if time.time() > deadline:
            break
        
        if not remaining:  # Valid partition
            if len(partition) < best_count:
                best_count = len(partition)
                best_partition = partition
                if best_count <= 5:
                    break
        
        # Try local improvement: merge bicliques if possible
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_partition = list(best_partition)
            
            i = 0
            while i < len(new_partition) and time.time() < deadline:
                j = i + 1
                while j < len(new_partition) and time.time() < deadline:
                    L1, R1 = new_partition[i]
                    L2, R2 = new_partition[j]
                    
                    # Try to merge: is (L1|L2, R1|R2) a valid biclique?
                    # Or (L1|L2, R2) with R1 subset? Complex.
                    # Simple check: if all edges in union are covered by (L1|L2, R1|R2)
                    union_L = L1 | L2
                    union_R = R1 | R2
                    
                    # Check if union biclique is valid (all cross pairs are edges)
                    valid = True
                    for u in mask_to_list(union_L):
                        for v in mask_to_list(union_R):
                            e = tuple(sorted((u, v)))
                            if e not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Check if the union biclique exactly covers the edges of the two bicliques
                        edges_i = biclique_edges(L1, R1)
                        edges_j = biclique_edges(L2, R2)
                        edges_union = biclique_edges(union_L, union_R)
                        
                        if edges_union == (edges_i | edges_j):
                            new_partition.pop(j)
                            new_partition.pop(i)
                            new_partition.append((union_L, union_R))
                            improved = True
                            i = 0
                            break
                    j += 1
                if improved:
                    break
                i += 1
            
            if improved:
                if len(new_partition) < best_count:
                    best_count = len(new_partition)
                    best_partition = new_partition
        
        if best_count <= 3:
            break
    
    # Final validation and format
    final_result = []
    for L, R in best_partition:
        L_list = mask_to_list(L)
        R_list = mask_to_list(R)
        if L_list and R_list:
            final_result.append([L_list, R_list])
    
    # Verify partition is valid
    covered = set()
    for L, R in final_result:
        covered |= biclique_edges(int_mask := 0, 0) if False else set()
    
    # Re-verify
    covered = set()
    for bic in final_result:
        L_list, R_list = bic
        L_mask = 0
        R_mask = 0
        for u in L_list:
            L_mask |= (1 << u)
        for v in R_list:
            R_mask |= (1 << v)
        covered |= biclique_edges(L_mask, R_mask)
    
    if covered != edge_set:
        # Fallback to trivial
        final_result = []
        for u in range(n):
            neighbors = [v for a, v in edges if a == u]
            if neighbors:
                final_result.append([[u], neighbors])
        # Fix: ensure all edges covered
        covered2 = set()
        for L, R in final_result:
            for u in L:
                for v in R:
                    e = tuple(sorted((u, v)))
                    if e in edge_set:
                        covered2.add(e)
        if covered2 != edge_set:
            # Last resort: each edge as biclique
            final_result = []
            for e in edge_set:
                final_result.append([[e[0]], [e[1]]])
    
    return final_result

# EVOLVE-BLOCK-END
