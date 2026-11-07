import time
import random
import sys
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    total_edges = len(edge_set)
    
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
    def expand_biclique(L_mask, R_mask):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                if (L_mask >> u) & 1:
                    continue
                if (adj[u] & R_mask) == R_mask and R_mask != 0:
                    L_mask |= (1 << u)
                    changed = True
            for v in range(n):
                if (R_mask >> v) & 1:
                    continue
                if (adj[v] & L_mask) == L_mask and L_mask != 0:
                    R_mask |= (1 << v)
                    changed = True
        return L_mask, R_mask
    
    # Generate candidate bicliques
    # We generate maximal bicliques starting from each edge
    candidates = []
    seen_bicliques = set()
    
    # For efficiency, we limit the number of candidates or use a smart selection
    # Since n=30, we can try all edges as seeds
    for u, v in edge_set:
        L_mask = 1 << u
        R_mask = 1 << v
        L_mask, R_mask = expand_biclique(L_mask, R_mask)
        
        # Canonical form for deduplication
        l_list = tuple(sorted(mask_to_list(L_mask)))
        r_list = tuple(sorted(mask_to_list(R_mask)))
        
        # A biclique is defined by its vertex sets. 
        # Note: (L, R) and (R, L) are effectively the same set of edges, but the partition structure matters.
        # However, for set cover, we just need the set of edges covered.
        # The set of edges is determined by L and R.
        # We store the edge set as a frozenset for deduplication? No, that's expensive.
        # We store the (l_list, r_list) tuple.
        
        key = (l_list, r_list)
        if key not in seen_bicliques:
            seen_bicliques.add(key)
            candidates.append((L_mask, R_mask))
            
        # Also try the other orientation? 
        # Actually, expand_biclique is symmetric in terms of the resulting edge set, but the L/R assignment might differ.
        # For set cover, we just care about the edges.
        # The edges covered are { (i, j) : i in L, j in R }.
        # This set is the same regardless of which side is called L or R.
        
    # Filter candidates: remove those that are subsets of others? 
    # For set cover, larger sets are generally better, but not always.
    # We'll keep all maximal ones found.
    
    # Pre-compute edge sets for candidates
    candidate_edges = []
    for L_mask, R_mask in candidates:
        l_list = mask_to_list(L_mask)
        r_list = mask_to_list(R_mask)
        e_set = set()
        for u in l_list:
            for v in r_list:
                if u < v:
                    e_set.add((u, v))
                else:
                    e_set.add((v, u))
        candidate_edges.append(e_set)
    
    # Greedy Set Cover with Randomization
    def greedy_cover(remaining_edges, candidates, candidate_edges, deadline):
        chosen = []
        remaining = set(remaining_edges)
        
        while remaining and time.time() < deadline:
            best_idx = -1
            best_count = 0
            best_edge_set = None
            
            # Randomize selection order to escape local optima
            indices = list(range(len(candidates)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                covered = len(candidate_edges[i] & remaining)
                if covered > best_count:
                    best_count = covered
                    best_idx = i
                    best_edge_set = candidate_edges[i]
            
            if best_idx == -1 or best_count == 0:
                # Fallback: pick any remaining edge
                if remaining:
                    e = next(iter(remaining))
                    u, v = e
                    chosen.append(([u], [v]))
                    remaining.remove(e)
                break
            
            L_mask, R_mask = candidates[best_idx]
            l_list = mask_to_list(L_mask)
            r_list = mask_to_list(R_mask)
            chosen.append((l_list, r_list))
            remaining -= best_edge_set
            
        return chosen
    
    # Local Search / Improvement
    def improve_partition(partition, deadline):
        current_count = len(partition)
        current_edges = set()
        for l, r in partition:
            for u in l:
                for v in r:
                    if u < v:
                        current_edges.add((u, v))
                    else:
                        current_edges.add((v, u))
        
        # Verify it covers all edges
        if current_edges != edge_set:
            return partition
        
        best_partition = partition
        best_count = current_count
        
        # Try to remove a biclique and see if we can cover its edges with existing ones
        # This is basically checking if any biclique is redundant.
        # If removing one doesn't uncover any edges, remove it.
        for i in range(len(best_partition)):
            if time.time() > deadline:
                break
            l, r = best_partition[i]
            temp_partition = [p for j, p in enumerate(best_partition) if j != i]
            
            temp_edges = set()
            for l2, r2 in temp_partition:
                for u in l2:
                    for v in r2:
                        if u < v:
                            temp_edges.add((u, v))
                        else:
                            temp_edges.add((v, u))
            
            if temp_edges == edge_set:
                best_partition = temp_partition
                best_count -= 1
                # Restart loop as partition changed
                i = 0
                current_count = best_count
        
        # Try to replace two bicliques with one larger one
        # This is expensive, so we only do it a few times or on small partitions
        if len(best_partition) > 2:
            for i in range(len(best_partition)):
                for j in range(i + 1, len(best_partition)):
                    if time.time() > deadline:
                        break
                    l1, r1 = best_partition[i]
                    l2, r2 = best_partition[j]
                    
                    # Try to merge
                    # The union of edges must be coverable by a single biclique
                    # This is rare, so we skip for now to save time
                    pass
        
        return best_partition
    
    best_result = None
    best_count = float('inf')
    
    # Run multiple random restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Randomize the order of candidates or use random weights?
        # Simple approach: Shuffle candidates before greedy
        shuffled_candidates = candidates[:]
        shuffled_edges = candidate_edges[:]
        
        # Sometimes, we can bias towards larger bicliques
        # Greedy naturally does this by counting coverage.
        
        partition = greedy_cover(edge_set, shuffled_candidates, shuffled_edges, deadline)
        
        # Improve
        partition = improve_partition(partition, deadline)
        
        if len(partition) < best_count:
            best_count = len(partition)
            best_result = partition
            
            if best_count == 1:
                break
    
    # Final Fallback: If best_result is None (shouldn't happen), use seed
    if best_result is None:
        best_result = []
        for u in range(n):
            right_side = [v for u2, v in edges if u2 == u]
            if right_side:
                best_result.append(([u], right_side))
    
    # Convert to final format
    final = []
    for l, r in best_result:
        left = [int(v) for v in l]
        right = [int(v) for v in r]
        final.append([left, right])
    
    return final

# EVOLVE-BLOCK-END
