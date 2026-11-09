"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    deadline = time.time() + 85  # 85s safety margin for 90s limit
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u < v:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
        else:
            adj[v].add(u)
            adj[u].add(v)
            edge_set.add((v, u))
    
    def biclique_edges(L, R):
        """Get the set of edges covered by biclique [L, R]."""
        es = set()
        for u in L:
            for v in R:
                if (u, v) in edge_set:
                    es.add((u, v))
                elif (v, u) in edge_set:
                    es.add((v, u))
        return es
    
    def is_valid_biclique(L, R):
        """Check if [L, R] is a valid biclique (all cross pairs are edges)."""
        if not L or not R:
            return False
        if set(L) & set(R):
            return False
        for u in L:
            for v in R:
                if (u, v) not in edge_set and (v, u) not in edge_set:
                    return False
        return True
    
    # Generate candidate bicliques
    candidates = []
    seen = set()
    
    # Single edges
    for u, v in edges:
        bc = (tuple(sorted([u])), tuple(sorted([v])))
        if bc not in seen:
            seen.add(bc)
            candidates.append(list(bc))
    
    # Try to grow bicliques
    for i in range(len(candidates)):
        L = list(candidates[i][0])
        R = list(candidates[i][1])
        # Try to add vertices to L
        for u in range(n):
            if u not in L and u not in R:
                new_L = L + [u]
                # Check if u connects to all in R
                ok = True
                for v in R:
                    if (u, v) not in edge_set and (v, u) not in edge_set:
                        ok = False
                        break
                if ok:
                    bc = (tuple(sorted(new_L)), tuple(R))
                    if bc not in seen:
                        seen.add(bc)
                        candidates.append(list(bc))
        # Try to add vertices to R
        for v in range(n):
            if v not in L and v not in R:
                new_R = R + [v]
                # Check if v connects to all in L
                ok = True
                for u in L:
                    if (u, v) not in edge_set and (v, u) not in edge_set:
                        ok = False
                        break
                if ok:
                    bc = (tuple(L), tuple(sorted(new_R)))
                    if bc not in seen:
                        seen.add(bc)
                        candidates.append(list(bc))
    
    # Also try random biclique generation
    for _ in range(200):
        L = random.sample(range(n), random.randint(1, 5))
        # Find vertices that connect to all in L
        R_candidates = [v for v in range(n) if v not in L and all((u, v) in edge_set or (v, u) in edge_set for u in L)]
        if R_candidates:
            R = random.sample(R_candidates, min(random.randint(1, 5), len(R_candidates)))
            if is_valid_biclique(L, R):
                bc = (tuple(sorted(L)), tuple(sorted(R)))
                if bc not in seen:
                    seen.add(bc)
                    candidates.append(list(bc))
    
    # Initial partition: one biclique per edge
    best_partition = [[[u], [v]] for u, v in edges]
    best_count = len(best_partition)
    
    # Local search: try to replace multiple bicliques with fewer
    def partition_covered_edges(partition):
        """Get the set of edges covered by the partition."""
        covered = set()
        for L, R in partition:
            covered |= biclique_edges(L, R)
        return covered
    
    def partition_is_valid(partition):
        """Check if partition exactly covers all edges."""
        covered = partition_covered_edges(partition)
        return covered == edge_set
    
    # Try to improve: replace a subset of bicliques with a single biclique
    for iter_count in range(50000):
        if time.time() > deadline:
            break
        
        # Try replacing 2-3 bicliques with 1
        subset_size = random.choice([2, 2, 3])
        if len(best_partition) < subset_size:
            continue
        
        indices = random.sample(range(len(best_partition)), subset_size)
        # Collect edges covered by these bicliques
        covered_edges = set()
        for idx in indices:
            L, R = best_partition[idx]
            covered_edges |= biclique_edges(L, R)
        
        # Try to find a single biclique that covers all these edges
        # The biclique must have L containing all left vertices from these bicliques' edges
        # and R containing all right vertices
        left_verts = set()
        right_verts = set()
        for e in covered_edges:
            u, v = e
            left_verts.add(u)
            right_verts.add(v)
            left_verts.add(v)
            right_verts.add(u)
        
        # Try a few candidate bicliques
        found = False
        for _ in range(20):
            # Random subset of left and right vertices
            L_size = random.randint(1, min(5, len(left_verts)))
            R_size = random.randint(1, min(5, len(right_verts)))
            L = random.sample(list(left_verts), L_size)
            R = random.sample(list(right_verts), R_size)
            
            if set(L) & set(R):
                continue
            
            if not is_valid_biclique(L, R):
                continue
            
            new_edges = biclique_edges(L, R)
            if new_edges == covered_edges:
                # Replace the subset with this single biclique
                new_partition = [best_partition[i] for i in range(len(best_partition)) if i not in indices]
                new_partition.append([L, R])
                if partition_is_valid(new_partition):
                    if len(new_partition) < best_count:
                        best_partition = new_partition
                        best_count = len(new_partition)
                        found = True
                        break
        
        # Try merging: replace 2 bicliques with 2 different bicliques (repartition)
        if not found and len(best_partition) >= 2 and random.random() < 0.3:
            i, j = random.sample(range(len(best_partition)), 2)
            L1, R1 = best_partition[i]
            L2, R2 = best_partition[j]
            edges1 = biclique_edges(L1, R1)
            edges2 = biclique_edges(L2, R2)
            all_edges = edges1 | edges2
            
            # Try to split all_edges into two bicliques
            for _ in range(10):
                # Random partition of edges
                edge_list = list(all_edges)
                random.shuffle(edge_list)
                split = random.randint(1, len(edge_list) - 1)
                group1 = set(edge_list[:split])
                group2 = set(edge_list[split:])
                
                # Try to find bicliques for each group
                bc1 = None
                bc2 = None
                for _ in range(5):
                    verts1 = set()
                    for u, v in group1:
                        verts1.add(u)
                        verts1.add(v)
                    if not verts1:
                        continue
                    L1_new = random.sample(list(verts1), random.randint(1, min(4, len(verts1))))
                    R1_new = [v for v in verts1 if v not in L1_new]
                    if not R1_new:
                        continue
                    if is_valid_biclique(L1_new, R1_new) and biclique_edges(L1_new, R1_new) == group1:
                        bc1 = [L1_new, R1_new]
                        break
                
                if bc1:
                    for _ in range(5):
                        verts2 = set()
                        for u, v in group2:
                            verts2.add(u)
                            verts2.add(v)
                        if not verts2:
                            continue
                        L2_new = random.sample(list(verts2), random.randint(1, min(4, len(verts2))))
                        R2_new = [v for v in verts2 if v not in L2_new]
                        if not R2_new:
                            continue
                        if is_valid_biclique(L2_new, R2_new) and biclique_edges(L2_new, R2_new) == group2:
                            bc2 = [L2_new, R2_new]
                            break
                
                if bc1 and bc2:
                    new_partition = [best_partition[k] for k in range(len(best_partition)) if k not in (i, j)]
                    new_partition.extend([bc1, bc2])
                    if partition_is_valid(new_partition) and len(new_partition) < best_count:
                        best_partition = new_partition
                        best_count = len(new_partition)
                        break
    
    # Convert to proper format
    result = []
    for L, R in best_partition:
        result.append([list(L), list(R)])
    
    return result
# EVOLVE-BLOCK-END
