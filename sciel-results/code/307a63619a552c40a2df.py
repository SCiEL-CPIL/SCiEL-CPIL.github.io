"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    t0 = time.time()
    deadline = t0 + 85  # safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as sets for fast lookup
    adj_sets = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
        edge_set.add(frozenset((u, v)))
    
    def is_valid_biclique(left, right):
        """Check if (left, right) forms a valid biclique: all cross pairs are edges."""
        if not left or not right:
            return False
        for u in left:
            for v in right:
                if v not in adj_sets[u]:
                    return False
        return True
    
    def biclique_edges(left, right):
        """Return the set of edges covered by this biclique."""
        es = set()
        for u in left:
            for v in right:
                es.add(frozenset((u, v)))
        return es
    
    def partition_covers_all(partition):
        """Check if the partition covers all edges exactly once."""
        covered = set()
        for left, right in partition:
            for u in left:
                for v in right:
                    e = frozenset((u, v))
                    if e in covered:
                        return False
                    if e not in edge_set:
                        return False
                    covered.add(e)
        return covered == edge_set
    
    def initial_partition():
        """Create initial partition: one biclique per vertex (star decomposition)."""
        partition = []
        for v in range(n):
            neighbors = list(adj_sets[v])
            if neighbors:
                partition.append(([v], neighbors))
        return partition
    
    def try_merge(partition, i, j):
        """Try to merge bicliques i and j. Returns new partition if successful, else None."""
        left_i, right_i = partition[i]
        left_j, right_j = partition[j]
        
        # Try all 4 combinations of sides
        candidates = [
            (left_i + left_j, right_i + right_j),
            (left_i + right_j, right_i + left_j),
            (right_i + left_j, left_i + right_j),
            (right_i + right_j, left_i + left_j),
        ]
        
        for left, right in candidates:
            # Remove duplicates
            left = list(set(left))
            right = list(set(right))
            if not left or not right:
                continue
            # Check no overlap between left and right
            if set(left) & set(right):
                continue
            if is_valid_biclique(left, right):
                new_partition = partition[:i] + partition[i+1:j] + partition[j+1:]
                new_partition.append((left, right))
                return new_partition
        return None
    
    def local_search(partition, max_iters=500):
        """Iteratively try merges until no more merges possible."""
        improved = True
        while improved and (time.time() - t0) < deadline:
            improved = False
            k = len(partition)
            for i in range(k):
                if (time.time() - t0) > deadline:
                    return partition
                for j in range(i+1, k):
                    new_part = try_merge(partition, i, j)
                    if new_part is not None:
                        partition = new_part
                        improved = True
                        break
                if improved:
                    break
        return partition
    
    def random_restart_partition():
        """Create a random-ish initial partition by grouping edges randomly."""
        edge_list = list(edge_set)
        random.shuffle(edge_list)
        
        # Assign edges to random bicliques
        num_bicliques = random.randint(8, 20)
        biclique_edges = [[] for _ in range(num_bicliques)]
        for e in edge_list:
            biclique_edges[random.randint(0, num_bicliques-1)].append(e)
        
        # Convert to bicliques: for each group, find max biclique
        partition = []
        for blist in biclique_edges:
            if not blist:
                continue
            # Try to form a biclique from these edges
            # Simple approach: pick a vertex, extend
            vertices_involved = set()
            for e in blist:
                u, v = tuple(e)
                vertices_involved.add(u)
                vertices_involved.add(v)
            
            best_left, best_right = None, None
            best_size = 0
            
            for anchor in list(vertices_involved)[:10]:  # limit for speed
                # Find neighbors in this edge subset
                neighbors = set()
                for e in blist:
                    u, v = tuple(e)
                    if u == anchor:
                        neighbors.add(v)
                    elif v == anchor:
                        neighbors.add(u)
                
                if not neighbors:
                    continue
                
                left = [anchor]
                right = list(neighbors)
                
                # Try to extend
                for v in list(neighbors):
                    # Check if v connects to all in right
                    v_neighbors_in_sub = set()
                    for e in blist:
                        u, w = tuple(e)
                        if u == v:
                            v_neighbors_in_sub.add(w)
                        elif w == v:
                            v_neighbors_in_sub.add(u)
                    
                    if set(right).issubset(v_neighbors_in_sub):
                        left.append(v)
                        right.remove(v)
                
                # Extend right
                for v in vertices_involved:
                    if v in left:
                        continue
                    v_neighbors_in_sub = set()
                    for e in blist:
                        u, w = tuple(e)
                        if u == v:
                            v_neighbors_in_sub.add(w)
                        elif w == v:
                            v_neighbors_in_sub.add(u)
                    
                    if set(left).issubset(v_neighbors_in_sub):
                        right.append(v)
                
                size = len(left) * len(right)
                if size > best_size and len(left) > 0 and len(right) > 0:
                    best_size = size
                    best_left = left
                    best_right = right
            
            if best_left and best_right and is_valid_biclique(best_left, best_right):
                partition.append((list(set(best_left)), list(set(best_right))))
        
        # Add any remaining edges as single-edge bicliques
        covered = set()
        for left, right in partition:
            for u in left:
                for v in right:
                    covered.add(frozenset((u, v)))
        
        for e in edge_list:
            if e not in covered:
                u, v = tuple(e)
                partition.append(([u], [v]))
        
        return partition
    
    # Start with star partition
    best_partition = initial_partition()
    best_partition = local_search(best_partition)
    best_count = len(best_partition)
    
    # ILS: multiple restarts
    random.seed(42)
    restarts = 0
    while (time.time() - t0) < deadline:
        restarts += 1
        
        # Generate new partition
        if restarts % 3 == 0:
            # Perturb best partition: remove a random biclique and re-add its edges as singles
            if len(best_partition) > 2:
                part = [list(bc) for bc in best_partition]
                idx = random.randint(0, len(part)-1)
                left, right = part[idx]
                part.pop(idx)
                # Add edges back as single-edge bicliques
                for u in left:
                    for v in right:
                        if frozenset((u, v)) in edge_set:
                            part.append(([u], [v]))
                current = part
            else:
                current = random_restart_partition()
        else:
            current = random_restart_partition()
        
        # Local search on current
        current = local_search(current)
        
        if len(current) < best_count:
            best_count = len(current)
            best_partition = current
        
        # If we've done enough restarts, try more aggressive perturbations
        if restarts > 20 and restarts % 5 == 0:
            # Swap: take a vertex from one biclique's left and try putting it in another
            if len(best_partition) > 2:
                part = [list(bc) for bc in best_partition]
                # Random swap
                i = random.randint(0, len(part)-1)
                j = random.randint(0, len(part)-1)
                if i != j and part[i][0] and part[j][0]:
                    v = random.choice(part[i][0])
                    # Remove v from biclique i's left
                    part[i][0].remove(v)
                    if not part[i][0]:
                        part.pop(i)
                        if j > i:
                            j -= 1
                    # Add v to biclique j's left
                    if v not in part[j][0] and v not in part[j][1]:
                        part[j][0].append(v)
                    current = local_search(part)
                    if len(current) < best_count:
                        best_count = len(current)
                        best_partition = current
    
    # Final: ensure all edges are covered exactly once
    # Verify and fix if needed
    covered = set()
    valid_partition = []
    for left, right in best_partition:
        left = list(set(left))
        right = list(set(right))
        if not left or not right:
            continue
        if set(left) & set(right):
            continue
        if not is_valid_biclique(left, right):
            continue
        ok = True
        for u in left:
            for v in right:
                e = frozenset((u, v))
                if e in covered or e not in edge_set:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            for u in left:
                for v in right:
                    covered.add(frozenset((u, v)))
            valid_partition.append((left, right))
    
    # Add any missing edges as single-edge bicliques
    for e in edge_set:
        if e not in covered:
            u, v = tuple(e)
            valid_partition.append(([u], [v]))
            covered.add(e)
    
    # Convert to required format
    result = []
    for left, right in valid_partition:
        left = [int(x) for x in sorted(left)]
        right = [int(x) for x in sorted(right)]
        result.append([left, right])
    
    return result
# EVOLVE-BLOCK-END
