# EVOLVE-BLOCK-START
import random
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add(frozenset((u, v)))
        
    def get_cross_edges(L, R):
        es = set()
        for u in L:
            for v in R:
                if adj[u][v]:
                    es.add(frozenset((u, v)))
        return es

    def is_valid_partition(bicliques):
        """Check if the partition is valid."""
        covered = set()
        for L, R in bicliques:
            if L & R:
                return False
            for u in L:
                for v in R:
                    if not adj[u][v]:
                        return False
                    e = frozenset((u, v))
                    if e in covered:
                        return False
                    covered.add(e)
        return covered == edge_set

    def can_merge(B1, B2):
        L1, R1 = B1
        L2, R2 = B2
        L_new = L1 | L2
        R_new = R1 | R2
        if L_new & R_new:
            return False
        E1 = get_cross_edges(L1, R1)
        E2 = get_cross_edges(L2, R2)
        E_new = get_cross_edges(L_new, R_new)
        return (E1 | E2) == E_new

    # Initialization: Random Greedy
    # This is different from the current program's specific greedy logic
    random.seed(42)
    remaining = set(edge_set)
    bicliques = []
    
    while remaining:
        # Pick a random edge
        e = random.choice(list(remaining))
        u, v = tuple(e)
        L = {u}
        R = {v}
        
        # Expand greedily
        changed = True
        while changed:
            changed = False
            # Try to add to L
            candidates = [x for x in range(n) if x not in L and x not in R]
            random.shuffle(candidates)
            for cand in candidates:
                can_add = True
                for rv in R:
                    if not adj[cand][rv] or frozenset((cand, rv)) not in remaining:
                        can_add = False
                        break
                if can_add:
                    L.add(cand)
                    changed = True
                    break
            # Try to add to R
            candidates = [x for x in range(n) if x not in L and x not in R]
            random.shuffle(candidates)
            for cand in candidates:
                can_add = True
                for lu in L:
                    if not adj[lu][cand] or frozenset((lu, cand)) not in remaining:
                        can_add = False
                        break
                if can_add:
                    R.add(cand)
                    changed = True
                    break
        
        # Add to bicliques
        bicliques.append([L, R])
        
        # Remove covered edges
        for lu in L:
            for rv in R:
                e = frozenset((lu, rv))
                if e in remaining:
                    remaining.remove(e)
                    
    # Local Search
    best_bicliques = bicliques
    best_count = len(bicliques)
    
    for _ in range(500): # Iterations
        improved = False
        
        # Try Merges
        merged = False
        for i in range(len(bicliques)):
            if merged: break
            for j in range(i + 1, len(bicliques)):
                if can_merge(bicliques[i], bicliques[j]):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    new_b = [L1 | L2, R1 | R2]
                    bicliques.pop(j)
                    bicliques.pop(i)
                    bicliques.append(new_b)
                    improved = True
                    merged = True
                    break
        
        if improved:
            if len(bicliques) < best_count:
                best_count = len(bicliques)
                best_bicliques = [b[:] for b in bicliques]
            continue
            
        # Try Splits (if stuck)
        # Pick a random biclique and try to split it
        if bicliques:
            idx = random.randint(0, len(bicliques) - 1)
            L, R = bicliques[idx]
            
            # Try to move a vertex from L to R (if valid) or to new biclique
            # We will try to split L into L1, L2
            if len(L) > 1:
                # Random split
                L_list = list(L)
                random.shuffle(L_list)
                mid = len(L_list) // 2
                L1 = set(L_list[:mid])
                L2 = set(L_list[mid:])
                
                # Check if L1, R and L2, R are valid bicliques
                # And if they cover the same edges
                # This is a valid split if no edges are lost or gained
                # Actually, any split of L is valid for the biclique structure, 
                # but we must ensure the partition remains valid.
                # If we split B into B1(L1,R) and B2(L2,R), 
                # the edges are (L1 x R) and (L2 x R). Union is L x R.
                # This is a valid partition of the edges of B.
                
                # Replace B with B1 and B2
                bicliques.pop(idx)
                bicliques.append([L1, R])
                bicliques.append([L2, R])
                
                # This increases count by 1. 
                # We accept this if it leads to a merge later?
                # For now, we just do it to escape local minima.
                
                # Check if new bicliques can merge with others
                # (This is handled in the next iteration's Merge phase)
                
                # If we split, we don't consider it an "improvement" immediately
                # but we allow it to happen to find better merges.
                
    # Return best found
    result = []
    for L, R in best_bicliques:
        L_list = [int(x) for x in sorted(L)]
        R_list = [int(x) for x in sorted(R)]
        result.append([L_list, R_list])
        
    return result
# EVOLVE-BLOCK-END
