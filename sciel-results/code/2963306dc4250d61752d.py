"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start = time.time()
    deadline = start + 85.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets and edge set
    adj = [set() for _ in range(n)]
    all_edges_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        all_edges_set.add((min(u, v), max(u, v)))
    
    remaining = set(all_edges_set)
    bicliques = []
    
    def is_valid_biclique(L, R):
        if L & R:
            return False
        for l in L:
            for r in R:
                if (min(l, r), max(l, r)) not in all_edges_set:
                    return False
        return True

    def expand_biclique(L, R, remaining_edges):
        """Expand L and R to be maximal with respect to remaining_edges."""
        changed = True
        while changed:
            changed = False
            # Add to L
            new_L = set()
            for x in range(n):
                if x in L or x in R:
                    continue
                # x must connect to all in R via remaining edges
                if all((min(x, r), max(x, r)) in remaining_edges for r in R):
                    new_L.add(x)
            if new_L:
                L.update(new_L)
                changed = True
            
            # Add to R
            new_R = set()
            for y in range(n):
                if y in L or y in R:
                    continue
                # y must connect to all in L via remaining edges
                if all((min(l, y), max(l, y)) in remaining_edges for l in L):
                    new_R.add(y)
            if new_R:
                R.update(new_R)
                changed = True
        return L, R

    def get_covered_edges(L, R, remaining_edges):
        covered = set()
        for l in L:
            for r in R:
                k = (min(l, r), max(l, r))
                if k in remaining_edges:
                    covered.add(k)
        return covered

    # Phase 1: Greedy Set Cover
    while remaining and time.time() < deadline:
        best_biclique = None
        best_covered = set()
        best_score = -1
        
        # Sample edges to seed bicliques
        rem_list = list(remaining)
        if len(rem_list) > 50:
            # Sample evenly
            step = len(rem_list) / 50.0
            idxs = set()
            for i in range(50):
                idxs.add(int(i * step))
            idxs.add(0)
            idxs.add(len(rem_list) - 1)
            seeds = [rem_list[i] for i in sorted(idxs)]
        else:
            seeds = rem_list
        
        for u, v in seeds:
            L = {u}
            R = {v}
            # Expand
            L, R = expand_biclique(L, R, remaining)
            
            # Ensure disjoint
            if L & R:
                continue
                
            covered = get_covered_edges(L, R, remaining)
            if not covered:
                continue
            
            # Score: maximize covered edges
            score = len(covered)
            if score > best_score:
                best_score = score
                best_biclique = (L, R)
                best_covered = covered
        
        if best_biclique is None or not best_covered:
            # Fallback: single edge
            k = next(iter(remaining))
            best_biclique = ({k[0]}, {k[1]})
            best_covered = {k}
        
        bicliques.append(best_biclique)
        remaining -= best_covered
    
    # Phase 2: Merge Bicliques
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Try merging: L = L1|L2, R = R1|R2
                L = set(L1) | set(L2)
                R = set(R1) | set(R2)
                
                if L & R:
                    continue
                
                if is_valid_biclique(L, R):
                    # Check if the union of edges is exactly the union of the two bicliques' edges
                    # This is always true if the bicliques were valid and disjoint in edges,
                    # but we must ensure no new edges are introduced that weren't in the original set
                    # Actually, if L x R is a biclique in G, and L1 x R1 and L2 x R2 are subsets,
                    # the new edges are (L1 x R2) U (L2 x R1).
                    # We just need to ensure these new edges were in the original graph (which is_valid_biclique checks)
                    # and that we are not covering edges that were not in the original problem?
                    # No, the problem is to partition the edges of G.
                    # If we merge, we are replacing two bicliques with one.
                    # The set of edges covered by the merged biclique is (L1 x R1) U (L2 x R2) U (L1 x R2) U (L2 x R1).
                    # The original two bicliques covered (L1 x R1) U (L2 x R2).
                    # The new biclique covers additional edges (L1 x R2) U (L2 x R1).
                    # These additional edges MUST have been in the original graph G.
                    # is_valid_biclique ensures all cross pairs are in G.
                    # So the merge is valid if the new biclique is a valid biclique in G.
                    
                    bicliques[i] = (L, R)
                    bicliques.pop(j)
                    changed = True
                    break
            if changed:
                break

    # Phase 3: Local Search / Vertex Moving
    # Try to move a vertex from one biclique to another to enable a merge
    if time.time() < deadline:
        for _iter in range(10):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(bicliques)):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    
                    # Try moving v from L1 to L2
                    for v in list(L1):
                        new_L1 = L1 - {v}
                        new_L2 = L2 | {v}
                        if not new_L1:
                            continue
                        # Check if new_L1, R1 is valid and new_L2, R2 is valid
                        if is_valid_biclique(new_L1, R1) and is_valid_biclique(new_L2, R2):
                            # Check if this enables a merge of the resulting bicliques
                            # Actually, we just replaced biclique i and j with potentially different ones.
                            # We want to see if we can merge new_L1, R1 with new_L2, R2? No, that would be merging i and j again.
                            # The goal is to reduce the number of bicliques.
                            # If we move v, we still have 2 bicliques.
                            # Unless... the move allows a merge with a THIRD biclique?
                            # Or unless the move makes one biclique empty (not allowed).
                            # Let's try a different local search:
                            # Can we merge biclique i with biclique j by moving vertices?
                            # Let's try to find a subset of vertices to move from L1 to L2 (or R1 to R2, etc)
                            # such that the merged biclique is valid.
                            pass
                    
                    # Alternative: Try to merge i and j by checking if there's a valid biclique
                    # that covers the union of their edges.
                    # This is essentially what Phase 2 did, but maybe we can expand the bicliques?
                    # If we merge L1, R1 and L2, R2, we get L, R.
                    # If L, R is not valid, can we shrink it?
                    # No, shrinking would mean we don't cover all edges.
                    
                    # Let's try a different move:
                    # Move v from L1 to R2?
                    # If v is in L1, it connects to all in R1.
                    # If we move v to R2, it must connect to all in L2.
                    # And L1\{v} must still connect to R1.
                    # And L2 must still connect to R2\{v} (if v was in R2, but it's not).
                    # And the new biclique (L1\{v}, R1) and (L2, R2 U {v}) must be valid.
                    
                    for v in list(L1):
                        new_L1 = L1 - {v}
                        new_R2 = R2 | {v}
                        if not new_L1:
                            continue
                        if is_valid_biclique(new_L1, R1) and is_valid_biclique(L2, new_R2):
                            # Check if we can now merge (new_L1, R1) and (L2, new_R2)?
                            # This is just swapping which biclique v belongs to.
                            # It doesn't reduce the count unless it enables a merge with a third biclique.
                            # For now, let's just accept the move if it doesn't break validity,
                            # and then re-run Phase 2 (Merging) to see if it helps.
                            bicliques[i] = (new_L1, R1)
                            bicliques[j] = (L2, new_R2)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
            
            # Re-run Phase 2 after local search moves
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for i in range(len(bicliques)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(bicliques)):
                        L1, R1 = bicliques[i]
                        L2, R2 = bicliques[j]
                        L = set(L1) | set(L2)
                        R = set(R1) | set(R2)
                        if L & R:
                            continue
                        if is_valid_biclique(L, R):
                            bicliques[i] = (L, R)
                            bicliques.pop(j)
                            changed = True
                            break
                    if changed:
                        break

    # Convert to lists of ints
    result = []
    for L, R in bicliques:
        result.append([list(int(x) for x in sorted(L)), list(int(x) for x in sorted(R))])
    
    return result
# EVOLVE-BLOCK-END
