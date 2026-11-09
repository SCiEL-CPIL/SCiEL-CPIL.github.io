"""Local search biclique partition with restarts."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 55.0
    
    # Build adjacency bitsets
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    def popcount(x):
        return bin(x).count('1')
    
    def iter_bits(x):
        while x:
            lsb = x & (-x)
            yield lsb.bit_length() - 1
            x ^= lsb
    
    def greedy_partition(rem_edges, adj, n, deadline, randomize=False):
        """Greedy maximal biclique partition."""
        if not rem_edges:
            return []
        
        # Build rem_adj
        rem_adj = [0] * n
        for (a, b) in rem_edges:
            rem_adj[a] |= (1 << b)
            rem_adj[b] |= (1 << a)
        
        bicliques = []
        
        while rem_edges:
            if time.time() > deadline:
                break
            
            # Find best seed edge
            best_L = None
            best_R = None
            best_count = -1
            
            # Get list of remaining edges for sampling
            edge_list = list(rem_edges)
            
            # If randomize, shuffle and sample
            if randomize and len(edge_list) > 50:
                random.shuffle(edge_list)
                edge_list = edge_list[:min(len(edge_list), 100)]
            
            for (a, b) in edge_list:
                if time.time() > deadline:
                    break
                
                # Try both orientations
                for (sl, sr) in ((a, b), (b, a)):
                    # Grow maximal biclique
                    Lmask = 1 << sl
                    Rmask = rem_adj[sl]
                    if not (Rmask & (1 << sr)):
                        continue
                    Rmask |= (1 << sr)
                    
                    changed = True
                    while changed:
                        changed = False
                        # Common neighborhood of L in rem graph
                        common = (1 << n) - 1
                        temp = Lmask
                        while temp:
                            lsb = temp & (-temp)
                            l = lsb.bit_length() - 1
                            common &= rem_adj[l]
                            temp ^= lsb
                        new_R = common
                        if new_R != Rmask:
                            Rmask = new_R
                            changed = True
                        
                        if Rmask:
                            commonL = (1 << n) - 1
                            temp = Rmask
                            while temp:
                                lsb = temp & (-temp)
                                r = lsb.bit_length() - 1
                                commonL &= rem_adj[r]
                                temp ^= lsb
                            new_L = commonL
                            if new_L != Lmask:
                                Lmask = new_L
                                changed = True
                        else:
                            break
                    
                    L = list(iter_bits(Lmask))
                    R = list(iter_bits(Rmask))
                    
                    if not L or not R:
                        continue
                    
                    # Count edges covered
                    cnt = 0
                    temp = Lmask
                    while temp:
                        lsb = temp & (-temp)
                        l = lsb.bit_length() - 1
                        common_with_rem = Rmask & rem_adj[l]
                        lt_mask = (1 << l) - 1
                        cnt += popcount(common_with_rem & lt_mask)
                        temp ^= lsb
                    
                    if cnt > best_count:
                        best_count = cnt
                        best_L = L
                        best_R = R
                        best_Lmask = Lmask
                        best_Rmask = Rmask
                
                if best_count >= len(rem_edges):
                    break
            
            if best_L is None or best_count <= 0:
                (a, b) = next(iter(rem_edges))
                best_L = [a]
                best_R = [b]
                best_Lmask = (1 << a)
                best_Rmask = (1 << b)
            
            # Remove covered edges
            temp_L = best_Lmask
            while temp_L:
                lsb_L = temp_L & (-temp_L)
                l = lsb_L.bit_length() - 1
                temp_R = best_Rmask
                while temp_R:
                    lsb_R = temp_R & (-temp_R)
                    r = lsb_R.bit_length() - 1
                    if l != r:
                        key = (min(l, r), max(l, r))
                        if key in rem_edges:
                            rem_edges.discard(key)
                            rem_adj[l] &= ~(1 << r)
                            rem_adj[r] &= ~(1 << l)
                    temp_R ^= lsb_R
                temp_L ^= lsb_L
            
            bicliques.append([sorted(best_L), sorted(best_R)])
        
        # Fallback for remaining edges
        if rem_edges:
            for (a, b) in list(rem_edges):
                bicliques.append([[a], [b]])
        
        return bicliques
    
    def local_search(bicliques, adj, n, edge_set, deadline):
        """Local search to improve biclique partition."""
        bicliques = [b[:] for b in bicliques]
        
        def get_biclique_edges(bic):
            """Get set of edges in a biclique."""
            L, R = bic
            edges_in = set()
            for l in L:
                for r in R:
                    if l != r:
                        key = (min(l, r), max(l, r))
                        if key in edge_set:
                            edges_in.add(key)
            return edges_in
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n_bic = len(bicliques)
            
            for i in range(n_bic):
                if time.time() > deadline:
                    break
                if i >= len(bicliques):
                    continue
                
                L_i, R_i = bicliques[i]
                edges_i = get_biclique_edges(bicliques[i])
                
                # Try removing a vertex from L_i and reassigning
                for l in L_i[:]:
                    if time.time() > deadline:
                        break
                    
                    # Find all edges incident to l in biclique i
                    l_edges = [(l, r) for r in R_i if l != r and (min(l, r), max(l, r)) in edge_set]
                    
                    if not l_edges:
                        continue
                    
                    # Try to reassign these edges to other bicliques
                    # Create new biclique i without l
                    new_L_i = [x for x in L_i if x != l]
                    if not new_L_i:
                        # Biclique i would be empty, so we can remove it
                        # Try to add l's edges to other bicliques
                        success = True
                        for (a, b) in l_edges:
                            # Find a biclique that can take this edge
                            placed = False
                            for j in range(len(bicliques)):
                                if j == i:
                                    continue
                                L_j, R_j = bicliques[j]
                                # Can we add (a,b) to biclique j?
                                # Need: a in L_j and b in R_j, or a in R_j and b in L_j
                                # And all cross edges must exist
                                if a in L_j and b in R_j:
                                    placed = True
                                    break
                                elif b in L_j and a in R_j:
                                    placed = True
                                    break
                                # Try adding to L_j or R_j
                                # Case 1: add a to L_j, b to R_j
                                if a not in L_j and b not in R_j:
                                    # Check if a can be added to L_j
                                    can_add_a_to_L = True
                                    for r in R_j:
                                        if a != r and (min(a, r), max(a, r)) not in edge_set:
                                            can_add_a_to_L = False
                                            break
                                    if can_add_a_to_L:
                                        # Check if b can be added to R_j
                                        can_add_b_to_R = True
                                        for ll in L_j:
                                            if b != ll and (min(b, ll), max(b, ll)) not in edge_set:
                                                can_add_b_to_R = False
                                                break
                                        if can_add_b_to_R:
                                            placed = True
                                            break
                                # Case 2: add b to L_j, a to R_j
                                if b not in L_j and a not in R_j:
                                    can_add_b_to_L = True
                                    for r in R_j:
                                        if b != r and (min(b, r), max(b, r)) not in edge_set:
                                            can_add_b_to_L = False
                                            break
                                    if can_add_b_to_L:
                                        can_add_a_to_R = True
                                        for ll in L_j:
                                            if a != ll and (min(a, ll), max(a, ll)) not in edge_set:
                                                can_add_a_to_R = False
                                                break
                                        if can_add_a_to_R:
                                            placed = True
                                            break
                            if not placed:
                                success = False
                                break
                        
                        if success:
                            # Remove biclique i
                            bicliques.pop(i)
                            improved = True
                            break
                    
                    # Try moving l to another biclique's L side
                    for j in range(len(bicliques)):
                        if j == i:
                            continue
                        L_j, R_j = bicliques[j]
                        # Check if l can be added to L_j
                        can_add = True
                        for r in R_j:
                            if l != r and (min(l, r), max(l, r)) not in edge_set:
                                can_add = False
                                break
                        if can_add and l not in L_j:
                            # Check if all edges from l to R_j are in edge_set
                            # (already checked above)
                            # Remove l from L_i, add to L_j
                            L_i.remove(l)
                            L_j.append(l)
                            # If L_i is now empty, remove biclique i
                            if not L_i:
                                bicliques.pop(i)
                            improved = True
                            break
                    
                    if improved:
                        break
                
                # Try similar moves for R_i
                if not improved:
                    R_i, L_i = bicliques[i][1], bicliques[i][0]
                    for r in R_i[:]:
                        if time.time() > deadline:
                            break
                        
                        r_edges = [(l, r) for l in L_i if l != r and (min(l, r), max(l, r)) in edge_set]
                        
                        if not r_edges:
                            continue
                        
                        new_R_i = [x for x in R_i if x != r]
                        if not new_R_i:
                            success = True
                            for (a, b) in r_edges:
                                placed = False
                                for j in range(len(bicliques)):
                                    if j == i:
                                        continue
                                    L_j, R_j = bicliques[j]
                                    if a in R_j and b in L_j:
                                        placed = True
                                        break
                                    elif b in R_j and a in L_j:
                                        placed = True
                                        break
                                    if a not in R_j and b not in L_j:
                                        can_add_a_to_R = True
                                        for ll in L_j:
                                            if a != ll and (min(a, ll), max(a, ll)) not in edge_set:
                                                can_add_a_to_R = False
                                                break
                                        if can_add_a_to_R:
                                            can_add_b_to_L = True
                                            for rr in R_j:
                                                if b != rr and (min(b, rr), max(b, rr)) not in edge_set:
                                                    can_add_b_to_L = False
                                                    break
                                            if can_add_b_to_L:
                                                placed = True
                                                break
                                    if b not in R_j and a not in L_j:
                                        can_add_b_to_R = True
                                        for ll in L_j:
                                            if b != ll and (min(b, ll), max(b, ll)) not in edge_set:
                                                can_add_b_to_R = False
                                                break
                                        if can_add_b_to_R:
                                            can_add_a_to_L = True
                                            for rr in R_j:
                                                if a != rr and (min(a, rr), max(a, rr)) not in edge_set:
                                                    can_add_a_to_L = False
                                                    break
                                            if can_add_a_to_L:
                                                placed = True
                                                break
                                if not placed:
                                    success = False
                                    break
                            
                            if success:
                                bicliques.pop(i)
                                improved = True
                                break
                        
                        if not improved:
                            for j in range(len(bicliques)):
                                if j == i:
                                    continue
                                L_j, R_j = bicliques[j]
                                can_add = True
                                for l in L_j:
                                    if r != l and (min(r, l), max(r, l)) not in edge_set:
                                        can_add = False
                                        break
                                if can_add and r not in R_j:
                                    R_i.remove(r)
                                    R_j.append(r)
                                    if not R_i:
                                        bicliques.pop(i)
                                    improved = True
                                    break
                        
                        if improved:
                            break
        
        # Merge pass
        merged = True
        while merged and time.time() < deadline:
            merged = False
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    break
                if i >= len(bicliques):
                    continue
                for j in range(i + 1, len(bicliques)):
                    if time.time() > deadline:
                        break
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    set_L1, set_R1 = set(L1), set(R1)
                    set_L2, set_R2 = set(L2), set(R2)
                    
                    if set_L1 & set_L2 or set_R1 & set_R2:
                        continue
                    
                    # Check all cross edges
                    all_exist = True
                    for l in L1:
                        for r in R2:
                            if l != r and (min(l, r), max(l, r)) not in edge_set:
                                all_exist = False
                                break
                        if not all_exist:
                            break
                    if all_exist:
                        for l in L2:
                            for r in R1:
                                if l != r and (min(l, r), max(l, r)) not in edge_set:
                                    all_exist = False
                                    break
                            if not all_exist:
                                break
                    
                    if all_exist:
                        new_L = sorted(set_L1 | set_L2)
                        new_R = sorted(set_R1 | set_R2)
                        bicliques[i] = [new_L, new_R]
                        bicliques.pop(j)
                        merged = True
                        break
                if merged:
                    break
        
        return bicliques
    
    # Run multiple restarts
    best_result = None
    best_count = float('inf')
    
    # First, deterministic greedy
    rem_edges_copy = set(edge_set)
    bicliques = greedy_partition(rem_edges_copy, adj, n, deadline, randomize=False)
    bicliques = local_search(bicliques, adj, n, edge_set, deadline)
    best_result = bicliques
    best_count = len(bicliques)
    
    # Then, randomized restarts
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        rem_edges_copy = set(edge_set)
        bicliques = greedy_partition(rem_edges_copy, adj, n, deadline, randomize=True)
        bicliques = local_search(bicliques, adj, n, edge_set, deadline)
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_result = bicliques
        
        if best_count <= 5:  # Good enough
            break
    
    # Ensure valid types
    out = []
    for L, R in best_result:
        out.append([[int(x) for x in L], [int(y) for y in R]])
    
    return out
# EVOLVE-BLOCK-END
