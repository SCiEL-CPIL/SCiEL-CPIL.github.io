import time
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build initial adjacency masks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Helper to check if any edges remain
    def edges_remain():
        for u in range(n):
            if adj[u]:
                return True
        return False

    # Helper to find max degree vertex
    def max_degree_vertex():
        max_deg = -1
        max_v = -1
        for u in range(n):
            deg = bin(adj[u]).count('1')
            if deg > max_deg:
                max_deg = deg
                max_v = u
        return max_v

    bicliques = []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 60  # 60s limit
    
    # Phase 1: Greedy Construction
    while edges_remain():
        if time.time() > deadline:
            break
            
        u = max_degree_vertex()
        if u == -1:
            break
            
        # Greedy biclique expansion
        L = [u]
        L_mask = (1 << u)
        R_mask = adj[u] & ~L_mask
        
        C_L_mask = 0
        for w in range(n):
            if not (L_mask & (1 << w)) and not (R_mask & (1 << w)):
                if (adj[w] & R_mask) == R_mask:
                    C_L_mask |= (1 << w)
                    
        C_R_mask = 0
        for v in range(n):
            if not (L_mask & (1 << v)) and not (R_mask & (1 << v)):
                if (adj[v] & L_mask) == L_mask:
                    C_R_mask |= (1 << v)
                    
        while True:
            if time.time() > deadline:
                break
                
            best_w = -1
            best_gain_w = -1
            m = C_L_mask
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                gain = bin(R_mask).count('1')
                if gain > best_gain_w:
                    best_gain_w = gain
                    best_w = w
                m ^= lsb
                
            best_v = -1
            best_gain_v = -1
            m = C_R_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                gain = bin(L_mask).count('1')
                if gain > best_gain_v:
                    best_gain_v = gain
                    best_v = v
                m ^= lsb
                
            if best_w == -1 and best_v == -1:
                break
                
            if best_gain_w >= best_gain_v:
                L.append(best_w)
                L_mask |= (1 << best_w)
                C_R_mask &= adj[best_w]
                C_L_mask &= ~(1 << best_w)
            else:
                R_mask |= (1 << best_v)
                C_L_mask &= adj[best_v]
                C_R_mask &= ~(1 << best_v)
                
        L_list = [i for i in range(n) if (L_mask & (1 << i))]
        R_list = [i for i in range(n) if (R_mask & (1 << i))]
        
        if not L_list or not R_list:
            found = False
            for u in range(n):
                m = adj[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if u < v:
                        L_list = [u]
                        R_list = [v]
                        found = True
                        break
                    m ^= lsb
                if found: break
            
            if not found:
                break
                
            if not L_list or not R_list:
                break

        bicliques.append([L_list, R_list])
        
        for u in L_list:
            for v in R_list:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
    # Phase 2: Local Search Refinement (Merging)
    # We try to merge bicliques if their union is a valid biclique.
    # A union of two bicliques [L1, R1] and [L2, R2] is a valid biclique [L1 U L2, R1 U R2] 
    # if and only if:
    # 1. All vertices in L1 are connected to all vertices in R2
    # 2. All vertices in L2 are connected to all vertices in R1
    # (L1-R1 and L2-R2 are already valid bicliques by construction)
    
    def is_valid_merge(b1, b2):
        L1, R1 = b1
        L2, R2 = b2
        
        # Check L1 vs R2
        for u in L1:
            for v in R2:
                if not (adj_orig[u] & (1 << v)):
                    return False
        # Check L2 vs R1
        for u in L2:
            for v in R1:
                if not (adj_orig[u] & (1 << v)):
                    return False
        return True

    # We need the original adjacency to check validity of merges, 
    # because 'adj' has been modified to remove covered edges.
    # Wait, the problem states: "Every cross pair must be a host edge."
    # This refers to the ORIGINAL graph edges.
    # The partition must cover ALL original edges exactly once.
    # My Phase 1 logic removed edges from 'adj' to track coverage.
    # To check if a merge is valid, I need to know if the cross-edges EXIST in the original graph.
    # Let's rebuild original adj masks.
    
    adj_orig = [0] * n
    for u, v in edges:
        adj_orig[u] |= (1 << v)
        adj_orig[v] |= (1 << u)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        found_merge = False
        for i in range(len(bicliques)):
            if found_merge:
                break
            for j in range(i + 1, len(bicliques)):
                if is_valid_merge(bicliques[i], bicliques[j]):
                    # Merge them
                    L_new = sorted(list(set(bicliques[i][0] + bicliques[j][0])))
                    R_new = sorted(list(set(bicliques[i][1] + bicliques[j][1])))
                    
                    # Replace i and j with the new biclique
                    # Remove j first to keep indices stable
                    bicliques.pop(j)
                    bicliques[i] = [L_new, R_new]
                    improved = True
                    found_merge = True
                    break

    # Convert to required format
    final_result = []
    for bic in bicliques:
        l = [int(x) for x in bic[0]]
        r = [int(x) for x in bic[1]]
        final_result.append([l, r])
        
    return final_result
