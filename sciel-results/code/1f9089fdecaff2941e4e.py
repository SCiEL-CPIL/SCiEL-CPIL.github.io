import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # 1. Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # 2. Precompute clique and independent set status for all masks
    # is_clique[m] = True if m is a clique
    # is_indep[m] = True if m is an independent set
    total_masks = 1 << n
    is_clique = [False] * total_masks
    is_indep = [False] * total_masks
    is_valid = [False] * total_masks
    
    is_clique[0] = True
    is_indep[0] = True
    is_valid[0] = True
    
    for m in range(1, total_masks):
        # Get lowest set bit
        lsb = m & -m
        v = lsb.bit_length() - 1
        m_rest = m ^ lsb
        
        # Check clique: m_rest is clique AND v connected to all in m_rest
        if is_clique[m_rest]:
            # v must be connected to all vertices in m_rest
            # adj[v] & m_rest == m_rest
            if (adj[v] & m_rest) == m_rest:
                is_clique[m] = True
        else:
            # If m_rest is not a clique, m cannot be a clique
            pass
            
        # Check independent: m_rest is independent AND v NOT connected to any in m_rest
        if is_indep[m_rest]:
            # v must not be connected to any vertex in m_rest
            # adj[v] & m_rest == 0
            if (adj[v] & m_rest) == 0:
                is_indep[m] = True
        else:
            pass
            
        if is_clique[m] or is_indep[m]:
            is_valid[m] = True
            
    # 3. Prepare valid masks for each vertex
    # valid_masks_for_vertex[v] = list of masks containing v that are valid
    # Sorted by size (popcount) descending to prioritize larger parts
    valid_masks_for_vertex = [[] for _ in range(n)]
    
    for m in range(1, total_masks):
        if is_valid[m]:
            # Find vertices in m
            mm = m
            while mm:
                lsb = mm & -mm
                v = lsb.bit_length() - 1
                valid_masks_for_vertex[v].append(m)
                mm ^= lsb
                
    # Sort by size descending
    for v in range(n):
        valid_masks_for_vertex[v].sort(key=lambda x: bin(x).count('1'), reverse=True)
        
    # 4. Branch and Bound Search
    best_count = n # Upper bound: each vertex is its own part
    best_partition = None
    current_path = []
    
    # Optional: Initial greedy solution to set a tighter upper bound
    # Greedy: pick vertex, pick largest valid mask containing it
    rem = (1 << n) - 1
    greedy_path = []
    while rem:
        v = (rem & -rem).bit_length() - 1
        # Find largest valid mask containing v and subset of rem
        best_mask = 0
        for m in valid_masks_for_vertex[v]:
            if (m & rem) == m:
                best_mask = m
                break # Since sorted by size desc, first one is largest
        if best_mask == 0:
             # Should not happen, single vertex is always valid
             best_mask = 1 << v
        greedy_path.append(best_mask)
        rem ^= best_mask
        
    best_count = len(greedy_path)
    best_partition = greedy_path.copy()
    
    start_time = time.time()
    deadline = start_time + 50.0 # 50s margin
    
    def search(rem_mask, count):
        nonlocal best_count, best_partition, current_path
        
        if time.time() > deadline:
            return
            
        if rem_mask == 0:
            if count < best_count:
                best_count = count
                best_partition = current_path.copy()
            return
            
        # Pruning
        if count >= best_count:
            return
            
        # Pick vertex v (lowest set bit)
        v = (rem_mask & -rem_mask).bit_length() - 1
        
        # Try all valid masks containing v and subset of rem_mask
        for m in valid_masks_for_vertex[v]:
            if (m & rem_mask) != m:
                continue
                
            current_path.append(m)
            search(rem_mask ^ m, count + 1)
            current_path.pop()
            
            # If we found a solution with 1 part, it's optimal (whole graph is clique or indep)
            if best_count == 1:
                return

    target = (1 << n) - 1
    search(target, 0)
    
    # 5. Convert best_partition (list of masks) to list of lists of vertices
    final_partition = []
    for mask in best_partition:
        verts = []
        mm = mask
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            verts.append(v)
            mm ^= lsb
        final_partition.append(verts)
        
    return final_partition
# EVOLVE-BLOCK-END
