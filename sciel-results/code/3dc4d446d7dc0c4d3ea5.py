import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n = 20
    # Precompute all edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Bitmasks for neighborhoods
    # mask[i] is an integer where bit j is 1 if (i,j) is an edge
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid(masks):
        # Check all triples
        for i in range(n):
            for j in range(i+1, n):
                # Optimization: if common neighbors of i,j < 3, skip?
                # No, we need common neighbors of i,j,k.
                # Intersect i and j first
                common_ij = masks[i] & masks[j]
                # If popcount(common_ij) < 3, then for any k, 
                # popcount(common_ij & masks[k]) <= popcount(common_ij) < 3.
                # So we only need to check k if popcount(common_ij) >= 3.
                if popcount(common_ij) < 3:
                    continue
                for k in range(j+1, n):
                    if popcount(common_ij & masks[k]) >= 3:
                        return False
        return True

    def find_violation(masks):
        # Returns (i, j, k, neighbor1, neighbor2, neighbor3) or None
        # We just need to find one triple with 3 common neighbors
        for i in range(n):
            for j in range(i+1, n):
                common_ij = masks[i] & masks[j]
                if popcount(common_ij) < 3:
                    continue
                # Find 3 common neighbors of i and j
                # We need a k such that k is also connected to these 3?
                # No, the condition is: exists i,j,k such that |N(i) \cap N(j) \cap N(k)| >= 3.
                # Let C = N(i) \cap N(j).
                # We need to find k in C such that |N(k) \cap C| >= 3?
                # No. N(k) must intersect C in at least 3 vertices.
                # But k itself is in C? No, k is a vertex index.
                # The common neighbors are vertices v such that v~i, v~j, v~k.
                # So we need to find i, j, k and 3 distinct vertices v1, v2, v3 such that
                # v1,v2,v3 are all connected to i, j, and k.
                
                # Let's iterate k > j
                for k in range(j+1, n):
                    common_ijk = common_ij & masks[k]
                    if popcount(common_ijk) >= 3:
                        # Found violation
                        return (i, j, k, common_ijk)
        return None

    best_edges = 0
    best_masks = [0] * n
    
    # Strategy 1: Repair from Complete
    # Strategy 2: Greedy Add from Empty/Random
    
    # We will run multiple iterations
    while time.time() < deadline:
        # Randomize starting point
        # Option A: Complete graph
        # Option B: Random graph with p=0.8
        
        if random.random() < 0.5:
            # Complete
            masks = [(1 << n) - 2] * n # All bits set except self?
            # Actually, let's just set all edges
            masks = [0] * n
            for i in range(n):
                for j in range(n):
                    if i != j:
                        masks[i] |= (1 << j)
        else:
            # Random dense
            masks = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.8:
                        masks[i] |= (1 << j)
                        masks[j] |= (1 << i)
        
        # Repair Phase
        # While invalid, remove an edge from a violation
        max_repair_steps = 500
        steps = 0
        while not is_valid(masks) and steps < max_repair_steps:
            v = find_violation(masks)
            if v is None:
                break
            i, j, k, common_mask = v
            # Get 3 vertices from common_mask
            # common_mask has bits set for neighbors
            neighbors = []
            m = common_mask
            while len(neighbors) < 3 and m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                neighbors.append(idx)
                m ^= lsb
            
            # We have K3,3 with parts {i,j,k} and neighbors
            # Pick a random edge to remove
            # Edges are (i, n1), (j, n1), (k, n1) etc.
            # Pick a random pair (u, v) from the 9 edges
            u = random.choice([i, j, k])
            v_neighbor = random.choice(neighbors)
            
            # Remove edge (u, v_neighbor)
            masks[u] &= ~(1 << v_neighbor)
            masks[v_neighbor] &= ~(1 << u)
            steps += 1
            
        if is_valid(masks):
            # Count edges
            edge_count = sum(popcount(m) for m in masks) // 2
            if edge_count > best_edges:
                best_edges = edge_count
                best_masks = masks[:]
                
            # Improvement Phase: Add edges
            # Try to add missing edges
            # Random order of missing edges
            missing = []
            for i in range(n):
                for j in range(i+1, n):
                    if not (masks[i] & (1 << j)):
                        missing.append((i, j))
                        
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline - 5:
                    break
                # Check if adding (u,v) is valid
                # Tentatively add
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                
                if is_valid(masks):
                    best_edges_local = sum(popcount(m) for m in masks) // 2
                    if best_edges_local > best_edges:
                        best_edges = best_edges_local
                        best_masks = masks[:]
                else:
                    # Revert
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)

    # Convert best_masks to edge list
    edges = []
    for i in range(n):
        m = best_masks[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            m ^= lsb
            
    return edges
