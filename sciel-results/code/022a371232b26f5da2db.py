import time
import random

def construct():
    d = 7
    n = 2 ** d
    deadline = time.time() + 140
    
    # 1. Generate edges
    edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
                
    num_edges = len(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # 2. Precompute conflicts
    # conflicts[i] is a list of triples (tuples of 3 indices)
    # If all 3 indices in a triple are selected, edge i cannot be added.
    conflicts = [[] for _ in range(num_edges)]
    
    for i, (u, v) in enumerate(edges):
        diff = u ^ v
        bit_i = diff
        # Find index i
        # Since diff is a power of 2, bit_length-1 gives the index
        idx_i = bit_i.bit_length() - 1
        
        for j in range(d):
            if j == idx_i:
                continue
            bit_j = 1 << j
            
            # Vertices of the face
            v0 = u
            v1 = u ^ bit_i
            v2 = u ^ bit_j
            v3 = u ^ bit_i ^ bit_j
            
            # Edges of the face
            # e0: v0-v1 (This is the edge i itself)
            # e1: v0-v2
            # e2: v1-v3
            # e3: v2-v3
            
            # We need the indices of e1, e2, e3
            
            e1 = (min(v0, v2), max(v0, v2))
            e2 = (min(v1, v3), max(v1, v3))
            e3 = (min(v2, v3), max(v2, v3))
            
            idx1 = edge_to_idx[e1]
            idx2 = edge_to_idx[e2]
            idx3 = edge_to_idx[e3]
            
            conflicts[i].append((idx1, idx2, idx3))
            
    best_count = 0
    best_edges = []
    
    # 3. Randomized Greedy
    order = list(range(num_edges))
    
    # Use a local variable for selected to avoid global lookups
    # bytearray is faster for boolean checks? 
    # selected[i] != 0
    
    while time.time() < deadline:
        random.shuffle(order)
        selected = bytearray(num_edges)
        count = 0
        current_edges = []
        
        for idx in order:
            # Check conflicts
            ok = True
            for t1, t2, t3 in conflicts[idx]:
                if selected[t1] and selected[t2] and selected[t3]:
                    ok = False
                    break
            if ok:
                selected[idx] = 1
                count += 1
                current_edges.append(edges[idx])
                
        if count > best_count:
            best_count = count
            best_edges = current_edges
            
    return [[u, v] for u, v in best_edges]
