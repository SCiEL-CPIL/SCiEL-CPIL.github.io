import random
import numpy as np

def construct():
    # Bitmask representation of adjacency
    # masks[i] is an integer where bit j is 1 if edge (i, j) exists
    
    best_edges = 0
    best_masks = None
    
    # Try multiple random orders
    for _ in range(50):
        masks = [0] * 16
        edge_count = 0
        
        # List of all possible edges
        edges = [(i, j) for i in range(16) for j in range(i+1, 16)]
        random.shuffle(edges)
        
        for u, v in edges:
            if is_q3_free_add(masks, u, v):
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                edge_count += 1
                
        if edge_count > best_edges:
            best_edges = edge_count
            best_masks = masks[:]
            
    # Convert best_masks back to list of edges
    result_edges = []
    for i in range(16):
        for j in range(i+1, 16):
            if (best_masks[i] >> j) & 1:
                result_edges.append([i, j])
                
    return result_edges

def is_q3_free_add(masks, u, v):
    """
    Check if adding edge (u, v) to the graph defined by masks creates a Q3 subgraph.
    Assumes the current graph is Q3-free.
    """
    Nu = masks[u]
    Nv = masks[v]
    
    # Get list of neighbors
    Nu_list = []
    for i in range(16):
        if i != v and ((Nu >> i) & 1):
            Nu_list.append(i)
            
    Nv_list = []
    for i in range(16):
        if i != u and ((Nv >> i) & 1):
            Nv_list.append(i)
            
    # Iterate over pairs of neighbors of u
    for i in range(len(Nu_list)):
        a = Nu_list[i]
        for j in range(i+1, len(Nu_list)):
            b = Nu_list[j]
            
            # Iterate over pairs of neighbors of v
            for k in range(len(Nv_list)):
                c = Nv_list[k]
                for l in range(k+1, len(Nv_list)):
                    d = Nv_list[l]
                    
                    # Ensure a, b, c, d are distinct
                    if a == c or a == d or b == c or b == d:
                        continue
                        
                    # Check edges a-c and b-d
                    if not ((masks[a] >> c) & 1):
                        continue
                    if not ((masks[b] >> d) & 1):
                        continue
                        
                    # Forbidden vertices for x and y
                    forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    
                    # x is common neighbor of a and b
                    common_ab = masks[a] & masks[b] & ~forbidden
                    # y is common neighbor of c and d
                    common_cd = masks[c] & masks[d] & ~forbidden
                    
                    # Check if there is an edge between any x in common_ab and any y in common_cd
                    x_mask = common_ab
                    while x_mask:
                        lsb = x_mask & -x_mask
                        xi = lsb.bit_length() - 1
                        x_mask ^= lsb
                        
                        # Check if xi is connected to any y in common_cd
                        if (masks[xi] & common_cd) != 0:
                            return False
                            
    return True
