import time
import random
import numpy as np

def has_k33(adj):
    """
    Check if graph adj (n x n boolean matrix) contains a K3,3.
    A K3,3 exists if there are 3 vertices with >= 3 common neighbors.
    """
    n = adj.shape[0]
    # Convert to list of integer bitmasks for faster intersection
    # Or just use numpy bitwise AND
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                common = adj[i] & adj[j] & adj[k]
                if np.sum(common) >= 3:
                    return True
    return False

def construct():
    """
    Construct a K3,3-free graph on 20 vertices with many edges.
    Uses the polarity graph of PG(2,4) with one absolute point removed,
    then attempts greedy edge addition.
    """
    # --- GF(4) Arithmetic ---
    add_table = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ]
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    def gf_add(a, b):
        return add_table[a][b]
    
    def gf_mul(a, b):
        return mul_table[a][b]
        
    def gf_inv(a):
        if a == 1:
            return 1
        if a == 2:
            return 3
        if a == 3:
            return 2
        raise ZeroDivisionError

    # --- Generate PG(2,4) Points ---
    points = []
    seen = set()
    for x in range(4):
        for y in range(4):
            for z in range(4):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = gf_inv(x)
                    p = [gf_mul(x, inv_x), gf_mul(y, inv_x), gf_mul(z, inv_x)]
                elif y != 0:
                    inv_y = gf_inv(y)
                    p = [gf_mul(x, inv_y), gf_mul(y, inv_y), gf_mul(z, inv_y)]
                else:
                    inv_z = gf_inv(z)
                    p = [gf_mul(x, inv_z), gf_mul(y, inv_z), gf_mul(z, inv_z)]
                    
                key = (p[0], p[1], p[2])
                if key not in seen:
                    seen.add(key)
                    points.append(p)
                    
    n_points = len(points)
    
    # --- Build Adjacency Matrix ---
    def bilinear(u, v):
        return gf_add(gf_add(gf_mul(u[0], v[0]), gf_mul(u[1], v[1])), gf_mul(u[2], v[2]))
    
    adj_21 = np.zeros((n_points, n_points), dtype=bool)
    absolute_points = []
    
    for i in range(n_points):
        u = points[i]
        if bilinear(u, u) == 0:
            absolute_points.append(i)
            
        for j in range(i+1, n_points):
            v = points[j]
            if bilinear(u, v) == 0:
                adj_21[i, j] = True
                adj_21[j, i] = True
                
    # --- Remove One Absolute Point ---
    if not absolute_points:
        remove_idx = 0
    else:
        remove_idx = absolute_points[0]
        
    keep_indices = [i for i in range(n_points) if i != remove_idx]
    n = 20
    adj = np.zeros((n, n), dtype=bool)
    
    for new_i, old_i in enumerate(keep_indices):
        for new_j in range(new_i + 1, n):
            old_j = keep_indices[new_j]
            if adj_21[old_i, old_j]:
                adj[new_i, new_j] = True
                adj[new_j, new_i] = True
                
    # --- Greedy Edge Addition ---
    # Try to add all non-edges
    # We iterate in a specific order, maybe by potential to maintain symmetry or just random?
    # Fixed order is deterministic and safe.
    # We can try to add edges that connect high-degree vertices?
    # Let's just try all non-edges.
    
    # Pre-compute common neighbors for speed?
    # has_k33 is O(n^3 * cost_of_sum). n=20, so 8000 checks.
    # Each check is 3 bitwise ANDs and a sum. Very fast.
    
    # We might want to be smart about the order.
    # If we add an edge, it might block others.
    # A simple greedy: try all non-edges. If valid, add.
    
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i, j]:
                # Try adding
                adj[i, j] = True
                adj[j, i] = True
                if has_k33(adj):
                    # Revert
                    adj[i, j] = False
                    adj[j, i] = False
                # Else keep

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
                
    return edges
