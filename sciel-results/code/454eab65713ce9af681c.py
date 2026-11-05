import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = 85.0
    
    N = 512
    limit = 512
    
    # 1. Build Compatibility Graph
    adj = [0] * (N + 1)
    degrees = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            l = (i * j) // g
            if l <= limit:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                degrees[i] += 1
                degrees[j] += 1
    
    # 2. Order vertices
    vertices = list(range(1, N + 1))
    vertices.sort(key=lambda x: degrees[x], reverse=True)
    
    rank = [0] * (N + 1)
    for i, v in enumerate(vertices):
        rank[v] = i
        
    # Precompute mask_after[v]: bitmask of u such that rank[u] > rank[v]
    mask_after = [0] * (N + 1)
    # Iterate from end of vertices list to start
    # mask_after[v] should contain all vertices that come AFTER v in the list.
    # Let's build it by accumulating a mask of "seen" vertices from the end.
    current_mask = 0
    for i in range(len(vertices) - 1, -1, -1):
        v = vertices[i]
        mask_after[v] = current_mask
        current_mask |= (1 << v)
        
    # 3. Greedy Initial Solution
    best_size = 0
    best_clique = []
    
    P = (1 << (N + 1)) - 2
    C = []
    while P:
        best_v = -1
        max_deg = -1
        for v in vertices:
            if (P >> v) & 1:
                local_deg = (adj[v] & P).bit_count()
                if local_deg > max_deg:
                    max_deg = local_deg
                    best_v = v
        if best_v == -1:
            break
        C.append(best_v)
        P &= adj[best_v]
        
    best_size = len(C)
    best_clique = C[:]
    
    # 4. Branch and Bound
    sys.setrecursionlimit(10000)
    
    def expand(C, P, depth):
        nonlocal best_size, best_clique
        
        if time.time() - start_time > deadline:
            return
            
        if P == 0:
            if len(C) > best_size:
                best_size = len(C)
                best_clique = C[:]
            return
            
        # Bound 1: Count
        if len(C) + P.bit_count() <= best_size:
            return
            
        # Get vertices in P
        verts = []
        temp = P
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            verts.append(v)
            temp ^= lsb
            
        # Sort by rank ascending (global order)
        verts.sort(key=lambda x: rank[x])
        
        # Color Bound
        # Greedy coloring
        color_masks = [0] * (len(verts) + 1)
        max_c = 0
        
        # To improve coloring, we can process verts in order of degree in P?
        # But we already sorted by global rank.
        # Let's stick to this order.
        
        for v in verts:
            assigned = False
            for c in range(1, max_c + 1):
                if (adj[v] & color_masks[c]) == 0:
                    color_masks[c] |= (1 << v)
                    assigned = True
                    break
            if not assigned:
                max_c += 1
                color_masks[max_c] |= (1 << v)
                
        if len(C) + max_c <= best_size:
            return
            
        # Iterate and branch
        # We iterate in the order of `verts` (rank ascending)
        for v in verts:
            if time.time() - start_time > deadline:
                return
                
            # Check if v is still in P
            # It is, because we haven't removed it yet from P in this loop iteration's scope
            # But we remove it at the end of the loop.
            
            # New P: compatible with v, and after v in order
            new_P = P & adj[v] & mask_after[v]
            
            expand(C + [v], new_P, depth + 1)
            
            # Remove v from P for next iteration
            P &= ~(1 << v)
            
            # Prune remaining in this level
            if len(C) + P.bit_count() <= best_size:
                break

    expand([], (1 << (N + 1)) - 2, 0)
    
    return best_clique

# Test
# print(construct())
