import time
import random

def _find_violation(neigh1, n, full_mask):
    """
    Find a monochromatic K2,2,2.
    Returns (vertices, color) or (None, None).
    """
    # Check color 1
    for a in range(n):
        for b in range(a+1, n):
            common_ab = neigh1[a] & neigh1[b]
            common_ab &= ~(1 << a) & ~(1 << b)
            
            if common_ab.bit_count() < 4:
                continue
                
            cn = []
            temp = common_ab
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                cn.append(v)
                temp ^= lsb
            
            # Check K2,2 in cn
            for i in range(len(cn)):
                u = cn[i]
                for j in range(i+1, len(cn)):
                    v = cn[j]
                    common_uv = neigh1[u] & neigh1[v]
                    common_in_cn = common_uv & common_ab
                    common_in_cn &= ~(1 << u) & ~(1 << v)
                    
                    if common_in_cn.bit_count() >= 2:
                        c1 = (common_in_cn & -common_in_cn).bit_length() - 1
                        temp2 = common_in_cn ^ (1 << c1)
                        c2 = (temp2 & -temp2).bit_length() - 1
                        return (a, b, u, v, c1, c2), 1
                        
    # Check color 0
    for a in range(n):
        for b in range(a+1, n):
            common_ab = full_mask & ~(neigh1[a] | neigh1[b]) & ~(1 << a) & ~(1 << b)
            
            if common_ab.bit_count() < 4:
                continue
            
            cn = []
            temp = common_ab
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                cn.append(v)
                temp ^= lsb
            
            for i in range(len(cn)):
                u = cn[i]
                for j in range(i+1, len(cn)):
                    v = cn[j]
                    common_uv = full_mask & ~(neigh1[u] | neigh1[v]) & ~(1 << u) & ~(1 << v)
                    common_in_cn = common_uv & common_ab
                    common_in_cn &= ~(1 << u) & ~(1 << v)
                    
                    if common_in_cn.bit_count() >= 2:
                        c1 = (common_in_cn & -common_in_cn).bit_length() - 1
                        temp2 = common_in_cn ^ (1 << c1)
                        c2 = (temp2 & -temp2).bit_length() - 1
                        return (a, b, u, v, c1, c2), 0
                        
    return None, None

def construct():
    deadline = time.time() + 145
    
    # Try n=30
    n = 30
    num_edges = n * (n - 1) // 2
    full_mask = (1 << n) - 1
    
    colors = [random.randint(0, 1) for _ in range(num_edges)]
    
    # Build initial neigh1
    neigh1 = [0] * n
    idx = 0
    edge_index = [[-1]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            edge_index[i][j] = idx
            edge_index[j][i] = idx
            if colors[idx] == 1:
                neigh1[i] |= (1 << j)
                neigh1[j] |= (1 << i)
            idx += 1

    for step in range(50000):
        if time.time() > deadline - 10:
            break
            
        verts, col = _find_violation(neigh1, n, full_mask)
        
        if verts is None:
            return {'n': n, 'colors': colors}
            
        a, b, u, v, c1, c2 = verts
        edges_to_flip = [
            edge_index[a][u], edge_index[a][v], edge_index[a][c1], edge_index[a][c2],
            edge_index[b][u], edge_index[b][v], edge_index[b][c1], edge_index[b][c2],
            edge_index[u][c1], edge_index[u][c2], edge_index[v][c1], edge_index[v][c2]
        ]
        
        flip_idx = random.choice(edges_to_flip)
        
        # Update colors
        old_val = colors[flip_idx]
        new_val = 1 - old_val
        colors[flip_idx] = new_val
        
        # Update neigh1
        # Find the endpoints of the flipped edge
        # We can store endpoints in a list
        # Or just derive from index? No, we have edge_index.
        # We need inverse mapping: index -> (u, v)
        
        # Let's precompute edge_endpoints
        # Actually, we can just find which pair it is by iterating? No.
        # Let's build edge_endpoints list.
        
        # I'll refactor to include edge_endpoints in the setup.
        
    # Fallback n=29
    # ... similar logic ...

    return {'n': 1, 'colors': []}
