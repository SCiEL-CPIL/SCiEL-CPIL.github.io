import time
import itertools
import numpy as np

# Pre-compute pairings for K2,2,2 detection
_PAIRINGS = [
    ((0,1),(2,3),(4,5)), ((0,1),(2,4),(3,5)), ((0,1),(2,5),(3,4)),
    ((0,2),(1,3),(4,5)), ((0,2),(1,4),(3,5)), ((0,2),(1,5),(3,4)),
    ((0,3),(1,2),(4,5)), ((0,3),(1,4),(2,5)), ((0,3),(1,5),(2,4)),
    ((0,4),(1,2),(3,5)), ((0,4),(1,3),(2,5)), ((0,4),(1,5),(2,3)),
    ((0,5),(1,2),(3,4)), ((0,5),(1,3),(2,4)), ((0,5),(1,4),(2,3)),
]

def _has_k222_fast(adj_mask, n):
    """
    Check if graph with adjacency bitmasks contains K2,2,2.
    adj_mask: numpy array of uint32, length n.
    """
    # For each 6-subset, check all 15 pairings
    for verts in itertools.combinations(range(n), 6):
        v = list(verts)
        for p1, p2, p3 in _PAIRINGS:
            a, b = v[p1[0]], v[p1[1]]
            c, d = v[p2[0]], v[p2[1]]
            e, f = v[p3[0]], v[p3[1]]
            
            # Check if A-B is complete bipartite: edges (a,c), (a,d), (b,c), (b,d)
            # adj_mask[a] has bit c and d set
            # adj_mask[b] has bit c and d set
            if not ((adj_mask[a] >> c) & 1) or not ((adj_mask[a] >> d) & 1):
                continue
            if not ((adj_mask[b] >> c) & 1) or not ((adj_mask[b] >> d) & 1):
                continue
            
            # Check if A-C is complete bipartite: edges (a,e), (a,f), (b,e), (b,f)
            if not ((adj_mask[a] >> e) & 1) or not ((adj_mask[a] >> f) & 1):
                continue
            if not ((adj_mask[b] >> e) & 1) or not ((adj_mask[b] >> f) & 1):
                continue
            
            # Check if B-C is complete bipartite: edges (c,e), (c,f), (d,e), (d,f)
            if not ((adj_mask[c] >> e) & 1) or not ((adj_mask[c] >> f) & 1):
                continue
            if not ((adj_mask[d] >> e) & 1) or not ((adj_mask[d] >> f) & 1):
                continue
            
            return True
    return False

def _count_k222(adj_mask, n):
    """
    Count number of K2,2,2 subgraphs in the graph defined by adj_mask.
    Returns count.
    """
    count = 0
    for verts in itertools.combinations(range(n), 6):
        v = list(verts)
        for p1, p2, p3 in _PAIRINGS:
            a, b = v[p1[0]], v[p1[1]]
            c, d = v[p2[0]], v[p2[1]]
            e, f = v[p3[0]], v[p3[1]]
            
            # Check if A-B, A-C, B-C are complete bipartite
            if ((adj_mask[a] >> c) & 1) and ((adj_mask[a] >> d) & 1) and \
               ((adj_mask[b] >> c) & 1) and ((adj_mask[b] >> d) & 1) and \
               ((adj_mask[a] >> e) & 1) and ((adj_mask[a] >> f) & 1) and \
               ((adj_mask[b] >> e) & 1) and ((adj_mask[b] >> f) & 1) and \
               ((adj_mask[c] >> e) & 1) and ((adj_mask[c] >> f) & 1) and \
               ((adj_mask[d] >> e) & 1) and ((adj_mask[d] >> f) & 1):
                count += 1
    return count

def _colors_to_adj_mask(colors, n):
    """Convert color list to adjacency bitmask array."""
    adj_mask = np.zeros(n, dtype=np.uint32)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if colors[idx] == 1:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
            idx += 1
    return adj_mask

def _paley_colors(n, p):
    """
    Generate colors for the first n vertices of the Paley graph on p vertices.
    """
    if p < n:
        return None
    squares = set((x*x) % p for x in range(1, p))
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            diff = (j - i) % p
            colors.append(1 if diff in squares else 0)
    return colors

def _get_paley_induced_subgraph(p, remove_vertex, n):
    """
    Get colors for an induced subgraph of Paley p, removing one vertex, 
    and taking the first n vertices (or just n vertices total).
    Here we take all p-1 vertices, so n = p-1.
    """
    if p - 1 != n:
        return None
    squares = set((x*x) % p for x in range(1, p))
    # Map original vertices to new indices
    # Original vertices: 0..p-1
    # Removed: remove_vertex
    # New vertices: 0..p-2
    # If orig < remove_vertex, new = orig
    # If orig > remove_vertex, new = orig - 1
    colors = []
    for new_i in range(n):
        orig_i = new_i if new_i < remove_vertex else new_i + 1
        for new_j in range(new_i + 1, n):
            orig_j = new_j if new_j < remove_vertex else new_j + 1
            diff = (orig_j - orig_i) % p
            colors.append(1 if diff in squares else 0)
    return colors

def construct():
    deadline = time.time() + 130  # Safety margin
    
    best_n = 0
    best_colors = None
    
    # 1. Try Paley 29 (Known valid)
    c29 = _paley_colors(29, 29)
    if c29:
        adj_mask_29 = _colors_to_adj_mask(c29, 29)
        if not _has_k222_fast(adj_mask_29, 29):
            # Check complement
            adj_mask_29_comp = np.zeros(29, dtype=np.uint32)
            for i in range(29):
                # Full mask for 29 bits
                full = (1 << 29) - 1
                # Remove self and neighbors
                adj_mask_29_comp[i] = full & ~adj_mask_29[i] & ~(1 << i)
            if not _has_k222_fast(adj_mask_29_comp, 29):
                best_n = 29
                best_colors = c29
    
    # 2. Try to find n=30
    # Strategy: Check induced subgraphs of Paley 31
    # Paley 31 is a graph on 31 vertices. Removing 1 vertex gives 30 vertices.
    # There are 31 such subgraphs.
    
    p = 31
    n_target = 30
    best_30_count = float('inf')
    best_30_colors = None
    
    for remove_v in range(p):
        if time.time() > deadline - 20:
            break
            
        c30 = _get_paley_induced_subgraph(p, remove_v, n_target)
        if not c30:
            continue
            
        adj_mask = _colors_to_adj_mask(c30, n_target)
        
        # Check color 1
        if _has_k222_fast(adj_mask, n_target):
            # Count them to find the "best" invalid one
            count1 = _count_k222(adj_mask, n_target)
            
            # Check color 0 (complement)
            adj_mask_comp = np.zeros(n_target, dtype=np.uint32)
            full_mask = (1 << n_target) - 1
            for i in range(n_target):
                adj_mask_comp[i] = full_mask & ~adj_mask[i] & ~(1 << i)
            
            if _has_k222_fast(adj_mask_comp, n_target):
                count0 = _count_k222(adj_mask_comp, n_target)
                total_count = count1 + count0
                
                if total_count < best_30_count:
                    best_30_count = total_count
                    best_30_colors = c30
            else:
                # Color 0 is valid, but Color 1 is not.
                # We need BOTH to be valid.
                # So this is not a solution.
                # But we can use it for local search.
                total_count = count1 # Just track the violations
                if total_count < best_30_count:
                    best_30_count = total_count
                    best_30_colors = c30
        else:
            # Color 1 is valid. Check Color 0.
            adj_mask_comp = np.zeros(n_target, dtype=np.uint32)
            full_mask = (1 << n_target) - 1
            for i in range(n_target):
                adj_mask_comp[i] = full_mask & ~adj_mask[i] & ~(1 << i)
            
            if not _has_k222_fast(adj_mask_comp, n_target):
                # Both valid!
                return {'n': 30, 'colors': c30}
            else:
                count0 = _count_k222(adj_mask_comp, n_target)
                if count0 < best_30_count:
                    best_30_count = count0
                    best_30_colors = c30

    # 3. Local Search on best_30_colors
    if best_30_colors is not None and time.time() < deadline - 5:
        # Simple local search: flip edges to reduce count
        cur_colors = list(best_30_colors)
        cur_n = 30
        adj_mask = _colors_to_adj_mask(cur_colors, cur_n)
        
        # We need a fast way to check validity or count.
        # Given time constraints, we might just try a few random flips and check validity.
        
        for step in range(100):
            if time.time() > deadline - 2:
                break
            
            # Flip a random edge
            edge_idx = np.random.randint(len(cur_colors))
            cur_colors[edge_idx] = 1 - cur_colors[edge_idx]
            
            adj_mask = _colors_to_adj_mask(cur_colors, cur_n)
            
            if not _has_k222_fast(adj_mask, cur_n):
                adj_mask_comp = np.zeros(cur_n, dtype=np.uint32)
                full_mask = (1 << cur_n) - 1
                for i in range(cur_n):
                    adj_mask_comp[i] = full_mask & ~adj_mask[i] & ~(1 << i)
                
                if not _has_k222_fast(adj_mask_comp, cur_n):
                    return {'n': 30, 'colors': cur_colors}
            
            # Undo flip
            cur_colors[edge_idx] = 1 - cur_colors[edge_idx]

    if best_colors is None:
        return {'n': 1, 'colors': []}
        
    return {'n': best_n, 'colors': best_colors}
