import time
import random
import numpy as np

def is_k222_free(colors, n):
    """
    Check if the coloring of K_n is K2,2,2-free in both colors.
    colors: list of 0/1 for pairs (i,j) i<j
    """
    # Build adjacency matrix for both colors
    # adj[c][i][j] = 1 if edge (i,j) is color c
    adj = [np.zeros((n, n), dtype=bool) for _ in range(2)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i][j] = True
            adj[c][j][i] = True
            idx += 1
    
    # Precompute partitions of 6 vertices into 3 pairs
    # Vertices are 0..5
    partitions = []
    def gen_partitions(remaining, current):
        if not remaining:
            partitions.append(tuple(sorted(tuple(sorted(p)) for p in current)))
            return
        first = remaining[0]
        for i in range(1, len(remaining)):
            second = remaining[i]
            new_remaining = remaining[1:i] + remaining[i+1:]
            gen_partitions(new_remaining, current + [(first, second)])
    gen_partitions(list(range(6)), [])
    
    # Check all 6-subsets
    # To speed up, we can iterate over combinations
    from itertools import combinations
    for verts in combinations(range(n), 6):
        # Map verts to 0..5
        v_map = {v: i for i, v in enumerate(verts)}
        
        for c in range(2):
            # Check if subgraph induced by verts contains K2,2,2 in color c
            # We check if there is a partition such that all edges between pairs are present
            # Note: K2,2,2 has no edges within pairs.
            # So we need:
            # 1. No edges within pairs (in color c)
            # 2. All edges between pairs (in color c)
            
            found = False
            for part in partitions:
                # part is a tuple of 3 pairs
                valid_part = True
                # Check no edges within pairs
                for p in part:
                    u, v = p
                    if adj[c][verts[u]][verts[v]]:
                        valid_part = False
                        break
                if not valid_part:
                    continue
                
                # Check all edges between pairs
                # Pairs are (0,1), (2,3), (4,5) in local indices
                # Edges between (0,1) and (2,3): (0,2), (0,3), (1,2), (1,3)
                # Edges between (0,1) and (4,5): (0,4), (0,5), (1,4), (1,5)
                # Edges between (2,3) and (4,5): (2,4), (2,5), (3,4), (3,5)
                
                pairs = part
                # pairs[0] is local indices of first pair, etc.
                # Actually, the partition generation gives local indices 0..5
                # We need to check edges between vertices in different pairs
                
                # Create a set of local indices for each pair
                # p0 = {pairs[0][0], pairs[0][1]}
                # p1 = {pairs[1][0], pairs[1][1]}
                # p2 = {pairs[2][0], pairs[2][1]}
                
                p0 = set(pairs[0])
                p1 = set(pairs[1])
                p2 = set(pairs[2])
                
                ok = True
                # Check edges between p0 and p1
                for u in p0:
                    for v in p1:
                        if not adj[c][verts[u]][verts[v]]:
                            ok = False
                            break
                    if not ok: break
                if not ok: continue
                
                # Check edges between p0 and p2
                for u in p0:
                    for v in p2:
                        if not adj[c][verts[u]][verts[v]]:
                            ok = False
                            break
                    if not ok: break
                if not ok: continue
                
                # Check edges between p1 and p2
                for u in p1:
                    for v in p2:
                        if not adj[c][verts[u]][verts[v]]:
                            ok = False
                            break
                    if not ok: break
                
                if ok:
                    found = True
                    break
            
            if found:
                return False
    
    return True

def get_paley_colors(n):
    # n must be prime
    squares = {x * x % n for x in range(1, n)}
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Try n=30 with Paley 31 subgraph
    try:
        n = 30
        # Paley 31
        colors_31 = get_paley_colors(31)
        # Remove vertex 30
        # Indices in colors_31:
        # (i, j) with i<j
        # We want to keep vertices 0..29
        # This means we exclude any pair involving 30
        colors_30 = []
        idx = 0
        for i in range(31):
            for j in range(i + 1, 31):
                if i != 30 and j != 30:
                    colors_30.append(colors_31[idx])
                idx += 1
        
        if time.time() < deadline:
            if is_k222_free(colors_30, 30):
                return {'n': 30, 'colors': colors_30}
    except Exception:
        pass

    # Try n=29 with Paley 29
    try:
        n = 29
        colors_29 = get_paley_colors(29)
        if time.time() < deadline:
            if is_k222_free(colors_29, 29):
                return {'n': 29, 'colors': colors_29}
    except Exception:
        pass

    # Fallback to n=17
    n = 17
    colors_17 = get_paley_colors(17)
    return {'n': 17, 'colors': colors_17}
