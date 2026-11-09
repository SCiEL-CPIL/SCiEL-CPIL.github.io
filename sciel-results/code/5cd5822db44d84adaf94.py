"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _get_idx(i, j, n):
    """Get index in the flat colors array for edge (i,j) where i<j."""
    if i > j:
        i, j = j, i
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _validate_coloring(n, colors):
    """Validate that no color class contains a C4."""
    # For each color, check if any pair has >= 2 common neighbors
    # k_c[i][j] = number of common neighbors in color c
    # If any k_c[i][j] >= 2, then there is a C4 (actually, it means there are >= 2 common neighbors,
    # which forms at least one C4 with the edge (i,j) if it existed? No, C4 is 4 vertices.
    # If i and j have 2 common neighbors k1, k2 in color c, then i-k1-j-k2-i is a C4.
    # Yes.
    
    for c in range(5):
        # Build adjacency lists or just count common neighbors
        # For efficiency, we can use the k_c matrix approach
        k_c = [[0]*n for _ in range(n)]
        
        # This O(n^3) check is fine for n=25
        for i in range(n):
            for j in range(i+1, n):
                count = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    idx_ik = _get_idx(i, k, n)
                    idx_jk = _get_idx(j, k, n)
                    if colors[idx_ik] == c and colors[idx_jk] == c:
                        count += 1
                if count >= 2:
                    return False
    return True

def _local_search(n, deadline, seed=42):
    """
    Local search with Simulated Annealing to find a 5-coloring of K_n without C4.
    """
    rng = random.Random(seed)
    n_edges = n * (n - 1) // 2
    colors = [rng.randint(0, 4) for _ in range(n_edges)]
    
    # Precompute edge indices for faster access
    # edge_idx[i][j] = index in colors
    edge_idx = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
            
    # k[c][i][j] = number of common neighbors of i and j in color c
    k = [[[0]*n for _ in range(n)] for _ in range(5)]
    
    def _calc_k():
        for c in range(5):
            for i in range(n):
                for j in range(i+1, n):
                    count = 0
                    for w in range(n):
                        if w == i or w == j:
                            continue
                        if colors[edge_idx[i][w]] == c and colors[edge_idx[j][w]] == c:
                            count += 1
                    k[c][i][j] = count
                    k[c][j][i] = count
        _calc_c4_count()
        
    c4_count = [0] * 5
    total_c4 = 0
    
    def _calc_c4_count():
        nonlocal total_c4
        total_c4 = 0
        for c in range(5):
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    count += k[c][i][j] * (k[c][i][j] - 1) // 2
            c4_count[c] = count
            total_c4 += count
            
    _calc_k()
    
    if total_c4 == 0:
        return colors
        
    T = 1.0
    cooling = 0.999
    min_T = 1e-4
    
    # Precompute list of edges for random selection
    edges = list(range(n_edges))
    
    while T > min_T and time.time() < deadline:
        # Pick a random edge
        e = rng.choice(edges)
        i = e // (2*n - e//n - 1) # This is wrong, need proper inverse
        # Better to just iterate or store (i,j) pairs
        # Let's store edge list as tuples
        pass
        
    # Re-implement with stored edges
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):
            edge_list.append((i, j, _get_idx(i, j, n)))
            
    # Reset k and c4_count
    k = [[[0]*n for _ in range(n)] for _ in range(5)]
    c4_count = [0] * 5
    total_c4 = 0
    
    def _init_k():
        nonlocal total_c4
        total_c4 = 0
        for c in range(5):
            for i in range(n):
                for j in range(i+1, n):
                    count = 0
                    for w in range(n):
                        if w == i or w == j:
                            continue
                        if colors[edge_idx[i][w]] == c and colors[edge_idx[j][w]] == c:
                            count += 1
                    k[c][i][j] = count
                    k[c][j][i] = count
            for i in range(n):
                for j in range(i+1, n):
                    c4_count[c] += k[c][i][j] * (k[c][i][j] - 1) // 2
            total_c4 += c4_count[c]
            
    _init_k()
    
    if total_c4 == 0:
        return colors
        
    T = 1.0
    cooling = 0.999
    min_T = 1e-4
    steps = 0
    max_steps = 100000
    
    while T > min_T and steps < max_steps and time.time() < deadline:
        steps += 1
        i, j, idx_ij = rng.choice(edge_list)
        c_old = colors[idx_ij]
        c_new = rng.randint(0, 4)
        if c_new == c_old:
            continue
            
        # Calculate delta
        delta = 0
        # Update k for c_old and c_new
        # For all w != i, j:
        for w in range(n):
            if w == i or w == j:
                continue
            idx_iw = edge_idx[i][w]
            idx_jw = edge_idx[j][w]
            c_iw = colors[idx_iw]
            c_jw = colors[idx_jw]
            
            # Effect on k[c][i][w]
            if c_iw == c_old:
                # k[c_old][i][w] changes if (j,w) is c_old?
                # k_c(i,w) counts x such that (i,x) and (w,x) are c.
                # The edge (i,j) is (i,x) with x=j.
                # So if (w,j) is c_old, then j was a common neighbor of i and w in c_old.
                # Removing (i,j) from c_old removes this common neighbor.
                if c_jw == c_old:
                    old_val = k[c_old][i][w]
                    new_val = old_val - 1
                    delta += (new_val*(new_val-1)//2) - (old_val*(old_val-1)//2)
                    k[c_old][i][w] = new_val
                    k[c_old][w][i] = new_val
            if c_iw == c_new:
                if c_jw == c_new:
                    old_val = k[c_new][i][w]
                    new_val = old_val + 1
                    delta += (new_val*(new_val-1)//2) - (old_val*(old_val-1)//2)
                    k[c_new][i][w] = new_val
                    k[c_new][w][i] = new_val
            
            # Effect on k[c][j][w]
            if c_jw == c_old:
                if c_iw == c_old:
                    old_val = k[c_old][j][w]
                    new_val = old_val - 1
                    delta += (new_val*(new_val-1)//2) - (old_val*(old_val-1)//2)
                    k[c_old][j][w] = new_val
                    k[c_old][w][j] = new_val
            if c_jw == c_new:
                if c_iw == c_new:
                    old_val = k[c_new][j][w]
                    new_val = old_val + 1
                    delta += (new_val*(new_val-1)//2) - (old_val*(old_val-1)//2)
                    k[c_new][j][w] = new_val
                    k[c_new][w][j] = new_val
            
        if delta == 0 or rng.random() < math.exp(-delta / T):
            colors[idx_ij] = c_new
            # c4_count and total_c4 are updated implicitly by k, but we need to track total_c4 for acceptance
            # Actually, we just need to accept/reject. If accepted, we keep the change.
            if total_c4 + delta < 0: total_c4 = 0 # Safety
            total_c4 += delta
            if total_c4 == 0:
                return colors
        else:
            # Reject: revert k changes
            for w in range(n):
                if w == i or w == j:
                    continue
                idx_iw = edge_idx[i][w]
                idx_jw = edge_idx[j][w]
                c_iw = colors[idx_iw]
                c_jw = colors[idx_jw]
                
                if c_iw == c_old and c_jw == c_old:
                    k[c_old][i][w] += 1
                    k[c_old][w][i] += 1
                if c_iw == c_new and c_jw == c_new:
                    k[c_new][i][w] -= 1
                    k[c_new][w][i] -= 1
                if c_jw == c_old and c_iw == c_old:
                    k[c_old][j][w] += 1
                    k[c_old][w][j] += 1
                if c_jw == c_new and c_iw == c_new:
                    k[c_new][j][w] -= 1
                    k[c_new][w][j] -= 1

        T *= cooling
        
    if total_c4 == 0:
        return colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    best_n = 1
    best_colors = [0]
    
    # Try local search for n from 25 down to 15
    for n in range(25, 14, -1):
        if time.time() > deadline - 5:
            break
            
        found = None
        # Try a few seeds
        for seed in [42, 123, 777, 2024, 999]:
            if time.time() > deadline - 2:
                break
            result = _local_search(n, deadline - 1, seed)
            if result is not None:
                found = result
                break
        
        if found is not None:
            best_n = n
            best_colors = found
            break # Found a valid one, return it
            
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
