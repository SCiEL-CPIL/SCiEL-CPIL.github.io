import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n_target = 50
    
    # Initialize with random colors
    rng = np.random.RandomState(42)
    # adj is n x n, only upper triangle matters
    adj = rng.randint(0, 4, size=(n_target, n_target))
    np.fill_diagonal(adj, 0)
    adj = (adj + adj.T) // 2  # Symmetrize, though random init doesn't need it strictly if we only use upper
    
    # Precompute indices for faster access
    # We will work with a flat array of edges for vectorization?
    # Or use the matrix directly.
    
    def count_mono_triangles(adj):
        # Vectorized count of mono triangles
        # For each color c, count triangles in the subgraph of color c
        total = 0
        for c in range(4):
            # M_c is the adjacency matrix of color c
            M = (adj == c).astype(np.int32)
            # Number of triangles is trace(M^3) / 6
            # But we can just sum M[i,j] * M[j,k] * M[k,i]
            # Efficient: trace(M @ M @ M) // 6
            # M is 50x50, so M@M is fast.
            M2 = M @ M
            M3 = M2 @ M
            total += np.trace(M3) // 6
        return int(total)

    cur_mono = count_mono_triangles(adj)
    
    # Local Search
    # We pick a random edge and change its color.
    # Delta calculation:
    # Changing edge (i,j) from c_old to c_new affects triangles containing (i,j).
    # A triangle (i,j,k) is mono if adj[i,j]==adj[i,k]==adj[j,k].
    
    # Precompute for each edge (i,j), the list of k such that (i,j,k) is a triangle candidate?
    # No, just iterate k.
    # To speed up, we can precompute for each pair (i,j) and color c, the number of k 
    # such that adj[i,k]==c and adj[j,k]==c.
    # Let deg_c[i][j] be the number of k such that adj[i,k]==c and adj[j,k]==c.
    # Then if we change (i,j) to c, the number of mono triangles of color c increases by deg_c[i][j] - (1 if adj[i,j]==c else 0)?
    # No, if adj[i,j] was c_old, and we change to c_new:
    # Triangles of color c_old: lose those where adj[i,k]==c_old and adj[j,k]==c_old.
    # Triangles of color c_new: gain those where adj[i,k]==c_new and adj[j,k]==c_new.
    
    # So we need to maintain deg_c[i][j] for all c, i, j.
    # deg_c[i][j] = sum_{k != i,j} (adj[i,k]==c) * (adj[j,k]==c)
    
    # Initialize deg
    # deg[c][i][j]
    deg = np.zeros((4, n_target, n_target), dtype=np.int32)
    for c in range(4):
        M = (adj == c).astype(np.int32)
        # deg_c[i][j] = sum_k M[i][k] * M[j][k]
        # This is M @ M.T
        deg_c = M @ M.T
        # Subtract self loops? M[i][i]=0, so no self loops.
        # But we need to exclude k=i and k=j?
        # M[i][i]=0, so M[i][i]*M[j][i]=0.
        # So deg_c[i][j] is correct for k != i,j.
        deg[c] = deg_c

    def update_deg(i, j, old_c, new_c):
        # Update deg for all k
        # For each k, the pair (i,k) and (j,k) are affected.
        # Actually, deg[c][i][j] depends on k.
        # When we change adj[i,j], it affects:
        # 1. deg[c][i][j] for all c? No, deg[c][i][j] counts triangles (i,j,k).
        #    Changing adj[i,j] does NOT change deg[c][i][j] directly, because deg[c][i][j] 
        #    is defined by adj[i,k] and adj[j,k].
        #    Wait, deg[c][i][j] is used to calculate the delta for changing (i,j).
        #    The delta for changing (i,j) to c_new is:
        #    delta = deg[c_new][i][j] - deg[c_old][i][j]
        #    (Assuming we don't count the edge itself in deg, which we don't).
        
        # However, changing adj[i,j] DOES affect deg[c][i][k] and deg[c][j][k] for all k.
        # Because deg[c][i][k] depends on adj[i,j] (when j is the 'other' vertex).
        # So we must update deg for all pairs involving i or j.
        
        # For each k != i, j:
        #   deg[old_c][i][k] changes?
        #   deg[old_c][i][k] = sum_m (adj[i,m]==old_c) * (adj[k,m]==old_c)
        #   The term m=j is (adj[i,j]==old_c) * (adj[k,j]==old_c).
        #   Before: adj[i,j]=old_c. So term was 1 * (adj[k,j]==old_c).
        #   After: adj[i,j]=new_c. So term is 0 * (adj[k,j]==old_c) = 0.
        #   So deg[old_c][i][k] decreases by (adj[k,j]==old_c).
        #   Similarly, deg[old_c][j][k] decreases by (adj[k,i]==old_c).
        
        #   For new_c:
        #   deg[new_c][i][k] increases by (adj[k,j]==new_c).
        #   deg[new_c][j][k] increases by (adj[k,i]==new_c).
        
        # We need to update deg for all k.
        # This is O(n) per edge change.
        
        for k in range(n_target):
            if k == i or k == j:
                continue
            
            # Update deg[old_c][i][k] and deg[old_c][k][i]
            if adj[k][j] == old_c:
                deg[old_c][i][k] -= 1
                deg[old_c][k][i] -= 1
            if adj[k][i] == old_c:
                deg[old_c][j][k] -= 1
                deg[old_c][k][j] -= 1
                
            # Update deg[new_c][i][k] and deg[new_c][k][i]
            if adj[k][j] == new_c:
                deg[new_c][i][k] += 1
                deg[new_c][k][i] += 1
            if adj[k][i] == new_c:
                deg[new_c][j][k] += 1
                deg[new_c][k][j] += 1

    it = 0
    max_iters = 5000000
    temp = 2.0
    cooling = 0.9999
    
    while cur_mono > 0 and it < max_iters and time.time() < deadline:
        it += 1
        temp *= cooling
        
        i = rng.randint(0, n_target - 1)
        j = rng.randint(0, n_target - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        old_c = adj[i][j]
        new_c = rng.randint(0, 3)
        if new_c == old_c:
            continue
            
        delta = deg[new_c][i][j] - deg[old_c][i][j]
        
        if delta < 0 or (delta == 0 and rng.random() < 0.1) or (delta > 0 and rng.random() < (0.5 ** (delta * 10))):
            adj[i][j] = new_c
            adj[j][i] = new_c
            cur_mono += delta
            update_deg(i, j, old_c, new_c)
            
    if cur_mono == 0:
        colors = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                colors.append(adj[i][j])
        return {'n': n_target, 'colors': colors}
    
    # Fallback: Return best valid subgraph
    # We can try to prune vertices.
    # For now, return n=32 Greenwood-Gleason if 50 fails.
    
    # Greenwood-Gleason n=32
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map, x = {}, 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    n = 32
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if i // 16 == j // 16:
                colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)

    return {'n': n, 'colors': colors}
