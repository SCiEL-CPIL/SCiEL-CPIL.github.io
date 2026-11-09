import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Generate the base n=32 coloring
    # Using the GF(16) construction from the seed code logic
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    def generate_seed_coloring(n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    def colors_to_matrix(colors, n):
        mat = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                mat[i, j] = c
                mat[j, i] = c
                idx += 1
        return mat

    def matrix_to_colors(mat, n):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(int(mat[i, j]))
        return colors

    current_n = 32
    base_colors = generate_seed_coloring(current_n)
    mat = colors_to_matrix(base_colors, current_n)
    
    best_n = current_n
    best_colors = base_colors
    
    # Helper to count triangles efficiently
    # We will maintain the total triangle count and update it incrementally.
    # But for simplicity and robustness in this short window, we can use a fast O(N^3) check 
    # only when we think we are close, or use the incremental update in the search loop.
    
    def count_triangles(mat):
        n = mat.shape[0]
        total = 0
        # Use numpy for speed if possible, but loop is okay for N<50 if we don't do it too often.
        # Actually, let's use a vectorized check for the whole matrix.
        # Triangles in color c:
        # For each c in 0..3:
        #   A = (mat == c).astype(int)
        #   tri = np.trace(A @ A @ A) / 6
        # This is 4 matrix multiplications of size N x N.
        # N=50 -> 50^3 = 125k ops. Very fast.
        for c in range(4):
            A = (mat == c).astype(np.int8)
            # Count triangles
            # trace(A^3) / 6
            # A^2 = A @ A
            # A^3 = A^2 @ A
            # We only need the diagonal sum.
            # diag(A @ A @ A) = sum_k (A @ A)_{ik} A_{ki}
            # Let's just do it.
            A2 = A @ A
            A3 = A2 @ A
            total += np.trace(A3) // 6
        return total

    # Local Search
    def local_search(mat, n, max_iters=5000):
        # Precompute neighbor color counts for each vertex to speed up gain calculation?
        # For each vertex v, and each color c, count neighbors u with mat[v,u] == c.
        # Then for an edge (v, u) with color c_old, 
        # old_gain = count of w where mat[v,w]==c_old and mat[u,w]==c_old
        # This is still O(N) to compute exactly if we don't precompute intersections.
        # Given N is small, O(N) per edge eval is fine.
        
        current_tris = count_triangles(mat)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if current_tris == 0:
                return mat, True
            
            # Pick a random vertex to optimize
            v = random.randint(0, n - 1)
            
            best_gain = 0
            best_u = -1
            best_c = -1
            
            # Evaluate all edges (v, u)
            for u in range(n):
                if u == v: continue
                old_c = mat[v, u]
                # Try all 4 colors
                for new_c in range(4):
                    if new_c == old_c: continue
                    
                    # Calculate change in triangles
                    # Triangles involving (v, u) are (v, u, w)
                    # Old: w such that mat[v,w]==old_c and mat[u,w]==old_c
                    # New: w such that mat[v,w]==new_c and mat[u,w]==new_c
                    
                    # We can compute this quickly using numpy
                    # mask_v_old = (mat[v] == old_c)
                    # mask_u_old = (mat[u] == old_c)
                    # old_count = np.sum(mask_v_old & mask_u_old)
                    # But we must exclude v and u themselves?
                    # mat[v,v] is 0. mat[v,u] is old_c.
                    # If we just sum over all w, we include w=v and w=u?
                    # For w=v: mat[v,v]=0, mat[u,v]=old_c. 
                    #   If old_c == 0, then mat[v,v]==0 and mat[u,v]==0 -> counted?
                    #   Triangle (v,u,v) is not a triangle.
                    #   The condition for triangle (v,u,w) is distinct v,u,w.
                    #   So we should exclude w=v and w=u.
                    
                    # Let's create a mask of valid w
                    # valid_w = np.ones(n, dtype=bool)
                    # valid_w[v] = False
                    # valid_w[u] = False
                    
                    # This is getting slow in Python loops.
                    # Let's use a simpler heuristic: 
                    # Just flip the edge if it reduces the "local" triangle count.
                    
                    # Approximate gain:
                    # We want to minimize triangles.
                    # Let's just pick a random edge and flip it if it looks good, or use a random walk.
                    # Actually, let's stick to the O(N) check but do it in Python for now, 
                    # it's fast enough for N=50.
                    
                    old_count = 0
                    new_count = 0
                    # Use local variables for speed
                    mv = mat[v]
                    mu = mat[u]
                    for w in range(n):
                        if w == v or w == u: continue
                        if mv[w] == old_c and mu[w] == old_c:
                            old_count += 1
                        if mv[w] == new_c and mu[w] == new_c:
                            new_count += 1
                    
                    gain = old_count - new_count
                    if gain > best_gain:
                        best_gain = gain
                        best_u = u
                        best_c = new_c
            
            if best_u != -1:
                # Apply change
                old_c = mat[v, best_u]
                # Update triangle count
                # We need to re-count or update.
                # For simplicity, re-count every K iterations?
                # Or just update the matrix and re-count at the end of the loop?
                # Re-counting is O(N^3) ~ 125k ops. 5000 iters -> 600M ops. Might be too slow in Python/NumPy overhead.
                # Let's try to update incrementally.
                
                # Delta triangles = (new_count - old_count)
                # But this only counts triangles involving (v, best_u).
                # Are there other triangles affected?
                # No, only triangles containing the edge (v, best_u).
                # So current_tris += (new_count - old_count)
                
                # Wait, the gain calculated was old_count - new_count.
                # So change = -gain.
                current_tris -= best_gain
                
                mat[v, best_u] = best_c
                mat[best_u, v] = best_c
            else:
                # Random kick
                u = random.randint(0, n - 1)
                if u != v:
                    mat[v, u] = random.randint(0, 3)
                    mat[u, v] = mat[v, u]
                    # Re-count or approximate?
                    # Let's just re-count to be safe if we kick.
                    current_tris = count_triangles(mat)
            
            # Periodic re-count to correct drift?
            # The incremental update is exact for the flipped edge.
            # So it should be fine.
            
        return mat, (current_tris == 0)

    # Try to expand
    for target_n in range(33, 51):
        if time.time() > deadline:
            break
            
        # Extend matrix
        new_mat = np.zeros((target_n, target_n), dtype=np.int8)
        new_mat[:current_n, :current_n] = mat
        # Initialize new edges randomly
        for i in range(current_n, target_n):
            for j in range(i + 1, target_n):
                c = random.randint(0, 3)
                new_mat[i, j] = c
                new_mat[j, i] = c
            for j in range(i):
                c = random.randint(0, 3)
                new_mat[i, j] = c
                new_mat[j, i] = c
        
        mat = new_mat
        current_n = target_n
        
        # Run local search
        mat, success = local_search(mat, current_n, max_iters=2000)
        
        if success:
            best_n = current_n
            best_colors = matrix_to_colors(mat, current_n)
        else:
            # If we fail, we can't trust this mat for further expansion
            # We should break or try to revert?
            # Breaking is safer.
            break

    return {'n': best_n, 'colors': best_colors}
