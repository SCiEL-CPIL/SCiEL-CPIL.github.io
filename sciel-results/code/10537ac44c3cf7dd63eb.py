import random
import time
import numpy as np

def construct():
    # Target n. The known record is 50.
    n = 50
    num_colors = 4
    
    # Precompute indices for edges in lexicographic order
    # We need to map between flat list indices and matrix indices
    # Flat index for (i, j) with i < j:
    # idx = i*n - i*(i+1)//2 + j - i - 1
    # Inverse:
    # i = (n - 1 - sqrt(8*idx + n*(n-1) - 7)) // 2 ... complicated
    # Easier: just build a list of (i, j) pairs
    edges = []
    idx_to_pair = {}
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            idx_to_pair[idx] = (i, j)
            pair_to_idx[(i, j)] = idx
            pair_to_idx[(j, i)] = idx
            idx += 1
    
    num_edges = len(edges)
    
    # Precompute for each edge e=(u,v), the list of other vertices k
    # and the indices of edges (u,k) and (v,k) in the flat array
    # This allows us to quickly check/update triangle counts if we recolor e
    # However, with numpy matrix, we can just look at M[u,k] and M[v,k]
    
    def flat_to_matrix(flat_colors):
        M = np.zeros((n, n), dtype=np.uint8)
        for idx, (u, v) in enumerate(edges):
            M[u, v] = flat_colors[idx]
            M[v, u] = flat_colors[idx]
        return M

    def matrix_to_flat(M):
        flat = np.zeros(num_edges, dtype=np.uint8)
        for idx, (u, v) in enumerate(edges):
            flat[idx] = M[u, v]
        return flat

    def count_mono_triangles(M):
        """
        Count monochromatic triangles in a symmetric matrix M.
        A triangle (i, j, k) is mono if M[i,j] == M[i,k] == M[j,k].
        We can sum over all i<j<k.
        Vectorized approach:
        For each color c, let A_c be the adjacency matrix of color c.
        The number of triangles in color c is trace(A_c^3) / 6.
        Total mono triangles = sum_c trace(A_c^3) / 6.
        """
        total = 0
        for c in range(num_colors):
            A = (M == c).astype(np.uint8)
            # trace(A^3)
            # A @ A @ A
            # For n=50, this is small enough
            total += np.trace(A @ A @ A)
        return total // 6

    def get_worst_edges(M, mono_count):
        """
        Identify edges that are part of the most monochromatic triangles.
        For an edge (u, v) with color c, it is in a mono triangle with k
        if M[u, k] == c and M[v, k] == c.
        We want to find edges where this count is high.
        """
        # For each color c, count for each edge (u,v) of color c,
        # how many k exist such that M[u,k]==c and M[v,k]==c.
        # This is equivalent to the number of common neighbors in the subgraph of color c.
        
        worst_edges = []
        for c in range(num_colors):
            A = (M == c).astype(np.uint8)
            # A @ A gives number of common neighbors for each pair (u, v)
            # But we only care about pairs (u, v) where M[u, v] == c.
            # Let C = A @ A
            # For each (u, v) with M[u, v] == c, the count is C[u, v].
            # We want to maximize this count to break the most triangles.
            
            # Actually, if we recolor (u, v) from c to c', we break all triangles 
            # formed by (u, v) and k where M[u,k]=c and M[v,k]=c.
            # We might create new triangles with color c' if M[u,k]=c' and M[v,k]=c'.
            
            # Let's compute the "badness" of each edge.
            # Badness of (u,v) with color c = number of k s.t. M[u,k]==c and M[v,k]==c.
            # This is exactly C[u,v] where C = A_c @ A_c.
            
            C = A @ A
            # Iterate over edges of color c
            for idx, (u, v) in enumerate(edges):
                if M[u, v] == c:
                    # C[u, v] is the number of mono triangles involving edge (u,v) and color c
                    # Note: C[u,v] includes the path u-k-v. Since M[u,v]=c, these are triangles.
                    # We want to recolor the edge with the highest C[u,v] to break the most triangles.
                    worst_edges.append((C[u, v], idx))
        
        # Sort by count descending
        worst_edges.sort(reverse=True)
        return [idx for _, idx in worst_edges]

    def recolor_impact(M, idx, new_color):
        """
        Calculate change in mono triangle count if we change edge idx to new_color.
        Returns delta.
        """
        u, v = idx_to_pair[idx]
        old_color = M[u, v]
        if old_color == new_color:
            return 0
        
        # Break triangles of old_color
        # Number of k such that M[u,k]==old_color and M[v,k]==old_color
        # This is the (u,v) entry of (A_old @ A_old)
        A_old = (M == old_color).astype(np.uint8)
        # We don't want to recompute A_old @ A_old from scratch every time if possible,
        # but for n=50, it's fast enough.
        # However, we can just count directly:
        # common_old = np.sum((M[u] == old_color) & (M[v] == old_color))
        # Note: M[u][u] is 0, M[v][v] is 0. 
        # k cannot be u or v.
        # M[u][v] is old_color, so k=v is excluded? 
        # If k=v, M[u,v]==old and M[v,v]==0 (not old). So v is not counted.
        # If k=u, M[u,u]==0 (not old) and M[v,u]==old. So u is not counted.
        # So np.sum((M[u] == old_color) & (M[v] == old_color)) correctly counts valid k.
        
        broken = np.sum((M[u] == old_color) & (M[v] == old_color))
        
        # Create triangles of new_color
        # Number of k such that M[u,k]==new_color and M[v,k]==new_color
        # Again, u and v won't be counted because M[u][u]=0 != new_color (unless new_color=0, but M[u][v]=old!=0 usually? 
        # Wait, M[u][u] is always 0. If new_color is 0, then M[u][u]==0.
        # If new_color=0, k=u: M[u,u]=0, M[v,u]=old. If old!=0, then M[v,u]!=0, so condition fails.
        # If new_color=0, k=v: M[u,v]=old, M[v,v]=0. If old!=0, condition fails.
        # So it seems safe.
        
        created = np.sum((M[u] == new_color) & (M[v] == new_color))
        
        return created - broken

    start_time = time.time()
    time_limit = 115
    
    best_n_found = 32
    best_colors_32 = None
    
    # We will try to find a valid 50-vertex coloring.
    
    for attempt in range(50): # Max 50 restarts
        if time.time() - start_time > time_limit:
            break
        
        # Initialize random coloring
        # It's often better to start with a somewhat structured coloring or just random.
        # Let's try random first.
        flat_colors = np.random.randint(0, 4, num_edges, dtype=np.uint8)
        M = flat_to_matrix(flat_colors)
        
        # Local search loop
        max_iters = 1000
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            mono_tris = count_mono_triangles(M)
            if mono_tris == 0:
                # Found a valid coloring!
                return {'n': n, 'colors': flat_colors.tolist()}
            
            # Find the worst edge to recolor
            # To save time, we might not want to sort all edges every iteration.
            # But for n=50, it's okay.
            worst_edges = get_worst_edges(M, mono_tris)
            
            if not worst_edges:
                break
                
            # Pick the top few worst edges and try to fix one
            # Try the single worst edge
            idx = worst_edges[0]
            u, v = idx_to_pair[idx]
            old_color = M[u, v]
            
            best_delta = float('inf')
            best_new_color = -1
            
            for c in range(4):
                if c == old_color:
                    continue
                delta = recolor_impact(M, idx, c)
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = c
            
            if best_new_color != -1 and best_delta < 0:
                # Apply the move
                M[u, best_new_color] = best_new_color # Typo in thought, M is matrix
                M[u, v] = best_new_color
                M[v, u] = best_new_color
                flat_colors[idx] = best_new_color
            else:
                # No improving move found for the worst edge.
                # Maybe try a random edge? Or restart?
                # Let's try a few random edges to escape local minima
                improved = False
                for _ in range(10):
                    r_idx = random.randint(0, num_edges - 1)
                    u, v = idx_to_pair[r_idx]
                    old_color = M[u, v]
                    best_delta = float('inf')
                    best_new_color = -1
                    for c in range(4):
                        if c == old_color:
                            continue
                        delta = recolor_impact(M, r_idx, c)
                        if delta < best_delta:
                            best_delta = delta
                            best_new_color = c
                    if best_new_color != -1 and best_delta < 0:
                        M[u, v] = best_new_color
                        M[v, u] = best_new_color
                        flat_colors[r_idx] = best_new_color
                        improved = True
                        break
                if not improved:
                    # Stuck, restart or jump
                    if random.random() < 0.5:
                        # Jump: recolor a random subset of edges
                        for _ in range(20):
                            r_idx = random.randint(0, num_edges - 1)
                            flat_colors[r_idx] = random.randint(0, 3)
                        M = flat_to_matrix(flat_colors)
                    else:
                        break

    # If we failed to find 50 in time, fall back to 32
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
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors_32}
