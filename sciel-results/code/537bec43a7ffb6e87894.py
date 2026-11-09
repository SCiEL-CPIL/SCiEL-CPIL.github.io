import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Construct base K32 coloring
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
        
    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            base_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
            
    current_n = 32
    current_colors = base_colors
    
    # Helper to get color of edge (i, j) from flat list
    def get_color(colors, n, i, j):
        if i > j: i, j = j, i
        # index = i*n - i*(i+1)/2 + (j - i - 1)
        # More efficient: precompute offsets
        return colors[i * n - i*(i+1)//2 + j - i - 1]
        
    # Precompute offsets for speed
    def get_offsets(n):
        offsets = [0] * n
        acc = 0
        for i in range(n):
            offsets[i] = acc
            acc += n - i - 1
        return offsets
        
    def try_extend(colors, n):
        """
        Try to extend K_n to K_{n+1}.
        Returns new colors list or None.
        """
        # Variables: c_0 ... c_{n-1}
        # Constraints: For each k, {i : c_i=k} is independent in G_k.
        # G_k is defined by edges in colors with value k.
        
        # Build adjacency lists for each color k
        # adj[k][i] is a list of j such that color(i,j)=k
        adj = [[[] for _ in range(n)] for _ in range(4)]
        offsets = get_offsets(n)
        
        for i in range(n):
            for j in range(i+1, n):
                c = colors[offsets[i] + j - i - 1]
                adj[c][i].append(j)
                adj[c][j].append(i)
                
        # Backtracking search for c_0 ... c_{n-1}
        c = [0] * n
        forbidden = [0] * n # bitmask of forbidden colors
        
        def backtrack(idx):
            if idx == n:
                return True
            # Variable ordering: pick the variable with fewest available colors
            # For simplicity and speed, we can just go 0..n-1, but MRV is better.
            # Let's try simple order first, with propagation.
            # Actually, MRV is crucial for CSP.
            
            # Find variable with minimum remaining values
            min_var = -1
            min_count = 5
            for v in range(idx, n):
                if c[v] != -1: continue
                avail = 15 & ~forbidden[v]
                count = bin(avail).count('1')
                if count < min_count:
                    min_count = count
                    min_var = v
                    if count == 0:
                        return False
            if min_var == -1:
                return True
            
            v = min_var
            # Try colors
            avail = 15 & ~forbidden[v]
            # Heuristic: try colors that cause least propagation? Or just random/0-3
            for k in range(4):
                if avail & (1 << k):
                    c[v] = k
                    # Propagate
                    # For all j in adj[k][v], forbid color k for c_j
                    new_forbidden = []
                    valid = True
                    for j in adj[k][v]:
                        if c[j] == k:
                            valid = False
                            break
                        if c[j] == -1:
                            if forbidden[j] & (1 << k):
                                # Already forbidden, no change
                                pass
                            else:
                                new_forbidden.append((j, forbidden[j] | (1 << k)))
                    if not valid:
                        c[v] = -1
                        continue
                    
                    # Apply changes
                    old_forbidden = [0]*n
                    for j, f in new_forbidden:
                        old_forbidden[j] = forbidden[j]
                        forbidden[j] = f
                        if forbidden[j] == 15:
                            # No colors left for j
                            valid = False
                            break
                            
                    if valid:
                        if backtrack(idx + 1):
                            return True
                            
                    # Undo
                    for j, f in new_forbidden:
                        forbidden[j] = old_forbidden[j]
                        
                    c[v] = -1
            return False
            
        # Initialize c with -1
        c = [-1] * n
        if backtrack(0):
            # Construct new color list
            new_colors = colors[:]
            # Add colors for edges (n, i)
            # The new vertex is index n.
            # Edges are (0, n), (1, n), ..., (n-1, n).
            # In lexicographic order for K_{n+1}, these come at the end.
            # Wait, the order is (i,j) for i<j.
            # The new edges are (0,n), (1,n), ..., (n-1,n).
            # They are appended to the list.
            for i in range(n):
                new_colors.append(c[i])
            return new_colors
        return None

    # Iterative extension
    # To handle failures, we can try multiple random permutations of the base graph.
    # Or just try to extend, and if it fails, we stop.
    # But we can also try to "shake" the solution if we get stuck.
    
    # For now, let's try to extend sequentially.
    # If it fails, we might try a different starting point or accept n.
    
    attempts = 0
    max_attempts = 50 # Limit attempts to avoid infinite loops if stuck
    
    while current_n < 51 and time.time() < deadline:
        if time.time() > deadline:
            break
            
        # Try to extend
        new_colors = try_extend(current_colors, current_n)
        if new_colors:
            current_colors = new_colors
            current_n += 1
            # Reset attempts on success
            attempts = 0
        else:
            # Failed to extend
            # We could try to restart with a random permutation of vertices
            # This requires re-indexing the colors, which is expensive.
            # Instead, we might just stop if we fail from the "canonical" order.
            # Or we can try to modify the last few steps.
            # For simplicity and safety, we will stop if we fail.
            # However, the record is 50. If we are at 40 and fail, we lose potential points.
            # A better strategy: if we fail, try a random restart of the CSP solver 
            # (randomize variable order or value order).
            
            # Let's try a few random restarts of the solver for the same n.
            # We need to modify try_extend to accept a random seed or order.
            # For now, let's just break. The base 32 is safe. 
            # If we want to be bolder, we can implement randomization.
            break

    return {'n': current_n, 'colors': current_colors}
