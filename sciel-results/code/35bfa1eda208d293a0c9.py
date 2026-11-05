import time
import random
import sys

def construct():
    """
    Finds a valid 2-coloring of K_n with no monochromatic K_{2,2,2}.
    Targets n=29.
    Uses incremental bitset updates and Simulated Annealing/Perturbation.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave buffer
    
    def edge_index(n, u, v):
        if u > v:
            u, v = v, u
        return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)
    
    def init_data(n, colors):
        """
        Initializes adjacency and common neighbor bitsets.
        colors: list of 0/1 of length n*(n-1)/2
        Returns: adj0, adj1, common0, common1
        """
        adj0 = [0] * n
        adj1 = [0] * n
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[k]
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                k += 1
        
        # Precompute common neighbors
        # common_c[u][v] is bitmask of common neighbors of u and v in color c
        # We only need it for u < v to save memory/time, but symmetric access is fine if we are careful.
        # Let's use a 2D list n x n for simplicity and speed of access.
        common0 = [[0] * n for _ in range(n)]
        common1 = [[0] * n for _ in range(n)]
        
        for u in range(n):
            for v in range(u + 1, n):
                c0 = adj0[u] & adj0[v]
                c1 = adj1[u] & adj1[v]
                common0[u][v] = c0
                common0[v][u] = c0
                common1[u][v] = c1
                common1[v][u] = c1
                
        return adj0, adj1, common0, common1

    def update_flip(n, u, v, color, adj0, adj1, common0, common1):
        """
        Flips edge (u, v) in the graph of given 'color' (0 or 1).
        Wait, flipping edge (u,v) means changing its color from 'color' to 1-'color'.
        So we remove it from adj[color] and add to adj[1-color].
        And update common neighbors accordingly.
        """
        # 1. Update adjacency
        bit_u = 1 << u
        bit_v = 1 << v
        
        if color == 0:
            adj0[u] &= ~bit_v
            adj0[v] &= ~bit_u
            adj1[u] |= bit_v
            adj1[v] |= bit_u
            # Update common0 for pairs involving u or v
            for w in range(n):
                if w == u or w == v:
                    continue
                # common0[u][w] changes
                common0[u][w] = adj0[u] & adj0[w]
                common0[w][u] = common0[u][w]
                common0[v][w] = adj0[v] & adj0[w]
                common0[w][v] = common0[v][w]
            common0[u][v] = adj0[u] & adj0[v]
            common0[v][u] = common0[u][v]
            
            # Update common1 for pairs involving u or v
            for w in range(n):
                if w == u or w == v:
                    continue
                common1[u][w] = adj1[u] & adj1[w]
                common1[w][u] = common1[u][w]
                common1[v][w] = adj1[v] & adj1[w]
                common1[w][v] = common1[v][w]
            common1[u][v] = adj1[u] & adj1[v]
            common1[v][u] = common1[u][v]
            
        else:
            adj1[u] &= ~bit_v
            adj1[v] &= ~bit_u
            adj0[u] |= bit_v
            adj0[v] |= bit_u
            
            for w in range(n):
                if w == u or w == v:
                    continue
                common0[u][w] = adj0[u] & adj0[w]
                common0[w][u] = common0[u][w]
                common0[v][w] = adj0[v] & adj0[w]
                common0[w][v] = common0[v][w]
            common0[u][v] = adj0[u] & adj0[v]
            common0[v][u] = common0[u][v]
            
            for w in range(n):
                if w == u or w == v:
                    continue
                common1[u][w] = adj1[u] & adj1[w]
                common1[w][u] = common1[u][w]
                common1[v][w] = adj1[v] & adj1[w]
                common1[w][v] = common1[v][w]
            common1[u][v] = adj1[u] & adj1[v]
            common1[v][u] = common1[u][v]

    def find_violation(n, adj0, adj1, common0, common1):
        """
        Finds a monochromatic K_{2,2,2}.
        Returns (color, u, v) where (u,v) is an edge in the violation, or None.
        """
        def check_color(adj, common):
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        continue
                    
                    common_uv = common[u][v]
                    # Check if common_uv has at least 2 bits
                    if (common_uv & (common_uv - 1)) == 0:
                        continue
                    
                    # Extract vertices
                    bits = common_uv
                    verts = []
                    while bits:
                        lsb = bits & (-bits)
                        idx = lsb.bit_length() - 1
                        verts.append(idx)
                        bits ^= lsb
                    
                    # If fewer than 2, skip (already checked, but safe)
                    if len(verts) < 2:
                        continue
                        
                    # Iterate pairs x, y
                    # Optimization: if verts is large, this is slow.
                    # But in a near-valid coloring, violations are rare, so we might find one quickly
                    # or the common neighbors are small.
                    
                    # We need x, y distinct from u, v.
                    # adj[u] and adj[v] contain each other, so u in adj[v], v in adj[u].
                    # common_uv = adj[u] & adj[v].
                    # u is not in adj[u] (no self loop). So u is not in common_uv.
                    # v is not in adj[v]. So v is not in common_uv.
                    # So x, y are automatically distinct from u, v.
                    
                    for i in range(len(verts)):
                        x = verts[i]
                        for j in range(i + 1, len(verts)):
                            y = verts[j]
                            
                            # Candidates for a': neighbors of v, x, y in this color
                            # Must be distinct from u, v, x, y
                            mask_excl = ~((1 << u) | (1 << v) | (1 << x) | (1 << y))
                            
                            cand_a = adj[v] & adj[x] & adj[y] & mask_excl
                            if cand_a == 0:
                                continue
                            
                            # Candidates for b': neighbors of u, x, y in this color
                            cand_b = adj[u] & adj[x] & adj[y] & mask_excl
                            if cand_b == 0:
                                continue
                            
                            return (u, v)
            return None
            
        res0 = check_color(adj0, common0)
        if res0:
            return (0, res0[0], res0[1])
        
        res1 = check_color(adj1, common1)
        if res1:
            return (1, res1[0], res1[1])
            
        return None

    # Target n=29
    for n in [29, 28, 27]:
        # Random restarts
        max_restarts = 200 if n == 29 else 50
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize
            colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            adj0, adj1, common0, common1 = init_data(n, colors)
            
            # Simulated Annealing / Perturbation Loop
            # We don't have an energy function, so we use "walk" with perturbations.
            stuck_count = 0
            last_violation = None
            
            for step in range(5000):
                if time.time() - start_time > time_limit:
                    break
                
                viol = find_violation(n, adj0, adj1, common0, common1)
                if viol is None:
                    return {'n': n, 'colors': colors}
                
                color, u, v = viol
                
                # If we are stuck on the same violation, perturb
                if viol == last_violation:
                    stuck_count += 1
                else:
                    stuck_count = 0
                    last_violation = viol
                
                # Strategy:
                # 1. Flip the violating edge.
                # 2. If stuck_count > threshold, do a random perturbation.
                
                if stuck_count > 20:
                    # Perturbation: flip 5 random edges
                    for _ in range(5):
                        idx = random.randint(0, len(colors) - 1)
                        # Decode idx to (i, j)
                        # Inverse of edge_index
                        # u*(n-1) - u*(u-1)/2 + (v-u-1) = idx
                        # Approx u ~ sqrt(2*idx)
                        u_guess = int((2 * idx) ** 0.5)
                        # Refine
                        while u_guess < n and edge_index(n, u_guess, u_guess + 1) > idx:
                            u_guess -= 1
                        while u_guess < n - 1 and edge_index(n, u_guess + 1, u_guess + 2) <= idx:
                            u_guess += 1
                        
                        # Calculate v
                        # idx - (u*(n-1) - u*(u-1)//2) = v - u - 1
                        base = u_guess * (n - 1) - u_guess * (u_guess - 1) // 2
                        v_guess = u_guess + 1 + (idx - base)
                        
                        if u_guess < v_guess and v_guess < n:
                            c = colors[idx]
                            colors[idx] = 1 - c
                            update_flip(n, u_guess, v_guess, c, adj0, adj1, common0, common1)
                    
                    stuck_count = 0
                    last_violation = None
                else:
                    # Normal flip
                    c = colors[edge_index(n, u, v)]
                    colors[edge_index(n, u, v)] = 1 - c
                    update_flip(n, u, v, c, adj0, adj1, common0, common1)

    # Fallback: return a known valid small graph (n=17)
    # Using the quadratic residues construction for n=17
    n = 17
    squares = {x * x % n for x in range(1, n)}
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                colors.append(0)
            else:
                colors.append(1)
    return {'n': n, 'colors': colors}
