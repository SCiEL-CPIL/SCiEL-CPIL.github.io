import time

def construct():
    """
    Uses algebraic construction (Paley graph) for base coloring, 
    then attempts to extend to larger n using greedy/local search 
    initialized from the algebraic structure.
    """
    start_time = time.time()
    time_limit = 110

    def is_valid(n, colors):
        """
        Checks validity using bitsets.
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
        
        def has_mono_k222(adj):
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        continue
                    common = adj[u] & adj[v]
                    if common & (common - 1) == 0:
                        continue
                    # Extract vertices
                    common_bits = common
                    verts = []
                    while common_bits:
                        lsb = common_bits & (-common_bits)
                        idx = (lsb.bit_length() - 1)
                        verts.append(idx)
                        common_bits ^= lsb
                    for i_x in range(len(verts)):
                        x = verts[i_x]
                        for i_y in range(i_x + 1, len(verts)):
                            y = verts[i_y]
                            mask_excl = ~((1 << u) | (1 << v) | (1 << x) | (1 << y))
                            cand_a = adj[v] & adj[x] & adj[y] & mask_excl
                            if cand_a == 0: continue
                            cand_b = adj[u] & adj[x] & adj[y] & mask_excl
                            if cand_b == 0: continue
                            return True
            return False

        if has_mono_k222(adj0):
            return False
        if has_mono_k222(adj1):
            return False
        return True

    def find_violation_edge(n, colors):
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
        
        def find_in_adj(adj):
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        continue
                    common = adj[u] & adj[v]
                    if common & (common - 1) == 0:
                        continue
                    common_bits = common
                    verts = []
                    while common_bits:
                        lsb = common_bits & (-common_bits)
                        idx = (lsb.bit_length() - 1)
                        verts.append(idx)
                        common_bits ^= lsb
                    for i_x in range(len(verts)):
                        x = verts[i_x]
                        for i_y in range(i_x + 1, len(verts)):
                            y = verts[i_y]
                            mask_excl = ~((1 << u) | (1 << v) | (1 << x) | (1 << y))
                            cand_a = adj[v] & adj[x] & adj[y] & mask_excl
                            if cand_a == 0: continue
                            cand_b = adj[u] & adj[x] & adj[y] & mask_excl
                            if cand_b == 0: continue
                            return (u, v)
            return None
        
        res0 = find_in_adj(adj0)
        if res0:
            return res0
        res1 = find_in_adj(adj1)
        if res1:
            return res1
        return None

    def edge_index(n, u, v):
        if u > v:
            u, v = v, u
        return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)

    # 1. Algebraic Construction: Paley Graph for primes p = 1 mod 4
    def generate_paley(n):
        # n must be prime and n % 4 == 1
        # Check if n is prime
        if n < 2:
            return None
        is_prime = True
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                is_prime = False
                break
        if not is_prime:
            return None
        if n % 4 != 1:
            return None
            
        # Quadratic residues mod n
        residues = set()
        for x in range(1, n):
            residues.add((x * x) % n)
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff in residues:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    # Try to find a valid algebraic coloring for n=29
    # If valid, try to extend to 30
    best_n = 17
    best_colors = None
    
    # Fallback for small n if algebraic fails
    def get_fallback(n):
        # Simple random or known small
        import random
        random.seed(42)
        for _ in range(1000):
            colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            if is_valid(n, colors):
                return colors
        return None

    # Target n=29 with Paley
    colors_29 = generate_paley(29)
    if colors_29 and is_valid(29, colors_29):
        best_n = 29
        best_colors = colors_29
        
        # Attempt to extend to n=30
        # Add vertex 29. 
        # Initial guess for new edges: random or based on some heuristic?
        # Let's try a few random assignments for the new edges and then local search.
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
            # Start with valid 29 coloring
            colors_30 = colors_29[:]
            # Append 29 new edges (29, 0) to (29, 28)
            # Note: lexicographic order: (0,1)..(0,28), (1,2)..(1,28), ... (28,29)
            # The new edges are (28, 29)? No.
            # Edges are (i, j) with i < j.
            # Existing: all pairs in 0..28.
            # New pairs: (0, 29), (1, 29), ..., (28, 29).
            # These are appended at the end of the list?
            # Lex order: 
            # ... (27, 28), (28, 29)? No.
            # The last edge in K_29 is (28, 29)? No, (28, 28+1) is not in K_29.
            # K_29 edges end with (27, 28).
            # K_30 edges:
            # First 29*28/2 edges are same as K_29.
            # Then edges involving 29: (0, 29), (1, 29), ..., (28, 29).
            # So we append 29 bits.
            
            import random
            new_edges = [random.randint(0, 1) for _ in range(29)]
            colors_30.extend(new_edges)
            
            # Local search on K_30
            valid = False
            for step in range(500):
                if time.time() - start_time > time_limit:
                    break
                viol = find_violation_edge(30, colors_30)
                if viol is None:
                    valid = True
                    break
                u, v = viol
                idx = edge_index(30, u, v)
                colors_30[idx] ^= 1
            
            if valid:
                return {'n': 30, 'colors': colors_30}
        
        # If extension failed, return 29
        return {'n': 29, 'colors': colors_29}
    else:
        # If Paley 29 is invalid, try other strategies
        # Try n=29 with random search (fallback to old method but limited)
        # Or try n=27?
        # For now, fallback to a known valid small n or random search for 29
        import random
        for n in [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]:
            if time.time() - start_time > time_limit:
                break
            for attempt in range(50):
                if time.time() - start_time > time_limit:
                    break
                colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
                # Quick local search
                for step in range(200):
                    if time.time() - start_time > time_limit:
                        break
                    viol = find_violation_edge(n, colors)
                    if viol is None:
                        return {'n': n, 'colors': colors}
                    u, v = viol
                    idx = edge_index(n, u, v)
                    colors[idx] ^= 1
        # Absolute fallback
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}
