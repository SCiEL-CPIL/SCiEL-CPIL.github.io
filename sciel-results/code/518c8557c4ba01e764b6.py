import random
import time

def construct():
    deadline = time.time() + 115
    rng = random.Random(42)
    
    # Target n=35
    n = 35
    # Initialize with random graph
    # p=0.5 is balanced
    A = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                A[i] |= (1 << j)
                A[j] |= (1 << i)
                
    def find_k4(n, A):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                # Common neighbors of a and b
                common = A[a] & A[b]
                # We need an edge (c,d) in common such that c,d > b? No, just any edge.
                # To avoid duplicates, we can just check if G[common] has any edge.
                # But we need the vertices to remove an edge.
                # Let's find one.
                c_mask = common
                while c_mask:
                    low_c = c_mask & -c_mask
                    c = low_c.bit_length() - 1
                    c_mask ^= low_c
                    # Check neighbors of c in common
                    d_mask = common & A[c]
                    while d_mask:
                        low_d = d_mask & -d_mask
                        d = low_d.bit_length() - 1
                        d_mask ^= low_d
                        if d > c: # Ensure unique pair
                            return (a, b, c, d)
                # Optimization: if common is small, this is fast.
        return None

    def find_ind6(n, A):
        # Find independent set of size 6
        # Complement graph
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
            
        found = [None]
        
        def bk(R, P, X):
            nonlocal found
            if found[0] is not None:
                return
            if len(R) == 6:
                found[0] = tuple(R)
                return
            if not P:
                return
            if len(R) + bin(P).count("1") < 6:
                return
            
            # Pivot
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                # Heuristic pivot: max degree in P
                max_deg = -1
                u = -1
                p_temp = P
                while p_temp:
                    v_bit = p_temp & -p_temp
                    v = v_bit.bit_length() - 1
                    deg = bin(comp[v] & P).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                    p_temp ^= v_bit
                candidates = P
                
            c = candidates
            while c and found[0] is None:
                low = c & -c
                v = low.bit_length() - 1
                R.append(v)
                bk(R, P & comp[v], X & comp[v])
                R.pop()
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], all_mask, 0)
        return found[0]

    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges

    # Main Loop
    best_A = A
    best_n = n
    
    while time.time() < deadline:
        # Check validity
        k4 = find_k4(n, A)
        if k4:
            a, b, c, d = k4
            edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            u, v = rng.choice(edges)
            A[u] &= ~(1 << v)
            A[v] &= ~(1 << u)
            continue
            
        ind6 = find_ind6(n, A)
        if ind6:
            verts = list(ind6)
            u, v = rng.sample(verts, 2)
            A[u] |= (1 << v)
            A[v] |= (1 << u)
            continue
            
        # If we reach here, graph is valid!
        best_A = A
        best_n = n
        break
        
        # If stuck (e.g. oscillating), add noise
        # This logic is implicit in the random choice of edge to remove/add.
        # To diversify, we can flip random edges if no progress?
        # The current repair is deterministic in structure but random in choice.
        
    # If we didn't find a valid 35, try 34, 33...
    # But 35 is the target.
    
    return {'n': best_n, 'edges': to_edges(best_n, best_A)}
