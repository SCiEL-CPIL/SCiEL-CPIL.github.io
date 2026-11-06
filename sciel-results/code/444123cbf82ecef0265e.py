import time

def construct():
    start_time = time.time()
    
    # The optimal known solution is n=29 using the Paley graph.
    n = 29
    
    # Generate the Paley graph adjacency for q=29
    # Quadratic residues mod 29
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
                
    # Optional: Verify validity if time permits, though it's a known result.
    # For safety and speed, we can skip heavy verification if we trust the construction.
    # However, a quick check is good practice.
    
    # Build adjacency bitmasks for verification
    adj = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            idx += 1
            
    # Check for K_{2,2,2} in color 0 (adj)
    # K_{2,2,2} exists if there are u,v adjacent such that N(u)∩N(v) contains a K_{2,2}
    # This is equivalent to checking if the induced subgraph on common neighbors has a K_{2,2}
    # A K_{2,2} exists if there are two vertices in the common neighborhood with >=2 common neighbors in that neighborhood.
    
    def has_k222(adj_list, n):
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj_list[a] >> b) & 1:
                    continue
                cn_mask = adj_list[a] & adj_list[b]
                cn_mask &= ~(1 << a)
                cn_mask &= ~(1 << b)
                if cn_mask.bit_count() < 4:
                    continue
                # Check if induced subgraph on cn_mask has K_{2,2}
                # This happens if there exist u, v in cn_mask such that |N(u) ∩ N(v) ∩ cn_mask| >= 2
                cn_list = []
                m = cn_mask
                while m:
                    lsb = m & (-m)
                    idx_v = lsb.bit_length() - 1
                    cn_list.append(idx_v)
                    m ^= lsb
                for i in range(len(cn_list)):
                    u = cn_list[i]
                    adj_u = adj_list[u]
                    for j in range(i + 1, len(cn_list)):
                        v = cn_list[j]
                        common = adj_u & adj_list[v] & cn_mask
                        if common.bit_count() >= 2:
                            return True
        return False

    if has_k222(adj, n):
        # Fallback to smaller n if 29 fails (should not happen)
        n = 17
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
        return {'n': n, 'colors': colors}
        
    # Check complement (color 1)
    comp_adj = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = full ^ adj[i] ^ (1 << i)
        
    if has_k222(comp_adj, n):
        # Fallback to smaller n if complement fails (should not happen)
        n = 17
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
        return {'n': n, 'colors': colors}

    return {'n': n, 'colors': colors}
