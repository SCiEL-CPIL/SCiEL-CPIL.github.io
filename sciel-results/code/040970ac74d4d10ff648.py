import itertools

def construct():
    def generate_paley(p):
        """Generates the Paley graph on p vertices (p must be 1 mod 4)."""
        if p % 4 != 1:
            return None
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        edges = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in residues:
                    edges.append((i, j))
        return p, edges

    def is_5_5_free(n, edges):
        """Checks if the graph has no K5 and no independent set of size 5."""
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Check for K5
        for combo in itertools.combinations(range(n), 5):
            is_clique = True
            for i in range(5):
                for j in range(i + 1, 5):
                    if not adj[combo[i]][combo[j]]:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                return False
        
        # Check for Independent Set of size 5 (Clique in complement)
        for combo in itertools.combinations(range(n), 5):
            is_independent = True
            for i in range(5):
                for j in range(i + 1, 5):
                    if adj[combo[i]][combo[j]]:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                return False
        
        return True

    # 1. Try the 38-vertex construction (2-fold cover of QR graph on 19)
    p_base = 19
    qr = set()
    for x in range(1, p_base):
        qr.add((x * x) % p_base)
    
    n_cover = 2 * p_base
    edges_cover = []
    
    def adj_base(u, v):
        diff = (v - u) % p_base
        return diff in qr
    
    for i in range(p_base):
        for j in range(i + 1, p_base):
            if adj_base(i, j):
                # Internal edges in copy 0
                edges_cover.append((i, j))
                # Internal edges in copy 1
                edges_cover.append((i + p_base, j + p_base))
                # Cross edges: i (copy 0) to j (copy 1)
                edges_cover.append((i, j + p_base))
                # Cross edges: i (copy 1) to j (copy 0)
                edges_cover.append((i + p_base, j))
    
    if is_5_5_free(n_cover, edges_cover):
        return n_cover, edges_cover

    # 2. Fallback to Paley graph on 37 vertices (known valid)
    p37 = generate_paley(37)
    if p37:
        return p37

    # 3. Final fallback to 37 vertices if generation failed
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
