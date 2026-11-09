import itertools
import numpy as np

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

    def build_2fold_cover(p):
        """
        Constructs a 2-fold cover of the Paley graph on p vertices.
        Vertex (copy, vertex) where copy in {0,1}.
        Within the same copy: edge iff base graph has edge.
        Between different copies: edge iff base graph does NOT have edge (complement adjacency).
        This is the "anti-cover" or complemented cross-connection.
        Total vertices: 2*p
        """
        # Generate base Paley graph
        base_edges = []
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        
        base_adj = [[False] * p for _ in range(p)]
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in residues:
                    base_edges.append((i, j))
                    base_adj[i][j] = True
                    base_adj[j][i] = True
        
        n = 2 * p
        edges = []
        
        # Map (copy, vertex) to global index
        def idx(copy, v):
            return copy * p + v
        
        # Within same copy: edges where base has edge
        for i in range(p):
            for j in range(i + 1, p):
                if base_adj[i][j]:
                    edges.append((idx(0, i), idx(0, j)))
                    edges.append((idx(1, i), idx(1, j)))
        
        # Between different copies: edges where base does NOT have edge (and i != j)
        for i in range(p):
            for j in range(p):
                if i != j and not base_adj[i][j]:
                    # Only add once to avoid duplicates
                    edges.append((idx(0, i), idx(1, j)))
        
        return n, edges

    # Try the 2-fold cover of Paley(19)
    # 19 is 1 mod 4, so Paley graph exists
    n38, edges38 = build_2fold_cover(19)
    if is_5_5_free(n38, edges38):
        return n38, edges38

    # Fallback: Try Paley graph on 41 vertices
    p41 = generate_paley(41)
    if p41:
        n, edges = p41
        if is_5_5_free(n, edges):
            return n, edges

    # Fallback to Paley graph on 37 vertices (known valid)
    p37 = generate_paley(37)
    if p37:
        return p37

    # Fallback to 37 vertices if generation failed (should not happen)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
