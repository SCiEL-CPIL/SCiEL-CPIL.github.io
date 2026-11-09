    def construct():
        n = 31
        # Quadratic residues mod 31
        residues = set()
        for i in range(1, n):
            residues.add((i * i) % n)
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff in residues:
                    edges.append([i, j])
        
        return {'n': n, 'edges': edges}
    