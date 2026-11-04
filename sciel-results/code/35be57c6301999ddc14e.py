"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p
    
    # Build the tensor as a 3D array of Fractions
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a, b, c] = 1 if a, b, c are consistent (same j, same k, same i)
    
    tensor = [[[Fraction(0)] * (p * n) for _ in range(m * p)] for _ in range(n * m)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a][b][c] = Fraction(1)
    
    # Greedy rank-1 extraction with improved anchoring strategy
    U = []  # list of vectors in R^{n*m}
    V = []  # list of vectors in R^{m*p}
    W = []  # list of vectors in R^{p*n}
    
    current = [[[tensor[i][j][k] for k in range(p*n)] for j in range(m*p)] for i in range(n*m)]
    
    def find_best_anchor(cur):
        """Find an anchor that maximizes the number of non-zero entries in the same row/column/layer."""
        best_score = -1
        best_anchor = None
        
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if cur[a][b][c] != 0:
                        # Score: count non-zeros in row a (for all b,c), column b, and layer c
                        row_count = sum(1 for b2 in range(m*p) for c2 in range(p*n) if cur[a][b2][c2] != 0)
                        col_count = sum(1 for a2 in range(n*m) for c2 in range(p*n) if cur[a2][b][c2] != 0)
                        layer_count = sum(1 for a2 in range(n*m) for b2 in range(m*p) if cur[a2][b2][c] != 0)
                        score = row_count + col_count + layer_count
                        if score > best_score:
                            best_score = score
                            best_anchor = (a, b, c)
        return best_anchor
    
    max_iter = 100
    extracted = 0
    for _ in range(max_iter):
        # Check if tensor is zero
        found = False
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if current[a][b][c] != 0:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if not found:
            break
        
        # Find best anchor
        anchor = find_best_anchor(current)
        if anchor is None:
            break
        
        anchor_a, anchor_b, anchor_c = anchor
        
        # Build rank-1 component anchored at (anchor_a, anchor_b, anchor_c)
        denom = current[anchor_a][anchor_b][anchor_c]
        if denom == 0:
            break
        
        u = [Fraction(0)] * (n * m)
        v = [Fraction(0)] * (m * p)
        w = [Fraction(0)] * (p * n)
        
        u[anchor_a] = Fraction(1)
        v[anchor_b] = Fraction(1)
        w[anchor_c] = Fraction(1)
        
        # Try to match more entries
        for a in range(n * m):
            if a != anchor_a:
                if current[a][anchor_b][anchor_c] != 0:
                    u[a] = current[a][anchor_b][anchor_c] / (v[anchor_b] * w[anchor_c])
        
        for b in range(m * p):
            if b != anchor_b:
                if current[anchor_a][b][anchor_c] != 0:
                    v[b] = current[anchor_a][b][anchor_c] / (u[anchor_a] * w[anchor_c])
        
        for c in range(p * n):
            if c != anchor_c:
                if current[anchor_a][anchor_b][c] != 0:
                    w[c] = current[anchor_a][anchor_b][c] / (u[anchor_a] * v[anchor_b])
        
        # Subtract this rank-1 component
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    current[a][b][c] -= u[a] * v[b] * w[c]
        
        U.append(u)
        V.append(v)
        W.append(w)
        extracted += 1
    
    # Check if we achieved a lower rank
    if extracted < rank:
        def convert_frac_list(lst):
            return [float(x) for x in lst]
        
        return {
            "U": [convert_frac_list(row) for row in U],
            "V": [convert_frac_list(row) for row in V],
            "W": [convert_frac_list(row) for row in W]
        }
    else:
        # Fall back to schoolbook
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
