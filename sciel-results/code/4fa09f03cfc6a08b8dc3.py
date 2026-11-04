"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p
    
    # Try to find a lower rank using greedy rank-1 extraction
    # The tensor T has entries T[i*m+j, k*n+l, j*p+k] = 1 for all valid i,j,k,l
    # Actually, T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a, b, c] = 1 if a, b, c are consistent (same j, same k, same i)
    
    # Build the tensor as a 3D array of Fractions
    tensor = [[[Fraction(0)] * (p * n) for _ in range(m * p)] for _ in range(n * m)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a][b][c] = Fraction(1)
    
    # Greedy rank-1 extraction
    # We'll try to subtract rank-1 components
    U = []  # list of vectors in R^{n*m}
    V = []  # list of vectors in R^{m*p}
    W = []  # list of vectors in R^{p*n}
    
    current = [[[tensor[i][j][k] for k in range(p*n)] for j in range(m*p)] for i in range(n*m)]
    
    # Try to extract rank-1 components greedily
    max_iter = 100
    extracted = 0
    for _ in range(max_iter):
        # Find a non-zero entry to anchor the rank-1 component
        anchor_a = anchor_b = anchor_c = None
        found = False
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if current[a][b][c] != 0:
                        anchor_a, anchor_b, anchor_c = a, b, c
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if not found:
            break
        
        # Build rank-1 component anchored at (anchor_a, anchor_b, anchor_c)
        # u[anchor_a] = 1, v[anchor_b] = 1, w[anchor_c] = 1
        # Then for each (a,b,c), the contribution is u[a]*v[b]*w[c]
        # We want to choose u, v, w to maximize the number of non-zero entries matched
        
        # Simple greedy: set u[a] = current[a][anchor_b][anchor_c] / current[anchor_a][anchor_b][anchor_c]
        # v[b] = current[anchor_a][b][anchor_c] / current[anchor_a][anchor_b][anchor_c]
        # w[c] = current[anchor_a][anchor_b][c] / current[anchor_a][anchor_b][anchor_c]
        
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
                # u[a] such that u[a] * v[anchor_b] * w[anchor_c] = current[a][anchor_b][anchor_c]
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
        # Convert to lists of Fractions (or integers if possible)
        U_out = [[float(x) if x.denominator == 1 else x for x in row] for row in U]
        V_out = [[float(x) if x.denominator == 1 else x for x in row] for row in V]
        W_out = [[float(x) if x.denominator == 1 else x for x in row] for row in W]
        
        # Actually, return as lists of numbers
        def convert_frac_list(lst):
            return [float(x) if x.denominator == 1 else float(x) for x in lst]
        
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
