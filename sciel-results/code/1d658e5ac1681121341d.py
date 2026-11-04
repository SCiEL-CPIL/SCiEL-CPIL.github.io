"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    n, m, p = DIMENSIONS
    dim1 = n * m  # 8
    dim2 = m * p  # 20
    dim3 = p * n  # 10
    schoolbook_rank = n * m * p  # 40
    
    # Build the multiplication tensor T[a,b,c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T[a,b,c] = 1 if indices are consistent, else 0
    tensor = [[[Fraction(0)] * dim3 for _ in range(dim2)] for _ in range(dim1)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a][b][c] = Fraction(1)
    
    # Greedy rank-1 extraction with exact rational arithmetic
    U = []
    V = []
    W = []
    
    # Work on a copy
    current = [[[tensor[a][b][c] for c in range(dim3)] for b in range(dim2)] for a in range(dim1)]
    
    max_iter = schoolbook_rank  # Don't exceed schoolbook rank
    
    for _ in range(max_iter):
        # Find first non-zero entry
        anchor_a = anchor_b = anchor_c = None
        found = False
        for a in range(dim1):
            for b in range(dim2):
                for c in range(dim3):
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
        
        # Construct rank-1 tensor u⊗v⊗w matching as many entries as possible
        # Anchor: u[anchor_a]=1, v[anchor_b]=1, w[anchor_c]=1
        # Then:
        #   u[a] = current[a][anchor_b][anchor_c]
        #   v[b] = current[anchor_a][b][anchor_c]
        #   w[c] = current[anchor_a][anchor_b][c]
        
        u = [Fraction(0)] * dim1
        v = [Fraction(0)] * dim2
        w = [Fraction(0)] * dim3
        
        u[anchor_a] = Fraction(1)
        v[anchor_b] = Fraction(1)
        w[anchor_c] = Fraction(1)
        
        # Fill u
        for a in range(dim1):
            if a != anchor_a:
                val = current[a][anchor_b][anchor_c]
                if val != 0:
                    u[a] = val
        
        # Fill v
        for b in range(dim2):
            if b != anchor_b:
                val = current[anchor_a][b][anchor_c]
                if val != 0:
                    v[b] = val
        
        # Fill w
        for c in range(dim3):
            if c != anchor_c:
                val = current[anchor_a][anchor_b][c]
                if val != 0:
                    w[c] = val
        
        # Verify this rank-1 component doesn't introduce errors where current is 0
        valid = True
        for a in range(dim1):
            if u[a] == 0:
                continue
            for b in range(dim2):
                if v[b] == 0:
                    continue
                for c in range(dim3):
                    if w[c] == 0:
                        continue
                    if current[a][b][c] == 0 and u[a]*v[b]*w[c] != 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Simple rank-1: only the anchor entry
            u = [Fraction(0)] * dim1
            v = [Fraction(0)] * dim2
            w = [Fraction(0)] * dim3
            u[anchor_a] = Fraction(1)
            v[anchor_b] = Fraction(1)
            w[anchor_c] = Fraction(1)
        
        # Subtract the rank-1 component
        for a in range(dim1):
            if u[a] == 0:
                continue
            for b in range(dim2):
                if v[b] == 0:
                    continue
                for c in range(dim3):
                    if w[c] == 0:
                        continue
                    current[a][b][c] -= u[a] * v[b] * w[c]
        
        U.append(u)
        V.append(v)
        W.append(w)
    
    # Check if we used fewer than schoolbook rank
    if len(U) < schoolbook_rank:
        U_out = [[float(x) for x in row] for row in U]
        V_out = [[float(x) for x in row] for row in V]
        W_out = [[float(x) for x in row] for row in W]
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Fall back to schoolbook decomposition
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(dim1)]
        V = [[0] * rank for _ in range(dim2)]
        W = [[0] * rank for _ in range(dim3)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
