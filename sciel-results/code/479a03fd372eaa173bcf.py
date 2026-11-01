"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 2x2 times 2x7: A is 2x2, B is 2x7, C = A*B is 2x7
    # Use schoolbook rank 28 as safe fallback, but try Strassen-based rank 25
    
    # Let's build the Strassen-based decomposition and verify it
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for 2x2x2 block
    def add_strassen_block(b_start):
        nonlocal col
        # A: 2x2, indices 0,1,2,3 (row-major)
        # B block: columns b_start, b_start+1
        # C block: columns b_start, b_start+1
        a11, a12, a21, a22 = 0, 1, 2, 3
        b11 = b_start, b21 = p + b_start
        b12 = b_start + 1, b22 = p + b_start + 1
        c11 = b_start, c12 = b_start + 1
        c21 = n + b_start, c22 = n + b_start + 1
        
        # M1 = (a11+a22)(b11+b22)
        U[a11][col] += 1; U[a22][col] += 1
        V[b11][col] += 1; V[b22][col] += 1
        W[c11][col] += 1; W[c22][col] += 1
        col += 1
        
        # M2 = (a21+a22)(b11)
        U[a21][col] += 1; U[a22][col] += 1
        V[b11][col] += 1
        W[c21][col] += 1
        col += 1
        
        # M3 = a11(b21-b22)
        U[a11][col] += 1
        V[b21][col] += 1; V[b22][col] -= 1
        W[c12][col] += 1
        col += 1
        
        # M4 = a22(b12-b11)
        U[a22][col] += 1
        V[b12][col] += 1; V[b11][col] -= 1
        W[c11][col] -= 1; W[c21][col] += 1
        col += 1
        
        # M5 = (a11+a12)(b22)
        U[a11][col] += 1; U[a12][col] += 1
        V[b22][col] += 1
        W[c11][col] += 1; W[c12][col] += 1
        col += 1
        
        # M6 = (a21+a22)(b12-b11)
        U[a21][col] += 1; U[a22][col] += 1
        V[b12][col] += 1; V[b11][col] -= 1
        W[c22][col] += 1
        col += 1
        
        # M7 = (a11+a22)(b11-b21)
        U[a11][col] += 1; U[a22][col] += 1
        V[b11][col] += 1; V[b21][col] -= 1
        W[c11][col] += 1
        col += 1
    
    # Schoolbook for 2x2x1
    def add_schoolbook_block(b_start):
        nonlocal col
        a11, a12, a21, a22 = 0, 1, 2, 3
        b1 = b_start, b2 = p + b_start
        c1 = b_start, c2 = n + b_start
        
        U[a11][col] += 1; V[b1][col] += 1; W[c1][col] += 1; col += 1
        U[a12][col] += 1; V[b2][col] += 1; W[c1][col] += 1; col += 1
        U[a21][col] += 1; V[b1][col] += 1; W[c2][col] += 1; col += 1
        U[a22][col] += 1; V[b2][col] += 1; W[c2][col] += 1; col += 1
    
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    add_schoolbook_block(6)
    
    # Verify
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Tensor T[i,j,k] = delta[i*m+j, j*p+k] * ... 
    # Actually the multiplication tensor: T[a,b,c] where a indexes A, b indexes B, c indexes C
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    
    # Check: sum_r U[a,r] * V[b,r] * W[c,r] should equal 1 if (a,b,c) is a valid triple, 0 otherwise
    # a = i*m+j, b = j*p+k, c = k*n+i
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U_np[a, r] * V_np[b, r] * W_np[c, r] for r in range(rank))
                if abs(val - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook
        rank = n * m * p
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
