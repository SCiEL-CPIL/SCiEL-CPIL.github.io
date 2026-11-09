import numpy as np

def construct():
    n, m, p = 2, 3, 9
    total_rank = 50
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Helper to set a single rank-1 term
    def set_term(u_idx, v_idx, w_idx, col, coeff):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
    
    # Helper to set a sum of terms in one column
    def set_sum(u_indices, v_indices, w_indices, col, coeff):
        for ui in u_indices:
            U[ui][col] += coeff
        for vi in v_indices:
            V[vi][col] += coeff
        for wi in w_indices:
            W[wi][col] += coeff
    
    col = 0
    
    # Process 4 pairs of columns (0,1), (2,3), (4,5), (6,7) using Strassen
    # A left block: rows 0-1, cols 0-1 (2x2)
    # B left block: rows 0-1, cols k0-k1 (2x2)
    # C block: rows i=0-1, cols k0-k1 (2x2)
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # A indices: A[i,j] at i*m + j
        a00, a01 = 0, 1      # A[0,0], A[0,1]
        a10, a11 = 3, 4      # A[1,0], A[1,1]
        
        # B indices: B[j,k] at j*p + k
        b0k0, b0k1 = 0*p+k0, 0*p+k1   # B[0,k0], B[0,k1]
        b1k0, b1k1 = 1*p+k0, 1*p+k1   # B[1,k0], B[1,k1]
        
        # C indices: C[k,i] at k*n + i
        c0k0, c0k1 = k0*n+0, k0*n+1   # C[0,k0], C[0,k1]
        c1k0, c1k1 = k1*n+0, k1*n+1   # C[1,k0], C[1,k1]
        
        # Strassen 7 products
        # M1 = (a00+a11)(b0k0+b1k1)
        set_sum([a00, a11], [b0k0, b1k1], [c0k0, c1k1], col, 1.0)
        # M2 = (a10+a11)(b0k0)
        set_sum([a10, a11], [b0k0], [c0k0, c1k0], col+1, 1.0)
        # M3 = (a00)(b0k1-b1k1)
        set_sum([a00], [b0k1], [c0k1, c1k1], col+2, 1.0)
        U[a00][col+2] += 1.0
        V[b0k1][col+2] += 1.0
        W[c0k1][col+2] += 1.0
        V[b1k1][col+2] -= 1.0
        W[c1k1][col+2] -= 1.0
        # M4 = (a11)(b1k0-b0k0)
        set_sum([a11], [b1k0], [c0k0, c1k0], col+3, 1.0)
        U[a11][col+3] += 1.0
        V[b1k0][col+3] += 1.0
        W[c0k0][col+3] += 1.0
        W[c1k0][col+3] += 1.0
        V[b0k0][col+3] -= 1.0
        W[c0k0][col+3] -= 1.0
        W[c1k0][col+3] -= 1.0
        # M5 = (a00+a01)(b1k1)
        set_sum([a00, a01], [b1k1], [c0k0, c1k1], col+4, 1.0)
        # M6 = (a10-a00)(b0k0+b0k1)
        set_sum([a10], [b0k0, b0k1], [c1k1], col+5, 1.0)
        U[a10][col+5] += 1.0
        V[b0k0][col+5] += 1.0
        V[b0k1][col+5] += 1.0
        W[c1k1][col+5] += 1.0
        U[a00][col+5] -= 1.0
        # M7 = (a01-a11)(b1k0+b1k1)
        set_sum([a01], [b1k0, b1k1], [c0k0, c1k1], col+6, 1.0)
        U[a01][col+6] += 1.0
        V[b1k0][col+6] += 1.0
        V[b1k1][col+6] += 1.0
        W[c0k0][col+6] += 1.0
        W[c1k1][col+6] += 1.0
        U[a11][col+6] -= 1.0
        
        col += 7
    
    # Handle column k=8 for left block
    k8 = 8
    a00, a01 = 0, 1
    a10, a11 = 3, 4
    b0k8, b1k8 = 0*p+k8, 1*p+k8
    c0k8, c1k8 = k8*n+0, k8*n+1
    
    # Direct 4 products
    set_term(a00, b0k8, c0k8, col, 1.0)
    set_term(a01, b1k8, c0k8, col+1, 1.0)
    set_term(a10, b0k8, c1k8, col+2, 1.0)
    set_term(a11, b1k8, c1k8, col+3, 1.0)
    
    col += 4
    
    # Process right block: A[:, 2] (2x1) * B[2, :] (1x9)
    a02, a12 = 0*m+2, 1*m+2  # A[0,2], A[1,2]
    
    for k in range(9):
        b2k = 2*p + k         # B[2,k]
        c0k = k*n + 0
        c1k = k*n + 1
        
        set_term(a02, b2k, c0k, col, 1.0)
        set_term(a12, b2k, c1k, col+1, 1.0)
        
        col += 2
    
    return {"U": U, "V": V, "W": W}
