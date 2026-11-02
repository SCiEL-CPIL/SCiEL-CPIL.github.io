import time

def construct():
    """
    Rank-44 tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Split B into four 2x2 blocks. For each block:
    - Strassen (rank 7) for A[0:2, :] slice @ B_block
    - Schoolbook (rank 4) for A[2, :] slice @ B_block
    Total: 4 * (7 + 4) = 44
    """
    _t0 = time.time()
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for b_r in range(2):
        for b_c in range(2):
            # Strassen: A[0:2, b_r*2:b_r*2+2] @ B[b_r*2:b_r*2+2, b_c*2:b_c*2+2]
            ac0 = b_r * 2
            ac1 = b_r * 2 + 1
            br0 = b_r * 2
            br1 = b_r * 2 + 1
            bc0 = b_c * 2
            bc1 = b_c * 2 + 1
            
            # m1 = (a+d)(e+h)
            U[0*m + ac0][col] = 1
            U[1*m + ac1][col] = 1
            V[br0*p + bc0][col] = 1
            V[br1*p + bc1][col] = 1
            W[bc0*n + 0][col] = 1
            W[bc1*n + 1][col] = 1
            col += 1
            
            # m2 = (c+d)e
            U[1*m + ac0][col] = 1
            U[1*m + ac1][col] = 1
            V[br0*p + bc0][col] = 1
            W[bc0*n + 1][col] = 1
            W[bc1*n + 1][col] = -1
            col += 1
            
            # m3 = a(f-h)
            U[0*m + ac0][col] = 1
            V[br0*p + bc1][col] = 1
            V[br1*p + bc1][col] = -1
            W[bc1*n + 0][col] = 1
            W[bc1*n + 1][col] = 1
            col += 1
            
            # m4 = d(g-e)
            U[1*m + ac1][col] = 1
            V[br1*p + bc0][col] = 1
            V[br0*p + bc0][col] = -1
            W[bc0*n + 0][col] = 1
            W[bc0*n + 1][col] = 1
            col += 1
            
            # m5 = (a+b)h
            U[0*m + ac0][col] = 1
            U[0*m + ac1][col] = 1
            V[br1*p + bc1][col] = 1
            W[bc0*n + 0][col] = -1
            W[bc1*n + 0][col] = 1
            col += 1
            
            # m6 = (c-a)(e+f)
            U[0*m + ac0][col] = -1
            U[1*m + ac0][col] = 1
            V[br0*p + bc0][col] = 1
            V[br0*p + bc1][col] = 1
            W[bc1*n + 1][col] = 1
            col += 1
            
            # m7 = (b-d)(g+h)
            U[0*m + ac1][col] = 1
            U[1*m + ac1][col] = -1
            V[br1*p + bc0][col] = 1
            V[br1*p + bc1][col] = 1
            W[bc0*n + 0][col] = 1
            col += 1
            
            # Schoolbook: A[2, ac0:ac1+1] @ B[br0:br1+1, bc0:bc1+1]
            for j_local in range(2):
                for k_local in range(2):
                    U[2*m + (ac0 + j_local)][col] = 1
                    V[(br0 + j_local)*p + (bc0 + k_local)][col] = 1
                    W[(bc0 + k_local)*n + 2][col] = 1
                    col += 1
    
    if time.time() - _t0 > 120:
        raise TimeoutError("Exceeded time limit")
    return {"U": U, "V": V, "W": W}
