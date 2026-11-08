"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (32, rank)
    # W index: k*n + i where k is col of B, i is row of A/output
    
    rank = 112
    U = [[0] * rank for _ in range(16)]  # n*m = 16
    V = [[0] * rank for _ in range(32)]  # m*p = 32
    W = [[0] * rank for _ in range(32)]  # p*n = 32
    
    def w_idx(col_b, row_a):
        return col_b * n + row_a
    
    col = 0
    for bi in range(2):  # block row of output (0 or 1)
        for bj in range(4):  # block col of output (0,1,2,3)
            for bk in range(2):  # which A block row multiplies which B block row
                # A_block = A[bi, bk], B_block = B[bk, bj]
                
                # Term 1: (a+d)*(e+h)
                U[2*bi+0, 2*bk+0][col] = 1  # a = A[2bi, 2bk]
                U[2*bi+1, 2*bk+1][col] = 1  # d = A[2bi+1, 2bk+1]
                V[2*bk+0, 2*bj+0][col] = 1  # e = B[2bk, 2bj]
                V[2*bk+0, 2*bj+1][col] = 1  # h = B[2bk, 2bj+1]
                W[w_idx(2*bj+0, 2*bi+0)][col] = 1  # C[2bi, 2bj]
                W[w_idx(2*bj+1, 2*bi+1)][col] = 1  # C[2bi+1, 2bj+1]
                col += 1
                
                # Term 2: (c+d)*e
                U[2*bi+1, 2*bk+0][col] = 1  # c = A[2bi+1, 2bk]
                U[2*bi+1, 2*bk+1][col] = 1  # d
                V[2*bk+0, 2*bj+0][col] = 1  # e
                W[w_idx(2*bj+0, 2*bi+1)][col] = 1  # C[2bi+1, 2bj]
                col += 1
                
                # Term 3: a*(f-h)
                U[2*bi+0, 2*bk+0][col] = 1  # a
                V[2*bk+0, 2*bj+1][col] = 1  # f = B[2bk, 2bj+1]
                V[2*bk+1, 2*bj+1][col] = -1  # -h = -B[2bk+1, 2bj+1]
                W[w_idx(2*bj+1, 2*bi+0)][col] = 1  # C[2bi, 2bj+1]
                col += 1
                
                # Term 4: d*(g-e)
                U[2*bi+1, 2*bk+1][col] = 1  # d
                V[2*bk+1, 2*bj+0][col] = 1  # g = B[2bk+1, 2bj]
                V[2*bk+0, 2*bj+0][col] = -1  # -e
                W[w_idx(2*bj+0, 2*bi+0)][col] = 1  # C[2bi, 2bj]
                W[w_idx(2*bj+0, 2*bi+1)][col] = 1  # C[2bi+1, 2bj]
                col += 1
                
                # Term 5: (a+b)*h
                U[2*bi+0, 2*bk+0][col] = 1  # a
                U[2*bi+0, 2*bk+1][col] = 1  # b = A[2bi, 2bk+1]
                V[2*bk+1, 2*bj+1][col] = 1  # h
                W[w_idx(2*bj+0, 2*bi+0)][col] = -1  # C[2bi, 2bj]
                W[w_idx(2*bj+1, 2*bi+0)][col] = 1   # C[2bi, 2bj+1]
                col += 1
                
                # Term 6: (c-a)*(e+f)
                U[2*bi+1, 2*bk+0][col] = 1  # c
                U[2*bi+0, 2*bk+0][col] = -1  # -a
                V[2*bk+0, 2*bj+0][col] = 1  # e
                V[2*bk+0, 2*bj+1][col] = 1  # f
                W[w_idx(2*bj+1, 2*bi+1)][col] = 1  # C[2bi+1, 2bj+1]
                col += 1
                
                # Term 7: (b-d)*(g+h)
                U[2*bi+0, 2*bk+1][col] = 1  # b
                U[2*bi+1, 2*bk+1][col] = -1  # -d
                V[2*bk+1, 2*bj+0][col] = 1  # g
                V[2*bk+1, 2*bj+1][col] = 1  # h
                W[w_idx(2*bj+0, 2*bi+0)][col] = 1  # C[2bi, 2bj]
                col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
