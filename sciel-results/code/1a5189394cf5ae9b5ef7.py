"""Optimized decomposition for 3x4x4 matrix multiplication using Strassen for 2x2 blocks."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    U_rows = n * m  # 12
    V_rows = m * p  # 16
    W_rows = p * n  # 12
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def make_u(row, col):
        v = [0] * U_rows
        v[row * m + col] = 1
        return v
    
    def make_v(row, col):
        v = [0] * V_rows
        v[row * p + col] = 1
        return v
    
    def make_w(row, col):
        v = [0] * W_rows
        v[col * n + row] = 1
        return v
    
    def combine_u(terms):
        v = [0] * U_rows
        for coeff, r, c in terms:
            v[r * m + c] += coeff
        return v
    
    def combine_v(terms):
        v = [0] * V_rows
        for coeff, r, c in terms:
            v[r * p + c] += coeff
        return v
    
    def combine_w(terms):
        v = [0] * W_rows
        for coeff, r, c in terms:
            v[c * n + r] += coeff
        return v
    
    def strassen_2x2(ar, ac, br, bc):
        """Add 7 Strassen terms for A[ar:ar+2, ac:ac+2] * B[br:br+2, bc:bc+2] -> C[ar:ar+2, bc:bc+2]"""
        # M1 = (A11+A22)(B11+B22) -> C11+=1, C22+=1
        add_term(
            combine_u([(1, ar, ac), (1, ar+1, ac+1)]),
            combine_v([(1, br, bc), (1, br+1, bc+1)]),
            combine_w([(1, ar, bc), (1, ar+1, bc+1)])
        )
        
        # M2 = (A21+A22)B11 -> C21+=1
        add_term(
            combine_u([(1, ar+1, ac), (1, ar+1, ac+1)]),
            combine_v([(1, br, bc)]),
            combine_w([(1, ar+1, bc)])
        )
        
        # M3 = A11(B21-B22) -> C12+=1, C22+=1
        add_term(
            combine_u([(1, ar, ac)]),
            combine_v([(1, br+1, bc), (-1, br+1, bc+1)]),
            combine_w([(1, ar, bc+1), (1, ar+1, bc+1)])
        )
        
        # M4 = A22(B12-B11) -> C11+=1, C21+=1
        add_term(
            combine_u([(1, ar+1, ac+1)]),
            combine_v([(1, br, bc+1), (-1, br, bc)]),
            combine_w([(1, ar, bc), (1, ar+1, bc)])
        )
        
        # M5 = (A11+A12)B22 -> C11+=-1, C12+=1
        add_term(
            combine_u([(1, ar, ac), (1, ar, ac+1)]),
            combine_v([(1, br+1, bc+1)]),
            combine_w([(-1, ar, bc), (1, ar, bc+1)])
        )
        
        # M6 = (A21-A11)(B11+B12) -> C22+=1
        add_term(
            combine_u([(1, ar+1, ac), (-1, ar, ac)]),
            combine_v([(1, br, bc), (1, br, bc+1)]),
            combine_w([(1, ar+1, bc+1)])
        )
        
        # M7 = (A12-A22)(B21+B22) -> C11+=1
        add_term(
            combine_u([(1, ar, ac+1), (-1, ar+1, ac+1)]),
            combine_v([(1, br+1, bc), (1, br+1, bc+1)]),
            combine_w([(1, ar, bc)])
        )
    
    # 2x4x4 part: A rows 0,1
    strassen_2x2(ar=0, ac=0, br=0, bc=0)   # A[0:2,0:2]*B[0:2,0:2] -> C[0:2,0:2]
    strassen_2x2(ar=0, ac=2, br=2, bc=0)   # A[0:2,2:4]*B[2:4,0:2] -> C[0:2,0:2]
    strassen_2x2(ar=0, ac=0, br=0, bc=2)   # A[0:2,0:2]*B[0:2,2:4] -> C[0:2,2:4]
    strassen_2x2(ar=0, ac=2, br=2, bc=2)   # A[0:2,2:4]*B[2:4,2:4] -> C[0:2,2:4]
    
    # 1x4x4 part: A row 2, schoolbook
    for j in range(4):
        for k in range(4):
            add_term(
                make_u(2, j),
                make_v(j, k),
                make_w(2, k)
            )
    
    rank = len(U_cols)
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
