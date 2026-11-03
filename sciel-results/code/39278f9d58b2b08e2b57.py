import time
import sys

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition block by block.
    # Partition n=3 into 2+1
    # Partition m=5 into 2+2+1
    # Partition p=6 into 2+2+2
    
    # We accumulate U, V, W rows/columns.
    # U is (n*m, rank)
    # V is (m*p, rank)
    # W is (p*n, rank)
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to add a term
    # u_vec: length n*m
    # v_vec: length m*p
    # w_vec: length p*n
    def add_term(u, v, w):
        U_rows.append(u)
        V_rows.append(v)
        W_rows.append(w)

    # Strassen 2x2x2 decomposition
    # C = AB where A, B are 2x2
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    #
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Indices for 2x2 block in A (rows r1, r2; cols c1, c2)
    # Indices for 2x2 block in B (rows c1, c2; cols d1, d2)
    # Indices for 2x2 block in C (rows r1, r2; cols d1, d2)
    
    def strassen_2x2(r1, r2, c1, c2, d1, d2):
        # A indices: (r1, c1), (r1, c2), (r2, c1), (r2, c2)
        # B indices: (c1, d1), (c1, d2), (c2, d1), (c2, d2)
        # C indices: (r1, d1), (r1, d2), (r2, d1), (r2, d2)
        
        # We need to create U, V, W vectors for each of the 7 products.
        # Product m1: (A11+A22)(B11+B22)
        # U: A11=1, A22=1
        # V: B11=1, B22=1
        # W: C11+=1, C22+=1
        
        def make_u(r1, c1, r2, c2, a11, a12, a21, a22):
            u = [0.0] * (n * m)
            if a11 != 0: u[r1 * m + c1] = a11
            if a12 != 0: u[r1 * m + c2] = a12
            if a21 != 0: u[r2 * m + c1] = a21
            if a22 != 0: u[r2 * m + c2] = a22
            return u

        def make_v(c1, d1, c2, d2, b11, b12, b21, b22):
            v = [0.0] * (m * p)
            if b11 != 0: v[c1 * p + d1] = b11
            if b12 != 0: v[c1 * p + d2] = b12
            if b21 != 0: v[c2 * p + d1] = b21
            if b22 != 0: v[c2 * p + d2] = b22
            return v

        def make_w(r1, d1, r2, d2, c11, c12, c21, c22):
            w = [0.0] * (p * n)
            # W index is k*n + i
            # C11 is (r1, d1) -> k=d1, i=r1
            # C12 is (r1, d2) -> k=d2, i=r1
            # C21 is (r2, d1) -> k=d1, i=r2
            # C22 is (r2, d2) -> k=d2, i=r2
            if c11 != 0: w[d1 * n + r1] = c11
            if c12 != 0: w[d2 * n + r1] = c12
            if c21 != 0: w[d1 * n + r2] = c21
            if c22 != 0: w[d2 * n + r2] = c22
            return w

        # m1
        u1 = make_u(r1, c1, r2, c2, 1, 0, 0, 1)
        v1 = make_v(c1, d1, c2, d2, 1, 0, 0, 1)
        w1 = make_w(r1, d1, r2, d2, 1, 0, 0, 1)
        add_term(u1, v1, w1)

        # m2
        u2 = make_u(r1, c1, r2, c2, 0, 0, 1, 1)
        v2 = make_v(c1, d1, c2, d2, 1, 0, 0, 0)
        w2 = make_w(r1, d1, r2, d2, 0, 0, 1, 0)
        add_term(u2, v2, w2)

        # m3
        u3 = make_u(r1, c1, r2, c2, 1, 0, 0, 0)
        v3 = make_v(c1, d1, c2, d2, 0, 1, 0, -1)
        w3 = make_w(r1, d1, r2, d2, 0, 1, 0, 0)
        add_term(u3, v3, w3)

        # m4
        u4 = make_u(r1, c1, r2, c2, 0, 0, 0, 1)
        v4 = make_v(c1, d1, c2, d2, -1, 0, 1, 0)
        w4 = make_w(r1, d1, r2, d2, 1, 0, 1, 0)
        add_term(u4, v4, w4)

        # m5
        u5 = make_u(r1, c1, r2, c2, 1, 1, 0, 0)
        v5 = make_v(c1, d1, c2, d2, 0, 0, 0, 1)
        w5 = make_w(r1, d1, r2, d2, -1, 1, 0, 0)
        add_term(u5, v5, w5)

        # m6
        u6 = make_u(r1, c1, r2, c2, -1, 0, 1, 0)
        v6 = make_v(c1, d1, c2, d2, 1, 1, 0, 0)
        w6 = make_w(r1, d1, r2, d2, 0, 0, 0, 1)
        add_term(u6, v6, w6)

        # m7
        u7 = make_u(r1, c1, r2, c2, 0, 1, 0, -1)
        v7 = make_v(c1, d1, c2, d2, 0, 0, 1, 1)
        w7 = make_w(r1, d1, r2, d2, 1, 0, 0, 0)
        add_term(u7, v7, w7)

    # Helper for 1x2x2 (A is 1x2, B is 2x2, C is 1x2)
    # C1 = A1 B1 + A2 B2
    # Rank 2
    def small_1x2x2(r, c1, c2, d1, d2):
        # Term 1: A1 B1
        u1 = [0.0] * (n * m)
        u1[r * m + c1] = 1
        v1 = [0.0] * (m * p)
        v1[c1 * p + d1] = 1
        v1[c1 * p + d2] = 0 # Wait, B1 is 2x2? No, B is 2x2.
        # B1 is top-left of B?
        # A is [A1, A2]. B is [[B11, B12], [B21, B22]].
        # C1 = A1 B11 + A2 B21? No.
        # C is 1x2.
        # C1 = A1 B11 + A2 B21
        # C2 = A1 B12 + A2 B22
        
        # Term 1: A1 B11 -> C1
        u1 = [0.0] * (n * m); u1[r * m + c1] = 1
        v1 = [0.0] * (m * p); v1[c1 * p + d1] = 1
        w1 = [0.0] * (p * n); w1[d1 * n + r] = 1
        add_term(u1, v1, w1)
        
        # Term 2: A2 B21 -> C1
        u2 = [0.0] * (n * m); u2[r * m + c2] = 1
        v2 = [0.0] * (m * p); v2[c2 * p + d1] = 1
        w2 = [0.0] * (p * n); w2[d1 * n + r] = 1
        add_term(u2, v2, w2)
        
        # Term 3: A1 B12 -> C2
        u3 = [0.0] * (n * m); u3[r * m + c1] = 1
        v3 = [0.0] * (m * p); v3[c1 * p + d2] = 1
        w3 = [0.0] * (p * n); w3[d2 * n + r] = 1
        add_term(u3, v3, w3)
        
        # Term 4: A2 B22 -> C2
        u4 = [0.0] * (n * m); u4[r * m + c2] = 1
        v4 = [0.0] * (m * p); v4[c2 * p + d2] = 1
        w4 = [0.0] * (p * n); w4[d2 * n + r] = 1
        add_term(u4, v4, w4)
        
    # This 1x2x2 is rank 4. Can we do rank 2?
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    # This is just schoolbook for 1x2x2. Rank 4.
    # Min rank of 1x2x2 is 2?
    # T_{i,j,k}. i=1.
    # T_{1,1,1}=1, T_{1,2,1}=1, T_{1,1,2}=1, T_{1,2,2}=1.
    # T = e1 \otimes (e1+e2) \otimes (e1+e2)?
    # (e1 \otimes (e1+e2) \otimes (e1+e2))_{1,j,k} = 1*1*1 = 1.
    # Yes, rank 1?
    # Wait, A is 1x2, B is 2x2.
    # A = [a1, a2]. B = [[b11, b12], [b21, b22]].
    # C1 = a1 b11 + a2 b21.
    # C2 = a1 b12 + a2 b22.
    # Tensor T has T_{1,1,1}=1 (a1 b11 -> c1), T_{1,2,1}=1 (a2 b21 -> c1), T_{1,1,2}=1 (a1 b12 -> c2), T_{1,2,2}=1 (a2 b22 -> c2).
    # T = e1 \otimes e1 \otimes e1 + e1 \otimes e2 \otimes e1 + e1 \otimes e1 \otimes e2 + e1 \otimes e2 \otimes e2.
    # T = e1 \otimes (e1 \otimes (e1+e2) + e2 \otimes (e1+e2)) = e1 \otimes (e1+e2) \otimes (e1+e2).
    # Rank 1.
    
    def small_1x2x2_rank1(r, c1, c2, d1, d2):
        u = [0.0] * (n * m)
        u[r * m + c1] = 1
        u[r * m + c2] = 1
        v = [0.0] * (m * p)
        v[c1 * p + d1] = 1
        v[c1 * p + d2] = 1
        v[c2 * p + d1] = 1
        v[c2 * p + d2] = 1
        w = [0.0] * (p * n)
        w[d1 * n + r] = 1
        w[d2 * n + r] = 1
        add_term(u, v, w)

    # Similarly for 2x1x2 (A is 2x1, B is 1x2, C is 2x2)
    # Rank 1?
    # C11 = A1 B1, C12 = A1 B2, C21 = A2 B1, C22 = A2 B2.
    # T = (e1+e2) \otimes e1 \otimes (e1+e2). Rank 1.
    
    def small_2x1x2_rank1(r1, r2, c, d1, d2):
        u = [0.0] * (n * m)
        u[r1 * m + c] = 1
        u[r2 * m + c] = 1
        v = [0.0] * (m * p)
        v[c * p + d1] = 1
        v[c * p + d2] = 1
        w = [0.0] * (p * n)
        w[d1 * n + r1] = 1
        w[d2 * n + r1] = 1
        w[d1 * n + r2] = 1
        w[d2 * n + r2] = 1
        add_term(u, v, w)

    # 1x1x1
    def small_1x1x1(r, c, d):
        u = [0.0] * (n * m); u[r * m + c] = 1
        v = [0.0] * (m * p); v[c * p + d] = 1
        w = [0.0] * (p * n); w[d * n + r] = 1
        add_term(u, v, w)

    # 1x1x2
    def small_1x1x2(r, c, d1, d2):
        u = [0.0] * (n * m); u[r * m + c] = 1
        v = [0.0] * (m * p)
        v[c * p + d1] = 1
        v[c * p + d2] = 1
        w = [0.0] * (p * n)
        w[d1 * n + r] = 1
        w[d2 * n + r] = 1
        add_term(u, v, w)

    # 2x1x1
    def small_2x1x1(r1, r2, c, d):
        u = [0.0] * (n * m)
        u[r1 * m + c] = 1
        u[r2 * m + c] = 1
        v = [0.0] * (m * p); v[c * p + d] = 1
        w = [0.0] * (p * n)
        w[d * n + r1] = 1
        w[d * n + r2] = 1
        add_term(u, v, w)

    # Block coordinates (0-indexed)
    # n=3: rows 0,1 (block 1), row 2 (block 2)
    # m=5: cols 0,1 (block 1), cols 2,3 (block 2), col 4 (block 3)
    # p=6: cols 0,1 (block 1), cols 2,3 (block 2), cols 4,5 (block 3)
    
    # C11 (0,0 to 1,1)
    strassen_2x2(0, 1, 0, 1, 0, 1) # A11 B11
    strassen_2x2(0, 1, 2, 3, 0, 1) # A12 B21
    small_2x1x2_rank1(0, 1, 4, 0, 1) # A13 B31
    
    # C12 (0,2 to 1,3)
    strassen_2x2(0, 1, 0, 1, 2, 3) # A11 B12
    strassen_2x2(0, 1, 2, 3, 2, 3) # A12 B22
    small_2x1x2_rank1(0, 1, 4, 2, 3) # A13 B32
    
    # C13 (0,4 to 1,5)
    strassen_2x2(0, 1, 0, 1, 4, 5) # A11 B13
    strassen_2x2(0, 1, 2, 3, 4, 5) # A12 B23
    small_2x1x2_rank1(0, 1, 4, 4, 5) # A13 B33
    
    # C21 (2,0 to 2,1)
    small_1x2x2_rank1(2, 0, 1, 0, 1) # A21 B11
    small_1x2x2_rank1(2, 2, 3, 0, 1) # A22 B21
    small_1x1x2(2, 4, 0, 1) # A23 B31
    
    # C22 (2,2 to 2,3)
    small_1x2x2_rank1(2, 0, 1, 2, 3) # A21 B12
    small_1x2x2_rank1(2, 2, 3, 2, 3) # A22 B22
    small_1x1x2(2, 4, 2, 3) # A23 B32
    
    # C23 (2,4 to 2,5)
    small_1x2x2_rank1(2, 0, 1, 4, 5) # A21 B13
    small_1x2x2_rank1(2, 2, 3, 4, 5) # A22 B23
    small_1x1x1(2, 4, 5) # A23 B33 -- Wait, B33 is 1x2? No, B33 is 1x2?
    # B block 3 is 1x2. So B33 is 1x2.
    # A23 is 1x1.
    # A23 B33 is 1x2.
    # So small_1x1x2.
    small_1x1x2(2, 4, 4, 5) # A23 B33
    
    # Convert lists to matrices
    rank = len(U_rows)
    U = [U_rows[i] for i in range(rank)] # Wait, U_rows are columns?
    # U is (n*m, rank). U_rows[i] is a vector of length n*m.
    # So U is a list of rows? No, U[i][j] is element i, j.
    # U_rows[j] is the j-th column.
    # So U[i][j] = U_rows[j][i].
    
    U_mat = [[0.0] * rank for _ in range(n * m)]
    V_mat = [[0.0] * rank for _ in range(m * p)]
    W_mat = [[0.0] * rank for _ in range(p * n)]
    
    for j in range(rank):
        for i in range(n * m):
            U_mat[i][j] = U_rows[j][i]
        for i in range(m * p):
            V_mat[i][j] = V_rows[j][i]
        for i in range(p * n):
            W_mat[i][j] = W_rows[j][i]
            
    return {"U": U_mat, "V": V_mat, "W": W_mat}

if __name__ == "__main__":
    construct()
