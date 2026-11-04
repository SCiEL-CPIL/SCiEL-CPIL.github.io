"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = 135.0  # safety margin before 150s

    n, m, p = DIMENSIONS

    def make_factor(rows, cols, entries):
        """Create a matrix from a list of (row, col, val) tuples."""
        mat = [[0.0] * cols for _ in range(rows)]
        for r, c, v in entries:
            mat[r][c] = v
        return mat

    def kron_factors(U1, V1, W1, U2, V2, W2):
        """
        Compute tensor product of two matrix multiplication decompositions.
        U1: (n1*m1, r1), V1: (m1*p1, r1), W1: (p1*n1, r1)
        U2: (n2*m2, r2), V2: (m2*p2, r2), W2: (p2*n2, r2)
        Returns U, V, W for (n1*n2, m1*m2, p1*p2) with rank r1*r2.
        """
        n1, m1, p1 = U1[0], V1[0], W1[0]
        n2, m2, p2 = U2[0], V2[0], W2[0]
        r1 = U1[1]
        r2 = U2[1]
        
        U1m = U1[2]  # n1*m1 x r1
        V1m = V1[2]  # m1*p1 x r1
        W1m = W1[2]  # p1*n1 x r1
        U2m = U2[2]  # n2*m2 x r2
        V2m = V2[2]  # m2*p2 x r2
        W2m = W2[2]  # p2*n2 x r2

        # We need to build U (n1*n2 * m1*m2) x (r1*r2)
        # Row index in U: ((i1*n2 + i2) * m1*m2 + j1*m2 + j2)
        #   = i1*n2*m1*m2 + i2*m1*m2 + j1*m2 + j2
        #   = i1*(n2*m1*m2) + i2*(m1*m2) + j1*m2 + j2
        #   = i1*(n2*m1*m2) + (i2*m1 + j1)*m2 + j2
        # Col index in U: c1*r2 + c2
        # U[row, col] = U1m[i1*m1+j1, c1] * U2m[i2*m2+j2, c2]
        
        n = n1 * n2
        m = m1 * m2
        p = p1 * p2
        r = r1 * r2
        
        U = [[0.0] * r for _ in range(n * m)]
        V = [[0.0] * r for _ in range(m * p)]
        W = [[0.0] * r for _ in range(p * n)]
        
        # This would be O(n^2*m^2*p^2 * r^2) which is too slow for direct iteration.
        # Let's use Kronecker product logic.
        # Actually, let's just do the direct construction since dimensions are small.
        # n=6, m=6, p=6. r=189.
        # n*m = 36. r = 189.
        # U is 36 x 189.
        
        # Let's re-verify the index mapping.
        # A is n1*n2 x m1*m2. A[i1*n2+i2, j1*m2+j2] = A1[i1, j1] * A2[i2, j2].
        # Wait, standard Kronecker product (A1 ⊗ A2) has:
        # (A1 ⊗ A2)[i1*n2 + i2, j1*m2 + j2] = A1[i1, j1] * A2[i2, j2].
        # This matches the block structure if A1 is top-level and A2 is bottom-level.
        
        # U row: i*n*m + j where i = i1*n2 + i2, j = j1*m2 + j2.
        # U row = (i1*n2 + i2)*m1*m2 + j1*m2 + j2
        #        = i1*n2*m1*m2 + i2*m1*m2 + j1*m2 + j2
        
        # U col: c1*r2 + c2.
        # U[row, col] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
        
        # Let's iterate over c1, c2, i1, j1, i2, j2.
        # For each (c1, c2, i1, j1, i2, j2), if U1[i1*m1+j1, c1] and U2[i2*m2+j2, c2] are non-zero,
        # then U[row, col] += U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
        
        # We can do this more efficiently.
        # U is essentially the Kronecker product of U1 and U2 in some sense, but permuted.
        # Let's just build it directly.
        
        for c1 in range(r1):
            for c2 in range(r2):
                col = c1 * r2 + c2
                for i1 in range(n1):
                    for j1 in range(m1):
                        u1_val = U1m[i1*m1+j1][c1]
                        if u1_val == 0: continue
                        for i2 in range(n2):
                            for j2 in range(m2):
                                u2_val = U2m[i2*m2+j2][c2]
                                if u2_val == 0: continue
                                row = (i1*n2 + i2)*m1*m2 + j1*m2 + j2
                                U[row][col] += u1_val * u2_val
                                
                                # Time check
                                if time.time() > deadline:
                                    pass # We can't easily return partial here, but we can stop early.
        
        # Similar for V
        for c1 in range(r1):
            for c2 in range(r2):
                col = c1 * r2 + c2
                for j1 in range(m1):
                    for k1 in range(p1):
                        v1_val = V1m[j1*p1+k1][c1]
                        if v1_val == 0: continue
                        for j2 in range(m2):
                            for k2 in range(p2):
                                v2_val = V2m[j2*p2+k2][c2]
                                if v2_val == 0: continue
                                row = (j1*m2 + j2)*p1*p2 + k1*p2 + k2
                                V[row][col] += v1_val * v2_val
                                
        # Similar for W
        for c1 in range(r1):
            for c2 in range(r2):
                col = c1 * r2 + c2
                for k1 in range(p1):
                    for i1 in range(n1):
                        w1_val = W1m[k1*n1+i1][c1]
                        if w1_val == 0: continue
                        for k2 in range(p2):
                            for i2 in range(n2):
                                w2_val = W2m[k2*n2+i2][c2]
                                if w2_val == 0: continue
                                row = (k1*p2 + k2)*n1*n2 + i1*n2 + i2
                                W[row][col] += w1_val * w2_val
                                
        return (n, m, p), r, U, V, W

    # 3x3 Schoolbook
    n3, m3, p3 = 3, 3, 3
    r3 = 27
    U3 = [[0.0]*r3 for _ in range(n3*m3)]
    V3 = [[0.0]*r3 for _ in range(m3*p3)]
    W3 = [[0.0]*r3 for _ in range(p3*n3)]
    col = 0
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                U3[i*m3+j][col] = 1
                V3[j*p3+k][col] = 1
                W3[k*n3+i][col] = 1
                col += 1
    T3 = ((n3, m3, p3), r3, U3, V3, W3)

    # 2x2 Strassen
    n2, m2, p2 = 2, 2, 2
    r2 = 7
    U2 = [[0.0]*r2 for _ in range(n2*m2)]
    V2 = [[0.0]*r2 for _ in range(m2*p2)]
    W2 = [[0.0]*r2 for _ in range(p2*n2)]
    
    # M1 = (A11+A22)(B11+B22)
    # U2[0,0]=1 (A11), U2[3,0]=1 (A22)
    # V2[0,0]=1 (B11), V2[3,0]=1 (B22)
    # W2[0,0]=1 (C11), W2[3,0]=1 (C22)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1: A11+A22 -> B11+B22
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2 = (A21+A22)B11
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = 1
    
    # M3 = A11(B12-B22)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # M4 = A22(B21-B11)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[2][3] = 1; W2[3][3] = 1
    
    # M5 = (A11+A12)B22
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    T2 = ((n2, m2, p2), r2, U2, V2, W2)

    # Combine T3 and T2
    # Resulting dims: 6x6x6
    # Rank: 27 * 7 = 189
    (n, m, p), r, U, V, W = kron_factors(T3, T2, T2)

    # Final sanity check (optional, but good for debugging)
    # if time.time() < deadline:
    #    ... verify ...

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
