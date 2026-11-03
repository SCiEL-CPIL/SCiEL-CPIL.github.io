"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Verify schoolbook is correct
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for c in range(U.shape[1]):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(lhs - rhs) > 1e-6:
                        return False
        return True

    # Try to find a lower-rank decomposition using Strassen on 2x2x2 blocks
    # We identified 6 disjoint 2x2x2 blocks that can be decomposed with rank 7 each.
    # The remaining 57 terms are schoolbook rank 1.
    # Total rank = 6 * 7 + 57 = 99.
    
    # Define the 6 blocks
    # Block format: (I_set, J_set, K_set)
    blocks = [
        ({0, 1}, {0, 1}, {0, 1}),
        ({1, 2}, {0, 1}, {2, 3}),
        ({0, 2}, {0, 1}, {4, 5}),
        ({0, 1}, {2, 3}, {0, 1}),
        ({1, 2}, {2, 3}, {2, 3}),
        ({0, 2}, {2, 3}, {4, 5}),
    ]
    
    U = []
    V = []
    W = []
    
    def add_strassen_block(I, J, K):
        # Map local indices 0,1 to global indices
        i0, i1 = sorted(I)
        j0, j1 = sorted(J)
        k0, k1 = sorted(K)
        
        # Strassen 7 terms for 2x2x2
        # Local indices: i, j, k in {0, 1}
        # A[i,j], B[j,k], C[i,k]
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to add 7 rank-1 terms to U, V, W
        # Each term is a vector u, v, w of size n*m, m*p, p*n respectively (sparse)
        
        terms = []
        
        # Term 1: (A00+A11)(B00+B11)
        # u: 1 at (i0,j0), 1 at (i1,j1)
        # v: 1 at (j0,k0), 1 at (j1,k1)
        # w: 1 at (k0,i0), 1 at (k1,i1), 1 at (k0,i1), 1 at (k1,i0) -> No, W is just a coefficient
        # Wait, the decomposition is sum u v w.
        # The product u v w corresponds to the scalar product of the linear forms.
        
        # Let's define the vectors explicitly.
        
        # M1
        u1 = np.zeros(n*m)
        v1 = np.zeros(m*p)
        w1 = np.zeros(p*n)
        u1[i0*m + j0] = 1.0
        u1[i1*m + j1] = 1.0
        v1[j0*p + k0] = 1.0
        v1[j1*p + k1] = 1.0
        # C00 += M1, C11 += M1
        w1[k0*n + i0] = 1.0
        w1[k1*n + i1] = 1.0
        terms.append((u1, v1, w1))
        
        # M2 = (A10+A11)B00
        u2 = np.zeros(n*m)
        v2 = np.zeros(m*p)
        w2 = np.zeros(p*n)
        u2[i1*m + j0] = 1.0
        u2[i1*m + j1] = 1.0
        v2[j0*p + k0] = 1.0
        # C10 += M2, C11 -= M2
        w2[k0*n + i1] = 1.0
        w2[k1*n + i1] = -1.0
        terms.append((u2, v2, w2))
        
        # M3 = A00(B01-B11)
        u3 = np.zeros(n*m)
        v3 = np.zeros(m*p)
        w3 = np.zeros(p*n)
        u3[i0*m + j0] = 1.0
        v3[j0*p + k1] = 1.0
        v3[j1*p + k1] = -1.0
        # C01 += M3, C11 += M3
        w3[k1*n + i0] = 1.0
        w3[k1*n + i1] = 1.0
        terms.append((u3, v3, w3))
        
        # M4 = A11(B10-B00)
        u4 = np.zeros(n*m)
        v4 = np.zeros(m*p)
        w4 = np.zeros(p*n)
        u4[i1*m + j1] = 1.0
        v4[j1*p + k0] = 1.0
        v4[j0*p + k0] = -1.0
        # C00 += M4, C10 += M4
        w4[k0*n + i0] = 1.0
        w4[k0*n + i1] = 1.0
        terms.append((u4, v4, w4))
        
        # M5 = (A00+A01)B11
        u5 = np.zeros(n*m)
        v5 = np.zeros(m*p)
        w5 = np.zeros(p*n)
        u5[i0*m + j0] = 1.0
        u5[i0*m + j1] = 1.0
        v5[j1*p + k1] = 1.0
        # C00 -= M5, C01 += M5
        w5[k1*n + i0] = 1.0
        w5[k1*n + i0] = 1.0 # C00 coeff is -1, C01 is 1
        # Correction:
        w5[k1*n + i0] = 1.0
        # Need separate terms or combined?
        # C00 gets -M5, C01 gets +M5.
        # This term contributes to both.
        w5 = np.zeros(p*n)
        w5[k1*n + i0] = 1.0 # for C01
        # But C00 needs -1.
        # We can't have both in one rank-1 term if they are different indices?
        # W is a vector. w[c] is the coefficient for C[c].
        # So w[k1*n + i0] = 1 and w[k1*n + i0] = -1 is a contradiction.
        # Ah, C00 is (i0, k0). C01 is (i0, k1).
        # So w[k0*n + i0] = -1 and w[k1*n + i0] = 1.
        w5[k0*n + i0] = -1.0
        w5[k1*n + i0] = 1.0
        terms.append((u5, v5, w5))
        
        # M6 = (A10-A00)(B00+B01)
        u6 = np.zeros(n*m)
        v6 = np.zeros(m*p)
        w6 = np.zeros(p*n)
        u6[i1*m + j0] = 1.0
        u6[i0*m + j0] = -1.0
        v6[j0*p + k0] = 1.0
        v6[j0*p + k1] = 1.0
        # C11 += M6
        w6[k1*n + i1] = 1.0
        terms.append((u6, v6, w6))
        
        # M7 = (A01-A11)(B10+B11)
        u7 = np.zeros(n*m)
        v7 = np.zeros(m*p)
        w7 = np.zeros(p*n)
        u7[i0*m + j1] = 1.0
        u7[i1*m + j1] = -1.0
        v7[j1*p + k0] = 1.0
        v7[j1*p + k1] = 1.0
        # C00 += M7
        w7[k0*n + i0] = 1.0
        terms.append((u7, v7, w7))
        
        for u, v, w in terms:
            U.append(u)
            V.append(v)
            W.append(w)

    # Add Strassen blocks
    for I, J, K in blocks:
        add_strassen_block(I, J, K)
        
    # Add remaining schoolbook terms
    # We need to identify which (i,j,k) are NOT covered by the blocks.
    # Covered if (i,j,k) is in the union of the blocks' index sets.
    covered = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
                    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u = np.zeros(n*m)
                    v = np.zeros(m*p)
                    w = np.zeros(p*n)
                    u[i*m + j] = 1.0
                    v[j*p + k] = 1.0
                    w[k*n + i] = 1.0
                    U.append(u)
                    V.append(v)
                    W.append(w)
                    
    U_mat = np.array(U).T # Shape (rank, n*m) -> transpose to (n*m, rank)
    V_mat = np.array(V).T
    W_mat = np.array(W).T
    
    if verify(U_mat, V_mat, W_mat):
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
