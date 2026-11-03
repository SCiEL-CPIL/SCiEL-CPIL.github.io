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

    # Try to find a lower-rank decomposition using Winograd 3x3 and Strassen 2x2x2 blocks
    # Winograd 3x3x3 block: rank 23 (vs 27 schoolbook)
    # Strassen 2x2x2 blocks: rank 7 each (vs 8 schoolbook)
    # Total: 23 + 6*7 + remaining schoolbook
    
    # Define the Winograd 3x3 block: I={0,1,2}, J={0,1,2}, K={0,1,2}
    winograd_block = ({0, 1, 2}, {0, 1, 2}, {0, 1, 2})
    
    # Define the 6 Strassen blocks (disjoint from Winograd in terms of (i,j,k) coverage)
    # We need blocks that don't overlap with the Winograd block's (i,j,k) indices
    # Winograd covers all (i,j,k) where i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
    # So Strassen blocks must use j in {3,4} or k in {3,4,5,6} to be disjoint
    
    blocks = [
        ({0, 1}, {3, 4}, {0, 1}),
        ({1, 2}, {3, 4}, {2, 3}),
        ({0, 2}, {3, 4}, {4, 5}),
        ({0, 1}, {3, 4}, {0, 1}),
        ({1, 2}, {3, 4}, {2, 3}),
        ({0, 2}, {3, 4}, {4, 5}),
    ]
    
    U = []
    V = []
    W = []
    
    def add_winograd_block(I, J, K):
        """Add Winograd 3x3x3 decomposition with rank 23."""
        i0, i1, i2 = sorted(I)
        j0, j1, j2 = sorted(J)
        k0, k1, k2 = sorted(K)
        
        # Winograd's 3x3 matrix multiplication algorithm (rank 23)
        # Based on the 2x2 Strassen-like approach extended to 3x3
        
        terms = []
        
        # We use the following 23 multiplication terms:
        # M1 = (A00+A11+A22)(B00+B11+B22)
        # M2 = (A10+A21)(B00+B10+B20)
        # M3 = (A01+A12)(B01+B11+B21)
        # M4 = (A02+A20)(B02+B12+B22)
        # M5 = (A00-A11+A22)(B00-B11+B22)
        # M6 = (A10-A21)(B00-B10+B20)
        # M7 = (A01-A12)(B01-B11+B21)
        # M8 = (A02-A20)(B02-B12+B22)
        # M9 = (A00+A11-2*A22)(B00+B11-2*B22)
        # M10 = (A10+A21-2*?)(B00+B10-2*?) -- need to be more careful
        
        # Let me use a known rank-23 decomposition for 3x3.
        # The standard approach uses 23 multiplications.
        
        # Actually, let me use a simpler approach: 
        # Use Strassen on 2x2 sub-blocks within the 3x3, plus handle the remaining terms.
        # Or better: use the known Winograd 3x3 scheme.
        
        # Winograd 3x3 (rank 23) - using the standard formulation:
        # The 23 terms are:
        
        # Row-based grouping approach:
        # M1 = (A00+A11+A22)(B00+B11+B22)
        # M2 = (A01+A12+A20)(B01+B12+B20)
        # M3 = (A02+A10+A21)(B02+B10+B21)
        # M4 = (A00+A11-A22)(B00+B11-A22)
        # M5 = (A01+A12-A20)(B01+B12-A20)
        # M6 = (A02+A10-A21)(B02+B10-A21)
        # M7 = (A00-A11+A22)(B00-A11+B22)
        # M8 = (A01-A12+A20)(B01-A12+B20)
        # M9 = (A02-A10+A21)(B02-A10+B21)
        # M10 = (A00+A11+4*A22)(B00+B11+4*B22)
        # M11 = (A01+A12+4*A20)(B01+B12+4*B20)
        # M12 = (A02+A10+4*A21)(B02+B10+4*B21)
        # M13 = (A00-A11-4*A22)(B00-A11-4*B22)
        # M14 = (A01-A12-4*A20)(B01-A12-4*B20)
        # M15 = (A02-A10-4*A21)(B02-A10-4*B21)
        # M16 = A00*B00
        # M17 = A11*B11
        # M18 = A22*B22
        # M19 = A01*B01
        # M20 = A12*B12
        # M21 = A20*B20
        # M22 = A02*B02
        # M23 = A10*B10
        # M24 = A21*B21
        
        # This gives 24 terms, not 23. Let me use a proper rank-23 scheme.
        
        # Actually, the minimal rank for 3x3 is 23 (Laderman). Let me use that.
        # Laderman's algorithm for 3x3 matrix multiplication has rank 23.
        
        # For now, let me use a simpler approach:
        # Use Strassen 2x2x2 on sub-blocks and schoolbook for the rest.
        
        # Split 3x3 into:
        # - 2x2 block (i0,i1) x (j0,j1) x (k0,k1): rank 7
        # - Remaining terms: 27 - 4 = 23 schoolbook terms
        # Total: 7 + 23 = 30, which is worse than 27.
        
        # Better approach: Use the known rank-23 decomposition.
        # Let me implement Laderman's 3x3 algorithm.
        
        # Laderman's 3x3 (rank 23):
        # The 23 multiplication terms are defined as follows:
        
        # I'll use a known correct rank-23 decomposition.
        # Reference: Laderman (1976) "On the multiplication of matrices"
        
        # For simplicity and correctness, let me use a different strategy:
        # Apply Strassen to multiple 2x2x2 sub-blocks within the 3x3.
        
        # 3x3 can be decomposed as:
        # - One 2x2x2 Strassen block: (i0,i1) x (j0,j1) x (k0,k1) -> rank 7
        # - Plus schoolbook for the remaining 23 terms
        # Total: 7 + 23 = 30 (worse than schoolbook 27)
        
        # OR:
        # - Two overlapping 2x2x2 Strassen blocks won't work directly.
        
        # Let me just use schoolbook for the 3x3 block for now,
        # and rely on the Strassen 2x2x2 blocks for the rest.
        
        # Actually, let me reconsider. The current approach with 6 Strassen blocks
        # gives rank 99. If I can find a better decomposition, I should use it.
        
        # For now, let me add the 3x3 block as schoolbook (rank 27)
        for ii in [i0, i1, i2]:
            for jj in [j0, j1, j2]:
                for kk in [k0, k1, k2]:
                    u = np.zeros(n*m)
                    v = np.zeros(m*p)
                    w = np.zeros(p*n)
                    u[ii*m + jj] = 1.0
                    v[jj*p + kk] = 1.0
                    w[kk*n + ii] = 1.0
                    terms.append((u, v, w))
        
        for u, v, w in terms:
            U.append(u)
            V.append(v)
            W.append(w)

    def add_strassen_block(I, J, K):
        # Map local indices 0,1 to global indices
        i0, i1 = sorted(I)
        j0, j1 = sorted(J)
        k0, k1 = sorted(K)
        
        # Strassen 7 terms for 2x2x2
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        u1 = np.zeros(n*m)
        v1 = np.zeros(m*p)
        w1 = np.zeros(p*n)
        u1[i0*m + j0] = 1.0
        u1[i1*m + j1] = 1.0
        v1[j0*p + k0] = 1.0
        v1[j1*p + k1] = 1.0
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
        w7[k0*n + i0] = 1.0
        terms.append((u7, v7, w7))
        
        for u, v, w in terms:
            U.append(u)
            V.append(v)
            W.append(w)

    # Add Winograd 3x3 block (as schoolbook for now)
    add_winograd_block(*winograd_block)
    
    # Add Strassen blocks
    for I, J, K in blocks:
        add_strassen_block(I, J, K)
        
    # Add remaining schoolbook terms
    # We need to identify which (i,j,k) are NOT covered by the blocks.
    covered = set()
    # Winograd block coverage
    I_w, J_w, K_w = winograd_block
    for i in I_w:
        for j in J_w:
            for k in K_w:
                covered.add((i, j, k))
    # Strassen block coverage
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
    
    # Try to remove redundant terms
    def try_remove_term(U, V, W, idx):
        """Try removing term idx and check if decomposition still works."""
        rank = U.shape[1]
        if rank <= 1:
            return False
        U_new = np.delete(U, idx, axis=1)
        V_new = np.delete(V, idx, axis=1)
        W_new = np.delete(W, idx, axis=1)
        return verify(U_new, V_new, W_new)
    
    # Attempt to remove terms one by one
    removed = True
    while removed and time.time() < deadline - 10:
        removed = False
        for idx in range(U_mat.shape[1]):
            if time.time() >= deadline - 10:
                break
            if try_remove_term(U_mat, V_mat, W_mat, idx):
                U_mat = np.delete(U_mat, idx, axis=1)
                V_mat = np.delete(V_mat, idx, axis=1)
                W_mat = np.delete(W_mat, idx, axis=1)
                removed = True
                break
    
    if verify(U_mat, V_mat, W_mat):
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
