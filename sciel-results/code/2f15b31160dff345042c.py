import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    nm = n * m
    mp = m * p
    pn = p * n
    
    # We'll use a Strassen-like decomposition for the 4x4 part and extend to 4x7
    # For 4x4, we can use a decomposition with rank 27 (known result)
    # For 4x7, we need to handle the extra 3 columns
    
    # Strategy: Use the 4x4 decomposition for the first 4 columns,
    # and handle the remaining 3 columns separately
    
    # First, let's build a decomposition for 4x4x4 with rank 27
    # Using a known decomposition pattern
    
    # For simplicity and correctness, let's use a structured approach:
    # Split the 4x4 matrix into 2x2 blocks and use Strassen for each block
    # This gives rank 7 for each 2x2x2, but we need to be careful about overlaps
    
    # Actually, let's use a simpler approach: 
    # For 4x4x4, use 27 terms from known optimal decomposition
    # For the extra 3 columns (making it 4x4x7), add 3*4=12 more terms
    # Total: 27 + 12 = 39
    
    # But we need to be careful: the decomposition must be exact
    
    # Let's use a different approach: 
    # Use the fact that we can share some computations between columns
    
    # For now, let's implement a correct decomposition with reduced rank
    # using a block-based Strassen approach
    
    terms = []
    
    # Helper functions
    def make_u(idx_i, idx_j, val):
        u = [0] * nm
        u[idx_i * m + idx_j] = val
        return u
    
    def make_v(idx_j, idx_k, val):
        v = [0] * mp
        v[idx_j * p + idx_k] = val
        return v
    
    def make_w(idx_k, idx_i, val):
        w = [0] * pn
        w[idx_k * n + idx_i] = val
        return w
    
    # Use a Strassen-like decomposition for the 4x4 part
    # Split into 2x2 blocks
    # For each 2x2x2 block, use 7 terms instead of 8
    
    # Block indices: (bi, bj) where bi in {0,1}, bj in {0,1}
    # Each block is 2x2x2
    
    # For block (bi, bj), the rows are 2*bi to 2*bi+1, cols are 2*bj to 2*bj+1
    # The k-index for this block is 2*bj to 2*bj+1 (for the first 4 columns)
    
    # Strassen for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each block (bi, bj), we have 7 terms
    # Total for 4x4x4: 4 blocks * 7 = 28 terms
    # Plus 3 extra columns: 3*4 = 12 terms
    # Total: 40 terms
    
    # Let's implement this:
    
    # For each 2x2 block in the 4x4x4 part
    for bi in range(2):
        for bj in range(2):
            # Block rows: 2*bi, 2*bi+1
            # Block cols (in A): 2*bj, 2*bj+1
            # Block k: 2*bj, 2*bj+1 (for the first 4 columns)
            
            i1, i2 = 2*bi, 2*bi+1
            j1, j2 = 2*bj, 2*bj+1
            k1, k2 = 2*bj, 2*bj+1
            
            # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
            # This requires two rank-1 terms for U and V, but we can combine
            # Actually, each M is a product of a linear combination of A entries 
            # and a linear combination of B entries, so it's one rank-1 term
            # in the tensor sense (with coefficients in U, V, W)
            
            # M1: U has +1 at (i1,j1) and +1 at (i2,j2)
            #      V has +1 at (j1,k1) and +1 at (j2,k2)
            #      W has +1 at (k1,i1), +1 at (k2,i2), -1 at (k1,i2), +1 at (k2,i1)
            # Wait, this is getting complex. Let me use the standard formulation.
            
            # Actually, for Strassen, each M is a single product of two linear forms,
            # so it corresponds to one rank-1 term in the tensor decomposition.
            # The U vector has the coefficients of the A linear form,
            # the V vector has the coefficients of the B linear form,
            # and the W vector distributes the product to the correct C positions.
            
            # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
            # This contributes to:
            # C[i1,k1] += +1 * M1
            # C[i1,k2] += +1 * M1
            # C[i2,k1] += +1 * M1
            # C[i2,k2] += +1 * M1
            # But wait, that's not right. Let me re-derive.
            
            # Actually, in Strassen, M1 contributes to C11 and C22 with specific signs.
            # Let me use the standard Strassen formulation for 2x2:
            
            # A = [[A11, A12], [A21, A22]]
            # B = [[B11, B12], [B21, B22]]
            # C = [[C11, C12], [C21, C22]]
            
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B21-B22)
            # M4 = A22(B12-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # For block (bi, bj):
            # A11 = A[i1, j1], A12 = A[i1, j2]
            # A21 = A[i2, j1], A22 = A[i2, j2]
            # B11 = B[j1, k1], B12 = B[j1, k2]
            # B21 = B[j2, k1], B22 = B[j2, k2]
            # C11 = C[i1, k1], C12 = C[i1, k2]
            # C21 = C[i2, k1], C22 = C[i2, k2]
            
            # M1: U = {A11: +1, A22: +1}, V = {B11: +1, B22: +1}
            # W: C11 += +1, C22 += +1
            u = [0] * nm
            v = [0] * mp
            w = [0] * pn
            u[i1*m + j1] = 1
            u[i2*m + j2] = 1
            v[j1*p + k1] = 1
            v[j2*p + k2] = 1
            w[k1*n + i1] = 1
            w[k2*n + i2] = 1
            terms.append((u, v, w))
            
            # M2: U = {A21: +1, A22: +1}, V = {B11: +1}
            # W: C21 += +1, C22 += -1
            u = [0] * nm
            v = [0] * mp
            w = [0] * pn
            u[i2*m + j1] = 1
            u[i2*m + j2] = 1
            v[j1*p + k1] = 1
            w[k1*n + i2] = 1
            w[k2*n + i2] = -1
            terms.append((u, v, w))
            
            # M3: U = {A11: +1}, V = {B21: +1, B22: -1}
            # W: C12 += +1, C22 += +1
            u = [0] * nm
            v = [0] * mp
            w = [0] * pn
            u[i1*m + j1] = 1
            v[j2*p + k1] = 1
            v[j2*p + k2] = -1
            w[k2*n + i1] = 1
            w[k2*n + i2] = 1
            terms.append((u, v, w))
            
            # M4: U = {A22: +1}, V = {B12: +1, B11: -1}
            # W: C11 += +1, C21 += +1
            u = [0] * nm
            v = [0] * mp
            w = [0] * pn
            u[i2*m + j2] = 1
            v[j1*p + k2] = 1
            v[j1*p + k1] = -1
            w[k1*n + i1] = 1
            w[k1*n + i2] = 1
            terms.append((u, v, w))
            
            # M5: U = {A11: +1, A12: +1}, V = {B22: +1}
            # W: C11 += -1, C12 += +1
            u = [0] * nm
            v = [0] * mp
            w = [0] * pn
            u[i1*m + j1] = 1
            u[i1*m + j2] = 1
            v[j2*p + k2] = 1
            w[k1*n + i1] = -1
            w[k2*n + i1] = 1
            terms.append((u, v, w))
            
            # M6: U = {A21: +1, A11: -1}, V = {B11: +1, B12: +1}
            # W: C22 += +1
            u = [0] * nm
            v = [0] * mp
            w = [0] * pn
            u[i2*m + j1] = 1
            u[i1*m + j1] = -1
            v[j1*p + k1] = 1
            v[j1*p + k2] = 1
            w[k2*n + i2] = 1
            terms.append((u, v, w))
            
            # M7: U = {A12: +1, A22: -1}, V = {B21: +1, B22: +1}
            # W: C11 += +1
            u = [0] * nm
            v = [0] * mp
            w = [0] * pn
            u[i1*m + j2] = 1
            u[i2*m + j2] = -1
            v[j2*p + k1] = 1
            v[j2*p + k2] = 1
            w[k1*n + i1] = 1
            terms.append((u, v, w))
    
    # Now handle the extra 3 columns (k=4,5,6)
    # For each i in 0..3, k in 4..6: C[i,k] = sum_j A[i,j] * B[j,k]
    # This is 4*3 = 12 terms
    for i in range(n):
        for k in range(4, p):
            u = [0] * nm
            v = [0] * mp
            w = [0] * pn
            for j in range(m):
                u[i*m + j] = 1
                v[j*p + k] = 1
            w[k*n + i] = 1
            terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(nm):
            U[r][c] = u[r]
        for r in range(mp):
            V[r][c] = v[r]
        for r in range(pn):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
