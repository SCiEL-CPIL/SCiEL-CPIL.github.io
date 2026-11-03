import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Try to find a rank-16 decomposition using structured search
    # with a deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s limit
    
    # First, verify the rank-17 decomposition works, then try to improve
    
    # Rank-17 decomposition (known valid)
    rank_17_U = np.zeros((n * m, 17))
    rank_17_V = np.zeros((m * p, 17))
    rank_17_W = np.zeros((p * n, 17))
    
    # Index convention:
    # U[i, r] for i = row * m + col  (A has shape 2x3, indices 0..5)
    # V[j, r] for j = row * p + col  (B has shape 3x3, indices 0..8)
    # W[k, r] for k = col * n + row  (C has shape 3x2, indices 0..5)
    # C[i,j] index in W is j*n + i = 2*j + i
    
    # Strassen for the 2x2 block
    # M1: (A00+A11)(B00+B11)
    rank_17_U[0, 0] = 1; rank_17_U[4, 0] = 1
    rank_17_V[0, 0] = 1; rank_17_V[4, 0] = 1
    rank_17_W[0, 0] = 1; rank_17_W[3, 0] = 1
    
    # M2: (A10+A11)B00
    rank_17_U[3, 1] = 1; rank_17_U[4, 1] = 1
    rank_17_V[0, 1] = 1
    rank_17_W[1, 1] = 1; rank_17_W[3, 1] = -1
    
    # M3: A00(B01-B11)
    rank_17_U[0, 2] = 1
    rank_17_V[1, 2] = 1; rank_17_V[4, 2] = -1
    rank_17_W[2, 2] = 1; rank_17_W[3, 2] = 1
    
    # M4: A11(B10-B00)
    rank_17_U[4, 3] = 1
    rank_17_V[3, 3] = 1; rank_17_V[0, 3] = -1
    rank_17_W[0, 3] = 1; rank_17_W[1, 3] = 1
    
    # M5: (A00+A01)B11
    rank_17_U[0, 4] = 1; rank_17_U[1, 4] = 1
    rank_17_V[4, 4] = 1
    rank_17_W[0, 4] = -1; rank_17_W[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    rank_17_U[3, 5] = 1; rank_17_U[0, 5] = -1
    rank_17_V[0, 5] = 1; rank_17_V[1, 5] = 1
    rank_17_W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    rank_17_U[1, 6] = 1; rank_17_U[4, 6] = -1
    rank_17_V[3, 6] = 1; rank_17_V[4, 6] = 1
    rank_17_W[0, 6] = 1
    
    # Schoolbook: A[:, 2] * B[2, :2] -> C[:, :2]
    rank_17_U[2, 7] = 1; rank_17_V[6, 7] = 1; rank_17_W[0, 7] = 1
    rank_17_U[2, 8] = 1; rank_17_V[7, 8] = 1; rank_17_W[2, 8] = 1
    rank_17_U[5, 9] = 1; rank_17_V[6, 9] = 1; rank_17_W[1, 9] = 1
    rank_17_U[5, 10] = 1; rank_17_V[7, 10] = 1; rank_17_W[3, 10] = 1
    
    # Schoolbook: A[:, 0:3] * B[:, 2] -> C[:, 2]
    rank_17_U[0, 11] = 1; rank_17_V[2, 11] = 1; rank_17_W[4, 11] = 1
    rank_17_U[1, 12] = 1; rank_17_V[5, 12] = 1; rank_17_W[4, 12] = 1
    rank_17_U[2, 13] = 1; rank_17_V[8, 13] = 1; rank_17_W[4, 13] = 1
    rank_17_U[3, 14] = 1; rank_17_V[2, 14] = 1; rank_17_W[5, 14] = 1
    rank_17_U[4, 15] = 1; rank_17_V[5, 15] = 1; rank_17_W[5, 15] = 1
    rank_17_U[5, 16] = 1; rank_17_V[8, 16] = 1; rank_17_W[5, 16] = 1
    
    # Verify rank-17 decomposition
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all (i,j,k)"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(U.shape[1]):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify_decomposition(rank_17_U, rank_17_V, rank_17_W, n, m, p):
        # Fallback: return schoolbook rank-18
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to find rank-16 using structured approach
    # Key insight: try to merge some of the last 6 terms (k=2) with the j=2 terms
    
    # Strategy: Use Kronecker product structure
    # The 2x3x3 tensor can be viewed as combining:
    # - 2x2x2 (Strassen, rank 7)
    # - 2x1x2 (j=2, k in {0,1}, rank 4 schoolbook)  
    # - 2x3x1 (k=2, rank 6 schoolbook)
    
    # Try to reduce the 2x3x1 part from rank 6 to rank 5
    # C[:,2] = A @ B[:,2]
    # This is a 2x3 @ 3x1 = 2x1 multiplication
    # Can we do better than schoolbook rank 6?
    
    # Actually, 2x3 @ 3x1 has rank at least 6 in general (it's a full-rank matrix mult)
    # But we can try to share with the j=2 terms
    
    # Alternative: try rank 16 by modifying the decomposition
    # Let's try a different Strassen-like approach for the whole problem
    
    # Try: use 7 products for the 2x2x2 core
    # For the remaining 10 terms, try to use 9 products
    
    # The 10 remaining terms:
    # (0,2,0): A02*B20 -> C00
    # (0,2,1): A02*B21 -> C01
    # (1,2,0): A12*B20 -> C10
    # (1,2,1): A12*B21 -> C11
    # (0,0,2): A00*B02 -> C02
    # (0,1,2): A01*B12 -> C02
    # (0,2,2): A02*B22 -> C02
    # (1,0,2): A10*B02 -> C12
    # (1,1,2): A11*B12 -> C12
    # (1,2,2): A12*B22 -> C12
    
    # Idea: share A02 and A12 across multiple products
    # r=7: (A02+A12)*B20 -> covers (0,2,0) and (1,2,0) with W[0,7]=1, W[1,7]=1
    # r=8: (A02+A12)*B21 -> covers (0,2,1) and (1,2,1) with W[2,8]=1, W[3,8]=1
    # But this gives same coefficient to both, which is what we want!
    
    # Let's try this approach:
    U16 = np.zeros((n*m, 16))
    V16 = np.zeros((m*p, 16))
    W16 = np.zeros((p*n, 16))
    
    # Copy Strassen part (r=0..6)
    for r in range(7):
        U16[:, r] = rank_17_U[:, r]
        V16[:, r] = rank_17_V[:, r]
        W16[:, r] = rank_17_W[:, r]
    
    # r=7: (A02+A12)*B20 -> C00, C10
    U16[2, 7] = 1; U16[5, 7] = 1
    V16[6, 7] = 1
    W16[0, 7] = 1; W16[1, 7] = 1
    
    # r=8: (A02+A12)*B21 -> C01, C11
    U16[2, 8] = 1; U16[5, 8] = 1
    V16[7, 8] = 1
    W16[2, 8] = 1; W16[3, 8] = 1
    
    # r=9: A02*B22 -> C02
    U16[2, 9] = 1
    V16[8, 9] = 1
    W16[4, 9] = 1
    
    # r=10: A12*B22 -> C12
    U16[5, 10] = 1
    V16[8, 10] = 1
    W16[5, 10] = 1
    
    # r=11: (A00+A10)*B02 -> C02, C12
    U16[0, 11] = 1; U16[3, 11] = 1
    V16[2, 11] = 1
    W16[4, 11] = 1; W16[5, 11] = 1
    
    # r=12: (A01+A11)*B12 -> C02, C12
    U16[1, 12] = 1; U16[4, 12] = 1
    V16[5, 12] = 1
    W16[4, 12] = 1; W16[5, 12] = 1
    
    # That's only 13 terms (r=0..12). We need to check if this is valid.
    # But wait, this might not work because the Strassen products already use
    # A00, A01, A10, A11, B00, B01, B10, B11, and we're now using them again
    # in r=11, r=12. This should be fine as long as the tensor equation holds.
    
    # Let's verify
    if verify_decomposition(U16, V16, W16, n, m, p):
        return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # If rank 13 doesn't work (likely, as it's below known lower bound),
    # try other combinations
    
    # Try rank 16 with different sharing
    # r=7: A02*B20 -> C00
    # r=8: A12*B20 -> C10
    # r=9: A02*B21 -> C01
    # r=10: A12*B21 -> C11
    # r=11: A02*B22 -> C02
    # r=12: A12*B22 -> C12
    # r=13: (A00+A10)*B02 -> C02, C12
    # r=14: (A01+A11)*B12 -> C02, C12
    # That's 15 terms (r=0..14). Need one more.
    
    # Actually, let's just try to use linear algebra to find a rank-16 solution
    # by setting up the system of equations
    
    # For rank r, we have variables U (6xr), V (9xr), W (6xr)
    # The constraint is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of 18 polynomial equations
    
    # Given time constraints, let's try a simpler approach:
    # Use the rank-17 as base and try to eliminate one term
    
    # Check if any two terms in rank-17 can be merged
    # Term r and term s can be merged if there exist scalars alpha, beta such that
    # the combined term still satisfies all 18 equations
    
    # This is complex, so let's just return the best we have
    
    # Try one more structured approach: use the fact that 2x3x3 can be
    # decomposed using the formula for rectangular matrix multiplication
    
    # For 2x3 @ 3x3, we can view it as:
    # [A1 | A2] @ [B1; B2] where A1 is 2x2, A2 is 2x1, B1 is 2x3, B2 is 1x3
    # C = A1*B1 + A2*B2
    
    # A1*B1 is 2x2 @ 2x3: can use Strassen-like for the 2x2x2 part (7 products)
    # plus 2 more for the third column: total 9 for A1*B1
    # A2*B2 is 2x1 @ 1x3: this is just outer product, rank 2 (one product per output element)
    # Wait, 2x1 @ 1x3 = 2x3, which needs 6 products in schoolbook
    # But actually, (A2 @ B2)[i,j] = A2[i] * B2[j], so each output element is one product
    # Total: 6 products for A2*B2
    
    # So total: 7 (for 2x2x2) + 2 (for 2x2x1, third column of B1) + 6 (for 2x1x3) = 15?
    # Hmm, that seems too low. Let me recount.
    
    # A1 is 2x2 (rows 0,1 of A; cols 0,1 of A)
    # A2 is 2x1 (rows 0,1 of A; col 2 of A)
    # B1 is 2x3 (rows 0,1 of B; cols 0,1,2 of B)
    # B2 is 1x3 (row 2 of B; cols 0,1,2 of B)
    
    # C = A1 @ B1 + A2 @ B2
    
    # A1 @ B1: 2x2 @ 2x3
    # This can be done with: Strassen for the 2x2x2 part (7 products) gives C[:, 0:2]
    # Plus 2 more products for C[:, 2]: A1[0,:] @ B1[:,2] and A1[1,:] @ B1[:,2]
    # Wait, that's not right. Let me think again.
    
    # Actually, 2x2 @ 2x3:
    # C[i,j] = sum_{k=0}^{1} A1[i,k] * B1[k,j]
    # For j=0,1: this is the 2x2x2 problem, rank 7 (Strassen)
    # For j=2: C[i,2] = A1[i,0]*B1[0,2] + A1[i,1]*B1[1,2] for i=0,1
    # That's 4 products in schoolbook, or can we do better?
    # C[0,2] = A1[0,0]*B1[0,2] + A1[0,1]*B1[1,2]
    # C[1,2] = A1[1,0]*B1[0,2] + A1[1,1]*B1[1,2]
    # We can use: P1 = (A1[0,0]+A1[1,0]) * B1[0,2], P2 = (A1[0,1]+A1[1,1]) * B1[1,2]
    # Then C[0,2] = P1 + P2 - A1[1,0]*B1[0,2] - A1[1,1]*B1[1,2]... no, that doesn't work cleanly.
    # Actually, for 2x2 @ 2x1, the rank is 4 (schoolbook) or can it be less?
    # The 2x2x1 tensor has rank 4 in general. So 4 products.
    
    # A2 @ B2: 2x1 @ 1x3
    # C[i,j] = A2[i] * B2[j]
    # This is an outer product. Each of the 6 output elements is one product.
    # But we can share: P1 = A2[0]*B2[0], P2 = A2[0]*B2[1], P3 = A2[0]*B2[2],
    # P4 = A2[1]*B2[0], P5 = A2[1]*B2[1], P6 = A2[1]*B2[2]
    # That's 6 products. Can we do better? The 2x1x3 tensor has rank 6 in general.
    
    # Total: 7 + 4 + 6 = 17. Same as before.
    
    # Let's just return the rank-17 decomposition
    return {"U": rank_17_U.tolist(), "V": rank_17_V.tolist(), "W": rank_17_W.tolist()}
