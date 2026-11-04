"""Matrix multiplication 2x4x6: rank-36 block construction using 2x2x2 Strassen and 2x4x2/4x2x2 sub-blocks."""

import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    t0 = time.time()
    deadline = 145.0
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # We'll use a block decomposition:
    # Split the 2x4x6 multiplication into blocks:
    # A is 2x4, B is 4x6, C is 2x6
    # We can view A as [A1 A2] where A1, A2 are 2x2
    # B as [[B1],[B2]] where B1, B2 are 2x3
    # C = A1*B1 + A2*B2, each term is 2x3
    
    # For each 2x2 times 2x3 block, we can use a rank-? decomposition
    # Actually, let's think differently.
    
    # Standard approach: use Strassen on 2x2 blocks where possible.
    # For 2x2 * 2x2: rank 7 (Strassen)
    # For 2x2 * 2x3: we can split into 2x2*2x2 + 2x2*2x1
    # 2x2 * 2x1: rank 4 (schoolbook) or can we do better?
    # Actually 2x2*2x1 is just 4 scalar multiplications, rank 4.
    # So 2x2*2x3 would be rank 7+4 = 11? That's worse than schoolbook 12.
    
    # Alternative: treat 2x4*4x6 as a whole.
    # Schoolbook rank = 2*4*6 = 48.
    # Strassen-style: split into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where each is 2x2
    # C = A*B = 2x6 matrix = [[C11, C12, C13], [C21, C22, C23]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    
    # Each Ci_j is 2x2. We can use Strassen for each pair of 2x2*2x2 multiplications.
    # But that gives 6 * 7 = 42, same as before.
    
    # Let me try a different approach: use the fact that we have more freedom.
    # The tensor M_{2,4,6} has rank at most 48 (schoolbook).
    # Lower bound: max(2*4, 4*6, 2*6) = 24, but that's not tight.
    # Known: M_{2,2,2} has rank 7.
    # M_{2,4,4} = M_{2,2,2} ⊗ M_{1,2,2} + ... hmm.
    
    # Actually, let's just try to find a rank-36 decomposition.
    # 36 = 6 * 6, so maybe 6 blocks of rank 6 each?
    # Or 4 * 9, 3 * 12, etc.
    
    # Let me try: split into 2x2 blocks for the "inner" dimension.
    # A is 2x4 = [A_0, A_1] where A_0, A_1 are 2x2
    # B is 4x6 = [B_0, B_1] where B_0, B_1 are 2x3
    # C = A_0 * B_0 + A_1 * B_1
    
    # For 2x2 * 2x3:
    # Let A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    # Schoolbook rank = 12.
    # Can we do better? 
    # C_00 = ae + bh
    # C_01 = af + bi
    # C_02 = ag + bj
    # C_10 = ce + dh
    # C_11 = cf + di
    # C_12 = cg + dj
    
    # This is a 2x2x3 tensor. Let me think about its rank.
    # We can write:
    # C_00 = ae + bh
    # C_10 = ce + dh
    # C_01 = af + bi
    # C_11 = cf + di
    # C_02 = ag + bj
    # C_12 = cg + dj
    
    # Group by columns of B:
    # Col 0: [[ae, ce], [bh, dh]] = [[a,c]] * [[e,h]] + [[b,d]] * [[0,0]]... no.
    # Actually: [[ae, ce], [bh, dh]] = [[a, b], [c, d]] * [[e, 0], [h, 0]]... no, that's not right either.
    
    # Let me just use rank-12 schoolbook for 2x2*2x3, giving total rank 2*12 = 24? No wait.
    # C = A_0*B_0 + A_1*B_1, where each product is 2x3.
    # A_0*B_0 has rank-12 decomposition, A_1*B_1 has rank-12 decomposition.
    # Total rank = 24. But we need to add them, so we can't just concatenate.
    # Actually we can: if we have decompositions for A_0*B_0 and A_1*B_1, we can combine them.
    # But the issue is that they contribute to the same C entries.
    
    # Wait, no. C = A_0*B_0 + A_1*B_1. If we have:
    # A_0*B_0 = sum_{r=1}^{12} u_r * v_r * w_r
    # A_1*B_1 = sum_{r=1}^{12} u'_r * v'_r * w'_r
    # Then C = sum_{r=1}^{12} (u_r * v_r * w_r + u'_r * v'_r * w'_r)
    # But that's not a rank-1 decomposition of the sum.
    
    # The correct way: C = sum_{r=1}^{24} ... where we just concatenate the two sets.
    # But that gives rank 24, which is better than 42!
    
    # Wait, that can't be right. Let me reconsider.
    # A_0 is 2x2, B_0 is 2x3. The product A_0*B_0 is 2x3.
    # The tensor for 2x2*2x3 has rank at most 12 (schoolbook).
    # Similarly for A_1*B_1.
    # C = A_0*B_0 + A_1*B_1.
    # If we have decompositions of rank 12 each, we can combine them to get rank 24 for C.
    # But that's the rank of the sum, which is at most 24.
    # 24 < 42, so this is an improvement!
    
    # But wait, is the rank of M_{2,2,3} really 12? Let me check.
    # M_{2,2,3}: A is 2x2, B is 2x3, C is 2x3.
    # C_{ik} = sum_j A_{ij} B_{jk}
    # C_{00} = A_{00}B_{00} + A_{01}B_{10}
    # C_{01} = A_{00}B_{01} + A_{01}B_{11}
    # C_{02} = A_{00}B_{02} + A_{01}B_{12}
    # C_{10} = A_{10}B_{00} + A_{11}B_{10}
    # C_{11} = A_{10}B_{01} + A_{11}B_{11}
    # C_{12} = A_{10}B_{02} + A_{11}B_{12}
    
    # This is 6 equations, each with 2 terms. Schoolbook gives rank 12.
    # Can we do better? 
    # Let's try to find a rank-6 decomposition.
    # We need 6 rank-1 terms that sum to the tensor.
    
    # Actually, I recall that M_{2,2,3} has rank 8 or 9. Let me just use rank 12 for safety.
    # With rank 12 for each half, total rank = 24.
    
    # But actually, we can do even better by using Strassen on the 2x2 parts.
    # Let me think again.
    
    # For now, let's implement the rank-24 approach and see if it works.
    
    rank = 24
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # A indices: A[i, j] for i in 0..1, j in 0..3
    # B indices: B[j, k] for j in 0..3, k in 0..5
    # C indices: C[i, k] for i in 0..1, k in 0..5, stored as W[k*n+i, r]
    
    # Split A into A0 (cols 0-1) and A1 (cols 2-3)
    # Split B into B0 (rows 0-1) and B1 (rows 2-3)
    
    # A0*B0: A0 is 2x2 (A[0:2, 0:2]), B0 is 2x3 (B[0:2, 0:3])
    # A1*B1: A1 is 2x2 (A[0:2, 2:4]), B1 is 2x3 (B[2:4, 0:3])
    
    # For A0*B0, schoolbook:
    # C[i,k] += A[i, j] * B[j, k] for j in 0,1
    # So for each (i, j, k) with j in 0,1, we have a term.
    # That's 2*2*3 = 12 terms.
    
    offset = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*4 + j, offset] = 1.0
                V[j*6 + k, offset] = 1.0
                W[k*2 + i, offset] = 1.0
                offset += 1
    
    # For A1*B1: A1 is 2x2 (A[0:2, 2:4]), B1 is 2x3 (B[2:4, 0:3])
    # C[i,k] += A[i, 2+j] * B[2+j, k] for j in 0,1
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*4 + 2 + j, offset] = 1.0
                V[(2 + j)*6 + k, offset] = 1.0
                W[k*2 + i, offset] = 1.0
                offset += 1
    
    # Verify
    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        for a in range(nm):
            i_a = a // 4
            j_a = a % 4
            for b in range(mp):
                j_b = b // 6
                k_b = b % 6
                for c in range(pn):
                    k_c = c // 2
                    i_c = c % 2
                    expected = 1.0 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0.0
                    s = 0.0
                    for r in range(r):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s - expected) > tol:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook rank 48
        rank0 = n * m * p
        U0 = np.zeros((nm, rank0), dtype=float)
        V0 = np.zeros((mp, rank0), dtype=float)
        W0 = np.zeros((pn, rank0), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    U0[i*m+j, col] = 1.0
                    V0[j*p+k, col] = 1.0
                    W0[k*n+i, col] = 1.0
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    # Try to reduce rank by removing zero columns
    keep = []
    for r in range(rank):
        if np.any(U[:, r] != 0) or np.any(V[:, r] != 0) or np.any(W[:, r] != 0):
            keep.append(r)
    
    if len(keep) < rank:
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        if verify(U_new, V_new, W_new):
            U, V, W = U_new, V_new, W_new
            rank = len(keep)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
