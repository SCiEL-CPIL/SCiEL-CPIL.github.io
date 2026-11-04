"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    
    # Strategy: Use the rank-23 Strassen decomposition for 3x3 matrices.
    # Embed the 2x3 * 3x5 multiplication into the 3x3 * 3x3 multiplication.
    # This is possible by padding the matrices to 3x3.
    # Let A be 2x3, B be 3x5.
    # We want to compute C = A * B (2x5).
    
    # The standard 3x3 Strassen decomposition computes C_3x3 = A_3x3 * B_3x3.
    # We can define A_3x3 and B_3x3 such that the top-left 2x5 block of the result 
    # (or some mapping) corresponds to C.
    
    # However, B is 3x5, which is larger than 3x3.
    # We can split B into two parts: B1 (3x3) and B2 (3x2).
    # C = A * B1 + A * B2.
    
    # For A * B1:
    # Pad A to 3x3: A_pad = [[A], [0,0,0]]
    # B1 is 3x3.
    # Compute C1_pad = A_pad * B1 (3x3 * 3x3).
    # The top 2 rows of C1_pad are A * B1.
    # This uses rank 23.
    
    # For A * B2:
    # Pad A to 3x3: A_pad = [[A], [0,0,0]]
    # Pad B2 to 3x3: B2_pad = [[B2, 0], [0,0,0]]? No, B2 is 3x2.
    # B2_pad = [[b00, b01, 0], [b10, b11, 0], [b20, b21, 0]]
    # Compute C2_pad = A_pad * B2_pad (3x3 * 3x3).
    # The top 2 rows and left 2 columns of C2_pad are A * B2.
    # This uses rank 23.
    
    # Total rank = 23 + 23 = 46. This is worse than schoolbook (30).
    
    # Let's try a different mapping.
    # Can we map the 2x3 * 3x5 tensor directly into a 3x3 * 3x3 tensor?
    # The tensor T[i,j,k] for i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}.
    # The 3x3 tensor T3[i',j',k'] for i',j',k' in {0,1,2}.
    # The dimensions don't match directly.
    
    # However, there is a known technique: "embedding" or "restriction".
    # If the 2x3 * 3x5 tensor is a restriction of the 3x3 * 3x3 tensor, 
    # then rank(2x3*3x5) <= rank(3x3*3x3) = 23.
    # Is it a restriction?
    # A restriction means we fix some indices to 0 or identify them.
    # Here, we have fewer rows (2 vs 3) and more columns (5 vs 3).
    # So it's not a simple restriction.
    
    # Let's look at the "Inspiration 1" which scored 1.0714 (rank 28).
    # How to get rank 28?
    # Maybe by using a specific linear transformation?
    
    # Let's try to find a rank-28 decomposition explicitly.
    # One way is to use the schoolbook and see if any terms can be merged.
    # Or use a solver.
    
    # Given the time, let's implement a solver that searches for a rank-R decomposition
    # using a randomized approach with local optimization, similar to the current program
    # but perhaps with better initialization or a different objective.
    
    # Actually, let's try to construct a rank-28 decomposition manually if possible.
    # Or, let's try to use the fact that we can compute a 2x2 * 2x2 product with rank 7.
    # A (2x3) * B (3x5).
    # Split A into A1 (2x2) and A2 (2x1).
    # Split B into B1 (2x5) and B2 (1x5).
    # C = A1 * B1 + A2 * B2.
    # A1 * B1: 2x2 * 2x5.
    #   This is like computing 2x2 * 2x1 five times.
    #   Rank of 2x2 * 2x1 is 7 (Strassen for 2x2, but we only need half the output?).
    #   Actually, rank of 2x2 * 2x1 is 7? No, rank of 2x2 * 2x2 is 7.
    #   For 2x2 * 2x1, we can use the first 2 columns of the 2x2 * 2x2 Strassen.
    #   So rank <= 7.
    #   We have 5 columns, so we can compute A1 * B1 by doing 5 independent 2x2 * 2x1?
    #   No, the rows of B1 are shared.
    #   B1 is 2x5. B1 = [b00, b01, b02, b03, b04; b10, b11, b12, b13, b14].
    #   A1 * B1 = [a00 b00 + a01 b10, ...].
    #   This is a linear combination of the rows of B1 with coefficients from A1.
    #   The bilinear complexity of 2x2 * 2x5 is the same as 2x2 * 2x2? 
    #   No, it's larger.
    #   Actually, rank(2x2 * 2x5) <= 2 * rank(2x2 * 2x1) = 14?
    #   Or is it 7?
    #   If we can compute 2x2 * 2x1 with rank 7, then 2x2 * 2x5 can be done with rank 7?
    #   No, because the 5 outputs are distinct.
    #   Let's check: C[i,k] = sum_j A[i,j] B[j,k].
    #   For fixed i, C[i,:] is a linear combination of the rows of B.
    #   The coefficients are A[i,0] and A[i,1].
    #   So we need to compute 2 linear combinations of 5 vectors.
    #   This is trivial: C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    #   C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
    #   This takes 4 multiplications (A[0,0]*B[0,:], etc.) which are vector multiplications.
    #   In terms of scalar multiplications, it's 4 * 5 = 20.
    #   But we can do better?
    #   No, the rank of 2x2 * 2x5 is 14?
    #   Let's use Strassen for 2x2 * 2x2 to compute the first 2 columns, and schoolbook for the rest?
    #   C[:, 0:2] = A * B[:, 0:2]. Rank 7.
    #   C[:, 2:5] = A * B[:, 2:5]. Rank 2*2*3 = 12.
    #   Total 19.
    #   Plus A2 * B2: 2x1 * 1x5. Rank 2*1*5 = 10.
    #   Total 29.
    #   This is better than 30.
    
    # Let's refine this.
    # A = [A1, A2] where A1 is 2x2, A2 is 2x1.
    # B = [B1; B2] where B1 is 2x5, B2 is 1x5.
    # C = A1 * B1 + A2 * B2.
    
    # Compute A1 * B1:
    # B1 = [B1a, B1b] where B1a is 2x2, B1b is 2x3.
    # A1 * B1a: 2x2 * 2x2. Rank 7 (Strassen).
    # A1 * B1b: 2x2 * 2x3.
    #   Pad B1b to 2x4? Or 2x3?
    #   Rank of 2x2 * 2x3 is 12?
    #   Or can we do better?
    #   2x2 * 2x3 is like 2x2 * 2x2 + 2x2 * 2x1.
    #   Rank 7 + 7 = 14?
    #   Or 12 (schoolbook)?
    #   Let's assume 12 for now.
    #   So A1 * B1 rank <= 7 + 12 = 19.
    
    # Compute A2 * B2:
    # 2x1 * 1x5. Rank 10 (schoolbook).
    
    # Total rank <= 19 + 10 = 29.
    
    # Can we do better?
    # Maybe A1 * B1b can be done with rank 11?
    # Or A2 * B2 with rank 9?
    
    # Let's try to construct this rank-29 decomposition explicitly.
    
    # Strassen for 2x2 * 2x2:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # For A1 * B1a (2x2 * 2x2):
    # A1 = [[a00, a01], [a10, a11]]
    # B1a = [[b00, b01], [b10, b11]]
    # C1a = [[c00, c01], [c10, c11]]
    
    # For A1 * B1b (2x2 * 2x3):
    # B1b = [[b02, b03, b04], [b12, b13, b14]]
    # We can treat this as 2x2 * 2x2 + 2x2 * 2x1?
    # Or just schoolbook: 4 multiplications per column, 3 columns -> 12 multiplications.
    # c02 = a00*b02 + a01*b12
    # c03 = a00*b03 + a01*b13
    # c04 = a00*b04 + a01*b14
    # c12 = a10*b02 + a11*b12
    # c13 = a10*b03 + a11*b13
    # c14 = a10*b04 + a11*b14
    
    # For A2 * B2 (2x1 * 1x5):
    # A2 = [[a02], [a12]]
    # B2 = [[b20, b21, b22, b23, b24]]
    # C2 = [[a02*b20, a02*b21, a02*b22, a02*b23, a02*b24],
    #       [a12*b20, a12*b21, a12*b22, a12*b23, a12*b24]]
    # This is 10 multiplications.
    
    # Total multiplications: 7 + 12 + 10 = 29.
    
    # Now we need to construct the U, V, W matrices for this decomposition.
    # The decomposition is a sum of rank-1 tensors.
    # Each multiplication corresponds to a rank-1 term.
    
    # Let's list the terms:
    # 1. M1: (a00 + a11)(b00 + b11)
    #    U[0,0]=1, U[1,0]=1 (for a00, a11)
    #    V[0,0]=1, V[1,0]=1 (for b00, b11)
    #    W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1, W[4,0]=1? No.
    #    M1 contributes to c00 and c11.
    #    c00 = M1 + ...
    #    c11 = M1 + ...
    #    So W[0,0]=1 (for c00), W[1,0]=1 (for c11).
    #    Wait, the index for W is k*n+i.
    #    c00: i=0, k=0 -> W[0,0]
    #    c11: i=1, k=1 -> W[1*2+1, 0] = W[3,0]
    
    # This is getting complicated. Let's use a helper to build the U, V, W matrices.
    
    # We will define the terms as (U_row_indices, V_row_indices, W_row_indices, coefficient).
    # But since we are using half-integers, we can just add the contributions.
    
    # Let's create empty U, V, W matrices of size (6, 29), (15, 29), (10, 29).
    # And fill them in.
    
    rank = 29
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        nonlocal rank
        U[u_idx, rank-1] += val
        V[v_idx, rank-1] += val
        W[w_idx, rank-1] += val
        rank -= 1
    
    # This approach is messy because we need to track the rank index.
    # Let's just build the matrices directly.
    
    U = np.zeros((6, 29))
    V = np.zeros((15, 29))
    W = np.zeros((10, 29))
    
    # Indices:
    # U: i*m+j -> i*3+j
    # V: j*p+k -> j*5+k
    # W: k*n+i -> k*2+i
    
    # Term 1: M1 = (a00 + a11)(b00 + b11)
    # a00: U[0, 0] = 1
    # a11: U[1, 0] = 1
    # b00: V[0, 0] = 1
    # b11: V[1*5+1, 0] = V[6, 0] = 1
    # c00: W[0, 0] = 1
    # c11: W[1*2+1, 0] = W[3, 0] = 1
    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1; V[6, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: M2 = (a10 + a11)b00
    # a10: U[3, 1] = 1
    # a11: U[4, 1] = 1
    # b00: V[0, 1] = 1
    # c10: W[0*2+1, 1] = W[1, 1] = 1
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # Term 3: M3 = a00(b01 - b11)
    # a00: U[0, 2] = 1
    # b01: V[1, 2] = 1
    # b11: V[6, 2] = -1
    # c01: W[1*2+0, 2] = W[2, 2] = 1
    # c11: W[3, 2] = -1
    U[0, 2] = 1
    V[1, 2] = 1; V[6, 2] = -1
    W[2, 2] = 1; W[3, 2] = -1
    
    # Term 4: M4 = a11(b10 - b00)
    # a11: U[1, 3] = 1
    # b10: V[5, 3] = 1
    # b00: V[0, 3] = -1
    # c00: W[0, 3] = 1
    # c10: W[1, 3] = 1
    U[1, 3] = 1
    V[5, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 5: M5 = (a00 + a01)b11
    # a00: U[0, 4] = 1
    # a01: U[1, 4] = 1
    # b11: V[6, 4] = 1
    # c00: W[0, 4] = -1
    # c01: W[2, 4] = 1
    U[0, 4] = 1; U[1, 4] = 1
    V[6, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Term 6: M6 = (a10 - a00)(b00 + b01)
    # a10: U[3, 5] = 1
    # a00: U[0, 5] = -1
    # b00: V[0, 5] = 1
    # b01: V[1, 5] = 1
    # c11: W[3, 5] = 1
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7: M7 = (a01 - a11)(b10 + b11)
    # a01: U[1, 6] = 1
    # a11: U[4, 6] = -1
    # b10: V[5, 6] = 1
    # b11: V[6, 6] = 1
    # c00: W[0, 6] = 1
    U[1, 6] = 1; U[4, 6] = -1
    V[5, 6] = 1; V[6, 6] = 1
    W[0, 6] = 1
    
    # Terms 8-19: Schoolbook for A1 * B1b (2x2 * 2x3)
    # c02 = a00*b02 + a01*b12
    # c03 = a00*b03 + a01*b13
    # c04 = a00*b04 + a01*b14
    # c12 = a10*b02 + a11*b12
    # c13 = a10*b03 + a11*b13
    # c14 = a10*b04 + a11*b14
    
    # Term 8: a00*b02
    U[0, 7] = 1
    V[2, 7] = 1
    W[4, 7] = 1 # c02: k=2, i=0 -> 2*2+0=4
    
    # Term 9: a01*b12
    U[1, 8] = 1
    V[7, 8] = 1 # b12: j=1, k=2 -> 1*5+2=7
    W[4, 8] = 1
    
    # Term 10: a00*b03
    U[0, 9] = 1
    V[3, 9] = 1
    W[5, 9] = 1 # c03: k=3, i=0 -> 3*2+0=5
    
    # Term 11: a01*b13
    U[1, 10] = 1
    V[8, 10] = 1
    W[5, 10] = 1
    
    # Term 12: a00*b04
    U[0, 11] = 1
    V[4, 11] = 1
    W[6, 11] = 1 # c04: k=4, i=0 -> 4*2+0=6
    
    # Term 13: a01*b14
    U[1, 12] = 1
    V[9, 12] = 1
    W[6, 12] = 1
    
    # Term 14: a10*b02
    U[3, 13] = 1
    V[2, 13] = 1
    W[4, 13] = 1 # c12: k=2, i=1 -> 2*2+1=5? No, c12 is i=1, k=2. W[2*2+1, 13]=W[5,13]
    W[5, 13] = 1
    
    # Term 15: a11*b12
    U[4, 14] = 1
    V[7, 14] = 1
    W[5, 14] = 1
    
    # Term 16: a10*b03
    U[3, 15] = 1
    V[3, 15] = 1
    W[6, 15] = 1 # c13: k=3, i=1 -> 3*2+1=7? No, W[3*2+1, 15]=W[7,15]
    W[7, 15] = 1
    
    # Term 17: a11*b13
    U[4, 16] = 1
    V[8, 16] = 1
    W[7, 16] = 1
    
    # Term 18: a10*b04
    U[3, 17] = 1
    V[4, 17] = 1
    W[8, 17] = 1 # c14: k=4, i=1 -> 4*2+1=9? No, W[4*2+1, 17]=W[9,17]
    W[9, 17] = 1
    
    # Term 19: a11*b14
    U[4, 18] = 1
    V[9, 18] = 1
    W[9, 18] = 1
    
    # Terms 20-29: Schoolbook for A2 * B2 (2x1 * 1x5)
    # C2 = [[a02*b20, a02*b21, a02*b22, a02*b23, a02*b24],
    #       [a12*b20, a12*b21, a12*b22, a12*b23, a12*b24]]
    
    # Term 20: a02*b20
    U[5, 19] = 1 # a02: i=0, j=2 -> 0*3+2=2? No, U index is i*m+j. A2 is the third column of A.
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # So a02 is A[0,2] -> U[0*3+2, 19] = U[2, 19]
    # a12 is A[1,2] -> U[1*3+2, 19] = U[5, 19]
    # B2 is the third row of B.
    # B = [[b00, b01, b02, b03, b04], [b10, b11, b12, b13, b14], [b20, b21, b22, b23, b24]]
    # So b20 is B[2,0] -> V[2*5+0, 19] = V[10, 19]
    
    U[2, 19] = 1
    V[10, 19] = 1
    W[0, 19] = 1 # c00: k=0, i=0 -> 0
    
    # Term 21: a02*b21
    U[2, 20] = 1
    V[11, 20] = 1
    W[1, 20] = 1 # c01: k=1, i=0 -> 1
    
    # Term 22: a02*b22
    U[2, 21] = 1
    V[12, 21] = 1
    W[2, 21] = 1 # c02: k=2, i=0 -> 4
    
    # Term 23: a02*b23
    U[2, 22] = 1
    V[13, 22] = 1
    W[3, 22] = 1 # c03: k=3, i=0 -> 5
    
    # Term 24: a02*b24
    U[2, 23] = 1
    V[14, 23] = 1
    W[4, 23] = 1 # c04: k=4, i=0 -> 6
    
    # Term 25: a12*b20
    U[5, 24] = 1
    V[10, 24] = 1
    W[5, 24] = 1 # c10: k=0, i=1 -> 1
    
    # Term 26: a12*b21
    U[5, 25] = 1
    V[11, 25] = 1
    W[6, 25] = 1 # c11: k=1, i=1 -> 3
    
    # Term 27: a12*b22
    U[5, 26] = 1
    V[12, 26] = 1
    W[7, 26] = 1 # c12: k=2, i=1 -> 5
    
    # Term 28: a12*b23
    U[5, 27] = 1
    V[13, 27] = 1
    W[8, 27] = 1 # c13: k=3, i=1 -> 7
    
    # Term 29: a12*b24
    U[5, 28] = 1
    V[14, 28] = 1
    W[9, 28] = 1 # c14: k=4, i=1 -> 9
    
    # Verify the decomposition
    # We need to check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    
    # Let's do a quick check in code.
    T_recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(29):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_recon[i,j,k] = val
    
    # Check if it's all 1s
    if not np.allclose(T_recon, np.ones((n,m,p)), atol=1e-9):
        # Fallback to schoolbook if the manual construction is wrong
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
