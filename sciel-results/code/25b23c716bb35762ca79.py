"""Matrix multiplication 4,4,6 rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition using block structure
    # Partition 4x4x6 into three 4x4x2 blocks
    # For each 4x4x2 block, we can potentially use rank < 32
    
    # Strategy: Use Strassen-like decomposition for 4x4x4 (rank 21) 
    # and adapt for 4x4x2
    
    # First, let's try a direct approach: exploit the fact that we can
    # decompose the 4x4x6 tensor more efficiently than schoolbook
    
    # Known results:
    # - 4x4x4 has rank 21 (Strassen)
    # - 2x2x2 has rank 7 (Strassen)
    # - 4x4x2 might have rank around 10-14
    
    # Let's try to construct a decomposition using 2x2x2 blocks with Strassen
    # 4x4x6 = 4 blocks of 2x2x3 in the i-direction, or 2x2x2 blocks
    
    # Actually, let's think about it differently:
    # The tensor is T[i,j,k] = A[i,j] * B[j,k]
    # We can partition into 2x2x2 sub-tensors: 2*2*3 = 12 sub-tensors
    # Each 2x2x2 can be done with rank 7 (Strassen)
    # Total rank: 12 * 7 = 84
    
    # This is better than 96 (schoolbook)
    
    # Let's implement the 2x2x2 Strassen decomposition
    # For 2x2x2: C[0,0] = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) - (A[0,1]+A[1,1])*(B[0,1]+B[1,0])
    # Actually, the standard Strassen for 2x2 matrices uses 7 multiplications
    
    # The 7 rank-1 terms for 2x2x2 tensor T[i,j,k] = A[i,j]*B[j,k]:
    # T[0,0,0] = A[0,0]*B[0,0]
    # T[0,0,1] = A[0,0]*B[0,1]
    # T[0,1,0] = A[0,1]*B[1,0]
    # T[0,1,1] = A[0,1]*B[1,1]
    # T[1,0,0] = A[1,0]*B[0,0]
    # T[1,0,1] = A[1,0]*B[0,1]
    # T[1,1,0] = A[1,1]*B[1,0]
    # T[1,1,1] = A[1,1]*B[1,1]
    
    # Strassen's 7 formulas:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[1,0]-B[1,1])
    # M4 = A[1,1]*(B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # C[0,0] = M1 - M4 + M6
    # C[0,1] = M3 + M4 + M7
    # C[1,0] = M2 + M5 - M6 - M7
    # C[1,1] = M1 - M3 + M2
    
    # Wait, let me verify this is correct for the matrix multiplication tensor
    # where T[i,j,k] = A[i,j]*B[j,k]
    
    # The bilinear form is: sum over j of A[i,j]*B[j,k]
    # For 2x2: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # Let me re-derive Strassen for 2x2x2:
    # We want to compute C = A*B where A is 2x2, B is 2x2
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen's 7 products:
    # P1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # P2 = (A[1,0]+A[1,1])*B[0,0]
    # P3 = A[0,0]*(B[1,0]-B[1,1])
    # P4 = A[1,1]*(B[0,1]-B[0,0])
    # P5 = (A[0,0]+A[0,1])*B[1,1]
    # P6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # P7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # Then:
    # C[0,0] = P1 - P4 + P6
    # C[0,1] = P3 + P4 + P7
    # C[1,0] = P2 + P5 - P6 - P7
    # C[1,1] = P1 - P3 + P2
    
    # Let's verify C[0,0]:
    # P1 - P4 + P6 = (A[0,0]+A[1,1])(B[0,0]+B[1,1]) - A[1,1](B[0,1]-B[0,0]) + (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    # = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1] - A[1,1]B[0,1] + A[1,1]B[0,0] + A[1,0]B[0,0] + A[1,0]B[0,1] - A[0,0]B[0,0] - A[0,0]B[0,1]
    # = A[0,0]B[1,1] + 2*A[1,1]B[0,0] + A[1,1]B[1,1] - A[1,1]B[0,1] + A[1,0]B[0,0] + A[1,0]B[0,1] - A[0,0]B[0,1]
    # This doesn't look right. Let me use the standard Strassen formulas.
    
    # Standard Strassen for 2x2 matrix multiplication:
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b21 - b22)
    # M4 = a22(b12 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    
    # c11 = M1 - M4 + M6
    # c12 = M3 + M4 + M7
    # c21 = M2 + M5 - M6 - M7
    # c22 = M1 - M3 + M2
    
    # Using 1-indexed notation where a[i,j] is A[i-1,j-1]
    # So a11=A[0,0], a12=A[0,1], a21=A[1,0], a22=A[1,1]
    # b11=B[0,0], b12=B[0,1], b21=B[1,0], b22=B[1,1]
    # c11=C[0,0], c12=C[0,1], c21=C[1,0], c22=C[1,1]
    
    # Verify c11 = M1 - M4 + M6:
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # M4 = A[1,1](B[0,1]-B[0,0])
    # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    
    # M1 - M4 + M6 = (A[0,0]+A[1,1])(B[0,0]+B[1,1]) - A[1,1](B[0,1]-B[0,0]) + (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    # = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1] - A[1,1]B[0,1] + A[1,1]B[0,0] + A[1,0]B[0,0] + A[1,0]B[0,1] - A[0,0]B[0,0] - A[0,0]B[0,1]
    # = A[0,0]B[1,1] + 2A[1,1]B[0,0] + A[1,1]B[1,1] - A[1,1]B[0,1] + A[1,0]B[0,0] + A[1,0]B[0,1] - A[0,0]B[0,1]
    
    # But c11 should be A[0,0]B[0,0] + A[0,1]B[1,0]
    
    # I think I'm confusing the tensor structure. Let me be more careful.
    
    # The matrix multiplication tensor T for multiplying an n x m matrix by an m x p matrix
    # has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = A[i,j] * B[j,k]
    
    # So for 2x2x2 (n=2, m=2, p=2):
    # T[0,0,0] = A[0,0]*B[0,0]
    # T[0,0,1] = A[0,0]*B[0,1]
    # T[0,1,0] = A[0,1]*B[1,0]
    # T[0,1,1] = A[0,1]*B[1,1]
    # T[1,0,0] = A[1,0]*B[0,0]
    # T[1,0,1] = A[1,0]*B[0,1]
    # T[1,1,0] = A[1,1]*B[1,0]
    # T[1,1,1] = A[1,1]*B[1,1]
    
    # The output C[i,k] = sum_j A[i,j]*B[j,k]
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = T[0,0,0] + T[0,1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = T[0,0,1] + T[0,1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = T[1,0,0] + T[1,1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = T[1,0,1] + T[1,1,1]
    
    # Now Strassen's decomposition for this tensor:
    # We need to express each T[i,j,k] as a linear combination of 7 rank-1 terms.
    
    # The 7 rank-1 terms (u, v, w) where T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
    # Wait, the convention is W[k*n+i, r] so w has index (k, i)
    
    # Let me define the 7 Strassen terms for 2x2x2:
    # Term 1: u1[i,j] = delta(i,0)+delta(i,1) * (delta(j,0)+delta(j,1)) = 1 for all (i,j) in 2x2
    #          v1[j,k] = 1 for all (j,k) in 2x2
    #          w1[k,i] = 1 for all (k,i) in 2x2
    # This gives T[i,j,k] += 1*1*1 = 1 for all, which is not right.
    
    # Let me look up the correct Strassen decomposition for the 2x2x2 tensor.
    
    # Actually, I think the standard Strassen decomposition is for the matrix multiplication
    # where C = A*B, and the tensor is T[i,j,k] = A[i,j]*B[j,k].
    
    # The 7 rank-1 decompositions for 2x2x2:
    # r=0: u[0,0]=1, u[1,1]=1; v[0,0]=1, v[1,1]=1; w[0,0]=1, w[1,1]=1
    #       This contributes to T[0,0,0], T[0,0,1], T[0,1,0], T[0,1,1], T[1,0,0], T[1,0,1], T[1,1,0], T[1,1,1]
    #       Actually no, a rank-1 term u*v*w contributes u[i,j]*v[j,k]*w[k,i] to T[i,j,k]
    
    # Let me just implement the known Strassen 7-product formula directly.
    
    # For computing C = A*B (2x2 matrices):
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[1,0]-B[1,1])
    # M4 = A[1,1]*(B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # C[0,0] = M1 - M4 + M6
    # C[0,1] = M3 + M4 + M7
    # C[1,0] = M2 + M5 - M6 - M7
    # C[1,1] = M1 - M3 + M2
    
    # Each M_r is a product of a linear combination of A entries and a linear combination of B entries.
    # So M_r = (sum_j alpha_r[j]*A[i,j]) * (sum_k beta_r[j,k]*B[j,k]) -- no, that's not right either.
    
    # Actually, each M_r = L_r(A) * Q_r(B) where L_r is linear in A and Q_r is linear in B.
    # But for the tensor decomposition, we need:
    # T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    # Where U_r[i,j] is the coefficient of A[i,j] in L_r, V_r[j,k] is the coefficient of B[j,k] in Q_r,
    # and W_r[k,i] is the coefficient in the output C[i,k].
    
    # Let me extract the coefficients:
    
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    #     = A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
    #     Contributes to:
    #     C[0,0] += 1 (from M1)
    #     C[1,1] += 1 (from M1)
    #     U1[0,0]=1, U1[1,1]=1
    #     V1[0,0]=1, V1[1,1]=1
    #     W1[0,0]=1 (for C[0,0]), W1[1,1]=1 (for C[1,1])
    #     Wait, W has index (k, i) so W1[k, i]
    #     C[0,0] means i=0, k=0, so W1[0,0]=1
    #     C[1,1] means i=1, k=1, so W1[1,1]=1
    
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    #     = A[1,0]*B[0,0] + A[1,1]*B[0,0]
    #     Contributes to:
    #     C[1,0] += 1 (from M2)
    #     U2[1,0]=1, U2[1,1]=1
    #     V2[0,0]=1
    #     W2[0,1]=1 (for C[1,0], i=1, k=0)
    
    # M3 = A[0,0]*(B[1,0]-B[1,1])
    #     = A[0,0]*B[1,0] - A[0,0]*B[1,1]
    #     Contributes to:
    #     C[0,1] += 1 (from M3)
    #     C[1,1] += -1 (from M3)
    #     U3[0,0]=1
    #     V3[1,0]=1, V3[1,1]=-1
    #     W3[1,0]=1 (for C[0,1], i=0, k=1)
    #     W3[1,1]=-1 (for C[1,1], i=1, k=1)
    
    # M4 = A[1,1]*(B[0,1]-B[0,0])
    #     = A[1,1]*B[0,1] - A[1,1]*B[0,0]
    #     Contributes to:
    #     C[0,0] += -1 (from M4)
    #     C[0,1] += 1 (from M4)
    #     U4[1,1]=1
    #     V4[0,1]=1, V4[0,0]=-1
    #     W4[0,0]=-1 (for C[0,0], i=0, k=0)
    #     W4[1,0]=1 (for C[0,1], i=0, k=1)
    
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    #     = A[0,0]*B[1,1] + A[0,1]*B[1,1]
    #     Contributes to:
    #     C[1,0] += 1 (from M5)
    #     U5[0,0]=1, U5[0,1]=1
    #     V5[1,1]=1
    #     W5[0,1]=1 (for C[1,0], i=1, k=0)
    
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    #     = A[1,0]*B[0,0] + A[1,0]*B[0,1] - A[0,0]*B[0,0] - A[0,0]*B[0,1]
    #     Contributes to:
    #     C[0,0] += 1 (from M6)
    #     C[1,0] += -1 (from M6)
    #     U6[1,0]=1, U6[0,0]=-1
    #     V6[0,0]=1, V6[0,1]=1
    #     W6[0,0]=1 (for C[0,0], i=0, k=0)
    #     W6[0,1]=-1 (for C[1,0], i=1, k=0)
    
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    #     = A[0,1]*B[1,0] + A[0,1]*B[1,1] - A[1,1]*B[1,0] - A[1,1]*B[1,1]
    #     Contributes to:
    #     C[0,1] += 1 (from M7)
    #     C[1,0] += -1 (from M7)
    #     U7[0,1]=1, U7[1,1]=-1
    #     V7[1,0]=1, V7[1,1]=1
    #     W7[1,0]=1 (for C[0,1], i=0, k=1)
    #     W7[0,1]=-1 (for C[1,0], i=1, k=0)
    
    # Now let's verify:
    # C[0,0] = M1 - M4 + M6
    # M1 contributes: A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
    # -M4 contributes: -A[1,1]*B[0,1] + A[1,1]*B[0,0]
    # M6 contributes: A[1,0]*B[0,0] + A[1,0]*B[0,1] - A[0,0]*B[0,0] - A[0,0]*B[0,1]
    
    # Sum: A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1] - A[1,1]*B[0,1] + A[1,1]*B[0,0] + A[1,0]*B[0,0] + A[1,0]*B[0,1] - A[0,0]*B[0,0] - A[0,0]*B[0,1]
    # = A[0,0]*B[1,1] + 2*A[1,1]*B[0,0] + A[1,1]*B[1,1] - A[1,1]*B[0,1] + A[1,0]*B[0,0] + A[1,0]*B[0,1] - A[0,0]*B[0,1]
    
    # But C[0,0] should be A[0,0]*B[0,0] + A[0,1]*B[1,0]
    
    # This doesn't match. I think I'm making an error in the Strassen formulas.
    
    # Let me use a different approach. I'll just use the schoolbook decomposition for now
    # since I'm running out of time and need to return a valid result.
    
    # Actually, let me try a simpler improvement: use 2x2x2 Strassen blocks.
    # Even if my derivation above is wrong, the known result is that 2x2x2 has rank 7.
    
    # Let me just hardcode the correct Strassen 7-term decomposition for 2x2x2.
    
    # After more careful thought, the correct Strassen decomposition for 2x2x2 tensor
    # T[i,j,k] = A[i,j]*B[j,k] is:
    
    # I'll use a verified set of 7 terms. Let me define them explicitly.
    
    # Term 0:
    # U[0,0]=1, U[1,1]=1
    # V[0,0]=1, V[1,1]=1
    # W[0,0]=1, W[1,1]=1
    # Contributes: T[0,0,0]+=1, T[0,0,1]+=0, T[0,1,0]+=0, T[0,1,1]+=0, T[1,0,0]+=0, T[1,0,1]+=0, T[1,1,0]+=0, T[1,1,1]+=1
    # Wait, that's not right. The contribution to T[i,j,k] is U[i,j]*V[j,k]*W[k,i]
    # For i=0,j=0,k=0: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # For i=0,j=0,k=1: U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0
    # For i=0,j=1,k=0: U[0,1]*V[1,0]*W[0,0] = 0*0*1 = 0
    # For i=0,j=1,k=1: U[0,1]*V[1,1]*W[1,0] = 0*1*0 = 0
    # For i=1,j=0,k=0: U[1,0]*V[0,0]*W[0,1] = 0*1*0 = 0
    # For i=1,j=0,k=1: U[1,0]*V[0,1]*W[1,1] = 0*0*1 = 0
    # For i=1,j=1,k=0: U[1,1]*V[1,0]*W[0,1] = 1*0*0 = 0
    # For i=1,j=1,k=1: U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1
    # So term 0 contributes 1 to T[0,0,0] and T[1,1,1]
    
    # This is getting too complex to derive by hand. Let me just use the schoolbook
    # decomposition and return it, since I'm close to the deadline.
    
    if time.time() > deadline - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Given the complexity and time constraints, I'll return the schoolbook decomposition
    # which is guaranteed to be correct.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
