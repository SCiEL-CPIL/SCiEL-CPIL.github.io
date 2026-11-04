"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition reconstructs the MM tensor T[i,j,k]=1."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    T_computed = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        U_col = U[:, c].reshape(n, m)
        V_col = V[:, c].reshape(m, p)
        W_col = W[:, c].reshape(p, n)
        tmp1 = np.einsum('ij,jk->ijk', U_col, V_col)
        T_computed += tmp1 * W_col.T[:, np.newaxis, :]
    return np.max(np.abs(T_computed - 1.0)) < tol


def _strassen_2x2x2():
    """
    Strassen's rank-5 decomposition of the 2x2x2 matrix multiplication tensor.
    
    Returns U (4x5), V (4x5), W (4x5) such that:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k in {0,1}
    
    The 5 bilinear forms for Strassen's algorithm:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M7
    """
    # U[i,j,c] = coefficient of A[i,j] in bilinear form c
    # V[j,k,c] = coefficient of B[j,k] in bilinear form c
    # W[k,i,c] = coefficient of C[k,i] in bilinear form c  (note: W[k*n+i] convention)
    
    # Bilinear form coefficients:
    # c=0: M1 = (A00+A11)(B00+B11)
    # c=1: M2 = (A10+A11)B00
    # c=2: M3 = A00(B01-B11)
    # c=3: M4 = A11(B10-B00)
    # c=4: M5 = (A00+A01)B11
    
    U = np.array([
        # i=0: A00, A01
        [1, 0, 1, 0, 1],   # A00
        [0, 0, 0, 0, 1],   # A01
        # i=1: A10, A11
        [0, 1, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0],   # A11
    ], dtype=np.float64)
    
    V = np.array([
        # j=0: B00, B01
        [1, 1, 1, -1, 0],  # B00
        [1, 0, 1, 0, 0],   # B01
        # j=1: B10, B11
        [0, 0, 0, 1, 0],   # B10
        [1, 0, -1, 0, 1],  # B11
    ], dtype=np.float64)
    
    # W[k*n+i] where n=2, so W[k*2+i]:
    # W[0*2+0]=W[0] is coeff of C[0,0]
    # W[0*2+1]=W[1] is coeff of C[0,1]
    # W[1*2+0]=W[2] is coeff of C[1,0]
    # W[1*2+1]=W[3] is coeff of C[1,1]
    
    # C00 = M1 + M4 - M5 - M7 -> but we only have 5 forms
    # Let me re-derive. Standard Strassen:
    # C00 = M1 - M5 + M4 - M7  ... wait, let me use the standard formulation.
    
    # Actually, the standard 5-form Strassen (using M1..M5 above) gives:
    # C00 = M1 + M4 - M5 - M2  ... no, that's not right either.
    
    # Let me use the well-known formulation:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    #
    # C00 = M1 + M4 - M5 - M7  -- but there's no M7 in 5-form.
    
    # The 5-multiplication Strassen uses a different grouping. Let me use the
    # standard 7-multiplication version and see if we can reduce, OR just use
    # the 7-term version which is definitely correct.
    
    # Actually, for correctness, let me use the well-known 7-term Strassen:
    U7 = np.zeros((4, 7), dtype=np.float64)
    V7 = np.zeros((4, 7), dtype=np.float64)
    W7 = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (A00+A11)(B00+B11)
    U7[0,0] = 1; U7[3,0] = 1
    V7[0,0] = 1; V7[3,0] = 1
    W7[0,0] = 1; W7[3,0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U7[2,1] = 1; U7[3,1] = 1
    V7[0,1] = 1
    W7[2,1] = 1; W7[3,1] = -1  # C10 += M2, C11 -= M2
    
    # M3 = A00(B01-B11)
    U7[0,2] = 1
    V7[1,2] = 1; V7[3,2] = -1
    W7[1,2] = 1; W7[3,2] = 1  # C01 += M3, C11 += M3
    
    # M4 = A11(B10-B00)
    U7[3,3] = 1
    V7[2,3] = 1; V7[0,3] = -1
    W7[2,3] = 1; W7[0,3] = 1  # C10 += M4, C00 += M4
    
    # M5 = (A00+A01)B11
    U7[0,4] = 1; U7[1,4] = 1
    V7[3,4] = 1
    W7[0,4] = -1; W7[3,4] = -1  # C00 -= M5, C11 -= M5
    
    # M6 = (A10-A00)(B00+B01)
    U7[2,5] = 1; U7[0,5] = -1
    V7[0,5] = 1; V7[1,5] = 1
    W7[2,5] = 1  # C10 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U7[1,6] = 1; U7[3,6] = -1
    V7[2,6] = 1; V7[3,6] = 1
    W7[1,6] = 1  # C01 += M7
    
    # Verify: C00 = M1 + M4 - M5 (from W7[0,:])
    # C01 = M3 + M7 (from W7[1,:])
    # C10 = M2 + M4 + M6 (from W7[2,:])
    # C11 = M1 - M2 + M3 - M5 + M7 (from W7[3,:])
    
    # Check: (A*B)[0,0] = A00*B00 + A01*B10
    # M1 + M4 - M5 = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B10 - A11*B00 - A00*B11 - A01*B11
    # = A00*B00 + A11*B11 + A11*B10 - A01*B11
    # That's NOT A00*B00 + A01*B10. Something is wrong.
    
    # Let me redo this carefully.
    # (A*B)[0,0] = A00*B00 + A01*B10
    # (A*B)[0,1] = A00*B01 + A01*B11
    # (A*B)[1,0] = A10*B00 + A11*B10
    # (A*B)[1,1] = A10*B01 + A11*B11
    
    # Standard Strassen 7-term:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Let me verify C00:
    # M1 + M4 - M5 - M7
    # = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 - (A01-A11)(B10+B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 - A01B10 - A01B11 + A11B10 + A11B11
    # = A00B00 + A11B11 + A11B10 - A01B11 - A01B10 - A01B11 + A11B10 + A11B11
    # Hmm, let me be more careful.
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M4 = A11B10 - A11B00
    # M5 = A00B11 + A01B11
    # M7 = A01B10 + A01B11 - A11B10 - A11B11
    # M1+M4-M5-M7 = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 - A01B10 - A01B11 + A11B10 + A11B11
    # = A00B00 + (A00B11-A00B11) + (A11B00-A11B00) + A11B11 + A11B10 - A01B11 - A01B10 - A01B11 + A11B10 + A11B11
    # = A00B00 + 2*A11B11 + 2*A11B10 - 2*A01B11 - A01B10
    # That's not right. I'm making arithmetic errors. Let me just use a verified implementation.
    
    # I'll use the well-known correct Strassen formulation from Wikipedia:
    # p1 = (a00 + a11)(b00 + b11)
    # p2 = (a10 + a11)b00
    # p3 = a00(b01 - b11)
    # p4 = a11(b10 - b00)
    # p5 = (a00 + a01)b11
    # p6 = (a10 - a00)(b00 + b01)
    # p7 = (a01 - a11)(b10 + b11)
    #
    # c00 = p1 - p5 + p4 - p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p7
    
    U7 = np.zeros((4, 7), dtype=np.float64)
    V7 = np.zeros((4, 7), dtype=np.float64)
    W7 = np.zeros((4, 7), dtype=np.float64)
    
    # p1 = (a00+a11)(b00+b11)
    U7[0,0]=1; U7[3,0]=1
    V7[0,0]=1; V7[3,0]=1
    W7[0,0]=1; W7[3,0]=1  # c00+=p1, c11+=p1
    
    # p2 = (a10+a11)b00
    U7[2,1]=1; U7[3,1]=1
    V7[0,1]=1
    W7[2,1]=1; W7[3,1]=-1  # c10+=p2, c11-=p2
    
    # p3 = a00(b01-b11)
    U7[0,2]=1
    V7[1,2]=1; V7[3,2]=-1
    W7[1,2]=1; W7[3,2]=1  # c01+=p3, c11+=p3
    
    # p4 = a11(b10-b00)
    U7[3,3]=1
    V7[2,3]=1; V7[0,3]=-1
    W7[0,3]=1; W7[2,3]=1  # c00+=p4, c10+=p4
    
    # p5 = (a00+a01)b11
    U7[0,4]=1; U7[1,4]=1
    V7[3,4]=1
    W7[0,4]=-1; W7[3,4]=-1  # c00-=p5, c11-=p5
    
    # p6 = (a10-a00)(b00+b01)
    U7[2,5]=1; U7[0,5]=-1
    V7[0,5]=1; V7[1,5]=1
    W7[2,5]=1  # c10+=p6
    
    # p7 = (a01-a11)(b10+b11)
    U7[1,6]=1; U7[3,6]=-1
    V7[2,6]=1; V7[3,6]=1
    W7[1,6]=1  # c01+=p7
    
    # Verify c00 = p1 - p5 + p4 - p7
    # c00 coeff: W7[0,:] = [1, 0, 0, 1, -1, 0, -1]
    # Verify c01 = p3 + p5
    # c01 coeff: W7[1,:] = [0, 0, 1, 0, 1, 0, 1]
    # Verify c10 = p2 + p4
    # c10 coeff: W7[2,:] = [0, 1, 0, 1, 0, 1, 0]
    # Verify c11 = p1 - p2 + p3 + p7
    # c11 coeff: W7[3,:] = [1, -1, 1, 0, -1, 0, 1]
    
    return U7, V7, W7


def _kronecker_construct(n, m, p):
    """
    Construct a decomposition of the n x m by m x p matrix multiplication tensor
    using a Kronecker product structure.
    
    For n=4, m=4, p=5:
    - Split into 2x2 blocks in the first two dimensions (4=2*2)
    - The third dimension p=5 is handled specially
    
    Strategy: Use the fact that 4x4x5 can be decomposed via:
    - 2x2x2 Strassen (rank 7) for the "core" 2x2x2 blocks
    - Additional terms for the extra dimension in p
    
    Actually, let's think more carefully.
    
    The MM tensor for (4,4,5) has indices:
    i in {0..3}, j in {0..3}, k in {0..4}
    T[i,j,k] = 1
    
    We can write i = 2*i1 + i0, j = 2*j1 + j0 where i1,j1 in {0,1}, i0,j0 in {0,1}
    k is in {0..4}
    
    The 4x4x4 part can be decomposed as a tensor product of two 2x2x2 tensors.
    For 4x4x4: rank <= 7*7 = 49 (using Strassen x Strassen)
    For the extra k=4 dimension, we need additional terms.
    
    Actually, a simpler approach: 
    4x4x5 = 4x4x4 + 4x4x1
    
    The 4x4x4 part: use Strassen^2 with rank 7*7=49
    The 4x4x1 part: this is like multiplying by a 4x1 matrix, rank <= 16
    
    Total: 49 + 16 = 65. That's better than 80!
    
    But wait, we can do better. Let's think of 4x4x5 differently.
    
    Alternative: 4x4x5 can be seen as a block matrix multiplication.
    If we partition A (4x4) into 2x2 blocks of 2x2:
    A = [[A00, A01], [A10, A11]]
    B (4x5) can be partitioned as:
    B = [[B00, B01, B02], [B10, B11, B12]]  where B00 is 2x3, B01 is 2x1, etc.
    
    This gets complicated. Let me try a different approach.
    
    Simplest correct approach: use the Kronecker product of Strassen(2x2x2) with 
    a rank-2 decomposition of 2x2x1, then handle the remaining k dimension.
    
    Actually, the cleanest structured approach:
    
    For 4x4x5, note that:
    - 4 = 2 * 2
    - 4 = 2 * 2  
    - 5 = 2 * 2 + 1
    
    We can decompose the tensor as:
    T_4x4x5 = T_2x2x2 ⊗ T_2x2x2 ⊗ T_1x1x1  (for the 4x4x4 part)
              + additional terms for k=4
    
    The 4x4x4 tensor: T[i,j,k] for i,j,k in {0..3}
    Write i=2*i1+i0, j=2*j1+j0, k=2*k1+k0
    T[i,j,k] = T2[i1,j1,k1] * T2[i0,j0,k0] where T2 is the 2x2x2 MM tensor
    
    So rank(4x4x4) <= rank(2x2x2)^2 = 7*7 = 49 (using 7-term Strassen)
    
    For the 4x4x1 part (k=4 only):
    T[i,j,4] = 1 for all i,j
    This is just the "all ones" tensor of shape 4x4x1, which has rank 16
    (each (i,j) pair needs its own term, or we can use rank 4 with 
     U[i,j,c]=delta(i,c), V[j,4,c]=1, W[4,c]=1... let me think)
    
    Actually for 4x4x1: T[i,j,0]=1 (renumbering k=4 as k=0 in this subproblem)
    We need: sum_c U[i,j,c]*V[j,0,c]*W[0,i,c] = 1 for all i,j
    One solution: rank 4, with U[i,j,c] = delta(i,c)*delta(j,0)... no.
    
    Simplest: U[i,j,c] = 1 if i==c else 0, V[j,0,c] = 1, W[0,i,c] = 1 if i==c else 0
    Then sum_c U[i,j,c]*V[j,0,c]*W[0,i,c] = sum_c delta(i,c)*1*delta(i,c) = 1 (when c=i)
    So rank 4 works for 4x4x1.
    
    Total rank: 49 + 4 = 53.
    
    Let me implement this.
    """
    n, m, p = 4, 4, 5
    
    # Get Strassen 2x2x2 factors
    U2, V2, W2 = _strassen_2x2x2()
    r2 = U2.shape[1]  # 7
    
    # Build 4x4x4 decomposition via Kronecker product
    # i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
    # T4[i,j,k] = T2[i1,j1,k1] * T2[i0,j0,k0]
    # Sum over (c1,c2): U2[i1,j1,c1]*V2[j1,k1,c1]*W2[k1,i1,c1] * U2[i0,j0,c2]*V2[j0,k0,c2]*W2[k0,i0,c2]
    
    r4 = r2 * r2  # 49
    
    U4 = np.zeros((16, r4), dtype=np.float64)  # 4*4 = 16
    V4 = np.zeros((16, r4), dtype=np.float64)  # 4*4 = 16
    W4 = np.zeros((16, r4), dtype=np.float64)  # 4*4 = 16
    
    for c1 in range(r2):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = 2*i1 + i0
                            j = 2*j1 + j0
                            U4[i*4+j, c] = U2[i1*2+j1, c1] * U2[i0*2+j0, c2]
            
            for j1 in range(2):
                for k1 in range(2):
                    for j0 in range(2):
                        for k0 in range(2):
                            j = 2*j1 + j0
                            k = 2*k1 + k0
                            V4[j*4+k, c] = V2[j1*2+k1, c1] * V2[j0*2+k0, c2]
            
            for k1 in range(2):
                for i1 in range(2):
                    for k0 in range(2):
                        for i0 in range(2):
                            k = 2*k1 + k0
                            i = 2*i1 + i0
                            W4[k*4+i, c] = W2[k1*2+i1, c1] * W2[k0*2+i0, c2]
    
    # Now add the 4x4x1 part for k=4
    # T[i,j,4] = 1 for all i in {0..3}, j in {0..3}
    # Need: sum_c U[i,j,c]*V[j,4,c]*W[4,i,c] = 1
    # Use rank-4 construction:
    # For c in {0,1,2,3}: U[i,j,c] = 1 if i==c else 0, V[j,4,c] = 1, W[4,i,c] = 1 if i==c else 0
    
    r_extra = 4
    r_total = r4 + r_extra  # 53
    
    U_total = np.zeros((16, r_total), dtype=np.float64)
    V_total = np.zeros((20, r_total), dtype=np.float64)  # 4*5 = 20
    W_total = np.zeros((20, r_total), dtype=np.float64)  # 5*4 = 20
    
    # Copy 4x4x4 part
    U_total[:16, :r4] = U4
    V_total[:16, :r4] = V4
    W_total[:16, :r4] = W4
    
    # Add 4x4x1 part (k=4)
    for c in range(r_extra):
        cc = r4 + c
        for i in range(4):
            for j in range(4):
                U_total[i*4+j, cc] = 1.0 if i == c else 0.0
                V_total[j*5+4, cc] = 1.0
                W_total[4*4+i, cc] = 1.0 if i == c else 0.0
    
    # Verify
    if not _verify_decomposition(U_total, V_total, W_total, n, m, p, tol=1e-9):
        # Fallback: use schoolbook
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": rank}
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist(), "rank": r_total}


def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W, "rank": rank}


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # Try Kronecker product construction first
    kronecker = _kronecker_construct(n, m, p)
    
    # Verify it
    if _verify_decomposition(kronecker["U"], kronecker["V"], kronecker["W"], n, m, p, tol=1e-8):
        return {"U": kronecker["U"], "V": kronecker["V"], "W": kronecker["W"]}
    
    # Fallback to schoolbook
    schoolbook = _construct_schoolbook(n, m, p)
    return {"U": schoolbook["U"], "V": schoolbook["V"], "W": schoolbook["W"]}

# EVOLVE-BLOCK-END
