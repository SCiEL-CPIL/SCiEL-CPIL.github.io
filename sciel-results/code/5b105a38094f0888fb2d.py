"""Matrix multiplication 2,4,8 - Strassen-based rank-56 decomposition."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x8 matrix.
    
    Strategy: Use Strassen's algorithm structure.
    - Split the 2x4 * 4x8 multiplication into 8 independent 2x2 * 2x2 blocks.
    - Each 2x2 * 2x2 can be done with rank 7 (Strassen).
    - Total rank = 8 * 7 = 56, vs schoolbook 64.
    
    The 2x4 matrix A is split as [A1 A2] where A1, A2 are 2x2.
    The 4x8 matrix B is split as [B1; B2; B3; B4]... wait, that's not right.
    
    Let me reconsider. A is 2x4, B is 4x8.
    A = [[A11, A12], [A21, A22]] where each Ai is 2x2 (splitting columns into two groups of 2).
    B = [[B11, B12], [B21, B22]] where each Bj is 2x4... no.
    
    Actually, let's think of it differently.
    A is 2x4, B is 4x8.
    We can split B into two 2x8 blocks: B = [B_top; B_bottom] where B_top is 2x8 and B_bottom is 2x8.
    And A into two 2x2 blocks: A = [A_left | A_right] where A_left is 2x2 and A_right is 2x2.
    
    Then C = A * B = A_left * B_top + A_right * B_bottom.
    
    Now A_left * B_top is 2x2 * 2x8 = 2x8.
    A_right * B_bottom is 2x2 * 2x8 = 2x8.
    
    Each 2x2 * 2x8 can be viewed as 4 independent 2x2 * 2x2 multiplications (splitting the 8 columns into 4 groups of 2).
    
    So we have 8 total 2x2 * 2x2 multiplications.
    Each uses rank 7 via Strassen.
    Total rank = 8 * 7 = 56.
    
    For a 2x2 * 2x2 Strassen decomposition:
    Given matrices X (2x2) and Y (2x2), compute Z = X*Y.
    X = [[x11, x12], [x21, x22]]
    Y = [[y11, y12], [y21, y22]]
    
    Strassen's 7 products:
    M1 = (x11 + x22) * (y11 + y22)
    M2 = (x21 + x22) * y11
    M3 = x11 * (y12 - y22)
    M4 = x22 * (y21 - y11)
    M5 = (x11 + x12) * y22
    M6 = (x21 - x11) * (y11 + y12)
    M7 = (x12 - x22) * (y21 + y22)
    
    Then:
    z11 = M1 + M4 - M5 + M7
    z12 = M3 + M5
    z21 = M2 + M4
    z22 = M1 - M2 + M3 + M6
    
    For tensor decomposition, we need U, V, W such that:
    For each output element z_ij, z_ij = sum_c U[i,j,c] * V[c] * W[c]
    
    Actually, the tensor T[i,j,k,l] = delta_{ik} * delta_{jl} for matrix mult (i,j) from X, (k,l) from Y.
    
    The decomposition gives: T = sum_c u_c ⊗ v_c ⊗ w_c
    where u_c is a vector over (i,j) indices, v_c over (k,l) indices... 
    
    Wait, let me re-read the interface. U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    For n=2, m=4, p=8:
    U: 8 x rank
    V: 32 x rank  
    W: 16 x rank
    
    The tensor is T[i, j, k] where i in [0,n), j in [0,m), k in [0,p).
    T[i,j,k] = 1 if the (i,k) element of C = A*B depends on A[i,j] and B[j,k].
    
    So T[i,j,k] = 1 always (for valid i,j,k), and C[i,k] = sum_j A[i,j] * B[j,k].
    
    The decomposition: C[i,k] = sum_c U[i*m+j... 
    Hmm, let me re-read. U has shape (n*m, rank). So U is indexed by (i,j) pairs where i in [0,n), j in [0,m).
    V has shape (m*p, rank). So V is indexed by (j,k) pairs where j in [0,m), k in [0,p).
    W has shape (p*n, rank). So W is indexed by (k,i) pairs where k in [0,p), i in [0,n).
    
    The decomposition says: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    This gives C[i,k] = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c].
    And C[i,k] should equal sum_j A[i,j]*B[j,k].
    
    So for each c, we have a term: (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]... no that's not right either.
    
    Actually the tensor decomposition means: the multiplication tensor T has T[i,j,k] = 1 for all valid i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k]).
    
    Wait no. T[i,j,k] represents the coefficient such that C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k].
    Since C[i,k] = sum_j A[i,j]*B[j,k], we have T[i,j,k] = 1 for all i,j,k.
    
    The rank-r decomposition: T[i,j,k] = sum_{c=1}^r U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    
    So we need: 1 = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] for all i,j,k.
    
    Let me use the Strassen approach. For each of the 8 block multiplications, I'll assign 7 columns to U, V, W.
    
    Let me define the blocks:
    A is 2x4. Split columns: A = [A0 | A1] where A0 is cols 0-1 (2x2), A1 is cols 2-3 (2x2).
    B is 4x8. Split rows: B = [B0; B1] where B0 is rows 0-1 (2x8), B1 is rows 2-3 (2x8).
    
    C = A0*B0 + A1*B1.
    
    Now A0*B0: A0 is 2x2, B0 is 2x8. Split B0 into 4 blocks of 2x2: B0 = [B00 | B01 | B02 | B03].
    Similarly A1*B1: A1 is 2x2, B1 is 2x8. Split B1 into 4 blocks: B1 = [B10 | B11 | B12 | B13].
    
    So we have 8 sub-multiplications: A0*B0b for b=0..3, and A1*B1b for b=0..3.
    
    For each sub-multiplication of a 2x2 matrix X by a 2x2 matrix Y, producing 2x2 Z:
    The relevant indices in the full problem:
    - X corresponds to A[i, j] for specific i, j
    - Y corresponds to B[j, k] for specific j, k  
    - Z corresponds to C[i, k] for specific i, k
    
    Let me be more concrete.
    
    Block (block_a, block_b) where block_a in {0,1} (which 2x2 block of A) and block_b in {0,1,2,3} (which 2x2 column block of the corresponding B half).
    
    For block_a=0 (A0, cols 0-1 of A):
      A0[i, j_local] = A[i, j_local] for i in 0,1 and j_local in 0,1 (j_local=0 -> j=0, j_local=1 -> j=1)
    
    For block_a=1 (A1, cols 2-3 of A):
      A1[i, j_local] = A[i, 2+j_local] for i in 0,1 and j_local in 0,1
    
    For block_a=0, the B half is B0 (rows 0-1):
      B0[j_local, k_local] = B[j_local, 2*block_b + k_local] for j_local in 0,1 and k_local in 0,1
    
    For block_a=1, the B half is B1 (rows 2-3):
      B1[j_local, k_local] = B[2+j_local, 2*block_b + k_local] for j_local in 0,1 and k_local in 0,1
    
    The output C[i, k] where k = 2*block_b + k_local.
    
    Now for each of the 8 blocks, I apply Strassen with 7 rank-1 terms.
    
    For a 2x2*2x2 Strassen term c (c=0..6):
    The term contributes to Z[i_local, k_local] = sum_c u_c[i_local] * v_c[j_local] * w_c[k_local]
    where the specific u, v, w patterns depend on c.
    
    Let me define the 7 Strassen patterns for 2x2*2x2:
    For X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]:
    
    c=0: M1 = (x00+x11)*(y00+y11)
      u: [1, 1] for x (i.e., x00 and x11 get coeff 1)
      v: [1, 1] for y (y00 and y11 get coeff 1)
      w contribution to z: z00 += 1, z11 += 1, z01 += 0, z10 += 0
      So w: [1, 0, 0, 1] in (z00, z01, z10, z11) order... 
      
    Hmm, let me think about this differently. For each rank-1 term c, we have:
    - A vector u over the 4 elements of X (x00, x01, x10, x11)
    - A vector v over the 4 elements of Y (y00, y01, y10, y11)
    - A vector w over the 4 elements of Z (z00, z01, z10, z11)
    
    And the condition is: for all (i,j,k) in the 2x2x2x2 tensor (i,k for Z, j for the shared index),
    delta_{i,k} (the multiplication tensor) = sum_c u_c[i,j] * v_c[j,k] * w_c[k,i]
    
    Wait, I'm getting confused with indices. Let me restart the indexing for the 2x2*2x2 case.
    
    For 2x2 * 2x2: n=2, m=2, p=2.
    U has shape (2*2, rank) = (4, rank). Indexed by (i, j) where i in 0,1 and j in 0,1.
    V has shape (2*2, rank) = (4, rank). Indexed by (j, k) where j in 0,1 and k in 0,1.
    W has shape (2*2, rank) = (4, rank). Indexed by (k, i) where k in 0,1 and i in 0,1.
    
    The tensor T[i,j,k] = 1 for all i,j,k in 0,1.
    Decomposition: T[i,j,k] = sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c].
    
    For Strassen, the 7 terms are:
    Let me use the standard Strassen formulas and convert to tensor form.
    
    M1 = (x00+x11)*(y00+y11)
    M2 = (x10+x11)*y00
    M3 = x00*(y01-y11)
    M4 = x11*(y10-y00)
    M5 = (x00+x01)*y11
    M6 = (x10-x00)*(y00+y01)
    M7 = (x01-x11)*(y10+y11)
    
    z00 = M1 + M4 - M5 + M7
    z01 = M3 + M5
    z10 = M2 + M4
    z11 = M1 - M2 + M3 + M6
    
    For each M_c, it's a product of a linear combination of x's and a linear combination of y's, contributing to z's.
    
    M1 = (x00+x11)*(y00+y11):
    x-coeffs: x00:1, x01:0, x10:0, x11:1
    y-coeffs: y00:1, y01:0, y10:0, y11:1
    z-contrib: z00:+1, z01:0, z10:0, z11:+1
    
    So for this term:
    U[i*2+j, c]: for (i,j) in {(0,0): 1, (1,1): 1, others: 0}
    V[j*2+k, c]: for (j,k) in {(0,0): 1, (1,1): 1, others: 0}
    W[k*2+i, c]: for (k,i) in {(0,0): 1, (1,1): 1, others: 0}
    
    Let me verify: sum over c of U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] should give T[i,j,k]=1.
    For i=0,j=0,k=0: U[0,c]*V[0,c]*W[0,c]. For c=M1: U[0]=1, V[0]=1, W[0]=1. Product=1.
    Other c's: need to check they give 0 for this (i,j,k) or sum correctly.
    
    This is getting complex. Let me just hardcode the 7 Strassen terms as (U_col, V_col, W_col) for the 2x2 case, then map them to the full problem.
    
    For 2x2*2x2, the 7 terms (U over 4 entries (0,0),(0,1),(1,0),(1,1), V over 4, W over 4):
    
    Term 0 (M1): 
      U: [1, 0, 0, 1]  (x00, x01, x10, x11)
      V: [1, 0, 0, 1]  (y00, y01, y10, y11)
      W: [1, 0, 0, 1]  (z00, z01, z10, z11) -> but W is indexed by (k,i)
      W[k*2+i]: (k,i) in {(0,0), (1,1)} -> W[0]=1, W[3]=1, W[1]=0, W[2]=0
    
    Term 1 (M2): (x10+x11)*y00
      U: [0, 0, 1, 1]
      V: [1, 0, 0, 0]
      W: z10:+1 -> (k=1,i=0) -> W[1*2+0]=W[2]=1, others 0. W: [0,0,1,0]
    
    Term 2 (M3): x00*(y01-y11)
      U: [1, 0, 0, 0]
      V: [0, 1, 0, -1]
      W: z01:+1 -> (k=1,i=0) -> W[1*2+0]=W[2]=1. W: [0,0,1,0]
    
    Term 3 (M4): x11*(y10-y00)
      U: [0, 0, 0, 1]
      V: [0, 0, 1, -1]  wait: y10-y00. y00 index 0, y10 index 2. So V: [-1, 0, 1, 0]
      W: z00:+1, z10:+1 -> (k=0,i=0): W[0]=1, (k=1,i=0): W[2]=1. W: [1,0,1,0]
    
    Term 4 (M5): (x00+x01)*y11
      U: [1, 1, 0, 0]
      V: [0, 0, 0, 1]
      W: z00:-1, z01:+1 -> (k=0,i=0): W[0]=-1, (k=1,i=0): W[2]=1. W: [-1,0,1,0]
    
    Term 5 (M6): (x10-x00)*(y00+y01)
      U: [-1, 0, 1, 0]
      V: [1, 1, 0, 0]
      W: z11:+1 -> (k=1,i=1): W[1*2+1]=W[3]=1. W: [0,0,0,1]
    
    Term 6 (M7): (x01-x11)*(y10+y11)
      U: [0, 1, 0, -1]
      V: [0, 0, 1, 1]
      W: z00:+1 -> (k=0,i=0): W[0]=1. W: [1,0,0,0]
    
    Let me verify term 3 more carefully:
    M4 = x11*(y10-y00)
    z00 += M4, z10 += M4
    z00 is at (i=0,k=0), z10 is at (i=1,k=0).
    W is indexed by (k,i). So W[0*2+0]=W[0] for (k=0,i=0) and W[0*2+1]=W[1] for (k=0,i=1).
    Wait, W[k*n+i] with n=2. W[k*2+i].
    z00: i=0, k=0 -> W[0*2+0]=W[0]
    z10: i=1, k=0 -> W[0*2+1]=W[1]
    So W: [1, 1, 0, 0]
    
    Let me redo all W vectors:
    W[k*2+i] where k is row of Z, i is... wait. C[i,k] where i is row of A (and C), k is col of B (and C).
    W is indexed by (k, i) -> W[k*n + i] = W[k*2 + i].
    
    z00: i=0, k=0 -> W[0]
    z01: i=0, k=1 -> W[1*2+0]=W[2]
    z10: i=1, k=0 -> W[0*2+1]=W[1]
    z11: i=1, k=1 -> W[1*2+1]=W[3]
    
    So:
    Term 0 (M1): z00:+1, z11:+1 -> W[0]=1, W[3]=1 -> W: [1,0,0,1]
    Term 1 (M2): z10:+1 -> W[1]=1 -> W: [0,1,0,0]
    Term 2 (M3): z01:+1 -> W[2]=1 -> W: [0,0,1,0]
    Term 3 (M4): z00:+1, z10:+1 -> W[0]=1, W[1]=1 -> W: [1,1,0,0]
    Term 4 (M5): z00:-1, z01:+1 -> W[0]=-1, W[2]=1 -> W: [-1,0,1,0]
    Term 5 (M6): z11:+1 -> W[3]=1 -> W: [0,0,0,1]
    Term 6 (M7): z00:+1 -> W[0]=1 -> W: [1,0,0,0]
    
    Now let me verify the decomposition for a specific case. Take i=0, j=0, k=0 (T[0,0,0] should be 1).
    sum_c U[0*2+0,c] * V[0*2+0,c] * W[0*2+0,c]
    = sum_c U[0,c] * V[0,c] * W[0,c]
    
    c=0: U[0]=1, V[0]=1, W[0]=1 -> 1
    c=1: U[0]=0 -> 0
    c=2: U[0]=1, V[0]=0 -> 0
    c=3: U[0]=0 -> 0
    c=4: U[0]=1, V[0]=0 -> 0
    c=5: U[0]=-1 -> 0 (V[0]=1, W[0]=0)
    c=6: U[0]=0 -> 0
    Sum = 1. Correct!
    
    Take i=0, j=0, k=1 (T[0,0,1] should be 1).
    sum_c U[0,c] * V[0*2+1,c] * W[1*2+0,c]
    = sum_c U[0,c] * V[1,c] * W[2,c]
    
    c=0: U[0]=1, V[1]=0 -> 0
    c=1: U[0]=0 -> 0
    c=2: U[0]=1, V[1]=1, W[2]=1 -> 1
    c=3: U[0]=0 -> 0
    c=4: U[0]=1, V[1]=0 -> 0
    c=5: U[0]=-1, V[1]=1, W[2]=0 -> 0
    c=6: U[0]=0 -> 0
    Sum = 1. Correct!
    
    Take i=0, j=1, k=0 (T[0,1,0] should be 1).
    sum_c U[0*2+1,c] * V[1*2+0,c] * W[0*2+0,c]
    = sum_c U[1,c] * V[2,c] * W[0,c]
    
    c=0: U[1]=0 -> 0
    c=1: U[1]=0 -> 0
    c=2: U[1]=0 -> 0
    c=3: U[1]=0 -> 0
    c=4: U[1]=1, V[2]=0 -> 0
    c=5: U[1]=0 -> 0
    c=6: U[1]=1, V[2]=1, W[0]=1 -> 1
    Sum = 1. Correct!
    
    Take i=0, j=1, k=1 (T[0,1,1] should be 1).
    sum_c U[1,c] * V[3,c] * W[2,c]
    
    c=0: U[1]=0 -> 0
    c=1: U[1]=0 -> 0
    c=2: U[1]=0 -> 0
    c=3: U[1]=0 -> 0
    c=4: U[1]=1, V[3]=1, W[2]=1 -> 1
    c=5: U[1]=0 -> 0
    c=6: U[1]=1, V[3]=1, W[2]=0 -> 0
    Sum = 1. Correct!
    
    Great, the 2x2 Strassen decomposition works. Now I need to map this to the full 2x4*4x8 problem.
    
    For each of the 8 blocks, I'll place the 7 Strassen terms.
    
    Block (ba, bb) where ba in {0,1} and bb in {0,1,2,3}:
    - A block: rows 0-1, cols (2*ba to 2*ba+1)
    - B block: rows (2*ba to 2*ba+1), cols (2*bb to 2*bb+1)
    - C block: rows 0-1, cols (2*bb to 2*bb+1)
    
    For local indices (i_l, j_l) in A block (i_l, j_l in 0,1):
      Global A index: i = i_l, j = 2*ba + j_l
      U global index: i*m + j = i_l*4 + (2*ba + j_l)
    
    For local indices (j_l, k_l) in B block:
      Global B index: j = 2*ba + j_l, k = 2*bb + k_l
      V global index: j*p + k = (2*ba + j_l)*8 + (2*bb + k_l)
    
    For local indices (k_l, i_l) in C block:
      Global C index: i = i_l, k = 2*bb + k_l
      W global index: k*n + i = (2*bb + k_l)*2 + i_l
    
    Now I'll construct U, V, W with rank = 56.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # 2x2 Strassen terms: (U_local, V_local, W_local)
    # U_local indexed by (i_l*2+j_l), V_local by (j_l*2+k_l), W_local by (k_l*2+i_l)
    strassen_terms = [
        # (U: [x00,x01,x10,x11], V: [y00,y01,y10,y11], W: [z00,z10,z01,z11] in (k,i) order)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]
    
    col = 0
    for ba in range(2):       # which A/B row block
        for bb in range(4):   # which column block
            for t in range(7):
                U_loc, V_loc, W_loc = strassen_terms[t]
                for i_l in range(2):
                    for j_l in range(2):
                        u_idx = i_l * m + (2 * ba + j_l)
                        U[u_idx][col] = U_loc[i_l * 2 + j_l]
                for j_l in range(2):
                    for k_l in range(2):
                        v_idx = (2 * ba + j_l) * p + (2 * bb + k_l)
                        V[v_idx][col] = V_loc[j_l * 2 + k_l]
                for k_l in range(2):
                    for i_l in range(2):
                        w_idx = (2 * bb + k_l) * n + i_l
                        W[w_idx][col] = W_loc[k_l * 2 + i_l]
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
