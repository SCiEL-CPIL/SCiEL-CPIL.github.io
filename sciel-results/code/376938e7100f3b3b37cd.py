"""Rewritten program: Strassen-based rank-42 decomposition for 2x2 * 2x12."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-42 exact tensor decomposition for multiplying a 2x2 matrix
    by a 2x12 matrix.
    
    Strategy: Use Strassen's rank-7 decomposition for the 2x2*2x2 core, and
    replicate it across the 6 blocks of the 2x12 right-hand side.
    
    The multiplication is: C[2x12] = A[2x2] @ B[2x12]
    We can view B as having 6 blocks of size 2x2: B = [B_0, B_1, ..., B_5]
    Then C = [A@B_0, A@B_1, ..., A@B_5]
    
    For each block, we use Strassen's 7-term decomposition.
    The U factor (for A) is shared across all 6 blocks, while V and W are
    specific to each block.
    
    Strassen's decomposition for 2x2 * 2x2:
    Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C = [[M1+M4-M5+M7, M3+M5],
         [M2+M4, M1-M2+M3+M6]]
    
    We need to express this in the form:
    C[i,k] = sum_r U[i, r] * V[?, r] * W[k, r]
    
    Wait, the tensor decomposition form is:
    T[i,j,k] = sum_r U[i, r] * V[j, r] * W[k, r]
    
    For matrix multiplication C = A @ B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So T[i,j,k] = delta (it's the multiplication tensor)
    The decomposition: T[i,j,k] = sum_r U[i, r] * V[j, r] * W[k, r]
    
    Here n=2, m=2, p=12.
    U has shape (n*m, rank) = (4, rank) -- wait, let me re-read the interface.
    
    Interface: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    
    So U[i*m + j, r] corresponds to A[i,j]
    V[j*p + k, r] corresponds to B[j,k]
    W[k*n + i, r] corresponds to C[i,k] (with the trace convention W[k*n+i, r])
    
    The decomposition condition:
    For all i, j, k: delta[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    where delta[i,j,k] = 1 if this is a valid multiplication term, else 0.
    Actually, the multiplication tensor T has T[i,j,k] = 1 if the (i,j) element of A
    and (j,k) element of B contribute to (i,k) element of C, which is always 1 for
    valid indices. Wait, no. The tensor is defined such that:
    
    sum_{j} A[i,j] * B[j,k] = C[i,k]
    
    The tensor T[i,j,k] is 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    This is always 1 for all valid i, j, k. So T[i,j,k] = 1 for all i in [0,n), 
    j in [0,m), k in [0,p).
    
    Wait, that's not right either. Let me think again.
    
    The matrix multiplication tensor T has entries T[i,j,k] where:
    - i indexes rows of A and C: 0 <= i < n
    - j indexes columns of A / rows of B: 0 <= j < m  
    - k indexes columns of B and C: 0 <= k < p
    
    And T[i,j,k] = 1 for all valid i, j, k (since every A[i,j]*B[j,k] contributes
    to C[i,k]).
    
    The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    So we need: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k.
    
    For the schoolbook: rank = n*m*p, each term picks out one (i,j,k) triple.
    
    For Strassen on 2x2*2x2: rank 7.
    For 2x2*2x12: We can decompose B into 6 blocks of 2x2.
    C[i, 2*b + k'] = sum_j A[i,j] * B[j, 2*b + k']
    
    For each block b, we use Strassen with the same A but different B block.
    The U factor (which depends on A's indices i*m+j) is shared.
    The V factor (which depends on B's indices j*p+k) is different per block.
    The W factor (which depends on C's indices k*n+i) is different per block.
    
    So total rank = 7 * 6 = 42.
    
    Let me define the Strassen terms for a 2x2*2x2 multiplication:
    A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    
    U indices: i*m+j where i in {0,1}, j in {0,1}
    So U[0, r] = A00, U[1, r] = A01, U[2, r] = A10, U[3, r] = A11
    
    V indices: j*p+k. For a single 2x2 block at position b, k goes from 2*b to 2*b+1.
    So V[j*12 + 2*b + 0, r] = B[j, 2*b], V[j*12 + 2*b + 1, r] = B[j, 2*b+1]
    
    W indices: k*n + i. For block b, k' in {0,1} corresponds to k = 2*b + k'.
    So W[(2*b+k')*2 + i, r] = C[i, 2*b+k']
    
    Strassen's 7 terms:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B11-B01)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Now I need to express each M_r as sum of products U[...] * V[...] * W[...].
    Actually, each M_r is a single product of a linear form in A, a linear form in B,
    and contributes to linear forms in C.
    
    Let me define for each Strassen term r (0..6):
    - u_r: vector in R^4 (coefficients for A00, A01, A10, A11)
    - v_r: vector in R^2 (coefficients for B's row 0 and row 1, for a 2x2 block)
    - w_r: vector in R^2 (coefficients for C's col 0 and col 1, for a 2x2 block)
    
    But wait, the decomposition is:
    1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This means for each (i,j,k), the sum over r of the product equals 1.
    
    Let me think of it differently. The Strassen decomposition says:
    C[i,k] = sum_{r=1}^{7} (sum_{j} u_r[i,j] * A[i,j]) * (sum_{j} v_r[j,k] * B[j,k]) * w_r[i,k]
    
    No, that's not quite right either. Let me be more careful.
    
    The standard tensor decomposition for Strassen:
    T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[i,k]
    
    where for each r, U_r, V_r, W_r are rank-1 matrices (outer products of vectors).
    
    For Strassen:
    r=1: U_1 = (1,0,0,1) for (A00,A01,A10,A11), V_1 = (1,1) for (B00,B10) [column-major? or row?]
    
    Hmm, let me just directly construct the U, V, W matrices.
    
    For block b (b=0..5), and Strassen term r (r=0..6):
    The global column index is b*7 + r.
    
    U[i*m + j, b*7+r] = u_r[i, j]  (same for all b)
    V[j*p + (2*b + k'), b*7+r] = v_r[j, k']  (for k'=0,1)
    W[(2*b + k')*n + i, b*7+r] = w_r[i, k']  (for k'=0,1)
    
    Now I need to define u_r, v_r, w_r for r=0..6 such that:
    For all i, j, k': sum_r u_r[i,j] * v_r[j,k'] * w_r[i,k'] = 1
    
    Let me use the Strassen formulas:
    
    M1 = (A00+A11)(B00+B11)
    => u_1 = (1, 0, 0, 1), v_1 = (1, 1) [for B00, B10? or B00, B01?]
    
    Wait, I need to be careful about index conventions.
    
    A is 2x2: A[i,j] with i=row, j=col.
    B is 2x2 (within a block): B[j,k] with j=row, k=col.
    C is 2x2 (within a block): C[i,k] with i=row, k=col.
    
    U index: i*m + j = 2*i + j
    V index: j*p + k = j*12 + k (for the full matrix), within block b: j*12 + 2*b + k'
    W index: k*n + i = 2*k + i
    
    For the 2x2*2x2 sub-problem:
    U_r[2*i+j] for i,j in {0,1}
    V_r[j, k'] for j in {0,1}, k' in {0,1} -> maps to V[2*b+k'] in the block
    W_r[k', i] for k' in {0,1}, i in {0,1} -> maps to W[2*b+k'] in the block
    
    Strassen terms:
    M1 = (A00+A11)(B00+B11)
    u_1: A00->1, A01->0, A10->0, A11->1 => U[0]=1, U[1]=0, U[2]=0, U[3]=1
    v_1: B00->1, B10->1 (B[j,k] with j=0,k=0 and j=1,k=1) => V[0,0]=1, V[0,1]=0, V[1,0]=0, V[1,1]=1
    Wait, B[j,k]: B00 is j=0,k=0; B01 is j=0,k=1; B10 is j=1,k=0; B11 is j=1,k=1.
    B00+B11: v_1[j,k'] = 1 if (j,k') is (0,0) or (1,1), else 0.
    So v_1 = [[1,0],[0,1]] (as a 2x2 matrix with rows j, cols k')
    
    w_1: M1 contributes to C00 (+1) and C11 (+1).
    w_1[i,k'] = 1 if (i,k') is (0,0) or (1,1), else 0.
    So w_1 = [[1,0],[0,1]]
    
    Let me verify: sum_r u_r[i,j]*v_r[j,k']*w_r[i,k'] should equal 1 for all i,j,k'.
    
    For (i,j,k') = (0,0,0): 
    Only terms where u_r[0,0] != 0, v_r[0,0] != 0, w_r[0,0] != 0 contribute.
    This is getting complex. Let me just use the known Strassen decomposition.
    
    The 7 rank-1 tensors for Strassen:
    
    r=1: U=(1,0,0,1), V=[[1,0],[0,1]], W=[[1,0],[0,1]]  -> (A00+A11)(B00+B11) -> C00+=1, C11+=1
    r=2: U=(0,0,1,1), V=[[1,0],[0,0]], W=[[0,0],[1,0]]  -> (A10+A11)B00 -> C10+=1
    r=3: U=(1,0,0,0), V=[[0,1],[0,0]], W=[[0,1],[0,0]]  -> A00(B01) ... wait
    
    Let me just look up the standard Strassen decomposition in tensor form.
    
    Actually, let me just directly compute. I'll define the 7 terms and verify numerically.
    
    Let me define:
    For each r in 0..6:
      u_r = vector of length 4 (for A00, A01, A10, A11)
      v_r = 2x2 matrix (for B, rows=j, cols=k')
      w_r = 2x2 matrix (for C, rows=i, cols=k')
    
    And check that for all i,j,k': sum_r u_r[2*i+j] * v_r[j,k'] * w_r[i,k'] = 1
    
    Let me use the standard Strassen:
    
    M1 = (a+d)(e+h): u=(1,0,0,1), v=[[1,0],[0,1]], w=[[1,0],[0,1]]
    M2 = (c+d)e: u=(0,0,1,1), v=[[1,0],[0,0]], w=[[0,0],[1,0]]
    M3 = a(f-h): u=(1,0,0,0), v=[[0,1],[0,-1]], w=[[0,1],[0,0]]
    M4 = d(g-e): u=(0,0,0,1), v=[[0,-1],[1,0]], w=[[1,0],[0,0]]
    M5 = (a+b)h: u=(1,1,0,0), v=[[0,0],[0,1]], w=[[-1,1],[0,0]]
    M6 = (c-a)(e+f): u=(0,0,1,-1), v=[[1,1],[0,0]], w=[[0,0],[0,1]]
    M7 = (b-d)(g+h): u=(0,1,0,-1), v=[[0,1],[1,1]], w=[[1,0],[0,0]]
    
    Let me verify M1: (a+d)(e+h)
    a=A00, b=A01, c=A10, d=A11, e=B00, f=B01, g=B10, h=B11
    
    M1 = (A00+A11)(B00+B11)
    u_1 = [1,0,0,1] (coefficients for A00,A01,A10,A11)
    v_1[j,k']: B00 is j=0,k'=0; B11 is j=1,k'=1. So v_1 = [[1,0],[0,1]]
    w_1[i,k']: contributes to C00 (i=0,k'=0) and C11 (i=1,k'=1). w_1 = [[1,0],[0,1]]
    
    Check for (i,j,k')=(0,0,0): u_1[0]*v_1[0,0]*w_1[0,0] = 1*1*1 = 1. Good.
    Check for (i,j,k')=(0,1,1): u_1[1]*v_1[1,1]*w_1[0,1] = 0*1*0 = 0. 
    But we need sum over all r to give 1.
    
    Let me check (i,j,k')=(0,1,1) which corresponds to A01*B01 contributing to C01.
    We need: sum_r u_r[2*0+1] * v_r[1,1] * w_r[0,1] = sum_r u_r[1] * v_r[1,1] * w_r[0,1] = 1
    
    r=1: u_1[1]=0, so 0.
    r=2: u_2[1]=0, so 0.
    r=3: u_3[1]=0, so 0.
    r=4: u_4[1]=0, so 0.
    r=5: u_5[1]=1, v_5[1,1]=1, w_5[0,1]=1. Product = 1*1*1 = 1.
    r=6: u_6[1]=0, so 0.
    r=7: u_7[1]=1, v_7[1,1]=1, w_7[0,1]=0. Product = 1*1*0 = 0.
    
    Sum = 1. 
    
    Let me check another: (i,j,k')=(1,0,1) which is A10*B01 -> C11.
    sum_r u_r[2*1+0] * v_r[0,1] * w_r[1,1] = sum_r u_r[2] * v_r[0,1] * w_r[1,1]
    
    r=1: u_1[2]=0, so 0.
    r=2: u_2[2]=1, v_2[0,1]=0, so 0.
    r=3: u_3[2]=0, so 0.
    r=4: u_4[2]=0, so 0.
    r=5: u_5[2]=0, so 0.
    r=6: u_6[2]=1, v_6[0,1]=1, w_6[1,1]=1. Product = 1*1*1 = 1.
    r=7: u_7[2]=0, so 0.
    
    Sum = 1. 
    
    Let me check (i,j,k')=(1,1,0) which is A11*B10 -> C10.
    sum_r u_r[2*1+1] * v_r[1,0] * w_r[1,0] = sum_r u_r[3] * v_r[1,0] * w_r[1,0]
    
    r=1: u_1[3]=1, v_1[1,0]=0, so 0.
    r=2: u_2[3]=1, v_2[1,0]=0, so 0.
    r=3: u_3[3]=0, so 0.
    r=4: u_4[3]=1, v_4[1,0]=1, w_4[1,0]=0. Product = 1*1*0 = 0.
    r=5: u_5[3]=0, so 0.
    r=6: u_6[3]=-1, v_6[1,0]=0, so 0.
    r=7: u_7[3]=-1, v_7[1,0]=1, w_7[1,0]=0. Product = -1*1*0 = 0.
    
    Sum = 0. That's wrong! We need 1.
    
    Let me re-examine. A11*B10 contributes to C10. So (i=1, j=1, k'=0) should give 1.
    
    Hmm, let me recheck my Strassen terms.
    
    Actually, I think I may have the w matrices wrong. Let me re-derive.
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    So:
    M1: C00 += 1, C11 += 1 => w_1 = [[1,0],[0,1]]
    M2: C10 += 1 => w_2 = [[0,0],[1,0]]
    M3: C01 += 1, C11 += 1 => w_3 = [[0,1],[0,1]]
    M4: C00 += 1, C10 += 1 => w_4 = [[1,0],[1,0]]
    M5: C00 -= 1, C01 += 1 => w_5 = [[-1,1],[0,0]]
    M6: C11 += 1 => w_6 = [[0,0],[0,1]]
    M7: C00 += 1 => w_7 = [[1,0],[0,0]]
    
    Now let me recheck (i=1, j=1, k'=0): A11*B10 -> C10
    sum_r u_r[3] * v_r[1,0] * w_r[1,0]
    
    r=1: u_1[3]=1, v_1[1,0]=0, w_1[1,0]=0. Product = 0.
    r=2: u_2[3]=1, v_2[1,0]=0, w_2[1,0]=1. Product = 0.
    r=3: u_3[3]=0. Product = 0.
    r=4: u_4[3]=1, v_4[1,0]=1, w_4[1,0]=1. Product = 1*1*1 = 1.
    r=5: u_5[3]=0. Product = 0.
    r=6: u_6[3]=-1, v_6[1,0]=0, w_6[1,0]=0. Product = 0.
    r=7: u_7[3]=-1, v_7[1,0]=1, w_7[1,0]=0. Product = 0.
    
    Sum = 1. 
    
    Let me now verify all 16 cases (i in {0,1}, j in {0,1}, k' in {0,1}) programmatically in my head... actually, let me just code it up and trust the standard Strassen decomposition. I'll add a verification step.
    
    Let me code the full solution.
    """
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    num_blocks = p // 2  # 6 blocks of 2x2
    strassen_rank = 7
    rank = num_blocks * strassen_rank  # 42
    
    # Strassen decomposition components
    # u_r: length 4, for A indices (i*m+j) = (2*i+j)
    # v_r: 2x2, for B indices within a block (j, k')
    # w_r: 2x2, for C indices within a block (i, k')
    
    u = [
        [1, 0, 0, 1],   # M1: (a+d)
        [0, 0, 1, 1],   # M2: (c+d)
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: (a+b)
        [0, 0, 1, -1],  # M6: (c-a)
        [0, 1, 0, -1],  # M7: (b-d)
    ]
    
    v = [
        [[1, 0], [0, 1]],   # M1: (e+h)
        [[1, 0], [0, 0]],   # M2: e
        [[0, 1], [0, -1]],  # M3: (f-h)
        [[0, -1], [1, 0]],  # M4: (g-e)
        [[0, 0], [0, 1]],   # M5: h
        [[1, 1], [0, 0]],   # M6: (e+f)
        [[0, 1], [1, 1]],   # M7: (g+h)
    ]
    
    w = [
        [[1, 0], [0, 1]],   # M1: C00+=1, C11+=1
        [[0, 0], [1, 0]],   # M2: C10+=1
        [[0, 1], [0, 1]],   # M3: C01+=1, C11+=1
        [[1, 0], [1, 0]],   # M4: C00+=1, C10+=1
        [[-1, 1], [0, 0]],  # M5: C00-=1, C01+=1
        [[0, 0], [0, 1]],   # M6: C11+=1
        [[1, 0], [0, 0]],   # M7: C00+=1
    ]
    
    # Verify the Strassen decomposition for 2x2*2x2
    for i in range(2):
        for j in range(2):
            for kp in range(2):
                total = 0
                for r in range(7):
                    total += u[r][2*i+j] * v[r][j][kp] * w[r][i][kp]
                assert total == 1, f"Verification failed for i={i}, j={j}, kp={kp}: got {total}"
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    for b in range(num_blocks):
        for r in range(strassen_rank):
            col = b * strassen_rank + r
            # U is shared across blocks
            for idx in range(4):
                U[idx][col] = u[r][idx]
            # V for block b
            for j in range(2):
                for kp in range(2):
                    row_v = j * p + (2 * b + kp)
                    V[row_v][col] = v[r][j][kp]
            # W for block b
            for i in range(2):
                for kp in range(2):
                    row_w = (2 * b + kp) * n + i
                    W[row_w][col] = w[r][i][kp]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
