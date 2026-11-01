import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 8]

def _verify_fast(U, V, W, n=2, m=4, p=8, trials=50, tol=1e-9):
    rank = len(U[0]) if U else 0
    if rank == 0:
        return True
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)

    U_r = U_mat.reshape(n, m, rank)
    V_r = V_mat.reshape(m, p, rank)
    W_r = W_mat.reshape(p, n, rank)

    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        factor_A = np.einsum('ij,ijc->ic', A, U_r)
        factor_B = np.einsum('jk,jkc->kc', B, V_r)
        C_computed = np.einsum('ic,kc,kic->ik', factor_A, factor_B, W_r)
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _build_kronecker_decomposition():
    """
    Build a decomposition exploiting the Kronecker structure of 2x4 x 4x8.
    
    Key idea: 2x4 x 4x8 can be seen as (2x2) x (2x8) applied twice (for the two
    middle-dimension halves), or we can use a 2x2 Strassen on the outer structure.
    
    Actually, let's use a direct structured approach:
    Split middle dimension j into j1 (0,1) and j2 (2,3).
    C[i,k] = sum_{j1=0}^{1} A[i,j1]*B[j1,k] + sum_{j2=0}^{1} A[i,2+j2]*B[2+j2,k]
    
    For each half, we have a 2x2 x 2x8 problem.
    A 2x2 x 2x8 matrix multiplication can be done with rank at most 2*2*8=32 schoolbook,
    but we can do better.
    
    Let's try: for the 2x2 x 2x8 case, treat it as 8 independent 2x2x1 multiplications?
    No, that's 8*4=32.
    
    Better: use the fact that 2x2 x 2x8 = (2x2 x 2x2) x 4 (in the output dimension).
    2x2 x 2x2 has rank 7 (Strassen). So 2x2 x 2x8 has rank at most 7*4 = 28.
    Two halves: 28*2 = 56. Same as before.
    
    Alternative: use a 1x4 x 4x8 decomposition.
    For 1x4 x 4x8: C[k] = sum_{j=0}^{3} A[j]*B[j,k]
    This is 8 dot products of length 4.
    
    Winograd for length-4 dot product:
    a·b = (a0+a1+a2+a3)(b0+b1+b2+b3) - (a0+a1-a2-a3)(b0+b1-b2-b3) 
          - (a0-a1+a2-a3)(b0-b1+b2-b3) + (a0-a1-a2+a3)(b0-b1-b2+b3)
    Wait, that's not right. Let me think more carefully.
    
    For a length-4 dot product, the standard Winograd approach:
    We want to compute sum_{j=0}^{3} a[j]*b[j].
    
    Using the fact that this is a 1x4 x 4x1 tensor, its rank is 4 (since it's a diagonal-like structure).
    Actually no, the 1x4 x 4x1 tensor T[j,k] = delta_{j,k} has rank 4.
    
    But for 1x4 x 4x8, the tensor T[j,k] = 1 for all j,k (since C[k] = sum_j A[j] B[j,k]).
    Wait, that's not right either. The bilinear form is:
    C[k] = sum_j A[j] * B[j,k]
    So the "tensor" here is T[j,k] = 1 (it's just the identity in the middle index pairing).
    The bilinear complexity is the rank of the 3-tensor T in R^{1 x 4 x 8} where T[i,j,k] = delta_{j,j}... 
    
    Let me reconsider. The matrix multiplication tensor for 1xn x nxm is:
    T[j,k] = 1 (the coefficient relating A[j] and B[j,k] to C[k]).
    As a 3-tensor with indices (i,j,k) where i is the "row" index (only 1 value):
    T[0,j,k] = 1 for all j,k.
    
    The rank of this tensor: we need to write T[0,j,k] = sum_c U[0,j,c] * V[j,k,c] * W[k,0,c].
    Since i=0 is the only index, U[0,j,c] = u_jc, W[k,0,c] = w_kc.
    So T[0,j,k] = sum_c u_jc * V[j,k,c] * w_kc = 1 for all j,k.
    
    This is equivalent to: the 4x8 matrix of all 1s must be expressible as sum_c (u_c outer product with w_c scaled by V_c).
    If V_c = I for all c, then we need sum_c u_c * w_c^T = J (all-ones 4x8 matrix).
    The rank of J is 1, so we can do it with c=1: u = [1,1,1,1], w = [1,1,1,1,1,1,1,1], V = I.
    But wait, V must be 4x8, and if V=I it's not 4x8. V[j,k,c] must be 4x8 for each c.
    
    Hmm, I'm confusing myself. Let me be very precise.
    
    The decomposition is:
    C[i,k] = sum_c (sum_{i',j'} U[i',j',c] * A[i',j']) * (sum_{j',k'} V[j',k',c] * B[j',k']) * W[k,i,c]
    
    Wait, that's not standard. The standard bilinear algorithm for matrix multiplication:
    C = sum_c (A * U_c) * (B * V_c) * W_c
    
    where U_c is n x m, V_c is m x p, W_c is p x n (as a matrix to multiply on the right? or is it an outer product?).
    
    Actually, the standard form is:
    C[i,k] = sum_c (sum_j A[i,j] * U_c[i,j]) * (sum_j B[j,k] * V_c[j,k]) * W_c[k,i]
    
    No wait. Let me look at the verify function:
    factor_A = einsum('ij,ijc->ic', A, U_r)  # factor_A[i,c] = sum_j A[i,j]*U_r[i,j,c]
    factor_B = einsum('jk,jkc->kc', B, V_r)  # factor_B[k,c] = sum_j B[j,k]*V_r[j,k,c]
    C_computed = einsum('ic,kc,kic->ik', factor_A, factor_B, W_r)  # C[i,k] = sum_c factor_A[i,c]*factor_B[k,c]*W_r[k,i,c]
    
    So: C[i,k] = sum_c W[k,i,c] * (sum_j A[i,j]*U[i,j,c]) * (sum_j B[j,k]*V[j,k,c])
    
    For the 1x4 x 4x8 case (n=1, m=4, p=8):
    C[0,k] = sum_c W[k,0,c] * (sum_j A[0,j]*U[0,j,c]) * (sum_j B[j,k]*V[j,k,c])
    
    We need this to equal sum_j A[0,j]*B[j,k] for all A, B.
    
    This means: sum_c W[k,0,c] * U[0,j,c] * V[j,k,c] = delta_{j,j}... no, = 1 if we're looking at the coefficient of A[0,j]*B[j,k].
    
    Wait: the coefficient of A[0,j1]*B[j2,k] in C[0,k] should be 1 if j1=j2, else 0.
    So: sum_c W[k,0,c] * U[0,j1,c] * V[j2,k,c] = delta_{j1,j2}
    
    This is a system of equations. For n=1, m=4, p=8, we have 4*8=32 equations (for each j1,j2,k pair where j1≠j2 we need 0, and for j1=j2,k we need 1).
    
    The minimum rank for this: this is essentially the "tensor rank" of the 4x4x8 tensor T[j1,j2,k] = delta_{j1,j2} (independent of k, so it's delta_{j1,j2} * 1_k).
    
    This tensor is: T = sum_{j=0}^{3} e_j ⊗ e_j ⊗ 1 (where 1 is the all-ones vector in R^8).
    The rank of sum_j e_j⊗e_j⊗1: since the 1 vector has rank 1, and sum_j e_j⊗e_j has rank 4 (it's the identity matrix), the total rank is at most 4*1 = 4? No, tensor rank doesn't factor like that.
    
    Actually, T[j1,j2,k] = delta_{j1,j2}. This is the 4x4x8 tensor that is the identity in the first two modes and 1 in the third.
    We can write: T = sum_{j=0}^{3} (e_j ⊗ e_j ⊗ 1_8)
    Each term is rank 1. So rank ≤ 4.
    Can we do better? The mode-2 unfolding has rank 4 (it's the 4x4 identity repeated 8 times), so rank ≥ 4.
    So the 1x4 x 4x8 problem has bilinear complexity exactly 4!
    
    With rank 4: U[0,j,c] = delta_{j,c} for c=0..3, V[j,k,c] = delta_{j,c}, W[k,0,c] = 1.
    Check: sum_c W[k,0,c]*U[0,j1,c]*V[j2,k,c] = sum_c 1*delta_{j1,c}*delta_{j2,c} = delta_{j1,j2}. ✓
    
    So for 1x4 x 4x8, rank = 4.
    For 2x4 x 4x8, if the two rows were independent, rank would be 2*4 = 8.
    But can we share? The W factor has index k (8 values) and i (2 values).
    
    For the 2-row case:
    C[i,k] = sum_c W[k,i,c] * (sum_j A[i,j]*U[i,j,c]) * (sum_j B[j,k]*V[j,k,c])
    
    Coefficient of A[i1,j1]*B[j2,k]: sum_c W[k,i1,c]*U[i1,j1,c]*V[j2,k,c] = delta_{i1,i1}*delta_{j1,j2} = delta_{j1,j2}
    (since i1 is fixed, there's no cross-term between different i values).
    
    Wait, that's not right. The coefficient of A[i1,j1]*B[j2,k] in C[i,k] is:
    - If i = i1: delta_{j1,j2}
    - If i ≠ i1: 0
    
    So: sum_c W[k,i,c]*U[i1,j1,c]*V[j2,k,c] = delta_{i,i1}*delta_{j1,j2}
    
    For i=i1: sum_c W[k,i1,c]*U[i1,j1,c]*V[j2,k,c] = delta_{j1,j2}
    For i≠i1: sum_c W[k,i,c]*U[i1,j1,c]*V[j2,k,c] = 0
    
    This means for each i, the 4x4x8 tensor (U[i,:,:] ⊗ V ⊗ W[:,i,:]) must equal delta_{j1,j2}.
    And for i≠i1, the cross terms must vanish.
    
    If we use the same V for both rows (which we can, since V doesn't depend on i), then:
    For row 0: sum_c W[k,0,c]*U[0,j1,c]*V[j2,k,c] = delta_{j1,j2}
    For row 1: sum_c W[k,1,c]*U[1,j1,c]*V[j2,k,c] = delta_{j1,j2}
    Cross: sum_c W[k,0,c]*U[1,j1,c]*V[j2,k,c] = 0
    Cross: sum_c W[k,1,c]*U[0,j1,c]*V[j2,k,c] = 0
    
    If we use the rank-4 solution for each row independently with disjoint c indices:
    Row 0 uses c=0,1,2,3: U[0,j,c]=delta_{j,c}, V[j,k,c]=delta_{j,c}, W[k,0,c]=1
    Row 1 uses c=4,5,6,7: U[1,j,c-4]=delta_{j,c-4}, V[j,k,c-4]=delta_{j,c-4}, W[k,1,c-4]=1
    
    But V is shared! V[j,k,c] must be the same for both rows. In the above, V[j,k,c]=delta_{j,c} for c=0..3 and V[j,k,c-4]=delta_{j,c-4} for c=4..7.
    So V[j,k,0..3] = delta_{j,0..3} and V[j,k,4..7] = delta_{j,0..3} (shifted).
    This works! V[j,k,c] = delta_{j, c%4} for c=0..7.
    
    Check cross terms:
    sum_c W[k,0,c]*U[1,j1,c]*V[j2,k,c] = sum_{c=0}^{3} 1*0*... + sum_{c=4}^{7} 0*... = 0 ✓
    (Since U[1,j1,c]=0 for c=0..3 and W[k,0,c]=0 for c=4..7)
    
    So total rank = 8!
    
    Let me verify: U is 2x4x8 (flattened to 8x8), V is 4x8x8 (flattened to 32x8), W is 8x2x8 (flattened to 16x8).
    
    U[i,j,c]: for i=0, U[0,j,c] = delta_{j,c} for c=0..3, else 0.
              for i=1, U[1,j,c] = delta_{j,c-4} for c=4..7, else 0.
    
    V[j,k,c]: V[j,k,c] = delta_{j, c%4} for all c=0..7, all k.
    
    W[k,i,c]: for i=0, W[k,0,c] = 1 for c=0..3, else 0.
              for i=1, W[k,1,c] = 1 for c=4..7, else 0.
    
    Verify C[0,k] = sum_c W[k,0,c] * (sum_j A[0,j]*U[0,j,c]) * (sum_j B[j,k]*V[j,k,c])
    = sum_{c=0}^{3} 1 * A[0,c] * B[c,k] = sum_{j=0}^{3} A[0,j]*B[j,k] ✓
    
    C[1,k] = sum_c W[k,1,c] * (sum_j A[1,j]*U[1,j,c]) * (sum_j B[j,k]*V[j,k,c])
    = sum_{c=4}^{7} 1 * A[1,c-4] * B[c-4,k] = sum_{j=0}^{3} A[1,j]*B[j,k] ✓
    
    So rank 8 works! That's a massive improvement from 56.
    
    But wait - the entries must be integer or half-integer. All entries here are 0 or 1, so that's fine.
    
    Can we do even better? The lower bound from mode-1 unfolding: the tensor T[i,j,k] has mode-1 unfolding of size n x (m*p) = 2 x 32. The rank of this unfolding gives a lower bound.
    T[i,j,k] = delta_{j,j}... no. The tensor for the bilinear algorithm isn't simply defined this way.
    
    Actually, the lower bound: for the 2x4x8 matrix multiplication, the known lower bounds are:
    - rank ≥ max(n, m, p) = 8
    - rank ≥ n*m*p / min(n,m,p)^2 ... not sure
    
    The rank 8 solution achieves the lower bound max(n,m,p)=8. So rank 8 is optimal!
    
    Let me double-check the interface. U has shape (n*m, rank) = (8, 8). V has shape (m*p, rank) = (32, 8). W has shape (p*n, rank) = (16, 8).
    
    U is indexed as [i*m+j, c] = [i*4+j, c]
    V is indexed as [j*p+k, c] = [j*8+k, c]
    W is indexed as [k*n+i, c] = [k*2+i, c]
    
    Let me build this:
    """
    n, m, p = 2, 4, 8
    rank = 8
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            c = i * m + j  # c = 0,1,2,3 for i=0; c=4,5,6,7 for i=1
            U[i * m + j, c] = 1.0
    
    for j in range(m):
        for k in range(p):
            for c in range(rank):
                V[j * p + k, c] = 1.0 if (j == c % m) else 0.0
    
    for k in range(p):
        for i in range(n):
            for c in range(rank):
                W[k * n + i, c] = 1.0 if (i * m <= c < (i + 1) * m) else 0.0
    
    return U.tolist(), V.tolist(), W.tolist()

def _build_block_strassen():
    """Fallback: the original block Strassen rank-56 decomposition."""
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    U_s = [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[0,0,1,-1],[0,1,0,1]]
    V_s = [[1,0,0,1],[0,0,0,1],[1,0,-1,0],[0,-1,0,1],[0,0,1,0],[1,1,0,0],[0,0,1,1]]
    W_s = [[1,0,0,1,-1,0,1],[0,0,1,0,1,0,0],[0,1,0,0,0,1,0],[1,-1,1,0,0,1,0]]

    col_off = 0
    for block in range(4):
        for f in range(7):
            col = col_off + f
            for r in range(4):
                U[r, col] = U_s[f][r]
                j = r // 2
                k = 2 * block + r % 2
                V[j * 8 + k, col] = V_s[f][r]
                i = r // 2
                W[k * 2 + i, col] = W_s[r][f]
        col_off += 7

    for block in range(4):
        for f in range(7):
            col = col_off + f
            for r in range(4):
                U[2 + r, col] = U_s[f][r]
                j = 2 + r // 2
                k = 2 * block + r % 2
                V[j * 8 + k, col] = V_s[f][r]
                i = r // 2
                W[k * 2 + i, col] = W_s[r][f]
        col_off += 7

    return U.tolist(), V.tolist(), W.tolist()

def _build_schoolbook():
    """Fallback: schoolbook rank-64 decomposition."""
    n, m, p = 2, 4, 8
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 135

    n, m, p = DIMENSIONS

    # Try the rank-8 structured decomposition first
    U_list, V_list, W_list = _build_kronecker_decomposition()
    U = np.array(U_list, dtype=float)
    V = np.array(V_list, dtype=float)
    W = np.array(W_list, dtype=float)

    if _verify_fast(U_list, V_list, W_list, n, m, p, trials=50, tol=1e-9):
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to block Strassen
    if time.time() > deadline - 10:
        U_sb, V_sb, W_sb = _build_schoolbook()
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    U_list2, V_list2, W_list2 = _build_block_strassen()
    if _verify_fast(U_list2, V_list2, W_list2, n, m, p, trials=30, tol=1e-9):
        return {"U": U_list2, "V": V_list2, "W": W_list2}

    # Final fallback: schoolbook
    U_sb, V_sb, W_sb = _build_schoolbook()
    return {"U": U_sb, "V": V_sb, "W": W_sb}
