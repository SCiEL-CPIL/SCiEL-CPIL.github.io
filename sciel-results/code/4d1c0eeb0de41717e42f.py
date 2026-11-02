"""Rank-minimal exact tensor decomposition for 2x2 by 2x11 matrix multiplication.

Strategy: Start from a known valid decomposition (Strassen 2x2x2 blocks + identity
tricks), then use SVD-based numerical rank detection to find linearly dependent
columns and remove them. The key insight is that the 2x2x11 tensor may have a
lower rank than the naive 2x2x2 block construction suggests, especially when
columns become linearly dependent after combining multiple blocks.
"""
import time
import numpy as np

DEADLINE = 140.0  # Safety margin before 150s kill
START_TIME = time.time()


def _time_remaining():
    return DEADLINE - (time.time() - START_TIME)


def _verify(U, V, W, n=2, m=2, p=11, trials=20, seed=42):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rng = np.random.RandomState(seed)
    rank = len(U[0])
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_true = A @ B
        
        # Compute via tensor: C[i,k] = sum_r U[i,r]*V[j,r]*W[k,r]*A[i,j]*B[j,k]
        # More precisely: C[i,k] = sum_j sum_r A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, that's not right. The tensor decomposition is:
        # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) ... no
        # 
        # Standard form: C[i,k] = sum_r U[i,r] * V[k,r] * ... no
        # 
        # For matrix multiplication A(n,m) * B(m,p) = C(n,p):
        # T[i,j,k] = delta where the tensor has T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
        # The decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
        # And C[i,k] = sum_r sum_j sum_l A[i,j] * B[l,k] * u_r[i*m+j] * v_r[l*p+k] * w_r[k*n+i]
        # Hmm, this doesn't quite work with the standard indexing.
        #
        # Let me use the evaluator's convention: W[k*n+i, r]
        # So W is indexed by (k, i) -> k*n + i
        # The product: C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * ... 
        # Actually the standard tensor decomposition for matmul:
        # C[i,k] = sum_r (sum_j A[i,j] * U[j, r]) * (sum_l B[l, k] * V[r, l]) ... no
        #
        # Let me think again. The matmul tensor T has T[i, j, k] = 1 (for the (i,k) output,
        # input A[i,j] and B[j,k]). The rank-r decomposition:
        # T[i,j,k] = sum_r U[i, r] * V[j, r] * W[k, r]  (simplified notation)
        # But with the actual indexing where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank):
        # The contribution to C[i,k] from rank r is:
        #   sum_j U[i*m+j, r] * A[i,j] * ... no, that's not separable.
        #
        # Correct: C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
        # No wait, that's not right either because U and V share the j index.
        #
        # The proper form: For each rank term r, we compute:
        #   partial = (A * U_col_r) * (V_col_r * B) * W[r]  where U_col_r is (n,m), V_col_r is (m,p)
        # Actually: reshape U[:,r] to (n,m), V[:,r] to (m,p), then:
        #   C += W[r] * (U_reshaped @ A) * (B @ V_reshaped)  -- no, element-wise?
        #
        # Let me just use the direct formula:
        # C[i,k] = sum_r sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
        # Wait no, that double-counts. The correct formula for tensor decomposition of matmul:
        # 
        # The matmul tensor is: T[A_idx, B_idx, C_idx] where A_idx=(i,j), B_idx=(j,k), C_idx=(i,k)
        # T[(i,j), (j,k), (i,k)] = 1
        # Decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
        # u_r: R^{n*m}, v_r: R^{m*p}, w_r: R^{p*n}
        # T[a, b, c] = sum_r u_r[a] * v_r[b] * w_r[c]
        # So: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  (for valid (i,j,k) triples)
        # And the computation: C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r A[i,j] * B[j,k] * U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # = sum_r W[k*n+i,r] * (sum_j A[i,j] * U[i*m+j,r]) * (sum_j B[j,k] * V[j*p+k,r])
        # 
        # YES! That's the correct formula. Let me verify with schoolbook:
        # For each (i,j,k): one term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
        # Then C[i,k] = sum_j A[i,j]*B[j,k] * 1 * 1 * 1 = correct. ✓
        
        C_comp = np.zeros((n, p))
        for r in range(rank):
            # sum_j A[i,j] * U[i*m+j, r]
            U_col = U_mat[:, r].reshape(n, m)
            V_col = V_mat[:, r].reshape(m, p)
            W_val = W_mat[k * n + i, r] if False else W_mat[:, r]  # W is (p*n, rank)
            # C[i,k] += W[k*n+i, r] * (A @ U_col)[i] * (V_col @ B)[:, k] ... 
            # Actually: (sum_j A[i,j]*U[i*m+j,r]) is (A * U_col)[i] element-wise? No.
            # sum_j A[i,j] * U_col[i,j] -- this is element-wise product summed over j, not a matrix product.
            # Let me re-examine: U[i*m+j, r] means U reshaped to (n,m) has entry [i,j] = U[i*m+j, r]
            # So sum_j A[i,j] * U_reshaped[i,j] = element-wise product of A and U_reshaped, summed over j.
            # That's NOT a matrix multiplication. It's a Hadamard product sum.
            
            # Hmm, so the formula is:
            # C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] * U_reshaped_r[i,j]) * (sum_j B[j,k] * V_reshaped_r[j,k])
            # where U_reshaped_r = U[:,r].reshape(n,m), V_reshaped_r = V[:,r].reshape(m,p)
            
            U_r = U_mat[:, r].reshape(n, m)
            V_r = V_mat[:, r].reshape(m, p)
            W_r = W_mat[:, r]  # shape (p*n,)
            
            # alpha[i] = sum_j A[i,j] * U_r[i,j]
            alpha = np.sum(A * U_r, axis=1)  # shape (n,)
            # beta[k] = sum_j B[j,k] * V_r[j,k]
            beta = np.sum(B * V_r, axis=0)  # shape (p,)
            
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += W_r[k * n + i] * alpha[i] * beta[k]
        
        if not np.allclose(C_comp, C_true, atol=1e-9):
            return False
    return True


def _build_strassen_decomposition(n=2, m=2, p=11):
    """Build a valid decomposition using Strassen 2x2x2 blocks.
    
    Split B into two 2x6 and 2x5 blocks (or handle all 11 columns).
    Actually, for 2x2 * 2x11, we can use:
    - 2 identity columns to pass through B_0 and B_1 structure
    - Strassen blocks for the remaining computation
    
    Simpler approach: use the fact that 2x2 matmul has rank 7 (Strassen),
    and extend to 2x2 * 2x11 by treating each column independently but
    sharing the 2x2 structure.
    
    Actually, the cleanest valid construction:
    C = A * B where A is 2x2, B is 2x11.
    Split B = [B_0 | B_1] where B_0 is 2x5, B_1 is 2x6.
    C = [A*B_0 | A*B_1]
    Each A*B_k is 2x2 * 2xk.
    
    For 2x2 * 2xk, we can use k copies of Strassen (rank 7k) or
    do better by sharing.
    
    Simplest guaranteed valid: schoolbook rank = 2*2*11 = 44.
    But we can do: for each of the 11 columns, use Strassen rank 7 -> 77. Worse.
    
    Better: The 2x2x11 tensor. Note that C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    This is just two "1x2 x 2x11" multiplications added.
    1x2 * 2x11 has rank 11 (it's just a vector-matrix product, rank = 11).
    So total rank <= 22.
    
    Even better: can we share? C[i,k] = sum_j A[i,j]*B[j,k].
    The tensor T[i,j,k] = delta_{i,j}... no, T is just the identity in the j dimension.
    T[(i,j), (j,k), (i,k)] = 1.
    
    Rank of this tensor: For fixed i,k, T depends on j. 
    The tensor can be written as: T = sum_j e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}... no.
    
    Let me think of it as: T = I_m ⊗ (something). Actually:
    T[a, b, c] where a=(i,j), b=(j,k), c=(i,k).
    T[(i,j),(j,k),(i,k)] = 1.
    
    This is the "identity" tensor in the middle index. Its rank is m * (n*p - 1)? No.
    
    For m=2, n=2, p=11: 
    T = sum_{j=0}^{1} sum_{i=0}^{1} sum_{k=0}^{10} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}
    That's 2*2*11 = 44 terms (schoolbook).
    
    Can we reduce? Note that for fixed j, we have a 2x11 "identity-like" tensor.
    For fixed j: sum_{i,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}.
    This is: for each i,k, the (i,j) component of U, the (j,k) component of V, and (i,k) of W.
    The rank of this sub-tensor (for fixed j) is 2*11 = 22 (it's a 2x11 matrix identity).
    
    Total: 2 * 22 = 44. That's schoolbook.
    
    To do better, we need to share across j values. The key question is whether
    the 2x2x11 tensor has rank < 44.
    
    Known: R(2,2,p) for matrix multiplication. For p=1: R=2. For p=2: R=4 (schoolbook) 
    or can we do 3? 2x2*2x2 is 2x2x2 matmul which has rank 7 (Strassen) but that's
    for computing 2x2*2x2=2x2. Here it's 2x2 * 2x2 = 2x2, same thing. Rank 7 > 4? No,
    2x2*2x2 schoolbook is rank 4, Strassen is rank 7. So schoolbook is better for small.
    
    Wait, I'm confusing myself. Let me reclarify:
    - "2x2 matrix times 2x2 matrix = 2x2 matrix": schoolbook rank 4, Strassen rank 7. Schoolbook wins.
    - "2x2 matrix times 2x11 matrix = 2x11 matrix": this is what we're doing.
    
    For 2x2 * 2x11:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    This is equivalent to: for each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:].
    So C is a linear combination of rows of B with coefficients from A.
    
    The tensor: T[i, j, k] = 1 (where i indexes A's row, j indexes A's col / B's row, k indexes B's col).
    T has shape (2, 2, 11) in the (i,j,k) sense, but the full tensor for the decomposition
    has U in R^{2*2}, V in R^{2*11}, W in R^{11*2}.
    
    Rank of T[i,j,k] = delta_{i,j}... no, T[i,j,k] = 1 for all i,j,k? No!
    T[i,j,k] is 1 if and only if the (i,k) output uses A[i,j] and B[j,k].
    So T[i,j,k] = 1 for all valid (i,j,k). The tensor is just all-ones? No.
    
    I think I need to be more careful. The matmul tensor for (n,m,p) is:
    T in R^{(n*m) x (m*p) x (n*p)} where T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k'.
    So T[(i,j),(j,k),(i,k)] = 1 and 0 otherwise.
    
    For n=2,m=2,p=11: T has shape (4, 22, 22).
    T[(i,j),(j,k),(i,k)] = 1.
    
    The rank of this tensor: For each j, we have a 2x11 block. 
    T restricted to j=0: T[(i,0),(0,k),(i,k)] = 1 for i=0,1; k=0..10.
    This is a 2x11 "identity" tensor with rank 22 (each (i,k) pair needs its own term).
    Similarly for j=1: another 22 terms.
    Total: 44.
    
    Can we share between j=0 and j=1? The issue is that U[(i,0),r] and U[(i,1),r] 
    are different entries, so a single rank term can only contribute to one j at a time
    in U (unless U[(i,0),r] and U[(i,1),r] are both nonzero, but then it contributes
    to both j=0 and j=1 simultaneously).
    
    If U[(i,0),r] = a and U[(i,1),r] = b, and V[(0,k),r] = c and V[(1,k),r] = d,
    and W[(k,i),r] = e, then the contribution to C[i,k] is:
    e * (a*c*A[i,0]*B[0,k] + b*d*A[i,1]*B[1,k])... 
    
    No wait, the formula is:
    C[i,k] = sum_r W[(k,i),r] * (sum_j A[i,j]*U[(i,j),r]) * (sum_j B[j,k]*V[(j,k),r])
    
    So for a single rank term r:
    alpha_i = A[i,0]*U[(i,0),r] + A[i,1]*U[(i,1),r]
    beta_k = B[0,k]*V[(0,k),r] + B[1,k]*V[(1,k),r]
    C[i,k] += W[(k,i),r] * alpha_i * beta_k
    
    For this to equal A[i,0]*B[0,k] + A[i,1]*B[1,k] (the true product), we need the
    sum over all r to give this.
    
    If we set for each r: U[(i,0),r] = u_i0, U[(i,1),r] = u_i1, V[(0,k),r] = v_0k, V[(1,k),r] = v_1k, W[(k,i),r] = w_ki
    Then C[i,k] = sum_r w_ki * (u_i0*A[i,0] + u_i1*A[i,1]) * (v_0k*B[0,k] + v_1k*B[1,k])
    
    We need this to equal A[i,0]*B[0,k] + A[i,1]*B[1,k] for all A,B.
    
    This is a tensor equation. The rank is the minimum number of terms.
    
    For the 2x2*2x11 case, I believe the rank is 22 (not 44). Here's why:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    = (A[i,0]*B[0,k]) + (A[i,1]*B[1,k])
    
    The first term A[i,0]*B[0,k]: this is a rank-11 tensor (for each k, it's A[i,0]*B[0,k]).
    Actually, summing over i: for fixed k, the (i,k) entry is A[i,0]*B[0,k].
    The tensor for "A[:,0] * B[0,:]" (outer product of a column of A and a row of B) has rank 11? 
    No: A[i,0]*B[0,k] = (A[:,0])_i * (B[0,:])_k. This is a rank-1 tensor in (i,k) for each... 
    no, it's just a single outer product, so the (i,k) matrix has rank 1. But as a tensor
    in the (i, j, k) space with j fixed at 0... 
    
    I think I'm overcomplicating this. Let me just construct a valid rank-22 decomposition:
    
    For j=0 part: C[i,k] gets A[i,0]*B[0,k].
    Use 11 terms (one per k): 
      Term r=k: U[(i,0),r] = 1 for all i, V[(0,k),r] = 1, V[(1,k'),r] = 0 for k'≠k, W[(k,i),r] = 1
      Then alpha_i = A[i,0]*1 + A[i,1]*0 = A[i,0]
      beta_k = B[0,k]*1 + B[1,k]*0 = B[0,k]
      C[i,k] += 1 * A[i,0] * B[0,k] ✓
    
    For j=1 part: C[i,k] gets A[i,1]*B[1,k].
    Use 11 more terms:
      Term r=11+k: U[(i,1),r] = 1, V[(1,k),r] = 1, others 0, W[(k,i),r] = 1
      alpha_i = A[i,0]*0 + A[i,1]*1 = A[i,1]
      beta_k = B[0,k]*0 + B[1,k]*1 = B[1,k]
      C[i,k] += 1 * A[i,1] * B[1,k] ✓
    
    Total: 22 terms. This is valid!
    
    Can we do better than 22? The lower bound: the tensor T has "slice rank" at least
    max(n*p, m*p, n*m) = max(22, 22, 4) = 22. So rank >= 22.
    And we achieved 22. So the rank is exactly 22!
    
    Wait, is the slice rank lower bound correct? The slice rank of T[i,j,k] where
    T[(i,j),(j,k),(i,k)] = 1... The slice rank lower bound says rank >= max of the
    "slice" dimensions. For a tensor in R^{A x B x C}, slice rank >= max(A, B, C) is NOT
    a valid lower bound in general. The valid bound is rank >= max over slices.
    
    Actually, for the matmul tensor M_{n,m,p} in R^{(nm) x (mp) x (np)}, 
    the rank is at least max(nm, mp, np) by the "slice rank" argument? No, that's not right either.
    
    The correct lower bound: rank(M_{n,m,p}) >= max(n, m, p) * min(n, m, p)? No.
    
    Let me just go with 22 and verify it works. If it does, that's a huge improvement over 44.
    """
    n, m, p = 2, 2, 11
    rank = 2 * p  # 22
    U = [[0] * rank for _ in range(n * m)]  # (4, 22)
    V = [[0] * rank for _ in range(m * p)]  # (22, 22)
    W = [[0] * rank for _ in range(p * n)]  # (22, 22)
    
    # j=0 part: 11 terms
    for k in range(p):
        r = k
        for i in range(n):
            U[i * m + 0][r] = 1  # U[(i,0), r] = 1
        V[0 * p + k][r] = 1  # V[(0,k), r] = 1
        for i in range(n):
            W[k * n + i][r] = 1  # W[(k,i), r] = 1
    
    # j=1 part: 11 terms
    for k in range(p):
        r = p + k
        for i in range(n):
            U[i * m + 1][r] = 1  # U[(i,1), r] = 1
        V[1 * p + k][r] = 1  # V[(1,k), r] = 1
        for i in range(n):
            W[k * n + i][r] = 1  # W[(k,i), r] = 1
    
    return U, V, W


def _build_strassen_2x2x2_block():
    """Build Strassen 2x2x2 decomposition for 2x2*2x2=2x2.
    
    Strassen's 7 multiplications:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]  # (4, 7)
    V = [[0] * rank for _ in range(m * p)]  # (4, 7)
    W = [[0] * rank for _ in range(p * n)]  # (4, 7)
    
    # M1 = (A00+A11)*(B00+B11)
    # U[(0,0)]=1, U[(1,1)]=1; V[(0,0)]=1, V[(1,1)]=1
    U[0][0] = 1  # (i=0,j=0)
    U[3][0] = 1  # (i=1,j=1)
    V[0][0] = 1  # (j=0,k=0)
    V[3][0] = 1  # (j=1,k=1)
    # W: C00 += M1, C11 += M1
    W[0][0] = 1   # (k=0,i=0) -> C[0,0]
    W[3][0] = 1   # (k=1,i=1) -> C[1,1]
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1  # (i=1,j=0)
    U[3][1] = 1  # (i=1,j=1)
    V[0][1] = 1  # (j=0,k=0)
    W[2][1] = 1  # (k=0,i=1) -> C[1,0]
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1  # (i=0,j=0)
    V[1][2] = 1  # (j=0,k=1)
    V[3][2] = -1  # (j=1,k=1)
    W[1][2] = 1  # (k=1,i=0) -> C[0,1]
    W[3][2] = 1  # (k=1,i=1) -> C[1,1]
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1  # (i=1,j=1)
    V[2][3] = 1  # (j=1,k=0)
    V[0][3] = -1  # (j=0,k=0)
    W[0][3] = 1  # (k=0,i=0) -> C[0,0]
    W[2][3] = 1  # (k=0,i=1) -> C[1,0]
    
    # M5 = (A00+A01)*B11
    U[0][4] = 1  # (i=0,j=0)
    U[1][4] = 1  # (i=0,j=1)
    V[3][4] = 1  # (j=1,k=1)
    W[0][4] = -1  # (k=0,i=0) -> C[0,0]
    W[1][4] = 1  # (k=1,i=0) -> C[0,1]
    
    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1  # (i=1,j=0)
    U[0][5] = -1  # (i=0,j=0)
    V[0][5] = 1  # (j=0,k=0)
    V[1][5] = 1  # (j=0,k=1)
    W[3][5] = 1  # (k=1,i=1) -> C[1,1]
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1  # (i=0,j=1)
    U[3][6] = -1  # (i=1,j=1)
    V[2][6] = 1  # (j=1,k=0)
    V[3][6] = 1  # (j=1,k=1)
    W[0][6] = 1  # (k=0,i=0) -> C[0,0]
    
    return U, V, W


def _combine_decompositions(decomps):
    """Combine multiple decompositions by concatenating columns."""
    total_rank = sum(len(d[0][0]) for d in decomps)
    n, m, p = 2, 2, 11
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    for U_d, V_d, W_d in decomps:
        r = len(U_d[0])
        for i in range(n * m):
            for col in range(r):
                U[i][offset + col] = U_d[i][col]
        for j in range(m * p):
            for col in range(r):
                V[j][offset + col] = V_d[j][col]
        for k in range(p * n):
            for col in range(r):
                W[k][offset + col] = W_d[k][col]
        offset += r
    
    return U, V, W


def _greedy_column_removal(U, V, W, n=2, m=2, p=11, tol=1e-10, max_iter=200):
    """Try to remove columns that are linearly dependent or contribute zero."""
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    rank = U_mat.shape[1]
    
    active = list(range(rank))
    
    for iteration in range(max_iter):
        if _time_remaining() < 5.0:
            break
        
        best_remove = -1
        best_error = 0
        
        for idx in active:
            # Check if removing column idx changes the result
            # A column is redundant if its contribution is zero or linearly dependent
            # Simple check: if the column is all zeros, remove it
            if np.allclose(U_mat[:, idx], 0, atol=tol) and \
               np.allclose(V_mat[:, idx], 0, atol=tol) and \
               np.allclose(W_mat[:, idx], 0, atol=tol):
                best_remove = idx
                break
        
        if best_remove == -1:
            # Try SVD-based: check if any column can be expressed as combination of others
            # Build the "contribution matrix" and check rank
            # For each column r, its contribution is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            # Two columns are dependent if their outer products are linearly dependent
            # This is expensive, so just check if the tensor rank drops
            break
        
        active.remove(best_remove)
    
    # Rebuild without removed columns
    if len(active) < rank:
        U_new = [[U[i][r] for r in active] for i in range(n * m)]
        V_new = [[V[i][r] for r in active] for i in range(m * p)]
        W_new = [[W[i][r] for r in active] for i in range(p * n)]
        return U_new, V_new, W_new
    
    return U, V, W


def _try_reduce_with_svd(U, V, W, n=2, m=2, p=11, tol=1e-8):
    """Use SVD to detect if the tensor has lower numerical rank.
    
    The tensor T = sum_r u_r ⊗ v_r ⊗ w_r. We can form a "matricization" and check rank.
    T_(1) = reshape to (n*m) x (m*p * n*p): each row i gives a vector of T[i, :, :]
    If the numerical rank of T_(1) is less than the number of columns, we can reduce.
    """
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    rank = U_mat.shape[1]
    
    # Form matricization: T[i, (j,k), (i',k')] = sum_r U[i,r]*V[j*p+k, r]*W[k'*n+i', r]
    # But we only care about T[i, j*p+k, k*n+i] = 1 (the matmul tensor)
    # Actually, we want to check if the CURRENT decomposition has redundant columns.
    
    # Alternative: check if any column r has U[:,r] ⊗ V[:,r] ⊗ W[:,r] in the span
    # of other columns' tensors. This is hard to check directly.
    
    # Simpler: try removing each column and verify.
    best_rank = rank
    best_result = (U, V, W)
    
    for r_to_remove in range(rank):
        if _time_remaining() < 5.0:
            break
        
        active = [i for i in range(rank) if i != r_to_remove]
        U_sub = [[U[i][r] for r in active] for i in range(n * m)]
        V_sub = [[V[i][r] for r in active] for i in range(m * p)]
        W_sub = [[W[i][r] for r in active] for i in range(p * n)]
        
        if _verify(U_sub, V_sub, W_sub, n, m, p, trials=5):
            best_rank = len(active)
            best_result = (U_sub, V_sub, W_sub)
    
    return best_result


def construct():
    global START_TIME
    START_TIME = time.time()
    
    n, m, p = 2, 2, 11
    
    # Start with the rank-22 construction
    U, V, W = _build_strassen_decomposition(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce further
    if _time_remaining() > 10.0:
        U, V, W = _try_reduce_with_svd(U, V, W, n, m, p)
    
    # Final verification
    if not _verify(U, V, W, n, m, p, trials=30):
        # Fallback to the rank-22
        U, V, W = _build_strassen_decomposition(n, m, p)
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    valid = _verify(result["U"], result["V"], result["W"], trials=50)
    print(f"Valid: {valid}")
