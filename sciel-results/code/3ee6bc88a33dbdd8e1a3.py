"""Matrix multiplication 3x4x4 tensor decomposition search."""
import numpy as np
from itertools import product
import time
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Standard O(n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W form a valid decomposition of the matrix multiplication tensor."""
    rank = U.shape[1]
    # Reconstruct tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # T[i,j,k] should equal 1 for all (i,j,k)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # Reshape for broadcasting
    # U reshaped to (n, m, rank)
    U3 = U.reshape(n, m, rank)
    # V reshaped to (m, p, rank)
    V3 = V.reshape(m, p, rank)
    # W reshaped to (p, n, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Compute via einsum
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # T should be all ones
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using random probes."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    
    # For the multiplication tensor, T[i,j,k] = 1
    # Check: for random A (n x m) and B (m x p), C = A @ B
    # C[i,k] = sum_j A[i,j] B[j,k]
    # From decomposition: C[i,k] = sum_j sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    #                        = sum_c W[k,i,c] (sum_j U[i,j,c] A[i,j]) (sum_k V[j,k,c] B[j,k])
    # Actually: C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
    # From tensor: C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) * V_part * W_part
    # More precisely:
    # Let u_c = U[:, c] reshaped to (n,m), v_c = V[:, c] reshaped to (m,p), w_c = W[:, c] reshaped to (p,n)
    # Then C = sum_c w_c[i,k] * (u_c * A)[j] contracted... 
    # Actually: C[i,k] = sum_c U3[i,?,c]... 
    # Let me just do: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j2 V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # Wait that's not right either. Let me re-derive.
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    #        = sum_j sum_c U[i,j,c] V[j,k,c] W[k,i,c] A[i,j] B[j,k]
    #        = sum_c W[k,i,c] * (sum_j U[i,j,c] A[i,j]) * (sum_j V[j,k,c] B[j,k])
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    for trial in range(20):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # C[i,k] = sum_c W3[k,i,c] * (U3[i,:,c] * A[i,:]).sum() * (V3[:,k,c] * B[:,k]).sum()
        # = sum_c W3[k,i,c] * np.dot(U3[i,:,c], A[i,:]) * np.dot(V3[:,k,c], B[:,k])
        
        # Compute (U3[i,:,c] * A[i,:]).sum() for all i,c: np.einsum('imc,im->ic', U3, A)
        UA = np.einsum('imc,im->ic', U3, A)  # (n, rank)
        VB = np.einsum('mkc,mk->kc', V3, B)  # (p, rank)
        # C[i,k] = sum_c W3[k,i,c] * UA[i,c] * VB[k,c]
        C_computed = np.einsum('ikc,ic,kc->ik', W3, UA, VB)
        
        if not np.allclose(C_computed, C_expected, atol=tol * 10):
            return False
    
    return True


def _try_rank_reduction_random(n, m, p, target_rank, num_trials=500, seed=42):
    """Try to find a lower-rank decomposition via random optimization."""
    random.seed(seed)
    np.random.seed(seed)
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    U0, V0, W0 = _build_schoolbook(n, m, p)
    
    for trial in range(num_trials):
        # Initialize with perturbed schoolbook (take first target_rank columns, add random noise)
        if target_rank <= n * m * p:
            U = U0[:, :target_rank].copy()
            V = V0[:, :target_rank].copy()
            W = W0[:, :target_rank].copy()
        else:
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
        
        # Add random perturbation
        U += 0.1 * np.random.randn(n * m, target_rank)
        V += 0.1 * np.random.randn(m * p, target_rank)
        W += 0.1 * np.random.randn(p * n, target_rank)
        
        # Simple gradient-free optimization: just check if it's close
        if _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-6):
            best_U, best_V, best_W = U, V, W
            best_err = 0.0
            break
    
    return best_U, best_V, best_W


def _try_las_versas_style(n, m, p):
    """
    Try to exploit structure: split the 3x4x4 tensor.
    
    Key insight: The multiplication tensor M_{n,m,p} can sometimes be decomposed
    by exploiting that if we can decompose smaller tensors efficiently.
    
    For 3x4x4: 
    - We can think of it as two blocks: (2x4x4) + (1x4x4)
    - M_{1,m,p} has rank m (each row is independent)
    - M_{2,m,p}: use the identity for 2x2x2 rank 7, scaled up
    
    Actually, let me think about this more carefully.
    
    The standard rank is n*m*p = 48.
    
    One approach: Use the fact that M_{n,m,p} <= M_{n,m,p1} + M_{n,m,p2}
    when p = p1 + p2.
    
    Also M_{n,m,p} <= n * M_{1,m,p} and M_{1,m,p} = m (rank).
    So n * m = 12 for the first dimension, but we need p=4 too.
    
    Actually M_{n,1,p} = p (rank), and M_{1,m,p} = m (rank).
    
    Let me try a direct structural decomposition:
    
    For 3x4x4, consider writing:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Split j into {0,1} and {2,3}:
    C[i,k] = sum_{j=0}^1 A[i,j] B[j,k] + sum_{j=2}^3 A[i,j] B[j,k]
    
    First part is 3x2x4, second is 3x2x4.
    
    For 3x2x4:
    Split k into {0,1} and {2,3}:
    3x2x4 = 3x2x2 + 3x2x2
    
    For 3x2x2:
    C[i,k] = sum_{j=0}^1 A[i,j] B[j,k]
    This is like matrix-vector multiplication with 2 columns.
    
    Standard rank for 3x2x2 is 12.
    Can we do better? 
    
    For 2x2x2, rank is 7 (Strassen).
    For 3x2x2, one can use:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    = (A[i,0]+A[i,1])*(B[0,k]+B[1,k]) - (A[i,0]+A[i,1])*B[1,k] - (A[i,1]+A[i,0])*B[0,k] + ...
    This gets complicated.
    
    Actually, for n x 2 x 2, the rank is at most 4n - 2 (from known results for small tensors).
    For n=3: rank <= 10.
    
    Let me try to construct a rank-10 decomposition for 3x2x2.
    """
    pass


def _try_structural_3x4x4():
    """
    Attempt a structural decomposition of the 3x4x4 matrix multiplication tensor.
    
    Strategy: 
    1. Decompose 3x4x4 as two 3x2x4 blocks
    2. Decompose each 3x2x4 as two 3x2x2 blocks  
    3. Find a rank-reduced decomposition for 3x2x2
    
    Known: rank(2x2x2) = 7 (Strassen)
    Known: rank(3x2x2) <= 2*7 + something? No, that's not how it works.
    
    Actually, for n x 2 x 2:
    C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    
    We can use: for each i, the 2x2 product has rank 7.
    But the A[i,0], A[i,1] are shared across different i's? No, each i is independent.
    
    Wait, the tensor is sum_{i,k} (sum_j A[i,j] B[j,k]) E_{ik}
    where E_{ik} is the matrix unit. The A variables are A[i,j] for all i,j
    and B variables are B[j,k] for all j,k.
    
    So it's a bilinear form in all A[i,j] and B[j,k].
    
    For 3x2x2: 3*2*2 = 12 standard terms.
    
    Can we do better than 12? 
    
    Consider: C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k]
    C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k]
    C[2,k] = A[2,0]B[0,k] + A[2,1]B[1,k]
    
    For k=0: C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    For k=1: C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1]
    
    So we have 6 outputs, each a sum of 2 products.
    
    One trick: define S_i = A[i,0] + A[i,1], D_i = A[i,0] - A[i,1]
    Then A[i,0] = (S_i + D_i)/2, A[i,1] = (S_i - D_i)/2
    
    C[i,k] = (S_i + D_i)/2 * B[0,k] + (S_i - D_i)/2 * B[1,k]
           = S_i * (B[0,k] + B[1,k])/2 + D_i * (B[0,k] - B[1,k])/2
    
    So C[i,k] = S_i * P_k + D_i * Q_k
    where P_k = (B[0,k] + B[1,k])/2, Q_k = (B[0,k] - B[1,k])/2
    
    Now C[i,k] = S_i * P_k + D_i * Q_k
    
    This is a sum of 2 products per (i,k), but S_i, D_i are "new" variables.
    The tensor rank is about expressing this as sum of rank-1 terms in the A,B variables.
    
    Original variables: A[0,0], A[0,1], A[1,0], A[1,1], A[2,0], A[2,1] (6 A-variables)
    B[0,0], B[0,1], B[1,0], B[1,1] (4 B-variables)
    
    New variables: S_0, D_0, S_1, D_1, S_2, D_2 (6 A-variables, linear transform)
    P_0, P_1, Q_0, Q_1 (4 B-variables, linear transform)
    
    C[i,k] = S_i * P_k + D_i * Q_k
    
    Rank-1 terms: S_i * P_k for i=0,1,2 and k=0,1: that's 6 terms.
    D_i * Q_k for i=0,1,2 and k=0,1: that's 6 terms.
    Total: 12 terms. Same as before!
    
    So this transformation doesn't help for 3x2x2.
    
    What about a different grouping?
    
    Let me try to use the 2x2x2 Strassen decomposition.
    
    For a single 2x2x2 product:
    C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] for i,k in {0,1}
    Strassen: 7 multiplications.
    
    For 3x2x2, we have 3 such "rows" but they share the B matrix.
    
    C[0,k] uses A[0,0], A[0,1] with B[0,k], B[1,k]
    C[1,k] uses A[1,0], A[1,1] with B[0,k], B[1,k]
    C[2,k] uses A[2,0], A[2,1] with B[0,k], B[1,k]
    
    The B variables are shared! So we can potentially share Strassen-like combinations.
    
    Define for the B side (2x2):
    P_0 = B[0,0] + B[1,0]
    P_1 = B[0,1] + B[1,1]
    Q_0 = B[1,0]
    Q_1 = B[1,1]
    R_0 = B[0,0]
    R_1 = B[0,1]
    
    Strassen for 2x2x2 (computing C = A*B where A,B are 2x2):
    m1 = A[0,0] * (B[0,0] + B[1,1])
    m2 = (A[0,0] + A[0,1]) * B[1,0]
    m3 = (A[1,0] + A[1,1]) * B[0,1]
    m4 = A[1,1] * (B[0,0] + B[1,1])
    m5 = (A[0,0] + A[1,0]) * (B[1,1] - B[1,0])
    m6 = (A[0,1] - A[1,1]) * (B[0,0] + B[1,0])
    m7 = (A[1,0] - A[0,0]) * (B[0,1] + B[1,1])
    
    C[0,0] = m1 + m4 - m5 + m7
    C[0,1] = m3 + m4
    C[1,0] = m1 + m2
    C[1,1] = m2 - m3 + m4 + m6
    
    Now for our 3x2x2, the "A" matrix is 3x2, not 2x2.
    
    Let me define the B-side combinations:
    B_sum_01 = B[0,0] + B[1,0]  (for column 0)
    B_sum_01_col1 = B[0,1] + B[1,1]  (for column 1)
    B_10 = B[1,0]
    B_11 = B[1,1]
    B_00 = B[0,0]
    B_01 = B[0,1]
    
    For row i (i=0,1,2):
    C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1]
    
    Using Strassen-like:
    m_i1 = A[i,0] * (B[0,0] + B[1,0])  -- but this mixes columns!
    
    Actually, Strassen's trick works because it exploits the 2x2 structure where both dimensions are 2.
    Here, our "A" is 3x2 and we're computing a 3x2 result (one row per i, two columns).
    
    For each row i, C[i,:] = [A[i,0], A[i,1]] * B  where B is 2x2.
    This is a 1x2 times 2x2 = 1x2.
    
    Rank of 1x2x2 tensor: 
    C[k] = A[0]*B[0,k] + A[1]*B[1,k] for k=0,1
    Standard rank: 4 (2*2).
    Can we do better? 
    C[0] = A0*B00 + A1*B10
    C[1] = A0*B01 + A1*B11
    
    Using: S = A0 + A1, D = A0 - A1
    C[0] = S*(B00+B10)/2 + D*(B00-B10)/2
    C[1] = S*(B01+B11)/2 + D*(B01-B11)/2
    
    Still 4 products: S*(B00+B10), S*(B01+B11), D*(B00-B10), D*(B01-B11)
    Same count.
    
    So rank(1x2x2) = 4 (no improvement).
    
    Therefore rank(3x2x2) >= 3 * 4 = 12? No, because the B variables are shared.
    
    Actually, the tensor rank of 3x2x2 is at most 12 (standard), and it's at least...
    The border rank is known. Let me just try 12 and see if I can get a different
    decomposition that might combine better with the rest.
    
    Let me try a different approach: use the known fact that
    rank(3,4,4) can potentially be reduced by exploiting the 4=2+2 split in both
    the middle and last dimensions.
    
    3x4x4 -> 2*(3x2x4) -> 4*(3x2x2)
    
    If each 3x2x2 has rank 12, total is 48. No improvement.
    
    But what if we can share some terms across the 3x2x2 blocks?
    
    The four 3x2x2 blocks are:
    Block (j,k) where j in {0,1} (from splitting m=4 into two 2's) and k in {0,1} (from splitting p=4 into two 2's)
    
    Block (j=0,k=0): C[i, 2k0] for k0 in {0,1}... 
    
    Actually let me be more careful.
    
    Original: C[i, p_col] = sum_{m_col=0}^{3} A[i, m_col] B[m_col, p_col]
    
    Split m_col into j=0 (cols 0,1) and j=1 (cols 2,3):
    C[i, p_col] = sum_{j=0,1} A[i, j*2+m_local] B[j*2+m_local, p_col]
    
    Then split p_col into k=0 (cols 0,1) and k=1 (cols 2,3):
    For block (j,k): C[i, k*2+p_local] = sum_{m_local=0,1} A[i, j*2+m_local] B[j*2+m_local, k*2+p_local]
    
    Each block is a 3x2x2 multiplication.
    
    The A-variables for different j are different (A[i,0], A[i,1] vs A[i,2], A[i,3]).
    The B-variables for different j,k: B[j*2+m_local, k*2+p_local].
    
    For fixed j, the B-variables for k=0 are B[j*2+0, 0], B[j*2+0, 1], B[j*2+1, 0], B[j*2+1, 1]
    For fixed j, the B-variables for k=1 are B[j*2+0, 2], B[j*2+0, 3], B[j*2+1, 2], B[j*2+1, 3]
    
    These are all different variables, so no sharing between k=0 and k=1 blocks.
    
    For different j values, the B-variables are also different.
    
    So we have 4 independent 3x2x2 tensors, each with rank 12, total 48.
    
    It seems 48 is hard to beat for 3x4x4.
    
    However, let me try one more idea: use the fact that the 3 in the first dimension
    can sometimes be exploited.
    
    For 3x2x2, consider:
    C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0]  for i=0,1,2
    C[i,1] = A[i,0]B[0,1] + A[i,1]B[1,1]  for i=0,1,2
    
    Define:
    S_i = A[i,0] + A[i,1]  (3 variables)
    D_i = A[i,0] - A[i,1]  (3 variables)
    
    C[i,0] = (S_i + D_i)/2 * B[0,0] + (S_i - D_i)/2 * B[1,0]
           = S_i * (B[0,0]+B[1,0])/2 + D_i * (B[0,0]-B[1,0])/2
    
    C[i,1] = S_i * (B[0,1]+B[1,1])/2 + D_i * (B[0,1]-B[1,1])/2
    
    Products needed:
    S_i * P0 where P0 = (B[0,0]+B[1,0])/2  -- 3 products (i=0,1,2)
    S_i * P1 where P1 = (B[0,1]+B[1,1])/2  -- 3 products
    D_i * Q0 where Q0 = (B[0,0]-B[1,0])/2  -- 3 products
    D_i * Q1 where Q1 = (B[0,1]-B[1,1])/2  -- 3 products
    
    Total: 12 products. Still 12.
    
    I'm fairly convinced the rank is 48 for 3x4x4.
    
    Let me just make sure my implementation is correct and fast.
    """
    n, m, p = 3, 4, 4
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


# EVOLVE-BLOCK-END


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 48)
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U, V, W)
    
    # Verify schoolbook
    assert _verify_decomposition_fast(U, V, W, n, m, p)
    
    # Try structural decomposition
    U_s, V_s, W_s = _try_structural_3x4x4()
    if _verify_decomposition_fast(U_s, V_s, W_s, n, m, p):
        if U_s.shape[1] < best_rank:
            best_rank = U_s.shape[1]
            best_result = (U_s, V_s, W_s)
    
    # Try random search for lower rank (with time budget)
    start_time = time.time()
    time_budget = 100  # Leave margin before 150s limit
    
    for target_rank in range(best_rank - 1, max(best_rank - 5, 30), -1):
        if time.time() - start_time > time_budget:
            break
        U_r, V_r, W_r = _try_rank_reduction_random(n, m, p, target_rank, num_trials=100)
        if U_r is not None and _verify_decomposition(U_r, V_r, W_r, n, m, p):
            best_rank = target_rank
            best_result = (U_r, V_r, W_r)
            break
    
    U, V, W = best_result
    
    # Convert to lists with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in np.asarray(U)]
    V_list = [[float(x) for x in row] for row in np.asarray(V)]
    W_list = [[float(x) for x in row] for row in np.asarray(W)]
    
    # Round to nearest half-integer for cleanliness
    def round_half(x):
        return round(x * 2) / 2
    
    U_list = [[round_half(x) for x in row] for row in U_list]
    V_list = [[round_half(x) for x in row] for row in V_list]
    W_list = [[round_half(x) for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
