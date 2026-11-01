"""Matrix multiplication 2,3,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    Strategy: Use a Kronecker product structure.
    The 2x3 * 3x6 multiplication tensor can be factored as the Kronecker product
    of a 2x1 * 1x2 multiplication tensor (rank 2) and a 1x3 * 3x3 multiplication tensor (rank 3).
    This yields a structured decomposition of rank 2 * 3 = 6.
    
    Let's verify the dimensions and structure.
    Multiplication C = A * B where A is (n x m) = (2 x 3), B is (m x p) = (3 x 6).
    Tensor T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    
    We want to write T = sum_r U_r (x) V_r (x) W_r.
    
    Let's try to factor the indices.
    n=2, m=3, p=6.
    
    Factor 1: n1=2, m1=1, p1=2. Rank r1 = 2.
    Factor 2: n2=1, m2=3, p2=3. Rank r2 = 3.
    
    Wait, m must be consistent. m = m1 * m2? No.
    In Kronecker product of tensors, the dimensions are component-wise products?
    T_A (size n1*m1, m1*p1, p1*n1) and T_B (size n2*m2, m2*p2, p2*n2).
    The Kronecker product T_A x T_B has size (n1*m1*n2*m2, m1*p1*m2*p2, p1*n1*p2*n2).
    This corresponds to n = n1*n2, m = m1*m2, p = p1*p2.
    
    Here n=2, m=3, p=6.
    Can we write 2 = 2*1, 3 = 1*3, 6 = 2*3?
    Yes.
    Factor 1: n1=2, m1=1, p1=2.
    Factor 2: n2=1, m2=3, p2=3.
    
    Let's check the ranks.
    For factor 1 (2x1 * 1x2):
    T1[i, j, k] = 1 if i=i1, j=j1, k=k1 with i1 in [0,1], j1 in [0,0], k1 in [0,1].
    Actually, let's just construct the tensor and find its rank.
    For 2x1 * 1x2:
    A is 2x1, B is 1x2. C is 2x2.
    C[i, k] = sum_j A[i, j] B[j, k] = A[i, 0] B[0, k].
    Tensor T1 has size (2*1, 1*2, 2*2) = (2, 2, 4).
    Indices: a=i*1+0=i, b=0*2+k=k, c=k*2+i.
    T1[i, k, 2k+i] = 1.
    i in {0,1}, k in {0,1}.
    Non-zeros:
    i=0, k=0: T1[0, 0, 0] = 1
    i=0, k=1: T1[0, 1, 2] = 1
    i=1, k=0: T1[1, 0, 1] = 1
    i=1, k=1: T1[1, 1, 3] = 1
    This tensor is rank 2?
    Let's check.
    T1 = e_0 x e_0 x e_0 + e_0 x e_1 x e_2 + e_1 x e_0 x e_1 + e_1 x e_1 x e_3.
    This looks like it can be decomposed into 2 terms?
    U1 = e_0 + e_1, V1 = e_0 + e_1, W1 = e_0 + e_1 + e_2 + e_3? No.
    Actually, the rank of 2x1 * 1x2 is 2.
    Proof: C = A * B. A is column vector a (2x1), B is row vector b (1x2). C = a b^T.
    Tensor T(a, b, c) = 1 if c = a b^T.
    This is the rank-2 tensor for outer product.
    U = [e_0, e_1], V = [e_0, e_1], W = [e_0, e_1]? No.
    Let's just compute it.
    
    For factor 2 (1x3 * 3x3):
    n2=1, m2=3, p2=3.
    A is 1x3, B is 3x3. C is 1x3.
    C[i=0, k] = sum_j A[0, j] B[j, k].
    Tensor T2 has size (1*3, 3*3, 3*1) = (3, 9, 3).
    Indices: a=j, b=3j+k, c=k.
    T2[j, 3j+k, k] = 1.
    This is the standard matrix multiplication tensor for 1x3 * 3x3.
    The rank of 1xn * nxn multiplication is known to be 2n-1? No.
    For 1x1 * 1x1, rank 1.
    For 1x2 * 2x2, rank 3?
    For 1x3 * 3x3, rank 5?
    Let's check.
    C = A B. A is row vector (1x3), B is 3x3 matrix.
    C[k] = sum_j A[j] B[j, k].
    This is 3 scalar products of length 3? No.
    It's 3 outputs, each is a dot product of A (length 3) and column of B (length 3).
    But A is shared.
    This is the tensor for multiplying a vector by a matrix.
    The rank of the vector-matrix multiplication tensor (1xn, nxn) is 2n-1?
    Let's check n=1. Rank 1. 2*1-1=1. Correct.
    n=2. Rank 3?
    T[a, b, c] with a in 0..1, b in 0..3, c in 0..1.
    T[j, 2j+k, k] = 1.
    j=0, k=0: T[0, 0, 0]=1
    j=0, k=1: T[0, 1, 1]=1
    j=1, k=0: T[1, 2, 0]=1
    j=1, k=1: T[1, 3, 1]=1
    This is rank 3?
    U = [e_0, e_1, e_0+e_1]
    V = [e_0, e_2, e_1+e_3] ?
    W = [e_0, e_1, e_0+e_1] ?
    Let's check rank of 1x3 * 3x3.
    It is known that the rank of the (1, n, n) matrix multiplication tensor is 2n-1.
    So for n=3, rank is 5.
    
    So total rank = rank(T1) * rank(T2) = 2 * 5 = 10.
    
    Let's verify the Kronecker product construction.
    T = T1 (x) T2.
    Size of T1: (2*1, 1*2, 2*2) = (2, 2, 4).
    Size of T2: (1*3, 3*3, 3*1) = (3, 9, 3).
    Size of T: (2*3, 2*9, 4*3) = (6, 18, 12).
    But we need (2*3, 3*6, 6*2) = (6, 18, 12).
    Matches!
    
    So we can construct U, V, W as Kronecker products of U1, U2, etc.
    
    We need to find explicit decompositions for T1 and T2.
    
    T1 (2x1 * 1x2):
    U1 (2x2), V1 (2x2), W1 (4x2).
    T1[a, b, c] = sum_r U1[a, r] V1[b, r] W1[c, r].
    Non-zeros: T1[0,0,0]=1, T1[0,1,2]=1, T1[1,0,1]=1, T1[1,1,3]=1.
    Let's try:
    U1 = [[1, 1], [1, -1]]
    V1 = [[1, 0], [0, 1]]
    W1 = [[1, 0], [0, 1], [0, 1], [1, 0]] ?
    Let's check r=0:
    U1[:,0] = [1, 1]^T. V1[:,0] = [1, 0]^T. W1[:,0] = [1, 0, 0, 1]^T.
    Term 0: 1*1*1=1 at (0,0,0). 1*0*1=0 at (1,0,0).
    1*1*0=0 at (0,0,1). 1*0*0=0 at (1,0,1).
    1*1*0=0 at (0,0,2). 1*0*1=0 at (1,0,3).
    1*1*1=1 at (0,0,3)? No, W1[3,0]=1. T1[0,0,3] should be 0.
    Wait, T1[0,0,3] is not in the list.
    My W1 was wrong.
    
    Let's use a systematic approach.
    T1 is the tensor for C = A B where A is 2x1, B is 1x2.
    C[i, k] = A[i] B[k].
    This is the outer product tensor.
    It is well-known that the rank of the outer product tensor of size n x m is n + m - 1? No.
    For 2x2 outer product, rank is 3?
    Let's check 1x1 * 1x1. Rank 1.
    1x2 * 2x1. Rank 2.
    2x2 * 2x2. Rank 3.
    
    Actually, let's just use ALS on the small tensors to be safe, or find known decompositions.
    Given the small size, I will use ALS for T1 and T2 separately, then combine.
    
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Factor 1: n1=2, m1=1, p1=2
    n1, m1, p1 = 2, 1, 2
    # Factor 2: n2=1, m2=3, p2=3
    n2, m2, p2 = 1, 3, 3
    
    # Build T1
    T1 = np.zeros((n1 * m1, m1 * p1, p1 * n1))
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                a = i * m1 + j
                b = j * p1 + k
                c = k * n1 + i
                T1[a, b, c] = 1.0
    
    # Build T2
    T2 = np.zeros((n2 * m2, m2 * p2, p2 * n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                a = i * m2 + j
                b = j * p2 + k
                c = k * n2 + i
                T2[a, b, c] = 1.0
    
    vec_T1 = T1.reshape(-1)
    vec_T2 = T2.reshape(-1)
    
    def als_decompose(T, vec_T, dims, max_rank, start_rank, time_limit):
        n_d, m_d, p_d = dims
        N1, N2, N3 = n_d * m_d, m_d * p_d, p_d * n_d
        best_rank = None
        best_U = None
        best_V = None
        best_W = None
        
        for r in range(start_rank, max_rank + 1):
            if time.time() > time_limit:
                break
            for trial in range(20):
                if time.time() > time_limit:
                    break
                U = np.random.randn(N1, r)
                V = np.random.randn(N2, r)
                W = np.random.randn(N3, r)
                
                for it in range(200):
                    if time.time() > time_limit:
                        break
                    
                    # Solve U
                    # A_U size: (N2*N3) x (N1*r)
                    # A_U[(b,c), (a, r_idx)] = V[b, r_idx] * W[c, r_idx]
                    # This is too large to build explicitly if N1*r is large, but here it's small.
                    # N2*N3 = 2*4=8 or 9*3=27.
                    # N1*r = 2*r or 3*r.
                    # It's small.
                    
                    A_U = np.zeros((N2 * N3, N1 * r))
                    for a in range(N1):
                        for r_idx in range(r):
                            idx_u = a * r + r_idx
                            vec_v = np.kron(np.ones(N3), V[:, r_idx])
                            vec_w = np.kron(W[:, r_idx], np.ones(N2))
                            A_U[:, idx_u] = vec_v * vec_w
                    
                    try:
                        U_new, _, _, _ = np.linalg.lstsq(A_U, vec_T, rcond=None)
                    except:
                        break
                    U = U_new.reshape(N1, r)
                    
                    # Solve V
                    A_V = np.zeros((N1 * N3, N2 * r))
                    for b in range(N2):
                        for r_idx in range(r):
                            idx_v = b * r + r_idx
                            vec_u = np.kron(np.ones(N3), U[:, r_idx])
                            vec_w = np.kron(W[:, r_idx], np.ones(N1))
                            A_V[:, idx_v] = vec_u * vec_w
                            
                    try:
                        V_new, _, _, _ = np.linalg.lstsq(A_V, vec_T, rcond=None)
                    except:
                        break
                    V = V_new.reshape(N2, r)
                    
                    # Solve W
                    A_W = np.zeros((N1 * N2, N3 * r))
                    for c in range(N3):
                        for r_idx in range(r):
                            idx_w = c * r + r_idx
                            vec_u = np.kron(U[:, r_idx], np.ones(N2))
                            vec_v = np.kron(np.ones(N1), V[:, r_idx])
                            A_W[:, idx_w] = vec_u * vec_w
                            
                    try:
                        W_new, _, _, _ = np.linalg.lstsq(A_W, vec_T, rcond=None)
                    except:
                        break
                    W = W_new.reshape(N3, r)
                    
                    # Check error
                    recon = np.zeros((N1, N2, N3))
                    for r_idx in range(r):
                        recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
                    error = np.max(np.abs(recon - T))
                    if error < 1e-8:
                        best_rank = r
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        break
                if best_rank is not None:
                    break
            if best_rank is not None:
                break
                
        return best_rank, best_U, best_V, best_W
    
    start_time = time.time()
    deadline = start_time + 100
    
    # T1: 2x1 * 1x2. Rank should be 2.
    r1, U1, V1, W1 = als_decompose(T1, vec_T1, (n1, m1, p1), 5, 1, deadline)
    if r1 is None:
        r1 = n1 * m1 * p1 # Fallback
        U1 = np.zeros((n1*m1, r1))
        V1 = np.zeros((m1*p1, r1))
        W1 = np.zeros((p1*n1, r1))
        col = 0
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U1[i*m1+j, col] = 1
                    V1[j*p1+k, col] = 1
                    W1[k*n1+i, col] = 1
                    col += 1
                    
    # T2: 1x3 * 3x3. Rank should be 5.
    r2, U2, V2, W2 = als_decompose(T2, vec_T2, (n2, m2, p2), 9, 1, deadline)
    if r2 is None:
        r2 = n2 * m2 * p2 # Fallback
        U2 = np.zeros((n2*m2, r2))
        V2 = np.zeros((m2*p2, r2))
        W2 = np.zeros((p2*n2, r2))
        col = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U2[i*m2+j, col] = 1
                    V2[j*p2+k, col] = 1
                    W2[k*n2+i, col] = 1
                    col += 1
    
    # Combine
    r_total = r1 * r2
    U_total = np.zeros((n * m, r_total))
    V_total = np.zeros((m * p, r_total))
    W_total = np.zeros((p * n, r_total))
    
    # Kronecker product of matrices
    # U_total = U1 (x) U2
    U_total = np.kron(U1, U2)
    V_total = np.kron(V1, V2)
    W_total = np.kron(W1, W2)
    
    # Verify
    T_full = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1.0
                
    recon_full = np.zeros((n * m, m * p, p * n))
    for r_idx in range(r_total):
        recon_full += np.einsum('a,b,c->abc', U_total[:, r_idx], V_total[:, r_idx], W_total[:, r_idx])
        
    if np.max(np.abs(recon_full - T_full)) < 1e-8:
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
        
    # Fallback: Schoolbook
    rank = n * m * p
    U_fb = np.zeros((n*m, rank))
    V_fb = np.zeros((m*p, rank))
    W_fb = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
