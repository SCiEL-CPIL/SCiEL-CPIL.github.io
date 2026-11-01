"""Matrix multiplication 2,3,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    Strategy: Use the Kronecker product structure.
    n=2, m=3, p=6.
    Factor as n = n1*n2 = 2*1, m = m1*m2 = 3*1, p = p1*p2 = 6*1? No.
    Let's try:
    Factor 1: n1=2, m1=3, p1=3. (2x3 * 3x3)
    Factor 2: n2=1, m2=1, p2=2. (1x1 * 1x2)
    Then n=2*1=2, m=3*1=3, p=3*2=6.
    
    Rank of (2,3,3):
    Known rank for 2x3 * 3x2 is 8?
    Let's check.
    The rank of (2,3,2) is 8 (Strassen-like for rectangular?).
    Actually, let's just compute the rank of the (2,3,3) tensor and (1,1,2) tensor.
    
    (1,1,2) tensor:
    A is 1x1, B is 1x2. C is 1x2.
    C[0, k] = A[0,0] B[0,k].
    Tensor T2 size (1, 2, 2).
    T2[0, k, 2*0+k] = T2[0, k, k] = 1.
    T2 = e0 x e0 x e0 + e0 x e1 x e1.
    Rank 2.
    
    (2,3,3) tensor:
    n1=2, m1=3, p1=3.
    Size (6, 9, 6).
    Let's find the rank.
    If rank is 8, total rank = 8 * 2 = 16.
    If rank is 7, total rank = 14.
    Schoolbook is 36.
    
    Let's use a fast exact rank search for the (2,3,3) tensor.
    The tensor is small enough (6*9*6 = 324 entries) to use exact arithmetic or high precision.
    But we can also just use the known Strassen-style decomposition for 2x3 * 3x2 and extend?
    No, (2,3,3) is not square.
    
    Let's try to find the rank of (2,3,3) via ALS or known results.
    Actually, the rank of (2,3,2) is 8.
    The rank of (2,3,3) might be related.
    
    Let's try a different factorization.
    n=2, m=3, p=6.
    Factor 1: n1=1, m1=3, p1=6. (1x3 * 3x6)
    Factor 2: n2=2, m2=1, p2=1. (2x1 * 1x1)
    n=1*2=2, m=3*1=3, p=6*1=6.
    
    Rank of (1,3,6):
    A is 1x3, B is 3x6. C is 1x6.
    C[k] = sum_j A[j] B[j,k].
    This is a vector-matrix product.
    Rank of (1, n, p) tensor?
    For (1, 3, 6), it's like 6 dot products of length 3, but A is shared.
    This is equivalent to multiplying a row vector by a matrix.
    The rank is known to be n + p - 1?
    For (1, 1, 1), rank 1.
    For (1, 2, 2), rank 3.
    For (1, 3, 3), rank 5.
    For (1, 3, 6), rank 3 + 6 - 1 = 8?
    Let's check.
    
    Rank of (2,1,1):
    A is 2x1, B is 1x1. C is 2x1.
    C[i] = A[i,0] B[0,0].
    Tensor T2 size (2, 1, 2).
    T2[i, 0, 2*i + 0] = T2[i, 0, 2i] = 1.
    T2 = e0 x e0 x e0 + e1 x e0 x e2.
    Rank 2.
    
    So if rank(1,3,6) is 8, total rank = 8 * 2 = 16.
    
    Let's verify rank(1,3,6).
    Tensor T[a, b, c] where a in 0..2, b in 0..17, c in 0..5.
    T[j, 3j+k, k] = 1.
    j in 0..2, k in 0..5.
    
    We can decompose this using the identity:
    sum_j A[j] B[j,k]
    This is the tensor for matrix-vector multiplication.
    The rank of the (1, n, p) matrix multiplication tensor is n + p - 1.
    Proof:
    Let M be the matrix of the tensor flattened.
    The rank is n + p - 1.
    So for n=3, p=6, rank is 3 + 6 - 1 = 8.
    
    So total rank = 8 * 2 = 16.
    
    Let's construct the decomposition for (1,3,6) and (2,1,1).
    
    For (2,1,1):
    U2 (2x2), V2 (1x2), W2 (2x2).
    T2[i, 0, 2i] = 1.
    r=0: U2[0,0]=1, V2[0,0]=1, W2[0,0]=1. -> T2[0,0,0]=1.
    r=1: U2[1,1]=1, V2[0,1]=1, W2[2,1]=1. -> T2[1,0,2]=1.
    U2 = [[1, 0], [0, 1]]
    V2 = [[1, 1]]
    W2 = [[1, 0], [0, 1]]
    
    For (1,3,6):
    n1=1, m1=3, p1=6.
    U1 (3x8), V1 (18x8), W1 (6x8).
    T1[j, 3j+k, k] = 1.
    We need a rank 8 decomposition.
    Known decomposition for (1, n, p) of rank n+p-1:
    For k=0..p-1, j=0..n-1.
    We can use:
    For r in 0..n+p-2:
    If r < n:
      U1[r, r] = 1
      V1[3*r + 0, r] = 1 (only for k=0? No)
      This is not straightforward.
      
    Let's use the standard decomposition for vector-matrix multiplication.
    C[k] = sum_j A[j] B[j,k].
    Let's define the terms.
    For j=0..n-1, k=0..p-1.
    We can write:
    sum_{j,k} A[j] B[j,k] C[k]
    = A[0] B[0,0] C[0] + A[0] B[0,1] C[1] + ...
    
    A known rank n+p-1 decomposition:
    For r = 0 to n+p-2:
    If r < n:
      U[r, r] = 1
      V[3*r, r] = 1  (this is B[r, 0])
      W[0, r] = 1     (this is C[0])
      # Plus corrections for k > 0?
      
    Actually, let's just use ALS or a known formula.
    The tensor is:
    T = sum_{j=0}^{n-1} sum_{k=0}^{p-1} e_j (x) e_{jp+k} (x) e_k.
    We can group by k.
    T = sum_{k=0}^{p-1} ( sum_{j=0}^{n-1} e_j (x) e_{jp+k} ) (x) e_k.
    Let S_k = sum_{j=0}^{n-1} e_j (x) e_{jp+k}.
    This is a matrix in the (A, B) space.
    The rank of the tensor is the rank of the matrix formed by S_k?
    No, it's the tensor rank.
    
    Let's use the decomposition:
    For k=0:
      Term 0: e_0 (x) e_0 (x) e_0
      Term 1: e_1 (x) e_p (x) e_0
      ...
      Term n-1: e_{n-1} (x) e_{(n-1)p} (x) e_0
    For k=1:
      We want to add e_j (x) e_{jp+1} (x) e_1.
      We can use:
      (e_0 + e_1 + ... + e_{n-1}) (x) (e_p + e_{2p} + ... ) (x) e_1 ? No.
      
    Let's use the formula for rank n+p-1.
    Terms:
    1. For j=0..n-1:
       U[j, j] = 1
       V[j*p, j] = 1
       W[0, j] = 1
    2. For k=1..p-1:
       U[n-1, n-1+k-1] = 1  (reuse last U index?)
       No, we need n+p-1 terms.
       
    Let's try:
    r = 0..n-1:
      U[r, r] = 1
      V[r*p, r] = 1
      W[0, r] = 1
    r = n..n+p-2:
      k = r - n + 1
      U[n-1, r] = 1  # This reuses the last row of U
      V[0*p + k, r] = 1 # This uses B[0, k]
      W[k, r] = 1
      
    Let's check if this works for n=2, p=2. Rank 3.
    r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1. -> T[0,0,0]=1.
    r=1: U[1,1]=1, V[2,1]=1, W[0,1]=1. -> T[1,2,0]=1.
    r=2: U[1,2]=1, V[1,2]=1, W[1,2]=1. -> T[1,1,1]=1.
    Sum:
    T[0,0,0] = 1. (Correct)
    T[1,2,0] = 1. (Correct)
    T[1,1,1] = 1. (Correct)
    What about T[0,1,1]? Should be 1.
    My sum gives 0 for T[0,1,1].
    So this decomposition is wrong.
    
    Correct decomposition for (1, n, p):
    For k=0..p-1:
      For j=0..n-1:
        We need to produce e_j (x) e_{jp+k} (x) e_k.
        
    Let's use:
    For r=0..n+p-2:
      If r < n:
        U[r, r] = 1
        V[r*p, r] = 1
        W[0, r] = 1
      Else:
        k = r - n + 1
        U[n-1, r] = 1
        V[k, r] = 1  # B[0, k]
        W[k, r] = 1
        
    And we need to add terms for j>0 and k>0.
    Actually, the rank is n+p-1.
    The decomposition is:
    sum_{j=0}^{n-1} e_j (x) e_{jp} (x) e_0
    + sum_{k=1}^{p-1} ( sum_{j=0}^{n-1} e_j ) (x) e_k (x) e_k  ? No.
    
    Let's just use ALS for (1,3,6) to be safe, or find the exact decomposition.
    Given the time, I will use ALS for the (1,3,6) tensor and (2,1,1) tensor.
    (2,1,1) is trivial.
    (1,3,6) is small enough for ALS.
    
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Factor 1: n1=1, m1=3, p1=6
    n1, m1, p1 = 1, 3, 6
    # Factor 2: n2=2, m2=1, p2=1
    n2, m2, p2 = 2, 1, 1
    
    # T2 is trivial
    U2 = np.array([[1, 0], [0, 1]], dtype=float)
    V2 = np.array([[1, 1]], dtype=float)
    W2 = np.array([[1, 0], [0, 1]], dtype=float)
    
    # T1: (1,3,6)
    N1, N2, N3 = n1*m1, m1*p1, p1*n1
    T1 = np.zeros((N1, N2, N3))
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                a = i*m1 + j
                b = j*p1 + k
                c = k*n1 + i
                T1[a, b, c] = 1.0
    vec_T1 = T1.reshape(-1)
    
    # ALS for T1
    r1 = None
    U1 = None
    V1 = None
    W1 = None
    start_time = time.time()
    deadline = start_time + 100
    
    # Try ranks from 8 upwards
    for r in range(8, 15):
        for trial in range(50):
            if time.time() > deadline:
                break
            U = np.random.randn(N1, r)
            V = np.random.randn(N2, r)
            W = np.random.randn(N3, r)
            
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Solve U
                A_U = np.zeros((N2*N3, N1*r))
                for a in range(N1):
                    for r_idx in range(r):
                        idx_u = a*r + r_idx
                        vec_v = np.kron(np.ones(N3), V[:, r_idx])
                        vec_w = np.kron(W[:, r_idx], np.ones(N2))
                        A_U[:, idx_u] = vec_v * vec_w
                try:
                    U_new, _, _, _ = np.linalg.lstsq(A_U, vec_T1, rcond=None)
                except:
                    break
                U = U_new.reshape(N1, r)
                
                # Solve V
                A_V = np.zeros((N1*N3, N2*r))
                for b in range(N2):
                    for r_idx in range(r):
                        idx_v = b*r + r_idx
                        vec_u = np.kron(np.ones(N3), U[:, r_idx])
                        vec_w = np.kron(W[:, r_idx], np.ones(N1))
                        A_V[:, idx_v] = vec_u * vec_w
                try:
                    V_new, _, _, _ = np.linalg.lstsq(A_V, vec_T1, rcond=None)
                except:
                    break
                V = V_new.reshape(N2, r)
                
                # Solve W
                A_W = np.zeros((N1*N2, N3*r))
                for c in range(N3):
                    for r_idx in range(r):
                        idx_w = c*r + r_idx
                        vec_u = np.kron(U[:, r_idx], np.ones(N2))
                        vec_v = np.kron(np.ones(N1), V[:, r_idx])
                        A_W[:, idx_w] = vec_u * vec_v
                try:
                    W_new, _, _, _ = np.linalg.lstsq(A_W, vec_T1, rcond=None)
                except:
                    break
                W = W_new.reshape(N3, r)
                
                recon = np.zeros((N1, N2, N3))
                for r_idx in range(r):
                    recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
                if np.max(np.abs(recon - T1)) < 1e-8:
                    r1 = r
                    U1 = U.copy()
                    V1 = V.copy()
                    W1 = W.copy()
                    break
            if r1 is not None:
                break
        if r1 is not None:
            break
    
    if r1 is None:
        # Fallback for T1
        r1 = N1*N2*N3
        U1 = np.zeros((N1, r1))
        V1 = np.zeros((N2, r1))
        W1 = np.zeros((N3, r1))
        col = 0
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U1[i*m1+j, col] = 1
                    V1[j*p1+k, col] = 1
                    W1[k*n1+i, col] = 1
                    col += 1
    
    # Combine
    r_total = r1 * 2
    U_total = np.kron(U1, U2)
    V_total = np.kron(V1, V2)
    W_total = np.kron(W1, W2)
    
    # Verify
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_full[a, b, c] = 1.0
                
    recon_full = np.zeros((n*m, m*p, p*n))
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
