"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Falls back to schoolbook (rank 80) if no better decomposition is found in time."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n=4, m=4, p=5):
    """Check that the decomposition exactly reproduces matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    And for all other index combos (i,j,k) where the triple is not a valid
    product term, the sum must be 0.
    Actually the tensor T[i,j,k] = 1 iff we're computing (AB)[i,k] from A[i,j]*B[j,k].
    The multiplication tensor has T[i,j,k]=1 for all i in [0,n), j in [0,m), k in [0,p).
    Wait - the multiplication tensor for C=AB where C is n x p, A is n x m, B is m x p:
    C[i,k] = sum_j A[i,j] B[j,k]
    The tensor T has T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor on n x m x p).
    No wait - that's not right. The tensor is T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p).
    Actually the matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 for all i,j,k.
    The rank is the minimum r such that M = sum_{c=1}^r U_c * V_c * W_c where
    U_c[i,j] = U[i*m+j, c], V_c[j,k] = V[j*p+k, c], W_c[k,i] = W[k*n+i, c].
    So we need: for all (i,j,k) in [n]x[m]x[p]: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    """
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    rank = Ua.shape[1]
    # Compute the reconstructed tensor
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Efficiently: for each c, outer product
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s = s + Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W, n=4, m=4, p=5):
    """Fast verification using numpy with float64, checking tolerance."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # This is a tensor contraction. Build it step by step.
    # First: A[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c]  -> shape (n, m, p)
    # Then: T[i,j,k] = sum_c ... but we already summed over c in the first step.
    # Actually T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # = sum_c (Ur[i,j,c] * Vr[j,k,c]) * Wr[k,i,c]
    # Let's do it as: for each c, the contribution is Ur[:,:,c] * Vr[:,:,c] * Wr[:,:,c] reshaped
    # Ur[:,:,c] is (n,m), Vr[:,:,c] is (m,p), Wr[:,:,c] is (p,n)
    # Contribution to T[i,j,k] from term c: Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]  # (n, m)
        vc = Vr[:, :, c]  # (m, p)
        wc = Wr[:, :, c]  # (p, n)
        # contrib[i,j,k] = uc[i,j] * vc[j,k] * wc[k,i]
        # = (uc[i,j] * vc[j,k]) * wc[k,i]
        # uc[i,j] * vc[j,k] is (n, p) via outer-like: uc[:,j] * vc[j,:]
        # Actually: for fixed j, uc[:,j] is (n,), vc[j,:] is (p,), product is (n,p)
        # Then multiply by wc[k,i] which is (p,n) -> need to transpose
        # contrib[i,j,k] = uc[i,j] * vc[j,k] * wc[k,i]
        # Let's build (n, m, p):
        # For each j: uc[:,j] * vc[j,:] gives (n, p), then multiply element-wise by wc.T (which is (n,p))
        # wc[k,i] -> wc.T[i,k]
        wcT = wc.T  # (n, p), wcT[i,k] = wc[k,i]
        for j in range(m):
            # uc[:, j] is (n,), vc[j, :] is (p,)
            # product[i,k] = uc[i,j] * vc[j,k] * wcT[i,k]
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.allclose(T, 1.0, atol=1e-9)

def _random_sparse_attempt(max_rank, seed, n=4, m=4, p=5, max_iters=200):
    """Try to find a decomposition of given max_rank using random initialization + refinement."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    for attempt in range(max_iters):
        rank = max_rank
        # Random initialization with small integers/half-integers
        # Use a structured approach: random sparse matrices
        density = 0.3
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        
        for c in range(rank):
            # Random non-zero pattern
            for idx in range(n*m):
                if rng.random() < density:
                    U[idx, c] = rng.choice([-1, 1, 0.5, -0.5, 2, -2])
            for idx in range(m*p):
                if rng.random() < density:
                    V[idx, c] = rng.choice([-1, 1, 0.5, -0.5, 2, -2])
            for idx in range(p*n):
                if rng.random() < density:
                    W[idx, c] = rng.choice([-1, 1, 0.5, -0.5, 2, -2])
        
        # Verify
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 80
    
    # Try to find a better decomposition
    # Known results: for 4x4x4, best known is 49 (Strassen-based).
    # For 4x4x5, it's less studied. Let's try ranks below 80.
    
    best_result = None
    
    # Try a few structured approaches with decreasing rank
    # Start with moderate ranks and work down
    ranks_to_try = [79, 78, 77, 76, 75, 74, 73, 72, 71, 70]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        result = _random_sparse_attempt(target_rank, seed=target_rank * 1000, 
                                         n=n, m=m, p=p, max_iters=50)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
