import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin under 150s

    n, m, p = 2, 2, 10

    # Build rank-35 Strassen block-diagonal decomposition
    # Strassen terms: (a_vec, b_vec, c_vec) each of length 4
    # a_vec: coefficients on A = [A00, A01, A10, A11] (row-major, index i*m+j)
    # b_vec: coefficients on B block = [B00, B01, B10, B11]
    # c_vec: coefficients on C block = [C00, C10, C01, C11] (W[k*n+i] convention)
    #
    # Standard Strassen (c_vec order: C00, C10, C01, C11):
    # M1: a=[1,0,0,1] b=[1,0,0,1] c=[1,0,0,1]
    # M2: a=[0,0,1,1] b=[1,0,0,0] c=[0,1,0,-1]
    # M3: a=[1,0,0,0] b=[0,1,0,-1] c=[0,0,1,1]
    # M4: a=[0,0,0,1] b=[-1,0,1,0] c=[1,1,0,0]
    # M5: a=[1,1,0,0] b=[0,0,0,1] c=[-1,0,1,0]
    # M6: a=[-1,0,1,0] b=[1,1,0,0] c=[0,0,0,1]
    # M7: a=[0,1,0,-1] b=[0,0,1,1] c=[1,0,0,0]

    strassen = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,0,1,1]),
        ([0,0,0,1], [-1,0,1,0], [1,1,0,0]),
        ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]

    def build_rank35():
        U = np.zeros((n*m, 35))
        V = np.zeros((m*p, 35))
        W = np.zeros((p*n, 35))
        col = 0
        for k in range(5):
            for av, bv, cv in strassen:
                for i in range(4):
                    U[i, col] = av[i]
                V[2*k, col] = bv[0]
                V[2*k+1, col] = bv[1]
                V[10+2*k, col] = bv[2]
                V[10+2*k+1, col] = bv[3]
                W[4*k, col] = cv[0]
                W[4*k+1, col] = cv[1]
                W[4*k+2, col] = cv[2]
                W[4*k+3, col] = cv[3]
                col += 1
        return U, V, W

    def verify(U, V, W):
        """Check that the decomposition is exact."""
        n, m, p = 2, 2, 10
        rng = np.random.RandomState(42)
        for _ in range(3):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_direct = A @ B  # shape (n, p)
            
            A_flat = A.flatten()  # (n*m,)
            B_flat = B.flatten()  # (m*p,)
            
            UA = U.T @ A_flat  # (rank,)
            VB = V.T @ B_flat  # (rank,)
            
            # C_decomp[i,k] = sum_r W[k*n+i, r] * UA[r] * VB[r]
            WV = W * (UA * VB)[np.newaxis, :]  # (p*n, rank)
            C_decomp_flat = WV.sum(axis=1)  # (p*n,)
            
            # Compare: C_decomp_flat[k*n + i] should equal C_direct[i, k]
            C_decomp = np.zeros((n, p))
            for k in range(p):
                for i in range(n):
                    C_decomp[i, k] = C_decomp_flat[k*n + i]
            
            if not np.allclose(C_direct, C_decomp, atol=1e-10):
                return False
        return True

    def to_list(M):
        return M.tolist()

    # Start with rank-35
    U35, V35, W35 = build_rank35()
    
    best_rank = 35
    best_U, best_V, best_W = U35, V35, W35
    
    if verify(best_U, best_V, best_W):
        # Confirmed correct
        pass
    else:
        # Should not happen, but fallback
        raise RuntimeError("Rank-35 verification failed")
    
    # Try to find a rank-34 decomposition via randomized search
    # The idea: perturb the rank-35 decomposition and try to zero out one column
    # by finding a linear dependency among the 35 rank-1 terms.
    
    # A rank-34 decomposition would mean that the 35 rank-1 tensors are linearly dependent.
    # We can try to find a vector (alpha, beta, gamma) of size 35 such that
    # sum_r alpha[r] * U[:,r] = 0, sum_r beta[r] * V[:,r] = 0, sum_r gamma[r] * W[:,r] = 0
    # and alpha*beta*gamma != 0 (i.e., the dependency is non-trivial).
    
    # Actually, for a rank reduction, we need:
    # There exist vectors u in R^(n*m), v in R^(m*p), w in R^(p*n) such that
    # the tensor T + u*v*w has rank 34, where T is the rank-35 tensor.
    # This is equivalent to finding a rank-1 perturbation that reduces rank.
    
    # A simpler approach: check if any column of [U; V; W] (stacked) is in the span of the others.
    # But this is not the right condition for tensor rank.
    
    # Let's try a different approach: use the fact that the 5 blocks are independent.
    # If we can find a way to share computation between blocks, we might reduce rank.
    
    # For now, return rank 35 as it is the best known.
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
