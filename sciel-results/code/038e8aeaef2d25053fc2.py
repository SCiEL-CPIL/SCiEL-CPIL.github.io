"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    deadline = time.time() + 130.0
    n, m, p = DIMENSIONS  # 2, 4, 8

    # Build the multiplication tensor T of shape (n*m, m*p, p*n) = (8, 32, 16)
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Try to find a low-rank decomposition via randomized SVD / truncated tensor decomposition
    # We'll use a simple approach: try ranks from 32 upward and check if a valid decomposition exists
    
    def try_rank(rank):
        """Try to find a rank-`rank` decomposition. Returns (U, V, W) or None."""
        # Flatten T to a matrix and use SVD to get initial guess
        # T has shape (8, 32, 16). Flatten to (8, 512) and SVD.
        T_flat = T.reshape(n * m, m * p * p * n)
        try:
            U_svd, s, Vt = np.linalg.svd(T_flat, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        
        # Get top `rank` components
        U_init = U_svd[:, :rank]  # (8, rank)
        # Vt is (512, rank), reshape to (32, 16, rank)
        Vt_reshaped = Vt[:rank].T.reshape(rank, m * p, p * n)  # (rank, 32, 16)
        
        # This gives a rank-`rank` approximation, but we need EXACT decomposition.
        # For exact decomposition, we need the tensor to be exactly representable.
        # Check if the reconstruction is exact:
        recon = np.einsum('ir,jkr,lkr->ijl', U_init, Vt_reshaped, np.ones((rank, p * n, n)))
        # This isn't quite right. Let me think more carefully.
        
        # Actually, for an exact tensor decomposition of rank r:
        # T[i,j,k] = sum_{r=0}^{r-1} U[i,r] * V[j,r] * W[k,r]
        # This is a bilinear system. For exact solution, we need T to have tensor rank <= r.
        
        # Check: does T have tensor rank exactly equal to its "tensor rank"?
        # The tensor rank is the minimum r such that T = sum of r rank-1 tensors.
        # For our multiplication tensor, the tensor rank is what we're looking for.
        
        # A necessary condition: for any unfolding, the matrix rank <= tensor rank.
        # T unfolded to (8, 512) has rank <= 8 (since it's 8x512).
        # T unfolded to (32, 128) has rank <= 32.
        # T unfolded to (16, 256) has rank <= 16.
        
        # So tensor rank >= max(8, 32, 16) = 32.
        # The naive rank is 64. Strassen gives 56.
        
        # Let's check the actual matrix ranks of the unfoldings:
        T1 = T.reshape(n * m, m * p * p * n)  # (8, 512)
        T2 = T.reshape(m * p, n * m * p * n)  # (32, 128)
        T3 = T.reshape(p * n, n * m * m * p)  # (16, 256)
        
        r1 = np.linalg.matrix_rank(T1)
        r2 = np.linalg.matrix_rank(T2)
        r3 = np.linalg.matrix_rank(T3)
        
        # Lower bound on tensor rank
        lower_bound = max(r1, r2, r3)
        
        if rank < lower_bound:
            return None
        
        # For exact decomposition, try to solve the system
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # This is a system of 8*32*16 = 4096 equations with 8*r + 32*r + 16*r = 56*r unknowns.
        # For r=32: 4096 equations, 1792 unknowns. Underdetermined? No, 4096 > 1792.
        # Actually it's nonlinear (products of unknowns).
        
        # Use a different approach: fix W and solve for U, V.
        # Or use the fact that for fixed k, T[:,:,k] = sum_r U * V[r] * W[k,r]
        # T[:,:,k] = sum_r W[k,r] * (U * V[r])  where U*V[r] is outer product
        
        # This is a linear system in the unknowns if we fix the other two matrices.
        
        # Try iterative approach:
        # 1. Initialize W randomly
        # 2. For each k, solve: T[:,:,k] = sum_r W[k,r] * outer(U[:,r], V[:,r])
        #    This is linear in U and V if W is fixed.
        # 3. Update U, V
        # 4. Repeat
        
        # Actually, let's just return the Strassen result for now and note the lower bound.
        return None
    
    # The lower bound from matrix ranks:
    T1 = T.reshape(n * m, m * p * p * n)
    T2 = T.reshape(m * p, n * m * p * n)
    T3 = T.reshape(p * n, n * m * m * p)
    lb = max(np.linalg.matrix_rank(T1), np.linalg.matrix_rank(T2), np.linalg.matrix_rank(T3))
    
    # Use Strassen decomposition (rank 56) as fallback
    strassen_products = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(a_col0, b_row0, b_col0, col_start):
        c = col_start
        for u_coeff, v_coeff, w_coeff in strassen_products:
            for (li, lj), coeff in u_coeff.items():
                gi = li
                gj = a_col0 + lj
                U[gi * m + gj][c] = coeff
            for (lj, lk), coeff in v_coeff.items():
                gj = b_row0 + lj
                gk = b_col0 + lk
                V[gj * p + gk][c] = coeff
            for (lk, li), coeff in w_coeff.items():
                gk = b_col0 + lk
                gi = li
                W[gk * n + gi][c] = coeff
            c += 1
        return c

    col = 0
    col = add_strassen_block(0, 0, 0, col)
    col = add_strassen_block(2, 2, 0, col)
    col = add_strassen_block(0, 0, 2, col)
    col = add_strassen_block(2, 2, 2, col)
    col = add_strassen_block(0, 0, 4, col)
    col = add_strassen_block(2, 2, 4, col)
    col = add_strassen_block(0, 0, 6, col)
    col = add_strassen_block(2, 2, 6, col)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
