import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    
    n, m, p = 2, 3, 3
    DEADLINE = 145.0  # seconds, safety margin from 150s
    
    def _time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Rank 17 solution - verified exact
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: Strassen for 2x2x2 sub-block (indices 0,1 for rows/cols)
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # U indices: i*m+j. (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # V indices: k*p+l. (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # W indices: l*n+i. (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # r=0
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # r=2
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # r=3
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # r=4
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # r=5
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Cross terms involving A[:, 2] and B[2, :]
    # C[0,0] += A[0,2]B[2,0]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    # C[0,1] += A[0,2]B[2,1]
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    # C[1,0] += A[1,2]B[2,0]
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    # C[1,1] += A[1,2]B[2,1]
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    
    # Part 3: Cross terms involving A[:, 0:2] and B[:, 2]
    # C[0,2] += A[0,0]B[0,2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    # C[0,2] += A[0,1]B[1,2]
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    # C[0,2] += A[0,2]B[2,2]
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    # C[1,2] += A[1,0]B[0,2]
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    # C[1,2] += A[1,1]B[1,2]
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    # C[1,2] += A[1,2]B[2,2]
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    def _verify(U_mat, V_mat, W_mat, n_, m_, p_, rank_):
        for i in range(n_):
            for k in range(p_):
                for j1 in range(m_):
                    for j2 in range(m_):
                        s = 0.0
                        for r in range(rank_):
                            s += W_mat[k*n_+i, r] * U_mat[i*m_+j1, r] * V_mat[j2*p_+k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True
    
    # Verify rank-17 solution
    if not _verify(U, V, W, n, m, p, rank):
        # Should not happen, but fallback
        U_fb = np.zeros((n * m, n * m * p))
        V_fb = np.zeros((m * p, n * m * p))
        W_fb = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Save the rank-17 solution as best so far
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to find a rank-16 solution using randomized algebraic search
    # Strategy: Fix V to a generic sparse matrix with random ±1 entries,
    # then solve for U and W using the constraint equations.
    # The tensor equation is: for all (i,k,j1,j2),
    #   sum_r U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] = delta(j1,j2)
    
    # This is a system of 2*3*3*3 = 54 equations (but many are 0=0 or 1=1)
    # Actually: 2*3*3 = 18 equations where j1==j2 (target=1), 
    #           and 2*3*3*2 = 36 equations where j1!=j2 (target=0)
    # Total 54 equations, but the rank-1 solution has 3*3=9 free parameters per term.
    
    # For rank 16, we have 16 * (6 + 9 + 6) = 16 * 21 = 336 unknowns
    # and 54 equations. So it's underdetermined.
    
    # Let's try a direct approach: use the structure that the tensor is 
    # "block structured" - try to find a decomposition where the rank-16
    # terms respect some symmetry.
    
    # Alternative: try to reduce the cross-term part from 10 terms to 9
    # by using Strassen-like tricks on the cross terms.
    
    # The cross terms are:
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]  (3 terms)
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]  (3 terms)
    # C[0,0] += A[0,2]B[2,0]  (1 term)
    # C[0,1] += A[0,2]B[2,1]  (1 term)
    # C[1,0] += A[1,2]B[2,0]  (1 term)
    # C[1,1] += A[1,2]B[2,1]  (1 term)
    # Total: 10 cross terms
    
    # Can we combine some of these? The C[0,2] and C[1,2] are matrix-vector
    # products: [A[0,0], A[0,1], A[0,2]] * [B[0,2], B[1,2], B[2,2]]^T
    # and [A[1,0], A[1,1], A[1,2]] * [B[0,2], B[1,2], B[2,2]]^T
    # These are dot products of length 3, which have tensor rank 3 over reals.
    # So no savings there.
    
    # The C[0,0], C[0,1], C[1,0], C[1,1] additions are:
    # A[0,2]B[2,0], A[0,2]B[2,1], A[1,2]B[2,0], A[1,2]B[2,1]
    # These form a 2x2 matrix: [[A[0,2]B[2,0], A[0,2]B[2,1]], [A[1,2]B[2,0], A[1,2]B[2,1]]]
    # = [[A[0,2]], [A[1,2]]] * [[B[2,0], B[2,1]]]
    # This is outer product of a 2-vector and a 2-vector, which has rank 1!
    # Wait, that's a 2x2 matrix that's the outer product of column vector and row vector.
    # But we need to ADD these to existing C entries, not produce them separately.
    
    # Hmm, the issue is that these terms must be combined with the Strassen terms.
    # Let me think about this differently.
    
    # Actually, let me try a computational search: 
    # Set up the system of equations for rank-16 and try to solve it.
    
    if _time_left() > 10:
        # Try rank-16 search
        rank16 = 16
        U16 = np.zeros((n * m, rank16))
        V16 = np.zeros((m * p, rank16))
        W16 = np.zeros((p * n, rank16))
        
        # Initialize with the rank-17 solution, merging two terms
        # Try merging terms 7 and 9 (both involve V[6,:])
        # Or try a more systematic approach.
        
        # Let's try: take the rank-17 solution and try to express two of the 
        # rank-1 terms as a combination that saves one term.
        
        # Actually, let me try a numerical optimization approach.
        # Flatten U, V, W and use the constraint equations.
        
        # The constraint is: for each (i, k, j1, j2):
        #   sum_r U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] = delta(j1,j2)
        
        # This is a system of 54 polynomial equations in 336 unknowns.
        # Let's try a simple iterative approach: randomly initialize, then
        # use gradient descent on the residual.
        
        np.random.seed(42)
        
        # Initialize with random ±1 values
        U16 = np.random.choice([-1, 1], size=(n * m, rank16))
        V16 = np.random.choice([-1, 1], size=(m * p, rank16))
        W16 = np.random.choice([-1, 1], size=(p * n, rank16))
        
        # Compute residual
        def _residual(U_mat, V_mat, W_mat, n_, m_, p_, rank_):
            res = 0.0
            max_res = 0.0
            for i in range(n_):
                for k in range(p_):
                    for j1 in range(m_):
                        for j2 in range(m_):
                            s = 0.0
                            for r in range(rank_):
                                s += U_mat[i*m_+j1, r] * V_mat[j2*p_+k, r] * W_mat[k*n_+i, r]
                            target = 1.0 if j1 == j2 else 0.0
                            err = s - target
                            res += err * err
                            max_res = max(max_res, abs(err))
            return res, max_res
        
        res, max_res = _residual(U16, V16, W16, n, m, p, rank16)
        
        # Try a few random restarts
        best_res = res
        for restart in range(10):
            if _time_left() < 5:
                break
            
            U16 = np.random.choice([-1, 1], size=(n * m, rank16))
            V16 = np.random.choice([-1, 1], size=(m * p, rank16))
            W16 = np.random.choice([-1, 1], size=(p * n, rank16))
            
            res, max_res = _residual(U16, V16, W16, n, m, p, rank16)
            if res < best_res:
                best_res = res
        
        # The random ±1 approach is unlikely to work for rank 16.
        # Let me try a different approach: use the known rank-17 structure
        # and try to find a linear dependency that allows merging.
        
        # Check if any two columns of the rank-17 tensor can be combined.
        # The tensor T has entries T[i*m+j1, j2*p+k, k*n+i] = delta(j1,j2)
        
        # For a rank-r decomposition, we need vectors u_r, v_r, w_r such that
        # sum_r u_r ⊗ v_r ⊗ w_r = T
        
        # Let's check if the rank-17 solution has any linear dependencies
        # that could be exploited.
        
        # Actually, let me just return the verified rank-17 solution.
        # The probability of finding rank 16 in the remaining time is low.
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
