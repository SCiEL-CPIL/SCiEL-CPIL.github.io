"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    N = n * m          # 9
    M = m * p          # 12
    P = p * n          # 12

    # Target tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((N, M, P), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Schoolbook rank (fallback)
    rank_school = n * m * p
    U_sch = np.zeros((N, rank_school))
    V_sch = np.zeros((M, rank_school))
    W_sch = np.zeros((P, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i * m + j, col] = 1
                V_sch[j * p + k, col] = 1
                W_sch[k * n + i, col] = 1
                col += 1

    def to_output(U, V, W):
        return {"U": [list(map(int, np.round(U[i]))) for i in range(U.shape[0])],
                "V": [list(map(int, np.round(V[i]))) for i in range(V.shape[0])],
                "W": [list(map(int, np.round(W[i]))) for i in range(W.shape[0])]
        }

    def verify(U, V, W, tol=1e-9):
        # Reconstruct tensor: T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Efficient: for each r, outer product
        recon = np.zeros((N, M, P), dtype=np.int64)
        for r in range(U.shape[1]):
            recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        return np.allclose(recon, T, atol=tol)

    def solve_for_U(U_fixed, rank):
        """Given U_fixed (N x rank), solve for V (M x rank) and W (P x rank).
        Equations: for each (a,b,c): sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        This is bilinear in V,W. We fix U and use the structure:
        For fixed U, define for each (b,c): the equation over a is
        sum_a T[a,b,c] ... no. Let's think differently.
        
        Actually with U fixed, the equations are linear in the products V[b,r]*W[c,r].
        Let X[b,c] = sum_r V[b,r] W[c,r] ... no that's not right either.
        
        The constraint is: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        For fixed a: T[a,b,c] = sum_r (U[a,r] V[b,r]) W[c,r]
        This is linear in the vector of W if we define for each r: A_r[b] = U[a,r]*V[b,r]
        
        Better approach: Fix U. Then for each pair (b,c), we need:
        For all a: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
        
        Let's define Z_r = V[:,r] * W[:,r] element-wise? No.
        
        Alternative: Fix U. The problem becomes: find V (M x rank) and W (P x rank)
        such that for all a,b,c: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        
        This is still bilinear. But if we fix U to have a special structure (e.g., each row
        of U has exactly one non-zero entry, i.e., U is a permutation-like sparse matrix),
        then for each a, only one r contributes: U[a,r_a] != 0, and the equation becomes
        V[b, r_a] * W[c, r_a] = T[a,b,c] / U[a,r_a]
        
        Let me try: U has N rows, rank columns. If each row of U has exactly one 1
        (i.e., U is N x rank with each row being a unit vector), then N <= rank and
        for row a, let r(a) be the column where U[a, r(a)] = 1.
        Then: V[b, r(a)] * W[c, r(a)] = T[a,b,c] for all b,c.
        
        This means: for each a, the matrix T[a,:,:] must be rank-1, and
        V[:, r(a)] and W[:, r(a)] are the rank-1 factors.
        
        But T[a,:,:] is generally NOT rank-1. Let me check:
        T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
        For fixed a = i*m+j: b ranges over j*p+k for k in 0..p-1 (p values),
        c ranges over k*n+i for k in 0..p-1 (p values).
        T[a, j*p+k, k*n+i] = 1 for k=0..p-1, and 0 otherwise.
        So T[a,:,:] has p ones on a "diagonal-like" pattern. This is rank p (for p=4, rank 4).
        So a single unit vector in U cannot capture it. We need rank >= p per row group.
        
        Let me try a different U structure: U is block-diagonal.
        Group rows by i (n groups of m rows). For group i, use m*p columns.
        U[i*m+j, i*m*p + j*p + k] = 1 for j in 0..m-1, k in 0..p-1.
        Then for fixed a=i*m+j: U[a, i*m*p + j*p + k] = 1 for all k.
        Equation: sum_k V[b, i*m*p+j*p+k] * W[c, i*m*p+j*p+k] = T[a,b,c]
        With b=j*p+k', c=k'*n+i: T=1 when k'=k... 
        V[j*p+k, i*m*p+j*p+k] * W[k*n+i, i*m*p+j*p+k] = 1
        So V and W are identity-like in those positions. This gives rank = n*m*p = 36.
        
        Let me try a smarter search: fix U to be a random/sparse matrix and solve.
        """
        pass

    # Strategy: Try to find a decomposition of rank < 36.
    # Approach: Fix U to be a sparse matrix (each row has few non-zeros), 
    # then solve the resulting system for V and W.
    
    # Key insight: The tensor T for (3,3,4) matrix mult.
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Also rank >= n+m+p-2 = 8 (for square, but this is rectangular)
    # Known: for (2,2,2) rank is 7. For (3,3,3) it's 23 (Landsberg).
    # For (3,3,4), we don't know the exact minimum. Let's search.
    
    # Practical approach: Use the fact that if we fix U, the equations
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r] can be seen as:
    # For each (b,c): the vector (T[:,b,c]) must be in the span of
    # {U[:,r] * (V[b,r] * W[c,r]) for r} ... no.
    
    # Better: For fixed U and fixed V, W is determined by:
    # For each (a,c): sum_r U[a,r] W[c,r] = T[a,b,c] / V[b,r] ... no, still coupled.
    
    # Let me try: fix U. Then define for each r: the matrix B_r = V[:,r] (M-vector) 
    # and C_r = W[:,r] (P-vector). Constraint: for all (a,b,c):
    # sum_r U[a,r] B_r[b] C_r[c] = T[a,b,c]
    # 
    # If U is fixed, let's define M_a = U[a,:] (row vector of length rank).
    # Then: sum_r M_a[r] B_r[b] C_r[c] = T[a,b,c]
    # This is: (M_a ⊗ I) applied to vec of B_r[b]C_r[c]... 
    # 
    # Equivalently: Let X = B^T * C (rank x ... no). 
    # Actually: sum_r M_a[r] B_r[b] C_r[c] = sum_r (M_a[r] B_r[b]) C_r[c]
    # = (M_a ⊙ B_b)^T C_c where ⊙ is element-wise... 
    # Define D_b[r] = M_a[r] * B_r[b] ... depends on a.
    
    # Let me try a completely different approach: 
    # Use the "border rank" / symbolic approach won't work for exact.
    
    # Practical search: 
    # 1. Try rank 12 (theoretical lower bound). Probably impossible.
    # 2. Use random initialization + alternating least squares to find low-rank solutions.
    # 3. Once we find a floating-point solution, round to integers/halves.
    
    # Let's implement ALS (Alternating Least Squares) for tensor decomposition:
    # Minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||_F^2
    
    def tensor_to_mat(T, mode):
        """Reshape tensor to matrix. mode=0: (N, M*P), mode=1: (M, N*P), mode=2: (P, N*M)"""
        if mode == 0:
            return T.reshape(N, M * P)
        elif mode == 1:
            return T.transpose(1, 0, 2).reshape(M, N * P)
        else:
            return T.transpose(2, 0, 1).reshape(P, N * M)
    
    def mat_to_tensor(M, mode, shape):
        if mode == 0:
            return M.reshape(N, M, P)  # careful with M name
        elif mode == 1:
            return M.reshape(M, N, P).transpose(1, 0, 2)
        else:
            return M.reshape(P, N, M).transpose(1, 2, 0)
    
    def als_decompose(T, rank, max_iter=500, tol=1e-10, seed=42):
        """Alternating least squares for CP decomposition."""
        np.random.seed(seed)
        U = np.random.randn(N, rank) * 0.1
        V = np.random.randn(M, rank) * 0.1
        W = np.random.randn(P, rank) * 0.1
        
        T0 = tensor_to_mat(T, 0)  # (N, M*P)
        T1 = tensor_to_mat(T, 1)  # (M, N*P)
        T2 = tensor_to_mat(T, 2)  # (P, N*M)
        
        for it in range(max_iter):
            if time.time() > DEADLINE - 30:
                break
            
            # Update U: fix V, W. For each a: T0[a,:] = sum_r U[a,r] * (V_r ⊗ W_r)
            # V_r ⊗ W_r is a vector of length M*P
            # So: T0[a,:] = U[a,:] @ (V_r ⊗ W_r for each r)
            # Define C[r, b*p+c] = V[b,r] * W[c,r]
            C = np.zeros((rank, M * P))
            for r in range(rank):
                C[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            # T0[a,:] = sum_r U[a,r] * C[r,:]
            # So U = T0 @ C^T @ (C C^T)^{-1}
            try:
                Cinv = np.linalg.pinv(C)  # (M*P, rank)
                U_new = T0 @ Cinv  # (N, rank)
            except:
                break
            U = U_new
            
            # Update V: fix U, W. For each b: T1[b,:] = sum_r V[b,r] * (U_r ⊗ W_r)
            C2 = np.zeros((rank, N * P))
            for r in range(rank):
                C2[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            try:
                C2inv = np.linalg.pinv(C2)
                V_new = T1 @ C2inv
            except:
                break
            V = V_new
            
            # Update W: fix U, V. For each c: T2[c,:] = sum_r W[c,r] * (U_r ⊗ V_r)
            C3 = np.zeros((rank, N * M))
            for r in range(rank):
                C3[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            try:
                C3inv = np.linalg.pinv(C3)
                W_new = T2 @ C3inv
            except:
                break
            W = W_new
            
            # Check convergence
            if it % 10 == 0:
                recon = np.zeros((N, M, P))
                for r in range(rank):
                    recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
                err = np.linalg.norm(recon - T)
                if err < tol:
                    break
        
        # Compute final error
        recon = np.zeros((N, M, P))
        for r in range(rank):
            recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        err = np.linalg.norm(recon - T)
        return U, V, W, err
    
    # Try various ranks from low to high
    best_result = (rank_school, U_sch, V_sch, W_sch)
    
    # Try ranks: start from a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = 12, but likely higher
    # Let's try: 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
    # But we need to be time-efficient. Start with a few promising ones.
    
    candidate_ranks = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    
    # Try a few random seeds for each rank
    for rank in candidate_ranks:
        if time.time() > DEADLINE - 20:
            break
        
        best_err_for_rank = float('inf')
        best_UVW_for_rank = None
        
        for seed in range(5):  # 5 random restarts per rank
            if time.time() > DEADLINE - 20:
                break
            U, V, W, err = als_decompose(T, rank, max_iter=200, tol=1e-10, seed=seed)
            if err < best_err_for_rank:
                best_err_for_rank = err
                best_UVW_for_rank = (U, V, W)
        
        if best_UVW_for_rank is not None:
            U, V, W = best_UVW_for_rank
            # Check if it's close to exact
            if best_err_for_rank < 1e-6:
                # Try to round to integers or half-integers
                # First try integer rounding
                U_r = np.round(U).astype(np.int64)
                V_r = np.round(V).astype(np.int64)
                W_r = np.round(W).astype(np.int64)
                if verify(U_r, V_r, W_r):
                    return to_output(U_r, V_r, W_r)
                
                # Try half-integer: multiply by 2, round, divide by 2
                U_h = np.round(2 * U) / 2.0
                V_h = np.round(2 * V) / 2.0
                W_h = np.round(2 * W) / 2.0
                U_h_int = np.round(2 * U).astype(np.int64)
                V_h_int = np.round(2 * V).astype(np.int64)
                W_h_int = np.round(2 * W).astype(np.int64)
                # Verify with the half-integer values
                recon = np.zeros((N, M, P), dtype=np.float64)
                for r in range(rank):
                    recon += np.outer(np.outer(U_h[:, r], V_h[:, r]), W_h[:, r])
                if np.allclose(recon, T, atol=1e-9):
                    # Output as integers (the evaluator should handle the scale)
                    # Actually, we need to output the actual values.
                    # Let's output as the rounded integer values of 2*x
                    # and the evaluator will interpret them... 
                    # No, we need to output the actual U, V, W values.
                    # Let's output them as floats that are multiples of 0.5
                    return to_output(U_h, V_h, W_h)
            
            # Update best if lower rank
            if rank < best_result[0]:
                best_result = (rank, U, V, W)
    
    # If no better solution found, return schoolbook
    if best_result[0] < rank_school:
        U, V, W = best_result[1], best_result[2], best_result[3]
        # Try rounding
        U_r = np.round(U).astype(np.int64)
        V_r = np.round(V).astype(np.int64)
        W_r = np.round(W).astype(np.int64)
        if verify(U_r, V_r, W_r):
            return to_output(U_r, V_r, W_r)
        U_h = np.round(2 * U) / 2.0
        V_h = np.round(2 * V) / 2.0
        W_h = np.round(2 * W) / 2.0
        recon = np.zeros((N, M, P), dtype=np.float64)
        for r in range(best_result[0]):
            recon += np.outer(np.outer(U_h[:, r], V_h[:, r]), W_h[:, r])
        if np.allclose(recon, T, atol=1e-9):
            return to_output(U_h, V_h, W_h)
    
    # Fallback: schoolbook
    return to_output(U_sch, V_sch, W_sch)
# EVOLVE-BLOCK-END
