"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Schoolbook decomposition as baseline
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        U_mat = U.reshape(n, m, r)
        V_mat = V.reshape(m, p, r)
        W_mat = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=tol, rtol=0)

    # Structured Kronecker product decomposition
    # Key insight: Factor 4x7 multiplication using known rank-4 decomposition for 4x4x4
    # The 4x4x4 tensor has rank 4 (Strassen-like). We can use this to reduce the overall rank.
    
    # Strategy: 
    # 1. Use the rank-4 decomposition for 4x4x4 matrix multiplication
    # 2. Combine with the 2x4x4 tensor via direct sum
    # 3. This should give rank = 2*4 + 4*3 = 8 + 12 = 20 (vs schoolbook 56)
    
    # Known rank-4 decomposition for 4x4x4 (using Strassen's algorithm on 2x2 blocks)
    # Actually, let's use a simpler approach: exploit the rectangular structure
    
    # For 2x4 * 4x7:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can split B into two 4x4 and 4x3 blocks: B = [B1 | B2] where B1 is 4x4, B2 is 4x3
    # Then C = A*B1 + A*B2
    
    # The rank of the tensor T_{n,m,p} is at most rank(T_{n,m,q}) + rank(T_{n,m,p-q})
    # for any q. So rank(T_{2,4,7}) <= rank(T_{2,4,4}) + rank(T_{2,4,3})
    
    # Now we need rank(T_{2,4,4}) and rank(T_{2,4,3})
    # T_{2,4,4}: This is 2x4 * 4x4. We can use the fact that rank(T_{n,m,p}) <= n * min(m,p) * max(m,p) / min(m,p)
    # Actually, a better bound: rank(T_{n,m,p}) <= n * m * p / max(n,m,p) * something
    
    # Let's use a direct construction:
    # For T_{2,4,4}: We can view this as two 2x4 * 4x1 multiplications (for each output column)
    # But that gives rank 2*4*4 = 32, which is not helpful.
    
    # Better approach: Use the known result that rank(T_{n,m,p}) <= n * m * p - min(n,m,p) * (min(n,m,p) - 1)
    # For T_{2,4,4}: rank <= 2*4*4 - 2*(2-1) = 32 - 2 = 30
    # For T_{2,4,3}: rank <= 2*4*3 - 2*(2-1) = 24 - 2 = 22
    # Total: 30 + 22 = 52, still not great.
    
    # Let's try a different structured approach:
    # Use the fact that we can share computations across the 7 output columns.
    # Specifically, if we have a rank-r decomposition for 2x4 * 4xq, we can extend it.
    
    # Actually, the simplest structured improvement:
    # The tensor T_{2,4,7} can be decomposed as:
    # T_{2,4,7} = T_{2,4,1} ⊗ I_7 (not quite right)
    
    # Let me think differently. The key is that for fixed i,k, we need:
    # sum_j A[i,j] * B[j,k] = C[i,k]
    # This is a bilinear form. The rank of the tensor is the minimum number of rank-1 tensors needed.
    
    # A simple structured decomposition:
    # For each (i,k), C[i,k] = sum_j A[i,j] * B[j,k]
    # We can write this as: C[i,k] = sum_{j1=0}^{1} A[i,j1] * B[j1,k] + sum_{j2=2}^{3} A[i,j2] * B[j2,k]
    # This doesn't reduce rank directly.
    
    # Let's try: Use the fact that B has 7 columns. We can group them.
    # If we use a rank-4 decomposition for the "middle" 4x4 part, we can reduce the rank.
    
    # Actually, here's a concrete structured construction:
    # The tensor T_{2,4,7} has 2*4*7 = 56 entries.
    # We can try to find a rank-28 decomposition by exploiting symmetries.
    
    # For now, let's use a simpler approach: try alternating minimization with better initialization
    # based on the structure.
    
    def try_structured_am(target_rank, n_restarts=5, max_iters=200, tol=1e-10):
        """Alternating minimization with structured initialization."""
        best_result = None
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Structured initialization: start from schoolbook and randomly perturb
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Add schoolbook components if target_rank >= schoolbook rank
            if target_rank >= rank_schoolbook:
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            col = i * m * p + j * p + k
                            if col < target_rank:
                                U[i * m + j, col] += 1.0
                                V[j * p + k, col] += 1.0
                                W[k * n + i, col] += 1.0
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break

                U_mat = U.reshape(n, m, target_rank)
                V_mat = V.reshape(m, p, target_rank)
                W_mat = W.reshape(p, n, target_rank)

                # Solve for U
                for i in range(n):
                    for j in range(m):
                        A = V_mat[j] * W_mat[:, i]  # shape (p, target_rank)
                    b = np.ones(p)
                    try:
                        U_row, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U_mat[i, j] = U_row
                    except np.linalg.LinAlgError:
                        pass

                # Solve for V
                for j in range(m):
                    for k in range(p):
                        B = U_mat[:, j] * W_mat[k]  # shape (n, target_rank)
                        b = np.ones(n)
                        try:
                            V_row, _, _, _ = np.linalg.lstsq(B.T, b, rcond=None)
                            V_mat[j, k] = V_row
                        except np.linalg.LinAlgError:
                            pass

                # Solve for W
                for k in range(p):
                    for i in range(n):
                        C = U_mat[i] * V_mat[:, k]  # shape (m, target_rank)
                        b = np.ones(m)
                        try:
                            W_row, _, _, _ = np.linalg.lstsq(C.T, b, rcond=None)
                            W_mat[k, i] = W_row
                        except np.linalg.LinAlgError:
                            pass

                # Check convergence
                T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
                err = np.max(np.abs(T - 1.0))
                if err < tol:
                    return U_mat.reshape(n*m, target_rank), V_mat.reshape(m*p, target_rank), W_mat.reshape(p*n, target_rank)

                if it % 20 == 19 and err > 1.0:
                    break

            T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
            err = np.max(np.abs(T - 1.0))
            if err < 1e-6:
                return U_mat.reshape(n*m, target_rank), V_mat.reshape(m*p, target_rank), W_mat.reshape(p*n, target_rank)

        return best_result

    # Try target ranks from just below schoolbook downward
    for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook - 20, 0), -1):
        if time.time() > deadline - 10:
            break
        
        result = try_structured_am(target_rank, n_restarts=3, max_iters=150, tol=1e-10)
        if result is not None:
            U_try, V_try, W_try = result
            if verify(U_try, V_try, W_try, tol=1e-9):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break

    # If still at schoolbook, try more aggressive search
    if best_rank == rank_schoolbook and time.time() < deadline - 20:
        for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook - 10, 0), -1):
            if time.time() > deadline - 10:
                break
            result = try_structured_am(target_rank, n_restarts=15, max_iters=400, tol=1e-10)
            if result is not None:
                U_try, V_try, W_try = result
                if verify(U_try, V_try, W_try, tol=1e-9):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_try, V_try, W_try
                    break

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
