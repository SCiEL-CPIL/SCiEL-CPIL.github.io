import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition of T."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T_approx = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_approx, T, atol=1e-8)

    def to_list(arr):
        return [[float(x) for x in row] for row in arr]

    # Fallback: Rank 28 solution from current program
    def get_rank_28_solution():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        # Part 1: A2 B2 (j=2, k=0..4)
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
        # Part 2: A1 B13 (j=0,1, k=4)
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
        # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen
        # M1
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1
        r += 1
        # M2
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1
        W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1
        r += 1
        # M3
        U[0 * m + 0][r] = 1
        V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1
        W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1
        r += 1
        # M4
        U[1 * m + 1][r] = 1
        V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1
        W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1
        r += 1
        # M5
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1
        r += 1
        # M6
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
        V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1
        W[1 * n + 1][r] = 1
        r += 1
        # M7
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
        V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1
        r += 1
        # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen
        # M1
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1
        r += 1
        # M2
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1
        W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1
        r += 1
        # M3
        U[0 * m + 0][r] = 1
        V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1
        W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1
        r += 1
        # M4
        U[1 * m + 1][r] = 1
        V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1
        W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1
        r += 1
        # M5
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1
        r += 1
        # M6
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
        V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1
        W[3 * n + 1][r] = 1
        r += 1
        # M7
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
        V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1
        r += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Rank 22 solution using block structure
    def get_rank_22_solution():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        # The 2x3 x 3x5 multiplication:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # For i=0,1 (n=2 rows), j=0,1,2 (m=3 cols), k=0..4 (p=5 cols)
        
        # Use Strassen-like for the 2x2 submatrix (rows 0,1 of A, cols 0,1 of B)
        # This gives 7 terms instead of 4 for the 2x2x2 block
        
        # For k=0: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        #         C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        # For k=1: C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        #         C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        
        # Group by B columns:
        # B col 0: contributes A[:,0:2] * B[0:2,0] to C[:,0], plus A[:,2]*B[2,0]
        # B col 1: contributes A[:,0:2] * B[0:2,1] to C[:,1], plus A[:,2]*B[2,1]
        # B col 2: A[:,0:2] * B[0:2,2] + A[:,2]*B[2,2]
        # B col 3: A[:,0:2] * B[0:2,3] + A[:,2]*B[2,3]
        # B col 4: A[:,0:2] * B[0:2,4] + A[:,2]*B[2,4]
        
        # Strategy: Use 7 Strassen terms for the 2x2x5 part (rows 0,1 of A, cols 0,1 of B, all 5 cols of B)
        # Then 5 terms for the 1x1x5 part (row 2 of A, all cols of B)
        # Total: 7 + 5 = 12... but that's not right either.
        
        # Actually, let me think more carefully. The tensor T[i*m+j, j*p+k, k*n+i] = 1
        # We need to decompose this.
        
        # For the 2x2x2 subproblem (i=0,1; j=0,1; k=0,1), we can use 7 Strassen terms.
        # For the remaining parts, we need direct terms.
        
        # Let's use a different approach: partition into blocks and use known optimal decompositions.
        
        # Actually, the minimum rank for 2x3x5 might be achievable at 22.
        # Let me construct it explicitly.
        
        # Terms for j=2 (A's third column times B's third row):
        # C[i,k] += A[i,2] * B[2,k] for i=0,1 and k=0..4
        # This is a rank-10 contribution (2*5=10 terms)
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
        
        # Terms for j=0,1 and k=0,1,2,3,4 (A's first two columns times B's first two rows)
        # This is a 2x2x5 subproblem
        # We can use Strassen for each pair of k values, but that's inefficient.
        # Instead, use a 2x2x5 decomposition.
        
        # For 2x2x5: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1, k=0..4
        # This can be decomposed using the fact that we need 2*2*5 = 20 scalar multiplications naively,
        # but we can do better with Strassen-like methods.
        
        # Actually, for 2x2xn, the minimum rank is known. For n=5, it might be 12.
        # Let me just use direct terms for now and see if we can get to 22.
        # 10 (for j=2) + 12 (for j=0,1,k=0..4) = 22
        
        # For the 2x2x5 part (i=0,1; j=0,1; k=0..4):
        # Use 12 terms
        # Term 1: M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,0]) -> contributes to C[0,0], C[1,0]
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1; V[1 * p + 0][r] = 1
        W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1
        r += 1
        
        # Term 2: M2 = (A[1,0]+A[1,1]) * B[0,0] -> contributes to C[1,0]
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1
        W[0 * n + 1][r] = 1
        r += 1
        
        # Term 3: M3 = A[0,0] * (B[0,1]+B[1,1]) -> contributes to C[0,1]
        U[0 * m + 0][r] = 1
        V[0 * p + 1][r] = 1; V[1 * p + 1][r] = 1
        W[1 * n + 0][r] = 1
        r += 1
        
        # Term 4: M4 = A[1,1] * (B[1,0]+B[0,0]) -> contributes to C[1,0]
        U[1 * m + 1][r] = 1
        V[1 * p + 0][r] = 1; V[0 * p + 0][r] = 1
        W[0 * n + 1][r] = 1
        r += 1
        
        # Term 5: M5 = (A[0,0]+A[0,1]) * B[1,1] -> contributes to C[0,1]
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
        V[1 * p + 1][r] = 1
        W[1 * n + 0][r] = 1
        r += 1
        
        # Term 6: M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1]) -> contributes to C[1,0], C[1,1]
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
        V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1
        W[0 * n + 1][r] = 1; W[1 * n + 1][r] = 1
        r += 1
        
        # Term 7: M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1]) -> contributes to C[0,0], C[0,1]
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
        V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1; W[1 * n + 0][r] = 1
        r += 1
        
        # Now for k=2,3,4, use direct terms for the 2x2 part
        # C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
        for k in [2, 3, 4]:
            for i in range(n):
                for j in range(2):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        
        # Total: 10 + 7 + 12 = 29... that's more than 28.
        # Let me reconsider.
        
        # Actually, I made an error. Let me just return the rank-28 solution for now
        # and fix the ALS to not break the product.
        return get_rank_28_solution()

    # Try to find a rank 27 solution using ALS
    def try_als_for_rank(target_rank, max_iters=100, n_restarts=5):
        if time.time() > deadline:
            return None
        
        best_solution = None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + restart)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U: for each row i, solve min ||T[i,:,:] - U[i,:] V^T W^T||
                # Reshape T to (n*m, m*p * p*n)
                T_flat = T.reshape(n * m, -1)
                # We want U[i,:] such that U[i,:] @ (V^T @ W^T) approx T_flat[i,:]
                # Let M = V @ W.T -> (m*p, p*n) @ (p*n, m*p) is not right.
                # Tensor T = sum_r U[:,r] V[:,r] W[:,r]
                # T[i, j, k] = sum_r U[i,r] V[j,r] W[k,r]
                # Fix V, W. Update U[i,r].
                # T[i, :, :] = sum_r U[i,r] (V[:,r] outer W[:,r])
                # This is a linear system for U[i,:].
                
                # Precompute the "basis" matrices for U update
                # For each r, B_r = V[:,r] outer W[:,r] -> shape (m*p, p*n)
                # We want to solve: T_flat[i] = sum_r U[i,r] B_r.flatten()
                # Let B_mat be shape (target_rank, m*p*p*n) where B_mat[r, :] = B_r.flatten()
                # Then T_flat[i] = U[i,:] @ B_mat.T
                # U[i,:] = T_flat[i] @ pinv(B_mat.T)
                
                # Construct B_mat
                # B_mat[r] = (V[:,r] * W[:,r]) reshaped
                # V[:,r] is (m*p, 1), W[:,r] is (p*n, 1)
                # Outer product is (m*p, p*n)
                
                # Efficient way:
                # T_flat is (n*m, m*p*p*n)
                # We need to solve U[i] = T_flat[i] @ (B_mat.T)^+
                
                # Let's construct B_mat.T which is (m*p*p*n, target_rank)
                # B_mat.T[col, r] = V[col // (p*n), r] * W[col % (p*n), r]
                # This is expensive to construct explicitly for each iteration if done naively.
                # But dimensions are small: m*p=15, p*n=10, target_rank=27.
                # Size of B_mat.T is 150 x 27.
                
                B_mat_T = np.zeros((m * p * p * n, target_rank))
                for r in range(target_rank):
                    # V[:, r] is (m*p, 1)
                    # W[:, r] is (p*n, 1)
                    # Outer product
                    outer_prod = np.outer(V[:, r], W[:, r])
                    B_mat_T[:, r] = outer_prod.flatten()
                
                # Solve for U
                # U[i, r] = sum_k T_flat[i, k] * (pinv(B_mat_T)[k, r])
                # pinv(B_mat_T) is (150, 27)
                # Let P = pinv(B_mat_T)
                # U = T_flat @ P
                try:
                    P_U = np.linalg.pinv(B_mat_T)
                    U_new = T_flat @ P_U
                except:
                    break
                
                # Update V
                # T_flat_V: reshape T to (m*p, n*m * p*n)
                # T[j, i, k] = sum_r V[j,r] U[i,r] W[k,r]
                # For fixed j, T[j, :, :] = sum_r V[j,r] (U[:,r] outer W[:,r])
                T_flat_V = T.reshape(m * p, n * m * p * n)
                B_mat_T_V = np.zeros((n * m * p * n, target_rank))
                for r in range(target_rank):
                    outer_prod = np.outer(U[:, r], W[:, r])
                    B_mat_T_V[:, r] = outer_prod.flatten()
                try:
                    P_V = np.linalg.pinv(B_mat_T_V)
                    V_new = T_flat_V @ P_V
                except:
                    break
                
                # Update W
                T_flat_W = T.reshape(p * n, n * m * m * p)
                B_mat_T_W = np.zeros((n * m * m * p, target_rank))
                for r in range(target_rank):
                    outer_prod = np.outer(U[:, r], V[:, r])
                    B_mat_T_W[:, r] = outer_prod.flatten()
                try:
                    P_W = np.linalg.pinv(B_mat_T_W)
                    W_new = T_flat_W @ P_W
                except:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence/validity
                if verify(U, V, W):
                    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
                
                # Check if we're getting close (for debugging)
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.max(np.abs(T_approx - T))
                if err < 1e-6:
                    # Try to polish
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        T_flat = T.reshape(n * m, -1)
                        B_mat_T = np.zeros((m * p * p * n, target_rank))
                        for rr in range(target_rank):
                            outer_prod = np.outer(V[:, rr], W[:, rr])
                            B_mat_T[:, rr] = outer_prod.flatten()
                        try:
                            P_U = np.linalg.pinv(B_mat_T)
                            U = T_flat @ P_U
                        except:
                            break
                        T_flat_V = T.reshape(m * p, n * m * p * n)
                        B_mat_T_V = np.zeros((n * m * p * n, target_rank))
                        for rr in range(target_rank):
                            outer_prod = np.outer(U[:, rr], W[:, rr])
                            B_mat_T_V[:, rr] = outer_prod.flatten()
                        try:
                            P_V = np.linalg.pinv(B_mat_T_V)
                            V = T_flat_V @ P_V
                        except:
                            break
                        T_flat_W = T.reshape(p * n, n * m * m * p)
                        B_mat_T_W = np.zeros((n * m * m * p, target_rank))
                        for rr in range(target_rank):
                            outer_prod = np.outer(U[:, rr], V[:, rr])
                            B_mat_T_W[:, rr] = outer_prod.flatten()
                        try:
                            P_W = np.linalg.pinv(B_mat_T_W)
                            W = T_flat_W @ P_W
                        except:
                            break
                        if verify(U, V, W):
                            return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
                
        return None

    # First, check if rank 27 is possible
    sol_27 = try_als_for_rank(27, max_iters=50, n_restarts=3)
    if sol_27:
        return sol_27
    
    # Try rank 26
    sol_26 = try_als_for_rank(26, max_iters=30, n_restarts=2)
    if sol_26:
        return sol_26
    
    # If not, return the known rank 28 solution
    return get_rank_28_solution()

# EVOLVE-BLOCK-END
