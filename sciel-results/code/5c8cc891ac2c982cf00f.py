"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16

    start_time = time.time()
    deadline = start_time + 140

    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    def check_decomp(U, V, W, tol=1e-8):
        # Vectorized check: for each r, term = U[:,r] * V[:,r] * W[:,r] broadcast
        # Sum over r should give 1 for all (i,j,k)
        # Reshape: U is (n*m, r), V is (m*p, r), W is (p*n, r)
        # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i,j,k] = 1 for all i,j,k
        
        # Efficient check:
        # Construct the tensor from factors and compare to ones
        # But full tensor is 2x4x8=64 elements, rank is up to 64.
        # So we can just loop over the 64 elements.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    # U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
                    # val = dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                    val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def make_strassen_blocks():
        # Strassen for 2x2x2 has rank 7.
        # We decompose 2x4 x 4x8 into blocks.
        # A is 2x4, B is 4x8, C is 2x8.
        # We can view this as 4 blocks of 2x2x2 along the 'm' and 'p' dimensions?
        # A = [A0 A1] where A0, A1 are 2x2.
        # B = [B0; B1] where B0, B1 are 2x4.
        # C = A0 B0 + A1 B1.
        # B0 is 2x4, B1 is 2x4.
        # This splits B into two 2x4 matrices.
        # Then C = A0 B0 + A1 B1.
        # Each term A_i B_j is (2x2)(2x4).
        # (2x2)(2x4) can be viewed as two (2x2)(2x2) multiplications.
        # So total blocks:
        # A0 B00, A0 B01, A1 B10, A1 B11.
        # Each is 2x2 x 2x2.
        # There are 4 such blocks.
        # Rank = 4 * 7 = 28.
        
        # Let's verify indices.
        # A[i, j] where i in {0,1}, j in {0,1,2,3}.
        # B[j, k] where j in {0,1,2,3}, k in {0..7}.
        # C[i, k] where i in {0,1}, k in {0..7}.
        
        # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # Block 1: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        # Wait, A is 2x4. So A[0:2, 0:2] and A[0:2, 2:4].
        # B is 4x8. B[0:2, 0:8] and B[2:4, 0:8].
        
        # So:
        # C[:, 0:4] = A[:, 0:2] B[0:2, 0:4] + A[:, 2:4] B[2:4, 0:4]
        # C[:, 4:8] = A[:, 0:2] B[0:2, 4:8] + A[:, 2:4] B[2:4, 4:8]
        
        # This gives 4 sub-problems of size 2x2 * 2x4.
        # A 2x2 * 2x4 multiplication is equivalent to two 2x2 * 2x2 multiplications.
        # So total 8 sub-problems of 2x2 * 2x2.
        # Rank = 8 * 7 = 56.
        
        # Can we do better?
        # Maybe group differently.
        # A is 2x4. B is 4x8.
        # Think of A as a vector of 8 elements. B as vector of 32.
        # This is bilinear.
        
        # Let's construct the rank-56 Strassen decomposition.
        
        # Strassen 2x2x2 factors:
        # U_local (4x7), V_local (4x7), W_local (4x7)
        # Indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
        #          B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
        #          C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
        
        U_local = np.zeros((4, 7))
        V_local = np.zeros((4, 7))
        W_local = np.zeros((4, 7))
        
        # M1 = (a00+a11)(b00+b11)
        U_local[0, 0] = 1; U_local[3, 0] = 1
        V_local[0, 0] = 1; V_local[3, 0] = 1
        W_local[0, 0] = 1; W_local[3, 0] = 1
        
        # M2 = (a10+a11)b00
        U_local[2, 1] = 1; U_local[3, 1] = 1
        V_local[0, 1] = 1
        W_local[2, 1] = 1; W_local[3, 1] = -1
        
        # M3 = a00(b01-b11)
        U_local[0, 2] = 1
        V_local[1, 2] = 1; V_local[3, 2] = -1
        W_local[1, 2] = 1; W_local[3, 2] = 1
        
        # M4 = a11(b10-b00)
        U_local[3, 3] = 1
        V_local[2, 3] = 1; V_local[0, 3] = -1
        W_local[0, 3] = 1; W_local[2, 3] = 1
        
        # M5 = (a00+a01)b11
        U_local[0, 4] = 1; U_local[1, 4] = 1
        V_local[3, 4] = 1
        W_local[0, 4] = -1; W_local[1, 4] = 1
        
        # M6 = (a10-a00)(b00+b01)
        U_local[2, 5] = 1; U_local[0, 5] = -1
        V_local[0, 5] = 1; V_local[1, 5] = 1
        W_local[3, 5] = 1
        
        # M7 = (a01-a11)(b10+b11)
        U_local[1, 6] = 1; U_local[3, 6] = -1
        V_local[2, 6] = 1; V_local[3, 6] = 1
        W_local[0, 6] = 1
        
        total_rank = 56
        U = np.zeros((U_dim, total_rank))
        V = np.zeros((V_dim, total_rank))
        W = np.zeros((W_dim, total_rank))
        
        # 8 blocks of 2x2x2
        # Block (a_row, a_col, b_row, b_col)
        # A indices: i in {0,1}, j in {0..3}
        # B indices: j in {0..3}, k in {0..7}
        # C indices: i in {0,1}, k in {0..7}
        
        # Block 0: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
        # Block 1: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
        # Block 2: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
        # Block 3: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
        # Block 4: A[:, 0:2] * B[0:2, 4:6] -> C[:, 4:6]
        # Block 5: A[:, 0:2] * B[0:2, 6:8] -> C[:, 6:8]
        # Block 6: A[:, 2:4] * B[2:4, 4:6] -> C[:, 4:6]
        # Block 7: A[:, 2:4] * B[2:4, 6:8] -> C[:, 6:8]
        
        blocks = [
            (0, 2, 0, 2, 0, 2), # A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
            (0, 2, 0, 2, 2, 4), # A[0:2, 0:2], B[0:2, 2:4], C[0:2, 2:4]
            (2, 4, 2, 4, 0, 2), # A[0:2, 2:4], B[2:4, 0:2], C[0:2, 0:2]
            (2, 4, 2, 4, 2, 4), # A[0:2, 2:4], B[2:4, 2:4], C[0:2, 2:4]
            (0, 2, 0, 2, 4, 6), # A[0:2, 0:2], B[0:2, 4:6], C[0:2, 4:6]
            (0, 2, 0, 2, 6, 8), # A[0:2, 0:2], B[0:2, 6:8], C[0:2, 6:8]
            (2, 4, 2, 4, 4, 6), # A[0:2, 2:4], B[2:4, 4:6], C[0:2, 4:6]
            (2, 4, 2, 4, 6, 8), # A[0:2, 2:4], B[2:4, 6:8], C[0:2, 6:8]
        ]
        
        for b_idx, (a_j1, a_j2, b_j1, b_j2, c_k1, c_k2) in enumerate(blocks):
            base_r = b_idx * 7
            
            # A mapping: local (li, lj) -> global (li, a_j1+lj)
            # U index: i*m + j
            for li in range(2):
                for lj in range(2):
                    g_i = li
                    g_j = a_j1 + lj
                    u_idx = g_i * m + g_j
                    for l in range(7):
                        U[u_idx, base_r + l] += U_local[li*2+lj, l]
            
            # B mapping: local (lj, lk) -> global (b_j1+lj, c_k1+lk)
            # V index: j*p + k
            for lj in range(2):
                for lk in range(2):
                    g_j = b_j1 + lj
                    g_k = c_k1 + lk
                    v_idx = g_j * p + g_k
                    for l in range(7):
                        V[v_idx, base_r + l] += V_local[lj*2+lk, l]
            
            # C mapping: local (li, lk) -> global (li, c_k1+lk)
            # W index: k*n + i
            for li in range(2):
                for lk in range(2):
                    g_i = li
                    g_k = c_k1 + lk
                    w_idx = g_k * n + g_i
                    for l in range(7):
                        W[w_idx, base_r + l] += W_local[li*2+lk, l]
        
        return U, V, W, total_rank

    # Start with schoolbook
    U0, V0, W0, r0 = make_schoolbook()
    best_rank = r0
    best = (U0, V0, W0, r0)
    
    # Try Strassen-based
    U1, V1, W1, r1 = make_strassen_blocks()
    if check_decomp(U1, V1, W1):
        if r1 < best_rank:
            best_rank = r1
            best = (U1, V1, W1, r1)
    
    # Try to reduce rank of Strassen decomposition using null space
    # If we have a valid decomposition of rank R, we can try to find linear dependencies
    # among the columns of the factor matrices.
    # The condition is that the tensor formed by the outer products is unchanged.
    # This is hard to do directly. Instead, we can use random linear combinations.
    # If we replace column r1 with c1*col1 + c2*col2, we must adjust the other columns?
    # No, that changes the tensor.
    # The tensor is T = sum_r U[:,r] V[:,r]^T W[:,r]^T.
    # If we have a dependency sum_r alpha_r U[:,r] V[:,r]^T W[:,r]^T = 0,
    # then we can eliminate one term if we can express it in terms of others?
    # Not exactly, because it's a sum of rank-1 tensors.
    # If T = sum_{r=1}^R T_r, and T_k = sum_{r != k} beta_r T_r, then T = sum_{r != k} (1+beta_r) T_r.
    # This reduces rank by 1.
    
    # How to find such dependencies?
    # We can try to solve for beta such that T_k = sum_{r != k} beta_r T_r.
    # This is a system of linear equations on the entries of the tensors.
    # T_r is a 2x4x8 tensor. 64 entries.
    # We have R-1 unknowns.
    # We can set up a least squares problem.
    
    def try_reduce_rank(U, V, W, R):
        # For each column k, try to express T_k in terms of others.
        # This is expensive: R * (solve 64x(R-1) system).
        # R=56, so 56 * (64x55) solves. Feasible.
        
        # Precompute tensor entries for all r
        # T_r[i,j,k] = U[i,r] V[j,r] W[k,r]
        # Flatten to vector of length 64.
        
        tensors = np.zeros((R, n*m*p))
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensors[r, i*m*p + j*p + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R
        improved = True
        while improved and best_R > 1:
            improved = False
            for k in range(best_R):
                if time.time() > deadline:
                    return best_U, best_V, best_W, best_R
                # Try to express tensor k in terms of others
                # A * beta = b
                # A is 64 x (best_R-1), b is 64x1
                # Columns of A are tensors of indices != k
                
                idx = [i for i in range(best_R) if i != k]
                A = tensors[idx, :]
                b = tensors[k, :]
                
                # Solve least squares
                try:
                    beta, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                    # Check if residual is small
                    if residuals and residuals[0] < 1e-8:
                        # We can eliminate k
                        # New tensors: for i != k, T_i' = T_i + beta_{i-k} T_k
                        # But we need to update U, V, W.
                        # T_i' = U_i V_i^T W_i^T + beta_i U_k V_k^T W_k^T
                        # This is not a rank-1 tensor in general.
                        # So this method doesn't directly give a new rank-1 decomposition.
                        # We need a different approach.
                        pass
                except:
                    pass
        return best_U, best_V, best_W, best_R

    # The above reduction doesn't work directly because linear combinations of rank-1 tensors
    # are not rank-1.
    # We need a different strategy.
    
    # Strategy: Random search for rank < current best.
    # Start with a random initialization of U, V, W with rank r.
    # Use gradient descent to minimize error.
    
    def gradient_optimize(U, V, W, r, steps=1000, lr=0.01):
        nonlocal deadline
        for step in range(steps):
            if time.time() > deadline:
                break
            # Compute error and gradients
            # Error = sum_{i,j,k} (sum_r U[i,r]V[j,r]W[k,r] - 1)^2
            # dError/dU[i,r] = 2 * sum_{j,k} (sum_s U[i,s]V[j,s]W[k,s] - 1) V[j,r] W[k,r]
            
            # Compute T[i,j,k] = sum_s U[i,s]V[j,s]W[k,s]
            # This is expensive.
            # We can use the structure.
            
            # Let's just do a simple random perturbation if gradient is too slow.
            # Or compute gradient efficiently.
            
            # Efficient gradient:
            # For fixed i, r:
            # grad_U[i,r] = 2 * sum_{j,k} E[i,j,k] V[j,r] W[k,r]
            # where E[i,j,k] = T[i,j,k] - 1
            
            # We can compute T first.
            # T is 2x4x8.
            # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            
            # This is still O(n*m*p*r).
            # n*m*p = 64, r ~ 50. 3200 ops. Very fast.
            
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
            
            E = T - 1.0
            err = np.sum(E**2)
            if err < 1e-6:
                return U, V, W, r, True
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for r in range(r):
                # dU[:,r]
                # dU[i,r] = 2 * sum_{j,k} E[i,j,k] V[j,r] W[k,r]
                for i in range(n):
                    # sum_{j,k} E[i,j,k] V[j,r] W[k,r]
                    # This is like a tensor contraction.
                    # Let's just loop.
                    s = 0.0
                    for j in range(m):
                        for k in range(p):
                            s += E[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                    dU[i*m, r] = 2 * s # Wait, U index is i*m+j? No, U is (n*m, r).
                    # U row index is i*m+j.
                    # So dU[i*m+j, r] depends on j too.
                    # My formula was for fixed i.
                    # Actually dU[i*m+j, r] = 2 * sum_{i',k'} E[i', j, k'] U[i', r] W[k', r] ?? No.
                    
                    # Let's re-derive.
                    # Err = sum_{i,j,k} (sum_s U[i,j,s] V[j,k,s] W[k,i,s] - 1)^2
                    # dErr/dU[i,j,r] = 2 * sum_{k} (sum_s U[i,j,s] V[j,k,s] W[k,i,s] - 1) V[j,k,r] W[k,i,r]
                    
                    # So for fixed i,j,r:
                    # grad = 2 * sum_k E[i,j,k] V[j,k,r] W[k,i,r]
                    
                # Let's rewrite the loop correctly.
                for i in range(n):
                    for j in range(m):
                        s = 0.0
                        for k in range(p):
                            s += E[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                        dU[i*m+j, r] = 2 * s
                
                # dV[j*p+k, r] = 2 * sum_i E[i,j,k] U[i*m+j, r] W[k*n+i, r]
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for i in range(n):
                            s += E[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
                        dV[j*p+k, r] = 2 * s
                
                # dW[k*n+i, r] = 2 * sum_j E[i,j,k] U[i*m+j, r] V[j*p+k, r]
                for k in range(p):
                    for i in range(n):
                        s = 0.0
                        for j in range(m):
                            s += E[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
                        dW[k*n+i, r] = 2 * s
            
            # Update
            U -= lr * dU
            V -= lr * dW # Wait, V gradient is dV
            V -= lr * dV
            W -= lr * dW
            
        return U, V, W, r, False

    # Try ranks lower than current best
    # Current best is 56 (Strassen) or 64 (Schoolbook).
    # Lower bounds:
    # R(2x2x2) = 7.
    # R(2x4x8) >= ?
    # Asymptotic rank lower bound is not tight for small sizes.
    # Known bounds: R(2x4x4) >= 24? R(2x4x8) >= ?
    # Let's try 48, 44, 40, 36, 32.
    
    for r in [48, 44, 40, 36, 32, 28]:
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
        
        found = False
        for trial in range(10):
            if time.time() > deadline:
                break
            
            # Initialize
            # Scale matters.
            U = np.random.randn(U_dim, r) * 0.5
            V = np.random.randn(V_dim, r) * 0.5
            W = np.random.randn(W_dim, r) * 0.5
            
            # Symmetry breaking?
            # Fix one entry?
            # U[0,0] = 1?
            
            U, V, W, r_eff, success = gradient_optimize(U, V, W, r, steps=200, lr=0.1)
            
            if success and check_decomp(U, V, W):
                best_rank = r_eff
                best = (U, V, W, r_eff)
                found = True
                break
        
        if found:
            # Try to go lower immediately?
            # Or break and let the loop handle it?
            # The loop iterates over fixed list.
            # If we found 40, we should try 36 next.
            # The list is descending, so it will.
            pass
    
    U, V, W, r = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
