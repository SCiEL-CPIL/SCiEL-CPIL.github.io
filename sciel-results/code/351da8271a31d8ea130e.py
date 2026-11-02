"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5

    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    def verify(U_mat, V_mat, W_mat, n, m, p, num_trials=20, seed=42):
        """Verify that the decomposition correctly computes matrix multiplication."""
        rng = np.random.RandomState(seed)
        for _ in range(num_trials):
            A = rng.rand(n, m)
            B = rng.rand(m, p)
            C_expected = A @ B
            # Flatten in the required conventions
            a_flat = A.flatten()  # a[i*m + j]
            b_flat = B.flatten()  # b[j*p + k]
            c_flat_expected = np.zeros(p * n)
            for k in range(p):
                for i in range(n):
                    c_flat_expected[k * n + i] = C_expected[i, k]
            # Compute via decomposition: c = sum_r (a·U_r)(b·V_r) W_r
            c_flat = np.zeros(p * n)
            for r in range(U_mat.shape[1]):
                c_flat += (a_flat @ U_mat[:, r]) * (b_flat @ V_mat[:, r]) * W_mat[:, r]
            if not np.allclose(c_flat, c_flat_expected, atol=1e-9):
                return False
        return True

    def verify_fast(U_mat, V_mat, W_mat, n, m, p, num_trials=50, seed=123):
        """Faster vectorized verification."""
        rng = np.random.RandomState(seed)
        for _ in range(num_trials):
            A = rng.rand(n, m)
            B = rng.rand(m, p)
            C_expected = A @ B
            a_flat = A.flatten()
            b_flat = B.flatten()
            # c = sum_r (a·U_r)(b·V_r) W_r = W * (V^T b) * (U^T a) elementwise... 
            # Actually: c = sum_r (U^T a)_r * (V^T b)_r * W_r
            ua = U_mat.T @ a_flat  # shape (rank,)
            vb = V_mat.T @ b_flat  # shape (rank,)
            c_flat = (ua * vb) @ W_mat  # shape (p*n,)
            c_flat_expected = np.zeros(p * n)
            for k in range(p):
                for i in range(n):
                    c_flat_expected[k * n + i] = C_expected[i, k]
            if not np.allclose(c_flat, c_flat_expected, atol=1e-9):
                return False
        return True

    # ============================================================
    # Strategy: The tensor for M_{2,5,5} can be viewed as two copies
    # of the 1x5x5 matrix multiplication tensor (each row of the 2x5
    # matrix times the 5x5 matrix gives a 1x5 result).
    # 
    # For 1x5x5 matrix mult, the tensor is: C_k = sum_j A_j * B_{j,k}
    # This is the same as the "vector-matrix product" tensor.
    # 
    # The rank of the 1x5x5 tensor is 5 (it's just a matrix-vector product,
    # which has rank = dimension of the matrix = 5).
    # 
    # But we can potentially share structure between the two rows.
    # 
    # Actually, let's think about this differently. The full tensor is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k} for i=0,1 and k=0,...,4
    # 
    # This is a 3-tensor of size (2*5) x (5*5) x (5*2) = 10 x 25 x 10
    # 
    # The naive decomposition has rank 10 (one for each (i,j) pair):
    # For each (i,j): U picks out A_{i,j}, V picks out row j of B, W picks out column k of C for row i
    # 
    # Can we do better? Let's try to use the structure.
    # 
    # Key insight: The tensor T has the property that for fixed j, the slice
    # T[:,j,:] is a 10x10 matrix (in (i,k) and (i',k') indices) that equals
    # the identity in the appropriate sense.
    # 
    # Actually, let me think about this more carefully using the matrix unfolding.
    # 
    # The tensor T[i,k,j] (where i indexes row of A/C, k indexes col of B/C, j indexes shared dim)
    # T[i,k,j] = 1 for all i,k,j? No!
    # 
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is T_{(i,k), j} where the bilinear form is:
    # <T, A, B> = sum_{i,k,j} T_{(i,k),j} A_{i,j} B_{j,k} = sum_{i,k} C_{i,k}^2? No.
    # 
    # Let me be precise. The multiplication tensor M_{n,m,p} is defined by:
    # M_{(i,k), (i',j'), (j'',k'')} = delta_{i,i'} delta_{j',j''} delta_{k,k''}
    # 
    # So M[(k,i), (i',j'), (j'',k'')] = 1 iff i=i', j'=j'', k=k''
    # 
    # This means the tensor is essentially the "identity" tensor in a certain sense.
    # 
    # The rank of this tensor: We need to find the minimum r such that
    # M = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # where u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{n*p}
    # 
    # For the schoolbook: R = n*m*p = 50
    # For the "row-wise" decomposition: R = n*m = 10 (as shown in the seed)
    # 
    # Can we do better than 10? 
    # 
    # Lower bound: The tensor has "slice rank" considerations. 
    # For each fixed j, the slice M[:,j,:] is a (n*p) x (n*m) matrix... 
    # Actually, let me think about the matricization.
    # 
    # Matricize along the first mode: M_(1) is a (n*p) x (n*m * m*p) matrix
    # M_(1)[(k,i), ((i',j'),(j'',k''))] = delta_{i,i'} delta_{j',j''} delta_{k,k''}
    # 
    # The rank of M_(1) gives a lower bound on the tensor rank.
    # Let's compute: For fixed (k,i), the row of M_(1) has 1s at positions
    # where i'=i, j'=j'', k''=k. So for each (k,i), there are m positions with 1.
    # These rows for different (k,i) are linearly independent (they have 1s in
    # disjoint sets of columns... wait, no, they overlap in the j'=j'' constraint).
    # 
    # Actually, for (k,i) and (k',i') with i≠i', the rows have 1s in disjoint
    # column blocks (since i' must equal i or i'). So the rank is at least n*p = 10.
    # 
    # Similarly, matricizing along mode 2: M_(2) is (m*p) x (n*p * n*m)
    # For fixed (j,k), the row has 1s where j'=j, j''=j, and i',k'' free.
    # So for each (j,k), there are n*p = 10 positions. Different j give disjoint
    # support, so rank ≥ m*p = 25? No wait...
    # 
    # Let me just try to find a rank < 10 decomposition by optimization.
    # 
    # Actually, I recall that the rank of M_{1,n,n} (vector-matrix product) is n.
    # And M_{2,5,5} contains two copies of M_{1,5,5} that share the B factor.
    # The rank should be at least 5 (from the shared B structure) but the two
    # output rows need to be distinguished.
    # 
    # Let me try a different approach: use the fact that the tensor is a sum
    # of 5 "blocks" (one for each j), where block j is:
    # T_j[(k,i),(i',j'),(j'',k'')] = delta_{j,j'} delta_{j,j''} delta_{i,i'} delta_{k,k''}
    # 
    # Each block T_j has rank 1 as a 3-tensor? Let's check:
    # T_j = u_j ⊗ v_j ⊗ w_j where
    # u_j[(i',j')] = delta_{j,j'} (picks column j of A)
    # v_j[(j'',k'')] = delta_{j,j''} (picks row j of B)  
    # w_j[(k,i)] = 1 (all ones in the output)
    # 
    # Then (u_j ⊗ v_j ⊗ w_j)[(k,i),(i',j'),(j'',k'')] = delta_{j,j'} * delta_{j,j''} * 1
    # This gives 1 when j'=j''=j, regardless of i,k,i',k''.
    # But we want delta_{i,i'} delta_{k,k''} as well!
    # 
    # So T_j is NOT rank 1. It's rank n*p = 10 (need to separate each (i,k) output).
    # 
    # Total rank = sum over j of rank(T_j) = 5 * 10 = 50 (schoolbook).
    # 
    # But we can do better by sharing the u and v factors across j values
    # for the same i or k.
    # 
    # The rank-10 decomposition: For each (i,j), one rank-1 term:
    # u_{i,j}[(i',j')] = delta_{i,i'} delta_{j,j'}
    # v_{i,j}[(j'',k'')] = delta_{j,j''}
    # w_{i,j}[(k,i')] = delta_{i,i'}
    # 
    # Check: sum_{i,j} (a·u_{i,j})(b·v_{i,j}) w_{i,j}[(k,i')]
    # = sum_{i,j} a[i*m+j] b[j*p+k] delta_{i,i'}
    # = sum_j a[i'*m+j] b[j*p+k] = C_{i',k} ✓
    # 
    # So rank 10 works. Can we do rank 9 or less?
    # 
    # Lower bound argument: Consider the restriction to A having only one nonzero
    # entry A_{0,0} = 1, and B arbitrary. Then C_{0,k} = B_{0,k}, C_{1,k} = 0.
    # This is a linear map from B (dim 25) to C (dim 10) with rank 5.
    # Similarly for other single-entry A's.
    # 
    # A more careful lower bound: The tensor rank of M_{n,m,p} is at least
    # max(n*m, m*p, n*p) in some cases, but this isn't always tight.
    # 
    # For M_{2,5,5}: max(10, 25, 10) = 25? That can't be right since we have rank 10.
    # 
    # Let me just try to optimize for rank < 10 numerically.
    # If it's not possible, rank 10 is our answer.

    # First, establish the rank-10 solution
    rank_10 = n * m  # 10
    U_10 = np.zeros((n * m, rank_10))
    V_10 = np.zeros((m * p, rank_10))
    W_10 = np.zeros((p * n, rank_10))
    col = 0
    for i in range(n):
        for j in range(m):
            U_10[i * m + j, col] = 1.0
            for k in range(p):
                V_10[j * p + k, col] = 1.0
            for k in range(p):
                W_10[k * n + i, col] = 1.0
            col += 1

    best_U = U_10.copy()
    best_V = V_10.copy()
    best_W = W_10.copy()
    best_rank = rank_10

    if verify_fast(best_U, best_V, best_W, n, m, p):
        pass  # Confirmed rank 10 works
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_sb

    # Now try to find rank < 10 via optimization
    # We'll use a simple gradient-free approach: try random initializations
    # and use the fact that if a rank-r decomposition exists, we can find it
    # by solving a system of equations.
    # 
    # The condition is: for all a, b: sum_r (a·U_r)(b·V_r) W_r = T(a,b)
    # This is equivalent to: the tensor T = sum_r U_r ⊗ V_r ⊗ W_r
    # 
    # We can set up: for each (i,k) output, the coefficient of a_{i',j'} b_{j'',k''}
    # must be delta_{i,i'} delta_{j',j''} delta_{k,k''}
    # 
    # sum_r U_r[i*m+j] V_r[j'*p+k] W_r[k*n+i] = delta_{i,i'} delta_{j',j''} delta_{k,k''}
    # (where i' is the row index in U, j' is the col index in U / row in V, etc.)
    # 
    # This gives n*p*(n*m)*(m*p) = 10*10*25 = 2500 equations in r*40 unknowns.
    # For r=9: 360 unknowns, 2500 equations. Overdetermined but may have a solution.
    # For r=10: 400 unknowns, 2500 equations.
    # 
    # Let's try a different approach: use the structure to reduce the problem.
    # 
    # Actually, let me try a constructive approach. The key observation is:
    # The tensor M_{2,5,5} can be written as I_2 ⊗ M_{1,5,5} in a certain sense,
    # where the I_2 acts on the "row" index. This suggests the rank might be
    # 2 * rank(M_{1,5,5}) = 2*5 = 10, which matches our construction.
    # 
    # To get rank < 10, we'd need to share terms between the two rows.
    # Let's try rank 9 with a structured initialization.

    if time.time() < deadline - 60:
        # Try rank 9
        for target_rank in [9, 8]:
            if time.time() > deadline - 30:
                break
            
            # Initialize with the rank-10 solution, then try to merge two terms
            # by finding a direction in the null space
            success = False
            for attempt in range(50):
                if time.time() > deadline - 30:
                    break
                
                rng = np.random.RandomState(attempt * 7 + 13)
                # Start from rank-10 and try to perturb
                U_t = best_U[:, :target_rank].copy()
                V_t = best_V[:, :target_rank].copy()
                W_t = best_W[:, :target_rank].copy()
                
                # Add random perturbation to encourage finding a different solution
                U_t += rng.randn(n * m, target_rank) * 0.1
                V_t += rng.randn(m * p, target_rank) * 0.1
                W_t += rng.randn(p * n, target_rank) * 0.1
                
                # Try to solve using iterative refinement
                # The error tensor is: E = T - sum_r U_r ⊗ V_r ⊗ W_r
                # We want to minimize ||E||_F
                # 
                # Use a simple approach: fix two of the three matrices, solve for the third
                # This is a least squares problem.
                
                # Build the tensor T as a 3D array
                # T[(k,i), (i',j'), (j'',k'')] = delta_{i,i'} delta_{j',j''} delta_{k,k''}
                # In flattened form: T[k*n+i, i'*m+j', j''*p+k'']
                
                # For fixed U and V, the optimal W is:
                # W = argmin_W ||T - sum_r U_r ⊗ V_r ⊗ W_r||_F
                # This is a linear system: for each (k,i):
                # sum_r U_r[i'*m+j'] V_r[j''*p+k''] W_r[k*n+i] = T[k*n+i, i'*m+j', j''*p+k'']
                # 
                # Reshape: Let M be the matrix where M[(k,i), r] = W_r[k*n+i]
                # and let B be the matrix where B[(k,i), (i',j',j'',k'')] = T[...]
                # and let A be the matrix where A[r, (i',j',j'',k'')] = U_r[i'*m+j'] * V_r[j''*p+k'']
                # Then B = A * M, so M = pinv(A) * B
                # 
                # This is expensive. Let's use a simpler iterative method.
                
                # Alternating least squares style:
                # Fix V, W: solve for U (linear)
                # Fix U, W: solve for V (linear)
                # Fix U, V: solve for W (linear)
                
                U_cur = U_t.copy()
                V_cur = V_t.copy()
                W_cur = W_t.copy()
                
                converged = False
                for it in range(200):
                    if time.time() > deadline - 30:
                        break
                    
                    # Solve for W given U, V:
                    # For each output index (k,i):
                    # sum_r (U[:,r] @ a_vec) (V[:,r] @ b_vec) W[r, k*n+i] = T_{k*n+i}
                    # where a_vec, b_vec are the basis vectors
                    # 
                    # More directly: the (k*n+i)-th row of the "W equation" is:
                    # sum_r U_r[i*m+j] V_r[j*p+k] W_r[k*n+i] = delta_{j,j} (which is 1)
                    # Wait, T[k*n+i, i*m+j, j*p+k] = 1 for all i,j,k
                    # 
                    # So the equation for W is:
                    # For each (i,k): sum_{j} U[:,j_offset(i,j)] * V[:,j_offset(j,k)] * W[:,k*n+i] = 1
                    # where j_offset(i,j) = i*m+j in U, j*p+k in V
                    # 
                    # Let me define: for each (i,k), the equation is:
                    # sum_{j=0}^{m-1} U[i*m+j, :] * V[j*p+k, :] (elementwise) dot W[:, k*n+i] = 1
                    # 
                    # This is: (U_slice(i,:) * V_slice(k,:))^T @ W[:, k*n+i] = 1
                    # where U_slice(i,:) = U[i*m:(i+1)*m, :] and V_slice(k,:) = V[0:m*p:p, :]... 
                    # No, V_slice for fixed k is V[j*p+k] for j=0..m-1, which is not contiguous.
                    
                    # Let me just build the full system.
                    # Number of equations: n*p*m = 2*5*5 = 50 (one for each (i,k,j) where the tensor is nonzero)
                    # Actually, the tensor T has T[k*n+i, i*m+j, j*p+k] = 1 for all i,j,k.
                    # That's n*m*p = 50 nonzero entries.
                    # Each gives one equation: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                    # 
                    # Unknowns: target_rank * p * n = target_rank * 10
                    # For target_rank=9: 90 unknowns, 50 equations. Underdetermined!
                    # 
                    # So we have freedom. Let's set up the system properly.
                    
                    # Build matrix A_w of shape (50, target_rank * p * n)
                    # and vector b_w of shape (50,)
                    # Equation index: (i, j, k) -> eq_idx
                    # Unknown index: (r, k_out, i_out) -> unknown_idx
                    # 
                    # A_w[eq_idx, unknown_idx] = U[i*m+j, r] * V[j*p+k, r] if (k_out=k, i_out=i) else 0
                    
                    num_eq = n * m * p  # 50
                    num_unk_w = target_rank * p * n  # target_rank * 10
                    
                    A_w = np.zeros((num_eq, num_unk_w))
                    b_w = np.zeros(num_eq)
                    
                    eq_idx = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                for r in range(target_rank):
                                    unk_idx = r * p * n + k * n + i
                                    A_w[eq_idx, unk_idx] = U_cur[i * m + j, r] * V_cur[j * p + k, r]
                                b_w[eq_idx] = 1.0
                                eq_idx += 1
                    
                    # Solve: W_flat = pinv(A_w) @ b_w
                    W_flat, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                    W_cur = W_flat.reshape(target_rank, p * n).T  # shape (p*n, target_rank)
                    
                    # Solve for V given U, W:
                    # Equation: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                    # Unknowns: V[j*p+k, r] for all j,k,r
                    # A_v[eq_idx, unk_idx] = U[i*m+j, r] * W[k*n+i, r] if unk_idx = (j,k,r)
                    
                    num_unk_v = target_rank * m * p
                    A_v = np.zeros((num_eq, num_unk_v))
                    b_v = np.zeros(num_eq)
                    
                    eq_idx = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                for r in range(target_rank):
                                    unk_idx = r * m * p + j * p + k
                                    A_v[eq_idx, unk_idx] = U_cur[i * m + j, r] * W_cur[k * n + i, r]
                                b_v[eq_idx] = 1.0
                                eq_idx += 1
                    
                    V_flat, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                    V_cur = V_flat.reshape(target_rank, m * p).T  # shape (m*p, target_rank)
                    
                    # Solve for U given V, W:
                    num_unk_u = target_rank * n * m
                    A_u = np.zeros((num_eq, num_unk_u))
                    b_u = np.zeros(num_eq)
                    
                    eq_idx = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                for r in range(target_rank):
                                    unk_idx = r * n * m + i * m + j
                                    A_u[eq_idx, unk_idx] = V_cur[j * p + k, r] * W_cur[k * n + i, r]
                                b_u[eq_idx] = 1.0
                                eq_idx += 1
                    
                    U_flat, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                    U_cur = U_flat.reshape(target_rank, n * m).T  # shape (n*m, target_rank)
                    
                    # Check convergence
                    if it % 20 == 19:
                        if verify_fast(U_cur, V_cur, W_cur, n, m, p, num_trials=10, seed=it):
                            success = True
                            break
                    
                    if time.time() > deadline - 30:
                        break
                
                if success:
                    best_U, best_V, best_W = U_cur, V_cur, W_cur
                    best_rank = target_rank
                    break
            
            if best_rank < 10:
                break
    
    # Final verification
    if not verify_fast(best_U, best_V, best_W, n, m, p, num_trials=30, seed=999):
        # Fall back to the known-good rank-10
        best_U, best_V, best_W = U_10, V_10, W_10
        best_rank = 10
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
