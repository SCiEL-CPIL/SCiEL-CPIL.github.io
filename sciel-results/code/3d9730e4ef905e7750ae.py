"""Program for matrix multiplication tensor decomposition 3x4x7.
Uses a greedy rank-reduction approach starting from schoolbook (rank 84)
and attempting to reduce the rank by removing columns that can be
represented as linear combinations of others.
"""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]
n, m, p = DIMENSIONS

def construct():
    deadline = time.time() + 110  # safety margin before 150s limit
    
    # Schoolbook decomposition: rank = n*m*p = 84
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    current_rank = rank_school
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    # Greedy rank reduction: try to remove columns
    # A column c can be removed if its "fiber" can be represented
    # by a linear combination of other columns' fibers.
    # For a rank-r decomposition, removing column c means:
    #   U[:, c] V[:, c]^T W[:, c]^T = sum_{c' != c} alpha_{c'} U[:, c'] V[:, c']^T W[:, c']^T
    # This is a 4-way condition. A simpler sufficient condition:
    # Check if column c is linearly dependent in the Kronecker product space.
    # 
    # Alternative: use the fact that if the tensor is unchanged when we
    # zero out column c and adjust other columns, we can remove it.
    # 
    # Practical approach: try small ranks and verify.
    
    # Verify function
    def verify(U_mat, V_mat, W_mat, tol=1e-10):
        r = U_mat.shape[1]
        # Compute C = A*B for random A, B and compare with decomposition
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, that's not right. Let me re-derive.
        # Standard: C[i,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] is wrong.
        # 
        # The decomposition is: C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_r (sum_j A[i,j] U[j,i,r]) * V[?,r] * W[?,r]
        # 
        # Actually the standard bilinear form is:
        # C[i,k] = sum_r U[i,j,r] * A[i,j] ... no.
        #
        # Let me use the trace convention from the task:
        # W[k*n + i, r]
        # The decomposition satisfies:
        # C[i,k] = sum_r sum_j A[i,j] B[j,k] U[j,i,r] V[?,r] W[?,r]
        #
        # From the schoolbook code:
        # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
        # So: C[i,k] = sum_c A[?] ... 
        #
        # The correct verification:
        # For the decomposition to be valid:
        # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
        # No, that's not right either because U and V share index j.
        #
        # The proper way: the tensor M_{n,m,p} has entries M[i,j,(k,l)] = delta_{i,l} 
        # where we view the output as indexed by (k,l). The rank-r decomposition is:
        # M[i,j,(k,l)] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r]
        #
        # For matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
        # The bilinear form is: C[i,k] = sum_r (sum_j A[i,j] U[i*m+j,r]) * ... 
        # 
        # Actually, let's just verify by computing C = A @ B and comparing
        # with the reconstruction from the factors.
        #
        # The reconstruction formula given the factor shapes:
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # C[i,k] should equal sum_r sum_j A[i,j] B[j,k] * [some expression]
        #
        # From schoolbook: U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1
        # Then: sum_r A[i,j] B[j,k] U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] 
        #     = sum_r A[i,j] B[j,k] * 1 * 1 * 1 (for the specific r where all are 1)
        #     = A[i,j] B[j,k] (for that r)
        # Summing over j and r: sum_j A[i,j] B[j,k] = C[i,k]. Correct.
        #
        # So the verification is:
        # C_recon[i,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * ... no.
        #
        # It's: C_recon[i,k] = sum_j sum_r A[i,j] B[j,k] U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        C_recon = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    for r in range(r):
                        val += A[i, j] * B[j, k] * U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                C_recon[i, k] = val
        
        return np.allclose(C_true, C_recon, atol=tol)
    
    # Try to reduce rank by checking if we can zero out a column
    # and still maintain the decomposition (with the remaining columns adjusted)
    
    # Simpler approach: just verify schoolbook and return it
    # The greedy approach is complex; let's try a different strategy.
    
    # Try rank 83, 82, etc. by checking if the schoolbook tensor has
    # any linearly dependent "terms"
    
    # Actually, let's try a smarter approach: use the fact that for
    # M_{3,4,7}, the known upper bound is 32 (from laser method) and
    # lower bound is around 27-28. Let's just try to find a valid
    # decomposition with rank < 84 using optimization.
    
    # Given time constraints, let's try a simple greedy removal:
    # For each column, check if removing it and solving for the rest
    # gives a valid decomposition.
    
    best_U = U_cur.copy()
    best_V = V_cur.copy()
    best_W = W_cur.copy()
    best_rank = current_rank
    
    # Heuristic: Try to remove column c and adjust column c' to compensate.
    # This reduces the rank by 1 if the residual is small enough.
    
    # Precompute the tensor for the schoolbook decomposition
    # T = sum_l U_l o V_l o W_l
    # We need to solve for delta in column c' such that:
    # (U_{c'} + dU) o (V_{c'} + dV) o (W_{c'} + dW) - U_{c'} o V_{c'} o W_{c'} approx U_c o V_c o W_c
    # Linearizing:
    # dU o V_{c'} o W_{c'} + U_{c'} o dV o W_{c'} + U_{c'} o V_{c'} o dW approx U_c o V_c o W_c
    
    best_rank = current_rank
    best_U = U_cur.copy()
    best_V = V_cur.copy()
    best_W = W_cur.copy()
    
    # To speed up, we might not try all pairs. But 7000 pairs is manageable if fast.
    # Let's try all pairs.
    
    for c in range(current_rank):
        if time.time() > deadline - 10:
            break
            
        u_c = U_cur[:, c]
        v_c = V_cur[:, c]
        w_c = W_cur[:, c]
        
        target = np.einsum('a,b,c->abc', u_c, v_c, w_c).flatten()
        
        for cp in range(current_rank):
            if cp == c:
                continue
            
            u_cp = U_cur[:, cp]
            v_cp = V_cur[:, cp]
            w_cp = W_cur[:, cp]
            
            # Construct J for the 3 vectors dU, dV, dW
            # J is 7056 x 61
            J = np.zeros((n * m * m * p * p * n, n * m + m * p + p * n))
            
            # Block 1: dU
            # Term: dU[a] * v_cp[b] * w_cp[c]
            # For each a, the row block is v_cp * w_cp
            for a in range(n * m):
                start_row = a * (m * p) * (p * n)
                end_row = start_row + (m * p) * (p * n)
                J[start_row:end_row, a] = np.outer(v_cp, w_cp).flatten()
            
            # Block 2: dV
            # Term: u_cp[a] * dV[b] * w_cp[c]
            # For each b, the contribution is to rows (a, b, c)
            # We need to scatter u_cp * w_cp into the rows for index b
            for b in range(m * p):
                # Indices for fixed b: a * (mp*pn) + b * pn + c
                # We can generate these indices
                # a in 0..nm-1, c in 0..pn-1
                # This is a bit slow in Python loop.
                # Let's use a vectorized approach if possible.
                # Or just accept the loop cost. 28 * 12 * 21 = 7056 ops.
                # 7000 pairs * 7056 ops = 50M ops. Might be ~1-2 seconds in Python?
                # Let's try.
                
                vec = np.outer(u_cp, w_cp).flatten()
                # Indices:
                # a_idx = np.arange(n*m)
                # c_idx = np.arange(p*n)
                # rows = a_idx[:, None] * (m*p)*(p*n) + b*(p*n) + c_idx[None, :]
                # J[rows.flatten(), n*m + b] = vec
                
                # Optimization: Precompute the index offsets for each b?
                # The offset for b is b * p * n.
                # The stride for a is (m*p)*(p*n).
                
                # We can fill the column J[:, n*m + b] directly by iterating a, c
                col_idx = n * m + b
                for a in range(n * m):
                    base = a * (m * p) * (p * n) + b * (p * n)
                    J[base : base + p * n, col_idx] = u_cp[a] * w_cp
            
            # Block 3: dW
            # Term: u_cp[a] * v_cp[b] * dW[c]
            # For each c, the contribution is to rows (a, b, c)
            for c_idx in range(p * n):
                col_idx = n * m + m * p + c_idx
                # Rows for fixed c_idx: a * (mp*pn) + b * pn + c_idx
                # We can iterate a, b
                for a in range(n * m):
                    for b in range(m * p):
                        row = a * (m * p) * (p * n) + b * (p * n) + c_idx
                        J[row, col_idx] = u_cp[a] * v_cp[b]
            
            # Solve J x = target
            try:
                x, residuals, rank, sv = np.linalg.lstsq(J, target, rcond=None)
                # Check residual
                res = np.linalg.norm(J @ x - target)
                if res < 1e-9:
                    # Success! Update the matrices
                    dU = x[:n*m]
                    dV = x[n*m:n*m+m*p]
                    dW = x[n*m+m*p:]
                    
                    U_new = U_cur.copy()
                    V_new = V_cur.copy()
                    W_new = W_cur.copy()
                    
                    U_new[:, c] = 0
                    V_new[:, c] = 0
                    W_new[:, c] = 0
                    
                    U_new[:, cp] += dU
                    V_new[:, cp] += dV
                    W_new[:, cp] += dW
                    
                    # Verify
                    if verify(U_new, V_new, W_new):
                        return {
                            "U": U_new.tolist(),
                            "V": V_new.tolist(),
                            "W": W_new.tolist()
                        }
            except Exception:
                pass

    # Return schoolbook decomposition if no improvement found
    return {
        "U": U_cur.tolist(),
        "V": V_cur.tolist(),
        "W": W_cur.tolist()
    }
