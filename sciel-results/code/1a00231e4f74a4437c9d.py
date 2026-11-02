import time
import random
import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = 2, 2, 7
    deadline = time.time() + 115  # Safety margin

    def verify_decomp(U, V, W, trials=30):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rank = len(U[0]) if isinstance(U, list) else U.shape[1]
        for _ in range(trials):
            A = np.array([[random.randint(-10, 10) for _ in range(m)] for _ in range(n)])
            B = np.array([[random.randint(-10, 10) for _ in range(p)] for _ in range(m)])
            C_true = A @ B

            Ua = np.array(U) if isinstance(U, list) else U
            Va = np.array(V) if isinstance(V, list) else V
            Wa = np.array(W) if isinstance(W, list) else W

            C_calc = np.zeros((n, p))
            for r in range(rank):
                u_val = sum(Ua[i*m+j, r] * A[i, j] for i in range(n) for j in range(m))
                v_val = sum(Va[j*p+k, r] * B[j, k] for j in range(m) for k in range(p))
                prod = u_val * v_val
                for i in range(n):
                    for k in range(p):
                        C_calc[i, k] += Wa[k*n+i, r] * prod

            if not np.allclose(C_true, C_calc, atol=1e-9):
                return False
        return True

    def build_rank25():
        """Build the known rank-25 Strassen-based decomposition for 2x2 * 2x7."""
        rank = 25
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        def add_strassen(col_start, k_start):
            # Strassen for 2x2 * 2x2 block starting at column k_start
            # M1 = (A00+A11)*(B0,ks + B1,ks+1)
            U[0, col_start+0] += 1; U[3, col_start+0] += 1
            V[7*0+k_start, col_start+0] += 1; V[7*1+k_start+1, col_start+0] += 1
            W[k_start*n+0, col_start+0] += 1; W[(k_start+1)*n+1, col_start+0] += 1
            
            # M2 = (A10+A11)*B0,ks
            U[2, col_start+1] += 1; U[3, col_start+1] += 1
            V[7*0+k_start, col_start+1] += 1
            W[k_start*n+1, col_start+1] += 1; W[(k_start+1)*n+1, col_start+1] -= 1
            
            # M3 = A00*(B0,ks+1 - B1,ks+1)
            U[0, col_start+2] += 1
            V[7*0+k_start+1, col_start+2] += 1; V[7*1+k_start+1, col_start+2] -= 1
            W[(k_start+1)*n+0, col_start+2] += 1; W[(k_start+1)*n+1, col_start+2] += 1
            
            # M4 = A11*(B1,ks - B0,ks)
            U[3, col_start+3] += 1
            V[7*1+k_start, col_start+3] += 1; V[7*0+k_start, col_start+3] -= 1
            W[k_start*n+0, col_start+3] += 1; W[k_start*n+1, col_start+3] += 1
            
            # M5 = (A00+A01)*B1,ks+1
            U[0, col_start+4] += 1; U[1, col_start+4] += 1
            V[7*1+k_start+1, col_start+4] += 1
            W[k_start*n+0, col_start+4] -= 1; W[(k_start+1)*n+0, col_start+4] += 1
            
            # M6 = (A10-A00)*(B0,ks+B0,ks+1)
            U[2, col_start+5] += 1; U[0, col_start+5] -= 1
            V[7*0+k_start, col_start+5] += 1; V[7*0+k_start+1, col_start+5] += 1
            W[(k_start+1)*n+1, col_start+5] += 1
            
            # M7 = (A01-A11)*(B1,ks+B1,ks+1)
            U[1, col_start+6] += 1; U[3, col_start+6] -= 1
            V[7*1+k_start, col_start+6] += 1; V[7*1+k_start+1, col_start+6] += 1
            W[k_start*n+0, col_start+6] += 1

        add_strassen(0, 0)
        add_strassen(7, 2)
        add_strassen(14, 4)
        
        # Schoolbook for column 6
        cs = 21
        U[0, cs+0] += 1; V[0*7+6, cs+0] += 1; W[6*n+0, cs+0] += 1
        U[1, cs+1] += 1; V[1*7+6, cs+1] += 1; W[6*n+0, cs+1] += 1
        U[2, cs+2] += 1; V[0*7+6, cs+2] += 1; W[6*n+1, cs+2] += 1
        U[3, cs+3] += 1; V[1*7+6, cs+3] += 1; W[6*n+1, cs+3] += 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def search_rank24():
        """
        Search for a rank-24 decomposition by trying to merge the 4 schoolbook terms
        for column 6 with the Strassen terms for columns 4,5 (the last Strassen block).
        
        The idea: the last Strassen block handles columns 4,5 with 7 terms. The schoolbook
        handles column 6 with 4 terms. Total = 7+4 = 11 for the last 3 columns.
        We try to find a 2x2x3 tensor decomposition of rank 10 instead of 11.
        
        Actually, let's try a different approach: use a rank-10 decomposition for 2x2x3
        if one exists, or try to find a rank-24 by merging across all blocks.
        
        For now, try a systematic search: take the rank-25 solution and try to find
        linear dependencies among the 25 terms to reduce rank by 1.
        """
        # Get the rank-25 solution
        sol25 = build_rank25()
        U25 = np.array(sol25["U"], dtype=float)
        V25 = np.array(sol25["V"], dtype=float)
        W25 = np.array(sol25["W"], dtype=float)
        
        rank25 = U25.shape[1]
        
        # Try to find a rank-24 by checking if any column is a linear combination of others
        # in the tensor space. This is equivalent to checking if the 25 rank-1 tensors
        # are linearly dependent.
        
        # The tensor T is in R^(4 x 14 x 14). Each rank-1 term is u_r ⊗ v_r ⊗ w_r.
        # We check if there exist coefficients c_r (not all zero) such that
        # sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0
        
        # This is a system of 4*14*14 = 784 equations in 25 unknowns.
        # Build the matrix where each column is the vectorized rank-1 tensor.
        
        tensor_dim = (n*m) * (m*p) * (p*n)
        M = np.zeros((tensor_dim, rank25))
        
        for r in range(rank25):
            # Vectorize u_r ⊗ v_r ⊗ w_r
            # u_r is (4,), v_r is (14,), w_r is (14,)
            # The tensor element T[i, j, k] = u[i] * v[j] * w[k]
            # But we need to be careful about the indexing.
            # U row index: i*m + j where i is A row, j is A col
            # V row index: j*p + k where j is B row, k is B col
            # W row index: k*n + i where k is C col, i is C row
            
            # The multiplication tensor T has T[(i,j), (j,k), (i,k)] = 1
            # In our flattened indexing:
            # U index: i*m + j
            # V index: j*p + k
            # W index: k*n + i
            
            # So the rank-1 term contributes to T[a, b, c] where:
            # a = i*m + j, b = j'*p + k, c = k'*n + i'
            # For it to be a valid term, we need j = j' and i = i'
            # So T[a, b, c] = u[a] * v[b] * w[c] if the indices are consistent
            
            # Actually, the tensor decomposition is:
            # T = sum_r u_r ⊗ v_r ⊗ w_r
            # where T is the 3-way tensor with T[i, j, k] representing
            # the coefficient for A[i,j] * B[j,k] * C[i,k]
            
            # In our flattened form:
            # U has rows indexed by (i,j) -> idx = i*m + j
            # V has rows indexed by (j,k) -> idx = j*p + k
            # W has rows indexed by (k,i) -> idx = k*n + i
            
            # The tensor T has shape (4, 14, 14) where:
            # T[idx_u, idx_v, idx_w] = 1 if idx_u=(i,j), idx_v=(j,k), idx_w=(k,i)
            
            # For the rank-1 term r, it contributes u_r[idx_u] * v_r[idx_v] * w_r[idx_w]
            # to T[idx_u, idx_v, idx_w]
            
            # So the vectorized form is the Kronecker product of u_r, v_r, w_r
            # But we need to order them correctly.
            
            vec = np.kron(np.kron(U25[:, r], V25[:, r]), W25[:, r])
            M[:, r] = vec
        
        # Now check the rank of M
        # If rank(M) < 25, then there's a linear dependency, and we can reduce the rank.
        
        try:
            rank_M = np.linalg.matrix_rank(M, tol=1e-10)
        except:
            rank_M = rank25
        
        if rank_M < rank25:
            # There's a linear dependency. Find the null space.
            try:
                _, s, vh = np.linalg.svd(M, full_matrices=False)
                # The null space is the last (rank25 - rank_M) rows of vh
                null_dim = rank25 - rank_M
                null_space = vh[rank_M:, :]
                
                # Find the coefficient vector with the largest norm component
                # to identify which term to eliminate
                coeffs = null_space[0]
                
                # Find the index of the largest coefficient
                elim_idx = np.argmax(np.abs(coeffs))
                
                # Remove this term and scale the others
                remaining = [i for i in range(rank25) if i != elim_idx]
                
                U24 = np.zeros((n*m, rank25 - 1))
                V24 = np.zeros((m*p, rank25 - 1))
                W24 = np.zeros((p*n, rank25 - 1))
                
                for new_idx, old_idx in enumerate(remaining):
                    U24[:, new_idx] = U25[:, old_idx]
                    V24[:, new_idx] = V25[:, old_idx]
                    W24[:, new_idx] = W25[:, old_idx]
                
                result = {"U": U24.tolist(), "V": V24.tolist(), "W": W24.tolist()}
                
                if verify_decomp(result["U"], result["V"], result["W"], trials=20):
                    return result
            except:
                pass
        
        return None

    def build_rank24_alt():
        """
        Alternative rank-24 construction using a different block decomposition.
        
        Try: Strassen for columns 0,1 (rank 7), Strassen for columns 2,3 (rank 7),
        and a custom rank-10 decomposition for columns 4,5,6.
        
        The 2x2x3 tensor (2x2 matrix times 2x3 matrix) has a known minimal rank.
        For 2x2x3, the schoolbook rank is 6. Let's check if we can do better.
        
        Actually, let's try a different split:
        - Columns 0,1,2: use a rank-10 decomposition for 2x2x3
        - Columns 3,4,5: use a rank-10 decomposition for 2x2x3
        - Column 6: schoolbook rank 4
        Total: 10 + 10 + 4 = 24
        
        But does a rank-10 decomposition exist for 2x2x3? The schoolbook is rank 6.
        So this doesn't help.
        
        Let's try:
        - Columns 0,1: Strassen rank 7
        - Columns 2,3,4,5,6: custom decomposition
        
        The 2x2x5 tensor has schoolbook rank 10. Can we do better?
        
        Actually, let's just try the linear dependency search first.
        """
        return None

    # First, try to build the rank-25 solution
    result = build_rank25()
    if verify_decomp(result["U"], result["V"], result["W"], trials=20):
        # Now try to reduce to rank 24
        if time.time() < deadline - 30:
            r24 = search_rank24()
            if r24 is not None:
                return r24
        
        return result

    # Fallback: schoolbook rank 28
    rank_sb = 28
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
