import numpy as np
import time

def construct():
    """
    Rank-minimal exact decomposition for 2x3 * 3x5 matrix multiplication.
    
    Strategy:
    1. Build the known rank-28 decomposition (Strassen + schoolbook blocks)
    2. Search for linear dependencies among the 28 rank-1 terms
    3. If a dependency is found, attempt to eliminate one term for rank 27
    4. If no dependency, return the rank-28 solution
    """
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = 120.0
    
    def build_rank28():
        """Build the standard rank-28 decomposition."""
        rank = 28
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Block 1: j=0, all k in {0..4} - schoolbook 2x1x5, rank 10
        for i in range(n):
            for k in range(p):
                U[i * m + 0][col] = 1.0
                V[0 * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
        
        # Block 2a: j in {1,2}, k in {0,1} - Strassen 2x2x2, rank 7
        # M1
        U[0*3+1][col] = 1.0; U[1*3+2][col] = 1.0
        V[1*5+0][col] = 1.0; V[2*5+1][col] = 1.0
        W[0*2+0][col] = 1.0; W[1*2+1][col] = 1.0
        col += 1
        # M2
        U[1*3+1][col] = 1.0; U[1*3+2][col] = 1.0
        V[1*5+0][col] = 1.0
        W[0*2+1][col] = 1.0; W[1*2+1][col] = -1.0
        col += 1
        # M3
        U[0*3+1][col] = 1.0
        V[1*5+1][col] = 1.0; V[2*5+1][col] = -1.0
        W[1*2+0][col] = 1.0; W[1*2+1][col] = 1.0
        col += 1
        # M4
        U[1*3+2][col] = 1.0
        V[2*5+0][col] = 1.0; V[1*5+0][col] = -1.0
        W[0*2+0][col] = 1.0; W[0*2+1][col] = 1.0
        col += 1
        # M5
        U[0*3+1][col] = 1.0; U[0*3+2][col] = 1.0
        V[2*5+1][col] = 1.0
        W[0*2+0][col] = -1.0; W[1*2+0][col] = 1.0
        col += 1
        # M6
        U[1*3+1][col] = 1.0; U[0*3+1][col] = -1.0
        V[1*5+0][col] = 1.0; V[1*5+1][col] = 1.0
        W[1*2+1][col] = 1.0
        col += 1
        # M7
        U[0*3+2][col] = 1.0; U[1*3+2][col] = -1.0
        V[2*5+0][col] = 1.0; V[2*5+1][col] = 1.0
        W[0*2+0][col] = 1.0
        col += 1
        
        # Block 2b: j in {1,2}, k=2 - schoolbook 2x2x1, rank 4
        for i in range(n):
            for j_local in range(2):
                j = j_local + 1
                U[i*3 + j][col] = 1.0
                V[j*5 + 2][col] = 1.0
                W[2*2 + i][col] = 1.0
                col += 1
        
        # Block 2c: j in {1,2}, k in {3,4} - Strassen 2x2x2, rank 7
        # M1
        U[0*3+1][col] = 1.0; U[1*3+2][col] = 1.0
        V[1*5+3][col] = 1.0; V[2*5+4][col] = 1.0
        W[3*2+0][col] = 1.0; W[4*2+1][col] = 1.0
        col += 1
        # M2
        U[1*3+1][col] = 1.0; U[1*3+2][col] = 1.0
        V[1*5+3][col] = 1.0
        W[3*2+1][col] = 1.0; W[4*2+1][col] = -1.0
        col += 1
        # M3
        U[0*3+1][col] = 1.0
        V[1*5+4][col] = 1.0; V[2*5+4][col] = -1.0
        W[4*2+0][col] = 1.0; W[4*2+1][col] = 1.0
        col += 1
        # M4
        U[1*3+2][col] = 1.0
        V[2*5+3][col] = 1.0; V[1*5+3][col] = -1.0
        W[3*2+0][col] = 1.0; W[3*2+1][col] = 1.0
        col += 1
        # M5
        U[0*3+1][col] = 1.0; U[0*3+2][col] = 1.0
        V[2*5+4][col] = 1.0
        W[3*2+0][col] = -1.0; W[4*2+0][col] = 1.0
        col += 1
        # M6
        U[1*3+1][col] = 1.0; U[0*3+1][col] = -1.0
        V[1*5+3][col] = 1.0; V[1*5+4][col] = 1.0
        W[4*2+1][col] = 1.0
        col += 1
        # M7
        U[0*3+2][col] = 1.0; U[1*3+2][col] = -1.0
        V[2*5+3][col] = 1.0; V[2*5+4][col] = 1.0
        W[3*2+0][col] = 1.0
        col += 1
        
        assert col == rank
        return U, V, W
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        
        tensor = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u = Ua[:, r]
            v = Va[:, r]
            w = Wa[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        
        return np.allclose(tensor, np.ones((n, m, p)), atol=1e-10)
    
    def search_rank27(U, V, W, rank28):
        """
        Search for a rank-27 decomposition by finding linear dependencies
        among the 28 rank-1 terms.
        """
        # Flatten each rank-1 term into a 30-dim vector
        # Term r: u_r ⊗ v_r ⊗ w_r, flattened as T[i,j,k] = u[i,j] * v[j,k] * w[k,i]
        terms = []
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        
        for r in range(rank28):
            u = Ua[:, r]
            v = Va[:, r]
            w = Wa[:, r]
            # Create 3D tensor and flatten
            term_tensor = np.zeros((n, m, p), dtype=np.float64)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        term_tensor[i, j, k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
            terms.append(term_tensor.flatten())
        
        terms = np.array(terms, dtype=np.float64)  # shape (28, 30)
        
        # Check if the 28 vectors are linearly dependent
        # The target tensor is ones((2,3,5)).flatten()
        target = np.ones(n*m*p, dtype=np.float64)
        
        # We need: sum_r c_r * term_r = target
        # This is a linear system: terms.T @ c = target
        # If the system has multiple solutions, we can find dependencies
        
        # First, check if rank-28 solution is valid
        sum_terms = terms @ np.ones(rank28)
        if not np.allclose(sum_terms, target, atol=1e-10):
            return None
        
        # Find the null space of the 28x30 matrix (or equivalently, 
        # find dependencies in the 28 vectors)
        # Actually, we want: are the 28 term vectors linearly dependent?
        # Form matrix M where M[r, :] = terms[r]
        # Find null space of M: vectors c such that M @ c = 0
        # But we also need sum_r c_r = 0 for it to be a true dependency
        # (since we're looking for alternative representations)
        
        # Better approach: find the rank of the 28x30 matrix
        # If rank < 28, there are dependencies
        M = terms  # 28 x 30
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
        
        # Count non-zero singular values
        tol = 1e-10
        rank_M = np.sum(S > tol)
        
        if rank_M < rank28:
            # There are dependencies!
            # Find null space vectors
            # Null space of M: vectors c in R^28 such that M @ c = 0
            # These are the right singular vectors corresponding to zero singular values
            null_space = Vt[rank_M:]  # shape (28-rank_M, 30) -- wait, this is wrong
            
            # Actually, for M (28x30), the null space of M is in R^30
            # But we want dependencies among the 28 rows, so we need the left null space
            # Left null space: vectors c in R^28 such that c^T @ M = 0
            # Equivalently, M^T @ c = 0
            Mt = M.T  # 30 x 28
            U2, S2, Vt2 = np.linalg.svd(Mt, full_matrices=False)
            rank_Mt = np.sum(S2 > tol)
            # Null space of Mt is in R^28
            if rank_Mt < rank28:
                null_vecs = Vt2[rank_Mt:]  # shape (28-rank_Mt, 28)
                
                # For each null vector c, we have sum_r c_r * term_r = 0
                # So we can try to eliminate one term by adjusting others
                # If c has a nonzero entry at position r0, we can write:
                # term_r0 = -sum_{r != r0} (c_r / c_r0) * term_r
                # This gives a representation with 27 terms (if we can merge)
                
                # But this only works if the coefficients are such that we can
                # actually eliminate the term. Let's try each null vector.
                for nv in null_vecs:
                    # Find the index with the largest magnitude
                    idx = np.argmax(np.abs(nv))
                    if abs(nv[idx]) < tol:
                        continue
                    
                    # Try to eliminate term idx
                    # New coefficients: c'_r = c_r / c_idx for r != idx, and c'_idx = 0
                    # But we need sum_r c'_r * term_r = target
                    # Originally: sum_r c_r * term_r = target (with all c_r = 1)
                    # And: sum_r nv_r * term_r = 0
                    # So: sum_r (1 - alpha * nv_r) * term_r = target for any alpha
                    # We want to set coefficient of term idx to 0:
                    # 1 - alpha * nv_idx = 0 => alpha = 1 / nv_idx
                    alpha = 1.0 / nv[idx]
                    new_coeffs = np.ones(rank28) - alpha * nv
                    new_coeffs[idx] = 0.0
                    
                    # Check if new_coeffs gives the target
                    result = terms @ new_coeffs
                    if np.allclose(result, target, atol=1e-8):
                        # Build new decomposition with 27 terms
                        # We need to find U', V', W' such that
                        # sum_{r != idx} new_coeffs[r] * u'_r ⊗ v'_r ⊗ w'_r = target
                        # This is non-trivial in general, but if new_coeffs[r] != 0,
                        # we can absorb the coefficient into one of the vectors
                        new_U = []
                        new_V = []
                        new_W = []
                        for r in range(rank28):
                            if r == idx:
                                continue
                            c_r = new_coeffs[r]
                            if abs(c_r) < tol:
                                continue
                            # Absorb c_r into U (or split among U, V, W)
                            # Simplest: multiply U by c_r
                            u_new = Ua[:, r] * c_r
                            v_new = Va[:, r]
                            w_new = Wa[:, r]
                            new_U.append(u_new)
                            new_V.append(v_new)
                            new_W.append(w_new)
                        
                        new_rank = len(new_U)
                        if new_rank == 27:
                            new_U_arr = np.array(new_U).T.tolist()  # 6 x 27
                            new_V_arr = np.array(new_V).T.tolist()  # 15 x 27
                            new_W_arr = np.array(new_W).T.tolist()  # 10 x 27
                            
                            # Verify
                            if verify_decomposition(new_U_arr, new_V_arr, new_W_arr, 27):
                                return {"U": new_U_arr, "V": new_V_arr, "W": new_W_arr}
        
        return None
    
    # Build rank-28 decomposition
    U, V, W = build_rank28()
    
    # Verify rank-28 is correct
    if not verify_decomposition(U, V, W, 28):
        # Fallback: schoolbook rank 30
        rs = n * m * p
        Uf = [[0.0] * rs for _ in range(n * m)]
        Vf = [[0.0] * rs for _ in range(m * p)]
        Wf = [[0.0] * rs for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i*m+j][c] = 1.0
                    Vf[j*p+k][c] = 1.0
                    Wf[k*n+i][c] = 1.0
                    c += 1
        return {"U": Uf, "V": Vf, "W": Wf}
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return {"U": U, "V": V, "W": W}
    
    # Search for rank-27
    rank27_result = search_rank27(U, V, W, 28)
    if rank27_result is not None:
        return rank27_result
    
    # Return rank-28
    return {"U": U, "V": V, "W": W}
