import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = 2, 3, 3
    start_time = time.time()
    deadline = start_time + 140.0

    def to_list(mat):
        return mat.tolist()

    def verify_decomposition(U, V, W, n, m, p):
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        rank = U.shape[1]
        
        if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
            return False
        
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_target = k * n + i
                    
                    contrib = np.zeros(p*n)
                    for r in range(rank):
                        uv = U[u_idx, r] * V[v_idx, r]
                        if uv != 0:
                            contrib += uv * W[:, r]
                    
                    if abs(contrib[w_target] - 1.0) > 1e-9:
                        return False
                    for ww in range(p*n):
                        if ww != w_target and abs(contrib[ww]) > 1e-9:
                            return False
        return True

    # Build the rank-17 solution using Strassen for 2x2x2 core + boundary terms
    # Strassen's 7 terms for the 2x2x2 subproblem (i in {0,1}, j in {0,1}, k in {0,1})
    # U indices: i*3+j, V indices: j*3+k, W indices: k*2+i
    
    R = 17
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Strassen terms for 2x2x2 core (j in {0,1}, k in {0,1})
    # M1 = (A10+A11)(B01-B11) -> contributes to C01, C00
    # M2 = A10(B01+B11) -> contributes to C11
    # M3 = A01(B10-B00) -> contributes to C00, C10  
    # M4 = (A00+A01)(B00-B10) -> contributes to C00, C10
    # M5 = (A00+A10)(B11-B01) -> contributes to C01, C11
    # M6 = (A01+A11)(B00-B10) -> contributes to C10, C11
    # M7 = (A00+A10)(B01-B11) -> contributes to C01, C11
    
    # Term 0: M1 = (A[0,0]+A[0,1])*(B[0,1]-B[1,1])
    U[0, 0] = 1; U[1, 0] = 1
    V[1, 0] = 1; V[4, 0] = -1
    W[1, 0] = 1; W[0, 0] = -1
    
    # Term 1: M2 = A[1,0]*(B[0,1]+B[1,1])
    U[3, 1] = 1
    V[1, 1] = 1; V[4, 1] = 1
    W[3, 1] = 1
    
    # Term 2: M3 = A[0,1]*(B[1,0]-B[0,0])
    U[1, 2] = 1
    V[3, 2] = 1; V[0, 2] = -1
    W[0, 2] = 1; W[2, 2] = -1
    
    # Term 3: M4 = (A[0,0]+A[0,1])*(B[0,0]-B[1,0])
    U[0, 3] = 1; U[1, 3] = 1
    V[0, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1; W[2, 3] = -1
    
    # Term 4: M5 = (A[0,0]+A[1,0])*(B[1,1]-B[0,1])
    U[0, 4] = 1; U[3, 4] = 1
    V[4, 4] = 1; V[1, 4] = -1
    W[1, 4] = 1; W[3, 4] = -1
    
    # Term 5: M6 = (A[0,1]+A[1,1])*(B[0,0]-B[1,0])
    U[1, 5] = 1; U[4, 5] = 1
    V[0, 5] = 1; V[3, 5] = -1
    W[0, 5] = -1; W[2, 5] = 1
    
    # Term 6: M7 = (A[0,0]+A[1,0])*(B[0,1]-B[1,1])
    U[0, 6] = 1; U[3, 6] = 1
    V[1, 6] = 1; V[4, 6] = -1
    W[1, 6] = -1; W[3, 6] = 1
    
    # Now handle the 10 remaining terms involving j=2 or k=2
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[0,0] += A[0,2]*B[2,0]
    # C[1,0] += A[1,2]*B[2,0]
    # C[0,1] += A[0,2]*B[2,1]
    # C[1,1] += A[1,2]*B[2,1]
    
    # Term 7: A[0,2]*B[2,0] -> C[0,0]
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # Term 8: A[1,2]*B[2,0] -> C[1,0]
    U[5, 8] = 1
    V[6, 8] = 1
    W[2, 8] = 1
    
    # Term 9: A[0,2]*B[2,1] -> C[0,1]
    U[2, 9] = 1
    V[7, 9] = 1
    W[1, 9] = 1
    
    # Term 10: A[1,2]*B[2,1] -> C[1,1]
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Term 11: A[0,0]*B[0,2] -> C[0,2]
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # Term 12: A[1,0]*B[0,2] -> C[1,2]
    U[3, 12] = 1
    V[2, 12] = 1
    W[5, 12] = 1
    
    # Term 13: A[0,1]*B[1,2] -> C[0,2]
    U[1, 13] = 1
    V[5, 13] = 1
    W[4, 13] = 1
    
    # Term 14: A[1,1]*B[1,2] -> C[1,2]
    U[4, 14] = 1
    V[5, 14] = 1
    W[5, 14] = 1
    
    # Term 15: A[0,2]*B[2,2] -> C[0,2]
    U[2, 15] = 1
    V[8, 15] = 1
    W[4, 15] = 1
    
    # Term 16: A[1,2]*B[2,2] -> C[1,2]
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    # Verify the rank-17 solution
    if verify_decomposition(U, V, W, n, m, p):
        # Try to find rank 16 by attempting to merge or eliminate terms
        # Check if any term is linearly dependent
        best_rank = 17
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Try removing each term and seeing if we can fix with small adjustments
        for term_to_remove in range(R):
            if time.time() > deadline - 10.0:
                break
            
            U_try = np.delete(U, term_to_remove, axis=1)
            V_try = np.delete(V, term_to_remove, axis=1)
            W_try = np.delete(W, term_to_remove, axis=1)
            
            # Now we have rank 16, check if it still works
            if verify_decomposition(U_try, V_try, W_try, n, m, p):
                return {"U": to_list(U_try), "V": to_list(V_try), "W": to_list(W_try)}
        
        # Try a more systematic rank-16 search
        # Use the structure: fix U and V patterns, solve for W
        for trial in range(3000):
            if time.time() > deadline - 5.0:
                break
            
            np.random.seed(trial * 7 + 123)
            R16 = 16
            
            # Start from rank-17 and try to perturb/merge
            U_base = U.copy()
            V_base = V.copy()
            
            # Try to make one column zero by adjusting
            # Or try random sparse patterns with Strassen-like structure
            
            # Keep Strassen core (7 terms) and try 9 terms for boundary
            U16 = np.zeros((n*m, R16))
            V16 = np.zeros((m*p, R16))
            
            # Copy Strassen terms
            U16[:, :7] = U[:, :7]
            V16[:, :7] = V[:, :7]
            
            # For the 9 boundary terms, try sparse random patterns
            for r in range(7, R16):
                # Each boundary term should involve A[*,2] or B[2,*] or B[*,2]
                # Try different sparse patterns
                n_u = np.random.choice([1, 2], p=[0.6, 0.4])
                n_v = np.random.choice([1, 2], p=[0.6, 0.4])
                
                # Bias toward boundary indices
                u_candidates = [2, 5]  # A[0,2], A[1,2]
                v_candidates = [6, 7, 8, 2, 5]  # B[2,0], B[2,1], B[2,2], B[0,2], B[1,2]
                
                if np.random.random() < 0.5:
                    # Pure boundary
                    u_idx = np.random.choice(u_candidates)
                    v_idx = np.random.choice(v_candidates)
                    U16[u_idx, r] = np.random.choice([-1, 1])
                    V16[v_idx, r] = np.random.choice([-1, 1])
                else:
                    # Mixed
                    u_idx = np.random.choice([0, 1, 2, 3, 4, 5])
                    v_idx = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8])
                    U16[u_idx, r] = np.random.choice([-1, 1])
                    V16[v_idx, r] = np.random.choice([-1, 1])
            
            # Solve for W
            A_mat = np.zeros((18, p*n*R16))
            b_vec = np.ones(18)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        for r in range(R16):
                            coeff = U16[u, r] * V16[v, r]
                            A_mat[eq_idx, w * R16 + r] = coeff
                        eq_idx += 1
            
            try:
                W_sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                W_sol = W_sol.reshape((p*n, R16))
                
                # Check if close to integer or half-integer
                W_int = np.round(W_sol * 2) / 2
                if np.allclose(W_sol, W_int, atol=1e-6):
                    if verify_decomposition(U16, V16, W_int, n, m, p):
                        return {"U": to_list(U16), "V": to_list(V16), "W": to_list(W_int)}
            except:
                pass
        
        # Return the verified rank-17 solution
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    else:
        # Fallback: return a simple rank-18 solution
        R18 = 18
        U18 = np.zeros((n*m, R18))
        V18 = np.zeros((m*p, R18))
        W18 = np.zeros((p*n, R18))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U18[i*m+j, idx] = 1
                    V18[j*p+k, idx] = 1
                    W18[k*n+i, idx] = 1
                    idx += 1
        
        return {"U": to_list(U18), "V": to_list(V18), "W": to_list(W18)}
