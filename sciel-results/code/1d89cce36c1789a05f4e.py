"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U,V,W satisfy the bilinear constraints."""
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = np.sum(W[k*n+i] * U[i*m+j1] * V[j2*p+k])
                        target = 1.0 if j1 == j2 else 0.0
                        max_err = max(max_err, abs(s - target))
        return max_err < tol
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p  # 144
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Try to use Strassen-like decomposition for better rank
    # Split p=9 into p1=4, p2=4, p3=1
    # For each (n,m,pi) sub-problem, try to find low-rank decomposition
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen)."""
        n2, m2, p2 = 2, 2, 2
        rank = 7
        U = np.zeros((n2*m2, rank))
        V = np.zeros((m2*p2, rank))
        W = np.zeros((p2*n2, rank))
        
        # Strassen's algorithm for 2x2 matrices:
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # C = A*B
        # 
        # 7 multiplications:
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b12-b22)
        # m4 = a22(b21-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        #
        # C11 = m1 - m4 + m6 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m3 + m2 + m5
        #
        # In our tensor notation:
        # U[i*m+j, c] corresponds to element (i,j) of A in factor c
        # V[j*p+k, c] corresponds to element (j,k) of B in factor c  
        # W[k*n+i, c] corresponds to element (k,i) of C in factor c
        #
        # For n=m=p=2:
        # U shape (4, 7): rows are (i,j) pairs: (0,0), (0,1), (1,0), (1,1)
        # V shape (4, 7): rows are (j,k) pairs: (0,0), (0,1), (1,0), (1,1)
        # W shape (4, 7): rows are (k,i) pairs: (0,0), (0,1), (1,0), (1,1)
        
        # Factor 1: m1 = (a11+a22)(b11+b22)
        # a11 is U[0,0], a22 is U[3,0]
        # b11 is V[0,0], b22 is V[3,0]
        # C11 = m1, C22 = m1
        U[0, 0] = 1  # a11
        U[3, 0] = 1  # a22
        V[0, 0] = 1  # b11
        V[3, 0] = 1  # b22
        W[0, 0] = 1  # C[0,0] = C11
        W[3, 0] = 1  # C[1,1] = C22
        
        # Factor 2: m2 = (a21+a22)b11
        # a21 is U[2,0], a22 is U[3,0]
        # b11 is V[0,0]
        # C21 = m2, C22 = m2
        U[2, 1] = 1  # a21
        U[3, 1] = 1  # a22
        V[0, 1] = 1  # b11
        W[2, 1] = 1  # C[1,0] = C21
        W[3, 1] = 1  # C[1,1] = C22
        
        # Factor 3: m3 = a11(b12-b22)
        # a11 is U[0,0]
        # b12 is V[1,0], b22 is V[3,0]
        # C12 = m3, C22 = -m3
        U[0, 2] = 1  # a11
        V[1, 2] = 1  # b12
        V[3, 2] = -1  # b22
        W[1, 2] = 1  # C[0,1] = C12
        W[3, 2] = -1  # C[1,1] = C22
        
        # Factor 4: m4 = a22(b21-b11)
        # a22 is U[3,0]
        # b21 is V[2,0], b11 is V[0,0]
        # C11 = -m4, C21 = m4
        U[3, 3] = 1  # a22
        V[2, 3] = 1  # b21
        V[0, 3] = -1  # b11
        W[0, 3] = -1  # C[0,0] = C11
        W[2, 3] = 1  # C[1,0] = C21
        
        # Factor 5: m5 = (a11+a12)b22
        # a11 is U[0,0], a12 is U[1,0]
        # b22 is V[3,0]
        # C12 = m5, C22 = m5
        U[0, 4] = 1  # a11
        U[1, 4] = 1  # a12
        V[3, 4] = 1  # b22
        W[1, 4] = 1  # C[0,1] = C12
        W[3, 4] = 1  # C[1,1] = C22
        
        # Factor 6: m6 = (a21-a11)(b11+b12)
        # a21 is U[2,0], a11 is U[0,0]
        # b11 is V[0,0], b12 is V[1,0]
        # C11 = m6
        U[2, 5] = 1  # a21
        U[0, 5] = -1  # a11
        V[0, 5] = 1  # b11
        V[1, 5] = 1  # b12
        W[0, 5] = 1  # C[0,0] = C11
        
        # Factor 7: m7 = (a12-a22)(b21+b22)
        # a12 is U[1,0], a22 is U[3,0]
        # b21 is V[2,0], b22 is V[3,0]
        # C11 = m7
        U[1, 6] = 1  # a12
        U[3, 6] = -1  # a22
        V[2, 6] = 1  # b21
        V[3, 6] = 1  # b22
        W[0, 6] = 1  # C[0,0] = C11
        
        return U, V, W
    
    def strassen_4x4x4():
        """Return U, V, W for 4x4x4 matrix multiplication.
        Use block structure with 2x2 Strassen at each level.
        This gives rank 49 for 4x4x4 (vs schoolbook 64).
        """
        n4, m4, p4 = 4, 4, 4
        # We'll use a hierarchical approach:
        # Split 4x4 into 2x2 blocks, apply Strassen to each
        # But we need to be careful with the tensor structure.
        
        # Alternative: use known rank-49 decomposition for 4x4x4
        # This is more complex. Let's use a simpler approach for now.
        
        # For simplicity, use schoolbook for 4x4x4 as fallback
        rank = n4 * m4 * p4
        U = np.zeros((n4*m4, rank))
        V = np.zeros((m4*p4, rank))
        W = np.zeros((p4*n4, rank))
        col = 0
        for i in range(n4):
            for j in range(m4):
                for k in range(p4):
                    U[i*m4+j, col] = 1
                    V[j*p4+k, col] = 1
                    W[k*n4+i, col] = 1
                    col += 1
        return U, V, W
    
    def schoolbook(n, m, p):
        """Return schoolbook decomposition for n x m times m x p."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # Try to find better decomposition using optimization
    def try_optimize(rank, max_iters=200, seed=42):
        """Try to find a rank-r decomposition using alternating least squares."""
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        for outer_iter in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Optimize U (fix V, W)
            for i in range(n):
                for j1 in range(m):
                    A_rows = []
                    b_vals = []
                    for k in range(p):
                        for j2 in range(m):
                            a_vec = V[j2*p+k] * W[k*n+i]
                            A_rows.append(a_vec)
                            b_vals.append(1.0 if j1 == j2 else 0.0)
                    
                    if len(A_rows) > 0:
                        A_mat = np.array(A_rows)
                        b_vec = np.array(b_vals)
                        try:
                            u_new, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            U[i*m+j1] = u_new
                        except:
                            pass
                    
                    if time.time() > deadline - 15:
                        break
                if time.time() > deadline - 15:
                    break
            
            if time.time() > deadline - 15:
                break
            
            # Optimize V (fix U, W)
            for j2 in range(m):
                for k in range(p):
                    A_rows = []
                    b_vals = []
                    for i in range(n):
                        for j1 in range(m):
                            a_vec = U[i*m+j1] * W[k*n+i]
                            A_rows.append(a_vec)
                            b_vals.append(1.0 if j1 == j2 else 0.0)
                    
                    if len(A_rows) > 0:
                        A_mat = np.array(A_rows)
                        b_vec = np.array(b_vals)
                        try:
                            v_new, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            V[j2*p+k] = v_new
                        except:
                            pass
                    
                    if time.time() > deadline - 15:
                        break
                if time.time() > deadline - 15:
                    break
            
            if time.time() > deadline - 15:
                break
            
            # Optimize W (fix U, V)
            for i in range(n):
                for k in range(p):
                    A_rows = []
                    b_vals = []
                    for j1 in range(m):
                        for j2 in range(m):
                            a_vec = U[i*m+j1] * V[j2*p+k]
                            A_rows.append(a_vec)
                            b_vals.append(1.0 if j1 == j2 else 0.0)
                    
                    if len(A_rows) > 0:
                        A_mat = np.array(A_rows)
                        b_vec = np.array(b_vals)
                        try:
                            w_new, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            W[k*n+i] = w_new
                        except:
                            pass
                    
                    if time.time() > deadline - 15:
                        break
                if time.time() > deadline - 15:
                    break
            
            # Check error
            err = check_decomposition(U, V, W, tol=1e-4)
            if err < 1e-6:
                return True, U, V, W
        
        return False, U, V, W
    
    # Try various ranks
    best_result = None
    best_rank = rank_schoolbook
    
    # Try ranks from 70 up to schoolbook
    for rank in [70, 80, 90, 100, 110, 120, 130]:
        if time.time() > deadline - 40:
            break
        if rank >= best_rank:
            continue
            
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 30:
                break
            success, U, V, W = try_optimize(rank, max_iters=150, seed=seed)
            if success:
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        if best_rank <= rank:
            break
    
    # If we found a better decomposition, verify it
    if best_result is not None and best_rank < rank_schoolbook:
        U_final = np.array(best_result["U"])
        V_final = np.array(best_result["V"])
        W_final = np.array(best_result["W"])
        if check_decomposition(U_final, V_final, W_final, tol=1e-5):
            return best_result
    
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
