"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    num_eq = n * m * p  # 40 equations
    num_U = n * m       # 8
    num_V = m * p       # 20
    num_W = p * n       # 10
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the multiplication tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def build_block_strassen():
        """Build the rank-36 block Strassen decomposition."""
        rank = 36
        U = np.zeros((num_U, rank))
        V = np.zeros((num_V, rank))
        W = np.zeros((num_W, rank))
        
        col = 0
        for block in range(2):
            j_off = block * 2
            
            # Strassen M1: (A00+A11)*(B00+B11)
            U[0*m + j_off, col] = 1
            U[1*m + j_off+1, col] = 1
            V[j_off*p + 0, col] = 1
            V[(j_off+1)*p + 1, col] = 1
            W[0*n + 0, col] = 1
            W[1*n + 0, col] = -1
            W[0*n + 1, col] = -1
            W[1*n + 1, col] = 1
            col += 1
            
            # Strassen M2: (A10+A11)*B00
            U[1*m + j_off, col] = 1
            U[1*m + j_off+1, col] = 1
            V[j_off*p + 0, col] = 1
            W[0*n + 1, col] = 1
            col += 1
            
            # Strassen M3: A00*(B10-B11)
            U[0*m + j_off, col] = 1
            V[(j_off+1)*p + 0, col] = 1
            V[(j_off+1)*p + 1, col] = -1
            W[0*n + 0, col] = 1
            col += 1
            
            # Strassen M4: A11*(B01-B10)
            U[1*m + j_off+1, col] = 1
            V[j_off*p + 1, col] = 1
            V[(j_off+1)*p + 0, col] = -1
            W[0*n + 1, col] = 1
            col += 1
            
            # Strassen M5: (A00+A01)*B11
            U[0*m + j_off, col] = 1
            U[0*m + j_off+1, col] = 1
            V[(j_off+1)*p + 1, col] = 1
            W[0*n + 1, col] = 1
            col += 1
            
            # Strassen M6: (A10-A00)*(B00+B01)
            U[1*m + j_off, col] = 1
            U[0*m + j_off, col] = -1
            V[j_off*p + 0, col] = 1
            V[j_off*p + 1, col] = 1
            W[0*n + 1, col] = 1
            col += 1
            
            # Strassen M7: (A01-A11)*(B10+B11)
            U[0*m + j_off+1, col] = 1
            U[1*m + j_off+1, col] = -1
            V[(j_off+1)*p + 0, col] = 1
            V[(j_off+1)*p + 1, col] = 1
            W[0*n + 1, col] = 1
            col += 1
            
            # Schoolbook for 2x2 * 2x1 (k=2)
            for ii in range(2):
                for jj in range(2):
                    U[ii*m + j_off+jj, col] = 1
                    V[(j_off+jj)*p + 2, col] = 1
                    W[2*n + ii, col] = 1
                    col += 1
        
        assert col == rank, f"Expected {rank}, got {col}"
        return U, V, W
    
    # Get rank-36 baseline
    U36, V36, W36 = build_block_strassen()
    assert check_decomposition(U36, V36, W36), "Block Strassen verification failed"
    
    best_rank = 36
    best_U, best_V, best_W = U36.copy(), V36.copy(), W36.copy()
    
    def try_rank(rank, max_iters=300, n_restarts=10):
        """Try to find a valid decomposition of given rank."""
        nonlocal best_rank, best_U, best_V, best_W
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return False
            
            np.random.seed(12345 + restart * 7)
            U = np.random.randn(num_U, rank) * 0.3
            V = np.random.randn(num_V, rank) * 0.3
            W = np.random.randn(num_W, rank) * 0.3
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return False
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            u_new = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                        except:
                            continue
                        U[i*m+j, :] = u_new
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            v_new = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                        except:
                            continue
                        V[j*p+k, :] = v_new
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            w_new = np.linalg.lstsq(A.T, np.ones(m), rcond=None)[0]
                        except:
                            continue
                        W[k*n+i, :] = w_new
                
                # Check
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-8:
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if check_decomposition(U_r, V_r, W_r, tol=1e-9):
                        if rank < best_rank:
                            best_rank = rank
                            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                        return True
                elif max_err < 1e-5:
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if check_decomposition(U_r, V_r, W_r, tol=1e-9):
                        if rank < best_rank:
                            best_rank = rank
                            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                        return True
                elif max_err < 1e-12:
                    # Already essentially exact, try rounding
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if check_decomposition(U_r, V_r, W_r, tol=1e-9):
                        if rank < best_rank:
                            best_rank = rank
                            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                        return True
        
        return False
    
    # Try ranks from 35 down
    for rank in range(35, 28, -1):
        if time.time() > deadline:
            break
        if try_rank(rank):
            break
    
    # Final verification
    if not check_decomposition(best_U, best_V, best_W, tol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((num_U, rank))
        V = np.zeros((num_V, rank))
        W = np.zeros((num_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
