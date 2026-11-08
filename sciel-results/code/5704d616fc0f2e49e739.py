"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def verify_decomposition(U, V, W, n=2, m=3, p=5, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    # Check: T[i,j,k,l] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+l,c] = delta_{i,l}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(n):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+l, c] for c in range(rank))
                    expected = 1.0 if i == l else 0.0
                    if abs(val - expected) > tol:
                        return False
    return True

def _strassen_2x2x2():
    """Return Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    
    Returns U, V, W each of shape (4, 7) where:
    U[i*2+j, c] corresponds to A[i,j]
    V[j*2+k, c] corresponds to B[j,k]
    W[k*2+i, c] corresponds to C[i,k] (note: W index is k*n+i with n=2)
    
    The seven Strassen products are:
    M1 = (a11 + a22)(b11 + b22)
    M2 = (a21 + a22)b11
    M3 = a11(b12 - b22)
    M4 = a22(b21 - b11)
    M5 = (a11 + a12)b22
    M6 = (a21 - a11)(b11 + b12)
    M7 = (a12 - a22)(b21 + b22)
    
    C11 = M1 - M4 + M5 + M6
    C12 = M3 + M4
    C21 = M2 + M5
    C22 = M1 - M2 + M3 + M7
    """
    # U matrix: rows indexed by (i,j) where i=0,1 and j=0,1
    # Row order: (0,0), (0,1), (1,0), (1,1)
    U = np.zeros((4, 7))
    # V matrix: rows indexed by (j,k) where j=0,1 and k=0,1
    # Row order: (0,0), (0,1), (1,0), (1,1)
    V = np.zeros((4, 7))
    # W matrix: rows indexed by (k,i) where k=0,1 and i=0,1
    # Row order: (0,0), (0,1), (1,0), (1,1)
    W = np.zeros((4, 7))
    
    # M1 = (a11 + a22)(b11 + b22)
    # a11 is row 0, a22 is row 3; b11 is row 0, b22 is row 3
    U[0, 0] = 1  # a11
    U[3, 0] = 1  # a22
    V[0, 0] = 1  # b11
    V[3, 0] = 1  # b22
    # C11 += M1, C22 += M1
    W[0, 0] = 1  # C11 (k=0, i=0)
    W[3, 0] = 1  # C22 (k=1, i=1)
    
    # M2 = (a21 + a22)b11
    # a21 is row 2, a22 is row 3; b11 is row 0
    U[2, 1] = 1  # a21
    U[3, 1] = 1  # a22
    V[0, 1] = 1  # b11
    # C21 += M2, C22 -= M2
    W[2, 1] = 1  # C21 (k=1, i=0)
    W[3, 1] = -1  # C22 (k=1, i=1)
    
    # M3 = a11(b12 - b22)
    # a11 is row 0; b12 is row 1, b22 is row 3
    U[0, 2] = 1  # a11
    V[1, 2] = 1  # b12
    V[3, 2] = -1  # b22
    # C11 += M3, C12 += M3
    W[0, 2] = 1  # C11
    W[1, 2] = 1  # C12 (k=0, i=1)
    
    # M4 = a22(b21 - b11)
    # a22 is row 3; b21 is row 2, b11 is row 0
    U[3, 3] = 1  # a22
    V[2, 3] = 1  # b21
    V[0, 3] = -1  # b11
    # C11 -= M4, C12 += M4
    W[0, 3] = -1  # C11
    W[1, 3] = 1  # C12
    
    # M5 = (a11 + a12)b22
    # a11 is row 0, a12 is row 1; b22 is row 3
    U[0, 4] = 1  # a11
    U[1, 4] = 1  # a12
    V[3, 4] = 1  # b22
    # C11 += M5, C21 += M5
    W[0, 4] = 1  # C11
    W[2, 4] = 1  # C21
    
    # M6 = (a21 - a11)(b11 + b12)
    # a21 is row 2, a11 is row 0; b11 is row 0, b12 is row 1
    U[2, 5] = 1  # a21
    U[0, 5] = -1  # a11
    V[0, 5] = 1  # b11
    V[1, 5] = 1  # b12
    # C11 += M6
    W[0, 5] = 1  # C11
    
    # M7 = (a12 - a22)(b21 + b22)
    # a12 is row 1, a22 is row 3; b21 is row 2, b22 is row 3
    U[1, 6] = 1  # a12
    U[3, 6] = -1  # a22
    V[2, 6] = 1  # b21
    V[3, 6] = 1  # b22
    # C22 += M7
    W[3, 6] = 1  # C22
    
    return U, V, W

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # Start with schoolbook rank 30
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook decomposition
    U = np.zeros((n*m, best_rank))
    V = np.zeros((m*p, best_rank))
    W = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    if verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try Kronecker-like decomposition: split 3x5 into two parts
    # Part 1: 2x2 * 2x5 (using Strassen for the 2x2 part, rank 7*5=35... too high)
    # Better approach: Use the structure more carefully
    
    # Alternative: Try to find a rank-22 decomposition using block structure
    # For 2x3 * 3x5, we can think of it as:
    # A is 2x3, B is 3x5
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Let's try a more systematic approach with the known lower bound of 15
    # and use optimization with better initialization
    
    # First, try a structured decomposition based on the idea that
    # we can share some computation between the two rows of A
    
    # For rank r, we need U (6, r), V (15, r), W (10, r)
    # The constraint is: for all i,j,k,l: sum_c U[i*3+j,c] V[j*5+k,c] W[k*2+l,c] = delta_{i,l}
    
    # Let's try ranks from 15 to 29 with multiple random restarts
    for target_rank in range(15, min(best_rank, 30)):
        if time.time() > deadline:
            break
        
        found = False
        n_restarts = 3 if target_rank <= 20 else 1
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + target_rank * 100 + restart)
            U_init = np.random.randn(n*m, target_rank) * 0.5
            V_init = np.random.randn(m*p, target_rank) * 0.5
            W_init = np.random.randn(p*n, target_rank) * 0.5
            
            # Use gradient-based optimization with a custom implementation
            # to avoid scipy overhead
            
            # Compute gradient of the objective function
            def compute_error_and_grad(U_mat, V_mat, W_mat):
                error = 0.0
                grad_U = np.zeros_like(U_mat)
                grad_V = np.zeros_like(V_mat)
                grad_W = np.zeros_like(W_mat)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for l in range(n):
                                val = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+l])
                                expected = 1.0 if i == l else 0.0
                                diff = val - expected
                                error += diff * diff
                                
                                # Gradient with respect to U[i*m+j, c]
                                # d/dc sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+l,c] = V[j*p+k,c] * W[k*n+l,c]
                                grad_U[i*m+j] += 2 * diff * (V_mat[j*p+k] * W_mat[k*n+l])
                                
                                # Gradient with respect to V[j*p+k, c]
                                grad_V[j*p+k] += 2 * diff * (U_mat[i*m+j] * W_mat[k*n+l])
                                
                                # Gradient with respect to W[k*n+l, c]
                                grad_W[k*n+l] += 2 * diff * (U_mat[i*m+j] * V_mat[j*p+k])
                
                return error, grad_U, grad_V, grad_W
            
            # Simple gradient descent with learning rate scheduling
            lr = 0.01
            U_mat = U_init.copy()
            V_mat = V_init.copy()
            W_mat = W_init.copy()
            
            max_iter = 200
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                error, grad_U, grad_V, grad_W = compute_error_and_grad(U_mat, V_mat, W_mat)
                
                if error < 1e-10:
                    break
                
                U_mat -= lr * grad_U
                V_mat -= lr * grad_V
                W_mat -= lr * grad_W
                
                # Decay learning rate
                if it % 50 == 49:
                    lr *= 0.5
            
            # Verify the result
            if verify_decomposition(U_mat.tolist(), V_mat.tolist(), W_mat.tolist()):
                best_rank = target_rank
                best_U, best_V, best_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
                found = True
                break
        
        if found:
            break  # Found minimal rank for this target, move to next
    
    # Convert to lists for return
    if best_U is None:
        # Fallback to schoolbook
        U = np.zeros((n*m, best_rank))
        V = np.zeros((m*p, best_rank))
        W = np.zeros((p*n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
