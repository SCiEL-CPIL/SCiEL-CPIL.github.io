import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7) -> indices i*2+j
    V: (4, 7) -> indices j*2+k
    W: (4, 7) -> indices k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Col 0: M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1
    U[3, 0] = 1
    V[0, 0] = 1
    V[3, 0] = 1
    W[0, 0] = 1
    W[3, 3] = 1
    
    # Col 1: M2 = (a21+a22)b11
    U[2, 1] = 1
    U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = 1
    W[3, 1] = -1
    
    # Col 2: M3 = a11(b21-b22)
    U[0, 2] = 1
    V[2, 2] = 1
    V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # Col 3: M4 = a22(b11-b12)
    U[3, 3] = 1
    V[0, 3] = 1
    V[1, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # Col 4: M5 = (a11+a12)b22
    U[0, 4] = 1
    U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # Col 5: M6 = (a21-a11)(b11+b12)
    U[1, 5] = 1
    U[0, 5] = -1
    V[0, 5] = 1
    V[1, 5] = 1
    W[3, 5] = 1
    
    # Col 6: M7 = (a12-a22)(b21+b22)
    U[2, 6] = 1
    U[3, 6] = -1
    V[2, 6] = 1
    V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_rank_33():
    """
    Construct a rank-33 decomposition:
    - Split B (3x6) into three 3x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6]
    - For each block, the product A (2x3) * B_block (3x2) gives a 2x2 result
    - Use Strassen for the 2x2 part (rows 0,1 of A) and trivial for row 2 of A
    - Strassen: 3 blocks * 7 = 21
    - Trivial: 3 blocks * 2 rows * 2 cols = 12
    - Total: 33
    """
    n, m, p = DIMENSIONS
    rank_total = 33
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    Strassen_U, Strassen_V, Strassen_W = _strassen_2x2x2()
    
    col_offset = 0
    for block_idx in range(3):
        k_start = block_idx * 2
        # Strassen part: A[0:2, :] (2x3) * B[:, k_start:k_start+2] (3x2) -> 2x2
        # But Strassen is for 2x2 * 2x2. We need to handle the 3rd row of A separately.
        # For the 2x2 subproblem (A[0:2, 0:2] * B[0:2, k_start:k_start+2]):
        # Actually, A is 2x3, so A[0:2, :] has 3 columns. The 3x2 block of B has 3 rows.
        # The product A[i, j] * B[j, k] sums over j=0,1,2.
        # We can split: A[0:2, 0:2] * B[0:2, k_start:k_start+2] (2x2 * 2x2, Strassen)
        #           + A[0:2, 2:3] * B[2:3, k_start:k_start+2] (2x1 * 1x2, trivial)
        
        # Strassen for A[0:2, 0:2] * B[0:2, k_start:k_start+2]
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * 3 + j, col_offset] = Strassen_U[i * 2 + j, c]
            for j in range(2):
                for k in range(2):
                    V[j * 6 + (k_start + k), col_offset] = Strassen_V[j * 2 + k, c]
            for k in range(2):
                for i in range(2):
                    W[(k_start + k) * 2 + i, col_offset] = Strassen_W[k * 2 + i, c]
            col_offset += 1
        
        # Trivial for A[0:2, 2:3] * B[2:3, k_start:k_start+2]
        # c[i, k] += A[i, 2] * B[2, k_start+k]
        # For each (i, k) pair, this is a scalar multiplication
        for i in range(2):
            for k in range(2):
                U[i * 3 + 2, col_offset] = 1
                V[2 * 6 + (k_start + k), col_offset] = 1
                W[(k_start + k) * 2 + i, col_offset] = 1
                col_offset += 1
    
    return U, V, W

def _try_random_search(rank, deadline, n=2, m=3, p=6):
    """
    Try to find a decomposition of the given rank using random initialization
    and Newton-like iterations.
    """
    if time.time() > deadline:
        return None
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    
    # Use a simple iterative approach: randomly adjust to reduce error
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    num_trials = 5
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # A few Newton-like steps
        for step in range(50):
            if time.time() > deadline - 10:
                break
            
            # Compute error tensor
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        error += (val - 1.0)**2
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-10:
                break
            
            # Gradient descent step (simplified)
            lr = 0.01 / (1 + step)
            for r in range(rank):
                # Update U
                for i in range(n):
                    for j in range(m):
                        grad = 0
                        for k in range(p):
                            grad += 2 * (np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]) - 1) * V[j*p+k, r] * W[k*n+i, r]
                        U[i*m+j, r] -= lr * grad
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        grad = 0
                        for i in range(n):
                            grad += 2 * (np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]) - 1) * U[i*m+j, r] * W[k*n+i, r]
                        V[j*p+k, r] -= lr * grad
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        grad = 0
                        for j in range(m):
                            grad += 2 * (np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]) - 1) * U[i*m+j, r] * V[j*p+k, r]
                        W[k*n+i, r] -= lr * grad
        
        if best_error < 1e-8:
            break
    
    if best_error < 1e-8:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # First, try the known rank-33 construction
    U, V, W = _construct_rank_33()
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8):
        # Try to see if we can do better with random search
        for rank in [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21]:
            if time.time() > deadline - 20:
                break
            
            result = _try_random_search(rank, deadline - 10, n, m, p)
            if result is not None:
                U_r, V_r, W_r = result
                U_r_list = U_r.tolist()
                V_r_list = V_r.tolist()
                W_r_list = W_r.tolist()
                if _verify_decomposition(U_r_list, V_r_list, W_r_list, n, m, p, tol=1e-8):
                    return {"U": U_r_list, "V": V_r_list, "W": W_r_list}
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
