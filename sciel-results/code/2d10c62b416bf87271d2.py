"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _check_decomposition(n, m, p, U, V, W, tol=1e-9):
    """Check if the decomposition is valid."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _compute_error(n, m, p, U, V, W):
    """Compute the maximum error of the decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _try_rank(n, m, p, rank, max_iters=100, seed=42):
    """Try to find a valid decomposition with given rank."""
    np.random.seed(seed)
    
    # Initialize with random values
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    # Define the objective: sum of squared errors
    def objective(params):
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U_mat = U_flat.reshape(n*m, rank)
        V_mat = V_flat.reshape(m*p, rank)
        W_mat = W_flat.reshape(p*n, rank)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+i])
                    err += (s - 1.0) ** 2
        return err
    
    # Try multiple random restarts
    best_error = float('inf')
    best_params = None
    
    for restart in range(max_iters):
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 50})
        
        if res.fun < best_error:
            best_error = res.fun
            best_params = res.x
        
        if best_error < 1e-12:
            break
    
    if best_params is None:
        return None
    
    U_mat = best_params[:n*m*rank].reshape(n*m, rank)
    V_mat = best_params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W_mat = best_params[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    # Check if valid
    if _check_decomposition(n, m, p, U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), tol=1e-8):
        return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()
    
    return None

def _strassen_2x2x2():
    """Return the Strassen decomposition for 2x2x2 matrix multiplication.
    
    Returns U, V, W each of shape (4, 7) where:
    U[i*2+j, r] for i,j in {0,1}
    V[j*2+k, r] for j,k in {0,1}
    W[k*2+i, r] for k,i in {0,1}
    """
    # Strassen's 7 multiplications for 2x2 matrices:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We need T[i,j,k] = 1 if (i,j,k) corresponds to a product term, else 0
    # Actually T[i,j,k] = 1 for all i,j,k in the 2x2x2 case (schoolbook)
    # But with Strassen, we use 7 terms.
    
    # Let me define the Strassen decomposition properly.
    # For 2x2x2, the tensor has entries T[i,j,k] = 1 for all i,j,k in {0,1}.
    # The rank-7 decomposition:
    
    # U[i,j,r] contributes A[i,j]
    # V[j,k,r] contributes B[j,k]
    # W[k,i,r] contributes to C[i,k]
    
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # So for r=0: U[0,0,0]=1, U[1,1,0]=1, V[0,0,0]=1, V[1,1,0]=1, W[0,0,0]=1, W[1,1,0]=1
    # Wait, W[k,i,r] means W[0,0,0] corresponds to C[0,0], W[1,1,0] to C[1,1]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    # A part: U[0,0]=1, U[1,1]=1
    # B part: V[0,0]=1, V[1,1]=1
    # C part: C00 += M1, C11 += M1 -> W[0,0]=1, W[1,1]=1
    U[0, 0] = 1; U[3, 0] = 1  # A[0,0], A[1,1]
    V[0, 0] = 1; V[3, 0] = 1  # B[0,0], B[1,1]
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0], C[1,1]
    
    # r=1: M2 = (A10+A11)B00
    # A part: U[1,0]=1, U[1,1]=1
    # B part: V[0,0]=1
    # C part: C10 += M2, C11 += M2 -> W[0,1]=1, W[1,1]=1
    U[2, 1] = 1; U[3, 1] = 1  # A[1,0], A[1,1]
    V[0, 1] = 1  # B[0,0]
    W[2, 1] = 1; W[3, 1] = 1  # C[1,0], C[1,1]
    
    # r=2: M3 = A00(B01-B11)
    # A part: U[0,0]=1
    # B part: V[0,1]=1, V[1,1]=-1
    # C part: C01 += M3 -> W[1,0]=1
    U[0, 2] = 1  # A[0,0]
    V[1, 2] = 1; V[3, 2] = -1  # B[0,1], B[1,1]
    W[1, 2] = 1  # C[0,1]
    
    # r=3: M4 = A11(B10-B00)
    # A part: U[1,1]=1
    # B part: V[1,0]=1, V[0,0]=-1
    # C part: C00 += M4, C10 += M4 -> W[0,0]=1, W[1,0]=1
    U[3, 3] = 1  # A[1,1]
    V[2, 3] = 1; V[0, 3] = -1  # B[1,0], B[0,0]
    W[0, 3] = 1; W[2, 3] = 1  # C[0,0], C[1,0]
    
    # r=4: M5 = (A00+A01)B11
    # A part: U[0,0]=1, U[0,1]=1
    # B part: V[1,1]=1
    # C part: C00 -= M5, C01 += M5 -> W[0,0]=-1, W[1,0]=1
    U[0, 4] = 1; U[1, 4] = 1  # A[0,0], A[0,1]
    V[3, 4] = 1  # B[1,1]
    W[0, 4] = -1; W[1, 4] = 1  # C[0,0], C[0,1]
    
    # r=5: M6 = (A10-A00)(B00+B01)
    # A part: U[1,0]=1, U[0,0]=-1
    # B part: V[0,0]=1, V[0,1]=1
    # C part: C11 += M6 -> W[1,1]=1
    U[2, 5] = 1; U[0, 5] = -1  # A[1,0], A[0,0]
    V[0, 5] = 1; V[1, 5] = 1  # B[0,0], B[0,1]
    W[3, 5] = 1  # C[1,1]
    
    # r=6: M7 = (A01-A11)(B10+B11)
    # A part: U[0,1]=1, U[1,1]=-1
    # B part: V[1,0]=1, V[1,1]=1
    # C part: C00 += M7 -> W[0,0]=1
    U[1, 6] = 1; U[3, 6] = -1  # A[0,1], A[1,1]
    V[2, 6] = 1; V[3, 6] = 1  # B[1,0], B[1,1]
    W[0, 6] = 1  # C[0,0]
    
    return U.tolist(), V.tolist(), W.tolist()

def _build_decomposition_3x4x5():
    """Build a rank-59 decomposition for 3x4x5 matrix multiplication.
    
    Strategy: Use Strassen (rank 7) for one 2x2x2 block, and schoolbook (rank 1) for the rest.
    Total schoolbook rank = 3*4*5 = 60.
    Replacing one 2x2x2 block (rank 4) with Strassen (rank 7) gives net change of +3,
    so rank = 60 - 4 + 7 = 63. That's worse.
    
    Actually, we need to think more carefully. The schoolbook decomposition uses 60 rank-1 terms.
    Each term corresponds to one (i,j,k) triple.
    
    If we can replace a group of terms with a lower-rank decomposition, we reduce the total rank.
    For 2x2x2, schoolbook uses 4 terms, Strassen uses 7. That's an increase.
    
    So Strassen doesn't help for reducing the rank in this context?
    Wait, the rank of the 2x2x2 multiplication tensor is 7, not 4. The schoolbook rank is 4,
    but the tensor rank is 7? No, the tensor rank of 2x2x2 is 7, meaning the minimum number
    of rank-1 tensors to sum to get the full tensor is 7.
    
    But the schoolbook decomposition uses 4 terms, each of which is a rank-1 tensor in the
    (i,j,k) space? No, each schoolbook term U[i,j]=1, V[j,k]=1, W[k,i]=1 for a specific (i,j,k)
    is indeed a rank-1 tensor. And there are 8 such terms for 2x2x2 (not 4).
    
    Wait, for 2x2x2, there are 2*2*2 = 8 terms in schoolbook. Strassen reduces this to 7.
    
    So for 3x4x5, schoolbook rank = 3*4*5 = 60.
    If we can find a 2x2x2 sub-block and replace its 8 terms with 7, we get rank 60 - 8 + 7 = 59.
    
    Let me verify: The 2x2x2 block involves indices i in {0,1}, j in {0,1}, k in {0,1}.
    The schoolbook terms for this block are 8 terms (one for each (i,j,k) in {0,1}^3).
    Strassen replaces these 8 terms with 7 terms.
    
    So total rank = 60 - 8 + 7 = 59.
    """
    n, m, p = 3, 4, 5
    rank = 59
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen block for i in {0,1}, j in {0,1}, k in {0,1}
    U_s, V_s, W_s = _strassen_2x2x2()
    U_s = np.array(U_s)
    V_s = np.array(V_s)
    W_s = np.array(W_s)
    
    # Place Strassen terms in columns 0-6
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m+j, r] = U_s[i*2+j, r]
        for j in range(2):
            for k in range(2):
                V[j*p+k, r] = V_s[j*2+k, r]
        for k in range(2):
            for i in range(2):
                W[k*n+i, r] = W_s[k*2+i, r]
    
    # Schoolbook terms for the remaining (i,j,k) triples
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this (i,j,k) is in the Strassen block
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Try the hardcoded rank-59 decomposition
    try:
        U, V, W = _build_decomposition_3x4x5()
        if _check_decomposition(n, m, p, U, V, W, tol=1e-9):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
