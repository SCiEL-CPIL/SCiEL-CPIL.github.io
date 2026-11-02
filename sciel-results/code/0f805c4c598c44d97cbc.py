"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _build_base_decomposition():
    """Build the rank-36 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 36
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        u00 = a_r * m + a_c
        u01 = a_r * m + a_c + 1
        u10 = (a_r + 1) * m + a_c
        u11 = (a_r + 1) * m + a_c + 1
        
        v00 = b_r * p + b_c
        v01 = b_r * p + b_c + 1
        v10 = (b_r + 1) * p + b_c
        v11 = (b_r + 1) * p + b_c + 1
        
        w00 = c_c * n + c_r
        w01 = (c_c + 1) * n + c_r
        w10 = c_c * n + c_r + 1
        w11 = (c_c + 1) * n + c_r + 1
        
        U[u00, col] += 1; U[u11, col] += 1
        V[v00, col] += 1; V[v11, col] += 1
        W[w00, col] += 1; W[w11, col] += 1
        col += 1
        
        U[u10, col] += 1; U[u11, col] += 1
        V[v00, col] += 1
        W[w10, col] += 1; W[w11, col] -= 1
        col += 1
        
        U[u00, col] += 1
        V[v01, col] += 1; V[v11, col] -= 1
        W[w01, col] += 1; W[w11, col] += 1
        col += 1
        
        U[u11, col] += 1
        V[v10, col] += 1; V[v00, col] -= 1
        W[w00, col] += 1; W[w10, col] += 1
        col += 1
        
        U[u00, col] += 1; U[u01, col] += 1
        V[v11, col] += 1
        W[w00, col] -= 1; W[w01, col] += 1
        col += 1
        
        U[u10, col] += 1; U[u00, col] -= 1
        V[v00, col] += 1; V[v01, col] += 1
        W[w11, col] += 1
        col += 1
        
        U[u01, col] += 1; U[u11, col] -= 1
        V[v10, col] += 1; V[v11, col] += 1
        W[w00, col] += 1
        col += 1

    def add_schoolbook_2x1(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        u00 = a_r * m + a_c
        u01 = a_r * m + a_c + 1
        u10 = (a_r + 1) * m + a_c
        u11 = (a_r + 1) * m + a_c + 1
        
        v0 = b_r * p + b_c
        v1 = (b_r + 1) * p + b_c
        
        w0 = c_c * n + c_r
        w1 = c_c * n + c_r + 1
        
        U[u00, col] = 1; V[v0, col] = 1; W[w0, col] = 1
        col += 1
        U[u01, col] = 1; V[v1, col] = 1; W[w0, col] = 1
        col += 1
        U[u10, col] = 1; V[v0, col] = 1; W[w1, col] = 1
        col += 1
        U[u11, col] = 1; V[v1, col] = 1; W[w1, col] = 1
        col += 1

    # C1 (2x2) from B cols 0,1
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # C2 (2x3) from B cols 2,3,4
    add_schoolbook_2x1(0, 0, 0, 2, 0, 2)
    add_schoolbook_2x1(0, 2, 2, 2, 0, 2)
    
    # C22 (2x2) from B cols 3,4
    add_strassen(0, 0, 0, 3, 0, 3)
    add_strassen(0, 2, 2, 3, 0, 3)
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    # The tensor for matrix multiplication: T[i,j,k,l] = delta_{i,l} * delta_{j,k}
    # where A is n x m, B is m x p, C = A*B is n x p
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n+i, r] gives coefficient for C[i,k]
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{i,k} * 1 (for all i,j,k)
    # Actually the decomposition should satisfy:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = delta_{i,k}... no.
    # The multiplication tensor: for each (i,j,k), coefficient is 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the identity in the appropriate sense)
    # Actually: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has entries T[i,j,k] = 1 for all valid i,j,k
    # Wait, that's not right either. Let me think more carefully.
    
    # U[i*m+j, r] multiplies with V[j'*p+k, r] and W[k'*n+i', r]
    # The product sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    # should give 1 if (i,j,j',k,k',i') corresponds to a valid multiplication term
    # i.e., j == j' (the inner dimension matches) and i == i' and k == k'
    # So the result should be delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    # More simply: for each (i, k), C[i,k] = sum_j A[i,j] * B[j,k]
    # In the decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]
    # This should equal sum_j A[i,j] * B[j,k]
    
    # So we need: sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) = sum_j A[i,j]*B[j,k]
    
    # For this to hold for all A, B, we need for each (i,k):
    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all j
    # and sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] = 0 for j' != j
    
    # Let's verify with random matrices
    np.random.seed(42)
    for trial in range(5):
        A = np.random.randint(-5, 5, size=(n, m))
        B = np.random.randint(-5, 5, size=(m, p))
        C_expected = A @ B
        
        # Compute via decomposition
        C_computed = np.zeros((n, p), dtype=int)
        for r in range(U.shape[1]):
            # Compute intermediate products
            prod_A = np.zeros(n, dtype=int)
            for i in range(n):
                for j in range(m):
                    prod_A[i] += A[i, j] * U[i * m + j, r]
            
            prod_B = np.zeros(p, dtype=int)
            for k in range(p):
                for j in range(m):
                    prod_B[k] += B[j, k] * V[j * p + k, r]
            
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += prod_A[i] * prod_B[k] * W[k * n + i, r]
        
        if not np.array_equal(C_computed, C_expected):
            return False
    return True

def _try_remove_columns(U, V, W, n, m, p, max_time=100):
    """Try to remove redundant columns from the decomposition."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    start_time = time.time()
    removed = []
    
    while time.time() - start_time < max_time:
        rank = U.shape[1]
        found = False
        for col_idx in range(rank):
            if col_idx in removed:
                continue
            # Try removing this column
            U_new = np.delete(U, col_idx, axis=1)
            V_new = np.delete(V, col_idx, axis=1)
            W_new = np.delete(W, col_idx, axis=1)
            
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                removed.append(col_idx)
                found = True
                break
        if not found:
            break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build base decomposition
    U, V, W = _build_base_decomposition()
    
    # Verify base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to simple schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by removing redundant columns
    time_remaining = 120 - (time.time() - start_time) - 10  # 10s safety margin
    if time_remaining > 5:
        U, V, W = _try_remove_columns(U, V, W, n, m, p, max_time=time_remaining)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Rebuild and don't try to reduce
        U, V, W = _build_base_decomposition()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
