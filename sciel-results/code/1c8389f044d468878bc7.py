import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def _time_remaining():
    return _DEADLINE - time.time()

def _verify_decomposition(U, V, W, n=2, m=2, p=7):
    """Verify that the decomposition correctly computes matrix multiplication."""
    # Create random test matrices
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Reconstruct C using the tensor decomposition
    # C[i,k] = sum_r W[2*k+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # Note: U is indexed by (i, j) flattened as i*m+j
    # V is indexed by (j, k) flattened as j*p+k
    # W is indexed by (k, i) flattened as 2*k+i (since n=2)
    
    C_reconstructed = np.zeros((n, p))
    
    for r in range(U.shape[1]):
        # Compute sum over j for A part
        SA = 0.0
        for i in range(n):
            for j in range(m):
                SA += U[i*m + j, r] * A[i, j]
        
        # Compute sum over j for B part
        SB = 0.0
        for j in range(m):
            for k in range(p):
                SB += V[j*p + k, r] * B[j, k]
        
        # Add contribution to C
        for k in range(p):
            for i in range(n):
                C_reconstructed[i, k] += W[2*k + i, r] * SA * SB
    
    return np.allclose(C_reconstructed, C_expected)

def _try_rank(R, n=2, m=2, p=7, max_iter=10000, tol=1e-10):
    """
    Attempt to find a rank-R decomposition using gradient-based optimization.
    Returns (U, V, W) if successful, None otherwise.
    """
    if _time_remaining() < 10.0:
        return None
    
    # Initialize with random values
    np.random.seed(42)  # Fixed seed for reproducibility
    U = np.random.randn(n*m, R) * 0.1
    V = np.random.randn(m*p, R) * 0.1
    W = np.random.randn(p*n, R) * 0.1
    
    # Build the target tensor T
    # T[a, b, c] = 1 if the product A_a * B_b contributes to C_c
    # a = i*m + j (A index)
    # b = j'*p + k (B index)
    # c = 2*k' + i' (C index, where i' is row, k' is col)
    # For matrix mult: C[i', k'] = sum_j A[i', j] * B[j, k']
    # So T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = 2*k+i and i=i', k=k'
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = 2*k + i
                T[a, b, c] = 1.0
    
    # Define loss function: ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
    def compute_loss(U, V, W):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T - T_approx)**2)
    
    # Simple gradient descent
    lr = 0.01
    best_loss = compute_loss(U, V, W)
    
    for it in range(max_iter):
        if _time_remaining() < 5.0:
            return None
        
        # Compute gradients
        # dL/dU[a,r] = 2 * sum_{b,c} (T[a,b,c] - T_approx[a,b,c]) * V[b,r] * W[c,r]
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T - T_approx
        
        dU = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
        dV = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
        dW = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        loss = compute_loss(U, V, W)
        if loss < best_loss:
            best_loss = loss
        
        if best_loss < tol:
            break
    
    if best_loss < tol:
        return U, V, W
    
    return None

def _schoolbook_decomposition():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_block_decomposition():
    """
    Block-diagonal Strassen decomposition with rank 25.
    Uses Strassen (rank 7) for columns 0-1, 2-3, 4-5 and schoolbook (rank 4) for column 6.
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,ks
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
        
    # Add Strassen for columns 0,1
    add_strassen(0, 0)
    
    # Add Strassen for columns 2,3
    add_strassen(7, 2)
    
    # Add Strassen for columns 4,5
    add_strassen(14, 4)
    
    # Add schoolbook for column 6 (4 terms)
    col_start = 21
    # A00 * B0,6 -> C0,6
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    
    # A01 * B1,6 -> C0,6
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    
    # A10 * B0,6 -> C1,6
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    
    # A11 * B1,6 -> C1,6
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    1. Try to find a lower-rank decomposition using global optimization (rank 22, 23, 24)
    2. Fall back to block-diagonal Strassen (rank 25) if global optimization fails
    3. Fall back to schoolbook (rank 28) if time runs out
    """
    if _time_remaining() < 5.0:
        return _schoolbook_decomposition()
    
    # Try global rank minimization starting from lower ranks
    for R in [22, 23, 24]:
        if _time_remaining() < 10.0:
            break
        
        result = _try_rank(R, max_iter=5000)
        if result is not None:
            U, V, W = result
            # Verify the decomposition
            if _verify_decomposition(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to block-diagonal Strassen (rank 25)
    if _time_remaining() < 5.0:
        return _schoolbook_decomposition()
    
    return _strassen_block_decomposition()

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    print(f"Rank: {U.shape[1]}")
    if _verify_decomposition(U, V, W):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
