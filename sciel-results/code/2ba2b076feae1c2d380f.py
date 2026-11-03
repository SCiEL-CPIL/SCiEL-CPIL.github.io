import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using NumPy broadcasting."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build tensors: U[i,j,c], V[j,k,c], W[k,i,c]
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Using einsum for efficiency
    T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
    
    # Check if T[i,j,k] = 1 for all i,j,k
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _construct_block_decomposition(n, m, p):
    """
    Constructs a decomposition by partitioning the matrices into blocks
    and using schoolbook multiplication for each block pair.
    
    Partition:
    A (n x m) into A11 (n1 x m1), A12 (n1 x m2), A21 (n2 x m1), A22 (n2 x m2)
    B (m x p) into B11 (m1 x p1), B12 (m1 x p2), B21 (m2 x p1), B22 (m2 x p2)
    
    C (n x p) = A B
    C11 = A11 B11 + A12 B21
    C12 = A11 B12 + A12 B22
    C21 = A21 B11 + A22 B21
    C22 = A21 B12 + A22 B22
    """
    # Dimensions
    n1, n2 = 3, 1
    m1, m2 = 3, 2
    p1, p2 = 3, 2
    
    # Verify dimensions
    assert n1 + n2 == n
    assert m1 + m2 == m
    assert p1 + p2 == p
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    def add_block(n_start, n_end, m_start, m_end, p_start, p_end):
        """
        Adds a schoolbook decomposition for the block multiplication
        A[n_start:n_end, m_start:m_end] * B[m_start:m_end, p_start:p_end]
        contributing to C[n_start:n_end, p_start:p_end].
        """
        bn = n_end - n_start
        bm = m_end - m_start
        bp = p_end - p_start
        
        for i in range(bn):
            for j in range(bm):
                for k in range(bp):
                    # Create U, V, W vectors for this term
                    # U is (n*m, 1), V is (m*p, 1), W is (p*n, 1)
                    u_vec = np.zeros(n * m)
                    v_vec = np.zeros(m * p)
                    w_vec = np.zeros(p * n)
                    
                    # Set the specific indices
                    u_vec[(n_start + i) * m + (m_start + j)] = 1.0
                    v_vec[(m_start + j) * p + (p_start + k)] = 1.0
                    w_vec[(p_start + k) * n + (n_start + i)] = 1.0
                    
                    U_blocks.append(u_vec)
                    V_blocks.append(v_vec)
                    W_blocks.append(w_vec)
                    
    # Block 1: A11 * B11 -> C11
    # A11: rows 0-2, cols 0-2. B11: rows 0-2, cols 0-2. C11: rows 0-2, cols 0-2.
    add_block(0, 3, 0, 3, 0, 3)
    
    # Block 2: A12 * B21 -> C11
    # A12: rows 0-2, cols 3-4. B21: rows 3-4, cols 0-2. C11: rows 0-2, cols 0-2.
    add_block(0, 3, 3, 5, 0, 3)
    
    # Block 3: A11 * B12 -> C12
    # A11: rows 0-2, cols 0-2. B12: rows 0-2, cols 3-4. C12: rows 0-2, cols 3-4.
    add_block(0, 3, 0, 3, 3, 5)
    
    # Block 4: A12 * B22 -> C12
    # A12: rows 0-2, cols 3-4. B22: rows 3-4, cols 3-4. C12: rows 0-2, cols 3-4.
    add_block(0, 3, 3, 5, 3, 5)
    
    # Block 5: A21 * B11 -> C21
    # A21: rows 3, cols 0-2. B11: rows 0-2, cols 0-2. C21: rows 3, cols 0-2.
    add_block(3, 4, 0, 3, 0, 3)
    
    # Block 6: A21 * B12 -> C21
    # A21: rows 3, cols 0-2. B12: rows 0-2, cols 3-4. C21: rows 3, cols 3-4.
    add_block(3, 4, 0, 3, 3, 5)
    
    # Block 7: A22 * B21 -> C22
    # A22: rows 3, cols 3-4. B21: rows 3-4, cols 0-2. C22: rows 3, cols 0-2.
    add_block(3, 4, 3, 5, 0, 3)
    
    # Block 8: A22 * B22 -> C22
    # A22: rows 3, cols 3-4. B22: rows 3-4, cols 3-4. C22: rows 3, cols 3-4.
    add_block(3, 4, 3, 5, 3, 5)
    
    U = np.column_stack(U_blocks)
    V = np.column_stack(V_blocks)
    W = np.column_stack(W_blocks)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Construct the block decomposition
    U, V, W = _construct_block_decomposition(n, m, p)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
