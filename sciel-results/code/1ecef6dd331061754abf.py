import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for matrix multiplication
    C = A @ B where A is 2x4 and B is 4x5.
    
    Returns:
        dict: {'U': U, 'V': V, 'W': W}
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit

    # We use the standard block decomposition:
    # Split A (2x4) into A_L (2x2) and A_R (2x2) along columns.
    # Split B (4x5) into B_T (2x5) and B_B (2x5) along rows.
    # C = A_L B_T + A_R B_B.
    # Each term is a 2x2 @ 2x5 multiplication.
    
    # Rank of 2x2 @ 2x5:
    # Split 2x5 into 2x2, 2x2, 2x1.
    # Rank(2x2@2x2) = 7 (Strassen).
    # Rank(2x2@2x1) = 4.
    # Total Rank(2x2@2x5) = 7 + 7 + 4 = 18.
    # Total Rank = 18 + 18 = 36.
    
    rank = 36
    
    # Initialize arrays
    # U: (n*m, rank) = (8, 36)
    # V: (m*p, rank) = (20, 36)
    # W: (p*n, rank) = (10, 36)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def u_idx(i, j):
        return i * m + j
        
    def v_idx(j, k):
        return j * p + k
        
    def w_idx(k, i):
        return k * n + i
        
    col = 0
    
    # Function to add a 2x2x2 Strassen decomposition
    # A_block: A[ai0:ai0+2, aj0:aj0+2]
    # B_block: B[bj0:bj0+2, bk0:bk0+2]
    # C_block: C[bk0:bk0+2, ai0:ai0+2]
    def add_strassen_block(ai0, aj0, bj0, bk0):
        nonlocal col
        # Indices
        # A: (ai0, aj0), (ai0, aj0+1), (ai0+1, aj0), (ai0+1, aj0+1)
        # B: (bj0, bk0), (bj0, bk0+1), (bj0+1, bk0), (bj0+1, bk0+1)
        # C: (bk0, ai0), (bk0, ai0+1), (bk0+1, ai0), (bk0+1, ai0+1)
        
        a00 = u_idx(ai0, aj0)
        a01 = u_idx(ai0, aj0 + 1)
        a10 = u_idx(ai0 + 1, aj0)
        a11 = u_idx(ai0 + 1, aj0 + 1)
        
        b00 = v_idx(bj0, bk0)
        b01 = v_idx(bj0, bk0 + 1)
        b10 = v_idx(bj0 + 1, bk0)
        b11 = v_idx(bj0 + 1, bk0 + 1)
        
        c00 = w_idx(bk0, ai0)
        c01 = w_idx(bk0, ai0 + 1)
        c10 = w_idx(bk0 + 1, ai0)
        c11 = w_idx(bk0 + 1, ai0 + 1)
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1

    # Function to add a 2x2 @ 2x1 direct decomposition (Rank 4)
    # A_block: 2x2
    # B_vec: 2x1
    # C_vec: 2x1
    def add_direct_2x2x1(ai0, aj0, bj0, bk):
        nonlocal col
        a00 = u_idx(ai0, aj0)
        a01 = u_idx(ai0, aj0 + 1)
        a10 = u_idx(ai0 + 1, aj0)
        a11 = u_idx(ai0 + 1, aj0 + 1)
        
        b0 = v_idx(bj0, bk)
        b1 = v_idx(bj0 + 1, bk)
        
        c0 = w_idx(bk, ai0)
        c1 = w_idx(bk, ai0 + 1)
        
        # A00 * B0
        U[a00, col] = 1; V[b0, col] = 1; W[c0, col] += 1; col += 1
        # A01 * B1
        U[a01, col] = 1; V[b1, col] = 1; W[c0, col] += 1; col += 1
        # A10 * B0
        U[a10, col] = 1; V[b0, col] = 1; W[c1, col] += 1; col += 1
        # A11 * B1
        U[a11, col] = 1; V[b1, col] = 1; W[c1, col] += 1; col += 1

    # Block 1: A_L (cols 0,1) @ B_T (rows 0,1, cols 0,1)
    add_strassen_block(0, 0, 0, 0)
    
    # Block 2: A_L (cols 0,1) @ B_T (rows 0,1, cols 2,3)
    add_strassen_block(0, 0, 0, 2)
    
    # Block 3: A_L (cols 0,1) @ B_T (rows 0,1, col 4)
    add_direct_2x2x1(0, 0, 0, 4)
    
    # Block 4: A_R (cols 2,3) @ B_B (rows 2,3, cols 0,1)
    add_strassen_block(0, 2, 2, 0)
    
    # Block 5: A_R (cols 2,3) @ B_B (rows 2,3, cols 2,3)
    add_strassen_block(0, 2, 2, 2)
    
    # Block 6: A_R (cols 2,3) @ B_B (rows 2,3, col 4)
    add_direct_2x2x1(0, 2, 2, 4)
    
    # Check time
    if time.time() - start_time > timeout:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Prune zero columns
    mask = ~(np.all(U == 0, axis=0) | np.all(V == 0, axis=0) | np.all(W == 0, axis=0))
    U = U[:, mask]
    V = V[:, mask]
    W = W[:, mask]
    
    # Convert to lists of lists for the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
