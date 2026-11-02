"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 130.0  # safety margin before 150s kill

def schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def verify_decomposition(U, V, W, rank):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for c in range(rank):
        u_col = U[:, c].reshape(n, m)
        v_col = V[:, c].reshape(m, p)
        w_val = W[:, c]
        
        for i in range(n):
            for k in range(p):
                u_dot = np.dot(u_col[i, :], A[i, :])
                v_dot = np.dot(v_col[:, k], B[:, k])
                C_computed[i, k] += w_val[k * n + i] * u_dot * v_dot
    
    error = np.max(np.abs(C_computed - C_expected))
    return error

def construct():
    """Construct the matrix multiplication decomposition using verified Strassen blocks."""
    n, m, p = DIMENSIONS
    U, V, W, rank = schoolbook()
    
    # Apply Strassen to multiple 2x2x2 blocks
    # For 3x4x8, we can apply Strassen to:
    # Block 1: i={0,1}, j={0,1}, k={0,1}
    # Block 2: i={0,1}, j={2,3}, k={0,1}
    # Block 3: i={0,1}, j={0,1}, k={2,3}
    # Block 4: i={0,1}, j={2,3}, k={2,3}
    # Block 5: i={0,1}, j={0,1}, k={4,5}
    # Block 6: i={0,1}, j={2,3}, k={4,5}
    # Block 7: i={0,1}, j={0,1}, k={6,7}
    # Block 8: i={0,1}, j={2,3}, k={6,7}
    # That's 8 blocks of 2x2x2, each saving 1 multiplication: 96 - 8 = 88
    
    # We need to apply Strassen to non-overlapping 2x2x2 blocks.
    # The 2x2x2 blocks use specific (i,j,k) triples.
    # In schoolbook, column for (i,j,k) is at index i*m*p + j*p + k.
    
    # Let's apply Strassen to blocks that don't overlap in their (i,j,k) tuples.
    # Block (i0,i1, j0,j1, k0,k1) uses columns:
    # (i0,j0,k0), (i0,j0,k1), (i0,j1,k0), (i0,j1,k1),
    # (i1,j0,k0), (i1,j0,k1), (i1,j1,k0), (i1,j1,k1)
    
    # For i={0,1}, j={0,1}, k={0,1}: columns 0,1,8,9,32,33,40,41
    # For i={0,1}, j={2,3}, k={0,1}: columns 2,3,10,11,34,35,42,43
    # For i={0,1}, j={0,1}, k={2,3}: columns 4,5,12,13,36,37,44,45
    # For i={0,1}, j={2,3}, k={2,3}: columns 6,7,14,15,38,39,46,47
    # For i={0,1}, j={0,1}, k={4,5}: columns 16,17,24,25,48,49,56,57
    # For i={0,1}, j={2,3}, k={4,5}: columns 18,19,26,27,50,51,58,59
    # For i={0,1}, j={0,1}, k={6,7}: columns 20,21,28,29,52,53,60,61
    # For i={0,1}, j={2,3}, k={6,7}: columns 22,23,30,31,54,55,62,63
    
    # These 8 blocks are non-overlapping! Each saves 1 rank.
    # Total rank: 96 - 8 = 88
    
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 0, 1, 6, 7),
        (0, 1, 2, 3, 6, 7),
    ]
    
    for (i0, i1, j0, j1, k0, k1) in blocks:
        # Get the 8 column indices for this block
        cols = []
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk
                    col = i * m * p + j * p + k
                    cols.append(col)
        
        # Zero out the 8 original columns
        for c in cols:
            U[:, c] = 0
            V[:, c] = 0
            W[:, c] = 0
        
        # Create 7 Strassen tensors using cols[0:7], leave cols[7] as zero
        new_cols = cols[:7]
        
        # M1 = (A00+A11)(B00+B11)
        c = new_cols[0]
        U[i0*m+j0, c] += 1
        U[i1*m+j1, c] += 1
        V[j0*p+k0, c] += 1
        V[j1*p+k1, c] += 1
        W[k0*n+i0, c] += 1
        W[k1*n+i1, c] += 1
        
        # M2 = (A10+A11)B00
        c = new_cols[1]
        U[i1*m+j0, c] += 1
        U[i1*m+j1, c] += 1
        V[j0*p+k0, c] += 1
        W[k0*n+i1, c] += 1
        W[k1*n+i1, c] -= 1
        
        # M3 = A00(B10-B00)
        c = new_cols[2]
        U[i0*m+j0, c] += 1
        V[j1*p+k0, c] += 1
        V[j0*p+k0, c] -= 1
        W[k1*n+i0, c] += 1
        W[k1*n+i1, c] += 1
        
        # M4 = A11(B01-B11)
        c = new_cols[3]
        U[i1*m+j1, c] += 1
        V[j0*p+k1, c] += 1
        V[j1*p+k1, c] -= 1
        W[k0*n+i0, c] += 1
        W[k0*n+i1, c] += 1
        
        # M5 = (A00+A01)B11
        c = new_cols[4]
        U[i0*m+j0, c] += 1
        U[i0*m+j1, c] += 1
        V[j1*p+k1, c] += 1
        W[k0*n+i0, c] -= 1
        W[k1*n+i0, c] += 1
        
        # M6 = (A10-A00)(B00+B10)
        c = new_cols[5]
        U[i1*m+j0, c] += 1
        U[i0*m+j0, c] -= 1
        V[j0*p+k0, c] += 1
        V[j1*p+k0, c] += 1
        W[k1*n+i1, c] += 1
        
        # M7 = (A01-A11)(B01+B11)
        c = new_cols[6]
        U[i0*m+j1, c] += 1
        U[i1*m+j1, c] -= 1
        V[j0*p+k1, c] += 1
        V[j1*p+k1, c] += 1
        W[k0*n+i0, c] += 1
    
    # Now remove all zero columns (the 8th column from each block)
    # The free columns are cols[7] from each block
    free_cols = set()
    for (i0, i1, j0, j1, k0, k1) in blocks:
        i = i1
        j = j1
        k = k1
        col = i * m * p + j * p + k
        free_cols.add(col)
    
    # Build mask
    mask = np.ones(rank, dtype=bool)
    for fc in free_cols:
        mask[fc] = False
    
    U = U[:, mask]
    V = V[:, mask]
    W = W[:, mask]
    
    new_rank = rank - len(free_cols)
    
    # Verify
    error = verify_decomposition(U, V, W, new_rank)
    if error > 1e-9:
        # Fall back to schoolbook
        U, V, W, rank = schoolbook()
        new_rank = rank
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
