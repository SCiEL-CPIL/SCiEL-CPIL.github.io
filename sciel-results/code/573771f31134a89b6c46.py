import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def _generate_strassen_2x2x2():
    """
    Programmatically generates the Strassen decomposition for 2x2 * 2x2.
    Returns U, V, W of shape (4, 7).
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Helper to add to matrices
    def add_u(i, j, r, val): U[i*2+j, r] += val
    def add_v(j, k, r, val): V[j*2+k, r] += val
    def add_w(k, i, r, val): W[k*2+i, r] += val
    
    # M1
    add_u(0,0,0,1); add_u(1,1,0,1)
    add_v(0,0,0,1); add_v(1,1,0,1)
    add_w(0,0,0,1); add_w(1,1,0,1)
    
    # M2
    add_u(1,0,1,1); add_u(1,1,1,1)
    add_v(0,0,1,1)
    add_w(0,1,1,1) # C10 = M2 + M4. Index in W: k=0, i=1 -> W[0*2+1] = W[1]
    # Wait, W index is k*n+i. n=2.
    # C10 is i=1, k=0. W[0*2+1] = W[1].
    
    # M3
    add_u(0,0,2,1)
    add_v(0,1,2,1); add_v(1,1,2,-1)
    add_w(1,0,2,1) # C01 is i=0, k=1. W[1*2+0] = W[2].
    
    # M4
    add_u(1,1,3,1)
    add_v(1,0,3,1); add_v(0,0,3,-1)
    add_w(0,0,3,1) # C00
    add_w(0,1,3,1) # C10
    
    # M5
    add_u(0,0,4,1); add_u(0,1,4,1)
    add_v(1,1,4,1)
    add_w(0,0,4,-1) # C00
    add_w(1,0,4,1)  # C01
    
    # M6
    add_u(1,0,5,1); add_u(0,0,5,-1)
    add_v(0,0,5,1); add_v(0,1,5,1)
    add_w(1,1,5,1)  # C11 is i=1, k=1. W[1*2+1] = W[3].
    
    # M7
    add_u(0,1,6,1); add_u(1,1,6,-1)
    add_v(1,0,6,1); add_v(1,1,6,1)
    add_w(0,0,6,1)  # C00
    add_w(1,1,6,1)  # C11
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Fast verification using numpy tensor contraction."""
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # We need to check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
    # This is equivalent to checking if the tensor T[i,j,k] is 1.
    # We can do this by iterating over r and adding outer products.
    R = U.shape[1]
    # Create a 3D array for the tensor
    # T[i, j, k]
    T = np.zeros((n, m, p), dtype=int)
    
    # This loop might be slow if R is large, but R=54 is small.
    for r in range(R):
        u = U[:, r] # (n*m)
        v = V[:, r] # (m*p)
        w = W[:, r] # (p*n)
        
        # u is vector of length n*m. Reshape to (n, m)
        u_mat = u.reshape((n, m))
        # v is vector of length m*p. Reshape to (m, p)
        v_mat = v.reshape((m, p))
        # w is vector of length p*n. Reshape to (p, n) -> Transpose to (n, p)?
        # W[k*n+i]. So w_mat[k, i].
        # We want to add u_mat[i,j] * v_mat[j,k] * w_mat[k,i] to T[i,j,k].
        
        # This is a tensor contraction.
        # For each r, we add a rank-1 tensor.
        # We can do this with einsum or loops.
        # Since n,m,p are small, loops are fine.
        for i in range(n):
            for j in range(m):
                u_val = u_mat[i, j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = v_mat[j, k]
                    if v_val == 0: continue
                    w_val = w.reshape((p, n))[k, i]
                    if w_val == 0: continue
                    T[i, j, k] += u_val * v_val * w_val
    
    return np.all(T == 1)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Strategy: Block decomposition
    # Split m=5 into 2, 2, 1
    # Split p=6 into 2, 2, 2 for the 2x2 blocks
    
    rank_total = 54
    U = np.zeros((n * m, rank_total), dtype=int)
    V = np.zeros((m * p, rank_total), dtype=int)
    W = np.zeros((p * n, rank_total), dtype=int)
    
    U_s, V_s, W_s = _generate_strassen_2x2x2()
    
    col = 0
    
    # Helper to place a 2x2x2 block
    def place_strassen(j_offset, k_offset):
        nonlocal col
        for r in range(7):
            # U part
            for i in range(2):
                for j_local in range(2):
                    val = U_s[i*2+j_local, r]
                    if val != 0:
                        U[i*m + (j_offset + j_local), col] = val
            # V part
            for j_local in range(2):
                for k_local in range(2):
                    val = V_s[j_local*2+k_local, r]
                    if val != 0:
                        V[(j_offset + j_local)*p + (k_offset + k_local), col] = val
            # W part
            for k_local in range(2):
                for i in range(2):
                    val = W_s[k_local*2+i, r]
                    if val != 0:
                        W[(k_offset + k_local)*n + i, col] = val
            col += 1
    
    # Group 1: j=0,1
    for k_offset in [0, 2, 4]:
        place_strassen(0, k_offset)
        
    # Group 2: j=2,3
    for k_offset in [0, 2, 4]:
        place_strassen(2, k_offset)
        
    # Group 3: j=4
    # A[i, 4] * B[4, k]
    for i in range(2):
        for k in range(6):
            U[i*m + 4, col] = 1
            V[4*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
            
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
