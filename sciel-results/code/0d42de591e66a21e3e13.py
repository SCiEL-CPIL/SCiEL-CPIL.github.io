import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2_terms():
    """
    Returns U, V, W for a single 2x2x2 Strassen decomposition.
    U: (4, 7) - indices 0..3 correspond to a,b,c,d
    V: (4, 7) - indices 0..3 correspond to e,f,g,h
    W: (4, 7) - indices 0..3 correspond to z00, z01, z10, z11
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # m2 = (c+d)(e)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # m5 = (a+b)(h)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W coefficients
    # z00 = m1 - m4 + m5 + m6
    W[0, 0] = 1; W[0, 3] = -1; W[0, 4] = 1; W[0, 5] = 1
    # z01 = m3 + m5
    W[1, 2] = 1; W[1, 4] = 1
    # z10 = m2 + m4
    W[2, 1] = 1; W[2, 3] = 1
    # z11 = m1 - m3 + m7
    W[3, 0] = 1; W[3, 2] = -1; W[3, 6] = 1
    
    return U, V, W

def _verify_strassen():
    """Verify the Strassen decomposition."""
    U, V, W = _strassen_2x2x2_terms()
    # Test with random 2x2 matrices
    for _ in range(10):
        A = np.random.randint(-10, 10, (2, 2))
        B = np.random.randint(-10, 10, (2, 2))
        C_expected = A @ B
        
        # Reshape A and B to vectors
        A_vec = A.flatten() # a,b,c,d
        B_vec = B.flatten() # e,f,g,h
        
        # Compute products
        m = np.zeros(7)
        for r in range(7):
            m[r] = np.dot(U[:, r], A_vec) * np.dot(V[:, r], B_vec)
        
        C_computed = np.zeros(4)
        for z in range(4):
            C_computed[z] = np.dot(W[z, :], m)
        
        C_computed = C_computed.reshape((2, 2))
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _strassen_block_decomposition(n, m, p):
    """
    Construct the full decomposition using 6 Strassen blocks.
    """
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_loc, V_loc, W_loc = _strassen_2x2x2_terms()
    
    # Define the 6 blocks
    # Block 0: A1 B11 -> C[0:2, 0:2]
    # A1: A[0:2, 0:2] -> indices 0..3
    # B11: B[0:2, 0:2] -> indices 0..5 (j*6+k, j=0,1; k=0,1)
    # C: C[0:2, 0:2] -> indices 0..3 (k*2+i, k=0,1; i=0,1)
    
    blocks = [
        # (a_start, b_start, c_start)
        # a_start: global index of A[0,0] in the block
        # b_start: global index of B[0,0] in the block
        # c_start: global index of C[0,0] in the block
        # We need to map local indices 0,1,2,3 to global indices.
        
        # Block 0: A1 B11
        # A local (i,j) -> global i*4 + j. i in 0,1; j in 0,1.
        # B local (j,k) -> global j*6 + k. j in 0,1; k in 0,1.
        # C local (i,k) -> global k*2 + i. k in 0,1; i in 0,1.
        
        # Block 1: A1 B12
        # B12: B[0:2, 2:4]. j in 0,1; k in 2,3.
        
        # Block 2: A1 B13
        # B13: B[0:2, 4:6]. j in 0,1; k in 4,5.
        
        # Block 3: A2 B21
        # A2: A[0:2, 2:4]. j in 2,3.
        # B21: B[2:4, 0:2]. j in 2,3; k in 0,1.
        
        # Block 4: A2 B22
        # B22: B[2:4, 2:4]. j in 2,3; k in 2,3.
        
        # Block 5: A2 B23
        # B23: B[2:4, 4:6]. j in 2,3; k in 4,5.
    ]
    
    # Helper to map local indices to global indices
    def map_A(i, j, block_id):
        if block_id in [0, 1, 2]: # A1
            return i * 4 + j
        else: # A2
            return i * 4 + (2 + j)
            
    def map_B(j, k, block_id):
        if block_id in [0, 1, 2]: # B_top
            if block_id == 0:
                return j * 6 + k
            elif block_id == 1:
                return j * 6 + (2 + k)
            else:
                return j * 6 + (4 + k)
        else: # B_bottom
            if block_id == 3:
                return (2 + j) * 6 + k
            elif block_id == 4:
                return (2 + j) * 6 + (2 + k)
            else:
                return (2 + j) * 6 + (4 + k)
                
    def map_C(i, k, block_id):
        if block_id in [0, 1, 2]: # C_left (cols 0-1? No, C is 2x6)
            # Block 0,1,2 correspond to B11, B12, B13.
            # B11 -> C cols 0,1
            # B12 -> C cols 2,3
            # B13 -> C cols 4,5
            if block_id == 0:
                return k * 2 + i
            elif block_id == 1:
                return (2 + k) * 2 + i
            else:
                return (4 + k) * 2 + i
        else: # C_right (same cols, but from A2)
            # Block 3,4,5 correspond to B21, B22, B23.
            # B21 -> C cols 0,1
            # B22 -> C cols 2,3
            # B23 -> C cols 4,5
            if block_id == 3:
                return k * 2 + i
            elif block_id == 4:
                return (2 + k) * 2 + i
            else:
                return (4 + k) * 2 + i

    for block_id in range(6):
        for r in range(7):
            global_r = block_id * 7 + r
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    local_idx = i * 2 + j # a,b,c,d order
                    global_idx = map_A(i, j, block_id)
                    U[global_idx, global_r] = U_loc[local_idx, r]
                    
            # Fill V
            for j in range(2):
                for k in range(2):
                    local_idx = j * 2 + k # e,f,g,h order
                    global_idx = map_B(j, k, block_id)
                    V[global_idx, global_r] = V_loc[local_idx, r]
                    
            # Fill W
            for i in range(2):
                for k in range(2):
                    local_idx = k * 2 + i # z00, z01, z10, z11 order
                    global_idx = map_C(i, k, block_id)
                    W[global_idx, global_r] = W_loc[local_idx, r]

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, max_samples=50):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    """
    rank = U.shape[1]
    for _ in range(max_samples):
        A = np.random.randint(-10, 10, (n, m))
        B = np.random.randint(-10, 10, (m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            alpha = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    alpha[i] += U[i*m + j, r] * A[i, j]
            
            beta = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    beta[k] += V[j*p + k, r] * B[j, k]
            
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += alpha[i] * beta[k] * W[k*n + i, r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Check if Strassen is valid
    if _verify_strassen():
        result = _strassen_block_decomposition(n, m, p)
        # Verify the full decomposition
        if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p, max_samples=20):
            return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
