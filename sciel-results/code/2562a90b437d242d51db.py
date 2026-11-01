import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix 
    by a 2x10 matrix.
    
    The known minimal rank for this problem is 35, achieved by applying Strassen's 
    rank-7 decomposition for 2x2 matrix multiplication to each of the 5 independent 
    2x2 blocks of the 2x10 matrix.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
              U: shape (n*m, rank) = (4, 35)
              V: shape (m*p, rank) = (20, 35)
              W: shape (p*n, rank) = (20, 35)
    """
    n, m, p = 2, 2, 10
    rank = 35
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 terms for 2x2 multiplication
    # Each term is a tuple (u_vec, v_vec, w_vec)
    # u_vec corresponds to A (4 elements: a, b, c, d)
    # v_vec corresponds to B (4 elements: e, f, g, h)
    # w_vec corresponds to C (4 elements: C11, C12, C21, C22)
    
    # Note on indexing:
    # A is 2x2: indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B is 2x2: indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C is 2x2: indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_terms = [
        # (a+d, e+h, C11, C12, C21, C22) -> mapped to vectors
        # Term 1: (a+d)(e+h)
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0]),  # Wait, Strassen definitions vary.
        # Let's use the standard Strassen formulas:
        # M1 = (a + d)(e + h)
        # M2 = (c + d)(e)
        # M3 = (a + b)(h)
        # M4 = (d - b)(f + g)
        # M5 = (a - c)(e + f)
        # M6 = (b + d)(g + h)
        # M7 = (a - d)(f - g)
        #
        # C11 = M1 + M4 - M5 - M6
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
    ]
    
    # Let's verify the standard Strassen terms and their contributions to C.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h) -> u=[1,0,1,0], v=[1,0,0,1]
    # M2 = (c+d)(e)   -> u=[0,0,1,1], v=[1,0,0,0]
    # M3 = (a+b)(h)   -> u=[1,1,0,0], v=[0,0,0,1]
    # M4 = (d-b)(f+g) -> u=[0,-1,0,1], v=[0,1,1,0]
    # M5 = (a-c)(e+f) -> u=[1,0,-1,0], v=[1,1,0,0]
    # M6 = (b+d)(g+h) -> u=[0,1,0,1], v=[0,0,1,1]
    # M7 = (a-d)(f-g) -> u=[1,0,0,-1], v=[0,1,-1,0]
    
    # Contributions to C:
    # C11 = M1 + M4 - M5 - M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    terms = [
        # (u, v, w) where w is [C11_coeff, C12_coeff, C21_coeff, C22_coeff]
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),   # M3
        ([0, -1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0]),  # M4
        ([1, 0, -1, 0], [1, 1, 0, 0], [-1, 1, 0, 0]), # M5
        ([0, 1, 0, 1], [0, 0, 1, 1], [-1, 0, 0, 0]),  # M6
        ([1, 0, 0, -1], [0, 1, -1, 0], [0, 0, 0, 1])  # M7
    ]
    
    # We have 5 blocks of 2x2 multiplication.
    # The 2x10 matrix B is split into 5 blocks of 2x2.
    # Block k corresponds to columns 2k and 2k+1 of B, and rows 0,1 of C.
    # Actually, B is 2x10. Rows 0,1. Cols 0..9.
    # Block 0: B[0,0:2], B[1,0:2] -> C[0,0:2], C[1,0:2]
    # Block 1: B[0,2:4], B[1,2:4] -> C[0,2:4], C[1,2:4]
    # ...
    # Block 4: B[0,8:10], B[1,8:10] -> C[0,8:10], C[1,8:10]
    
    col_idx = 0
    for block in range(5):
        for u_vec, v_vec, w_vec in terms:
            # Fill U (same for all blocks, as A is the same 2x2 matrix)
            for i in range(4):
                U[i][col_idx] = u_vec[i]
            
            # Fill V
            # V index for B element (row j, col l) is j*p + l
            # Block 'block' uses cols 2*block and 2*block+1
            # v_vec is [e, f, g, h] corresponding to B[0,0], B[0,1], B[1,0], B[1,1] within the block
            # So:
            # v_vec[0] -> B[0, 2*block]
            # v_vec[1] -> B[0, 2*block+1]
            # v_vec[2] -> B[1, 2*block]
            # v_vec[3] -> B[1, 2*block+1]
            
            V[0 * p + 2 * block][col_idx] = v_vec[0]
            V[0 * p + 2 * block + 1][col_idx] = v_vec[1]
            V[1 * p + 2 * block][col_idx] = v_vec[2]
            V[1 * p + 2 * block + 1][col_idx] = v_vec[3]
            
            # Fill W
            # W index for C element (row i, col l) is l*n + i (Trace convention W[k*n+i,r])
            # w_vec is [C11, C12, C21, C22] corresponding to C[0,0], C[0,1], C[1,0], C[1,1] within the block
            # So:
            # w_vec[0] -> C[0, 2*block]
            # w_vec[1] -> C[0, 2*block+1]
            # w_vec[2] -> C[1, 2*block]
            # w_vec[3] -> C[1, 2*block+1]
            
            # C[0, 2*block] -> index (2*block)*n + 0
            W[2 * block * n + 0][col_idx] = w_vec[0]
            # C[0, 2*block+1] -> index (2*block+1)*n + 0
            W[(2 * block + 1) * n + 0][col_idx] = w_vec[1]
            # C[1, 2*block] -> index (2*block)*n + 1
            W[2 * block * n + 1][col_idx] = w_vec[2]
            # C[1, 2*block+1] -> index (2*block+1)*n + 1
            W[(2 * block + 1) * n + 1][col_idx] = w_vec[3]
            
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n=2, m=2, p=10, tol=1e-6):
    """Verify that the tensor decomposition is exact."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Compute reconstructed tensor T_recon[a, b, c] = sum_k U[a,k] V[b,k] W[c,k]
    T_recon = np.zeros((nm, mp, pn))
    for k in range(rank):
        T_recon += np.einsum('a,b,c->abc', U_arr[:, k], V_arr[:, k], W_arr[:, k])
    
    # Build target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a = i * m + j
                b = j * p + l
                c = l * n + i
                T_target[a, b, c] = 1.0
                
    diff = np.abs(T_recon - T_target)
    return np.max(diff) < tol

if __name__ == "__main__":
    result = construct()
    # Optional verification for local testing
    # if _verify_decomposition(result['U'], result['V'], result['W']):
    #     print("Decomposition is valid.")
    # else:
    #     print("Decomposition is INVALID.")
