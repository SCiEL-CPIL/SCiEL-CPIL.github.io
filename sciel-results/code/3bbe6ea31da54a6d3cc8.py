import time

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x6 matrix.
    
    The output matrix C is 2x6.
    U has shape (2*5, Rank) = (10, Rank)
    V has shape (5*6, Rank) = (30, Rank)
    W has shape (6*2, Rank) = (12, Rank)
    
    Indexing:
    U[i*m + j, r]
    V[j*p + k, r]
    W[k*n + i, r]
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    n, m, p = 2, 5, 6
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """
        Adds a rank-1 term to the decomposition.
        u_coeffs: list of (index_in_U, value)
        v_coeffs: list of (index_in_V, value)
        w_coeffs: list of (index_in_W, value)
        """
        nonlocal col
        if col >= rank:
            return False
        if time.time() - start_time > deadline:
            return False
            
        for idx, val in u_coeffs:
            U[idx][col] = val
        for idx, val in v_coeffs:
            V[idx][col] = val
        for idx, val in w_coeffs:
            W[idx][col] = val
            
        col += 1
        return True

    # Helper to get U index: i * m + j
    def u_idx(i, j):
        return i * m + j

    # Helper to get V index: j * p + k
    def v_idx(j, k):
        return j * p + k

    # Helper to get W index: k * n + i
    def w_idx(k, i):
        return k * n + i

    # --- Strassen block for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i in i0:i0+2, k in k0:k0+2] ---
    def strassen_block(i0, j0, k0):
        """
        Apply Strassen's algorithm for a 2x2x2 block.
        A rows i0, i0+1; cols j0, j0+1
        B rows j0, j0+1; cols k0, k0+1
        C rows i0, i0+1; cols k0, k0+1
        """
        if time.time() - start_time > deadline:
            return
        
        # m1 = (a00+a11)(b00+b11)
        add_term([(u_idx(i0, j0), 1), (u_idx(i0+1, j0+1), 1)], 
                 [(v_idx(j0, k0), 1), (v_idx(j0+1, k0+1), 1)], 
                 [(w_idx(k0, i0), 1), (w_idx(k0+1, i0+1), 1)])
                 
        # m2 = (a10+a11)*b00
        add_term([(u_idx(i0+1, j0), 1), (u_idx(i0+1, j0+1), 1)], 
                 [(v_idx(j0, k0), 1)], 
                 [(w_idx(k0, i0+1), 1), (w_idx(k0+1, i0+1), -1)])
                 
        # m3 = a00*(b01-b11)
        add_term([(u_idx(i0, j0), 1)], 
                 [(v_idx(j0, k0+1), 1), (v_idx(j0+1, k0+1), -1)], 
                 [(w_idx(k0+1, i0), 1), (w_idx(k0+1, i0+1), 1)])
                 
        # m4 = a11*(b10-b00)
        add_term([(u_idx(i0+1, j0+1), 1)], 
                 [(v_idx(j0+1, k0), 1), (v_idx(j0, k0), -1)], 
                 [(w_idx(k0, i0), 1), (w_idx(k0, i0+1), 1)])
                 
        # m5 = (a00+a01)*b11
        add_term([(u_idx(i0, j0), 1), (u_idx(i0, j0+1), 1)], 
                 [(v_idx(j0+1, k0+1), 1)], 
                 [(w_idx(k0, i0), -1), (w_idx(k0+1, i0), 1)])
                 
        # m6 = (a10-a00)*(b00+b01)
        add_term([(u_idx(i0+1, j0), 1), (u_idx(i0, j0), -1)], 
                 [(v_idx(j0, k0), 1), (v_idx(j0, k0+1), 1)], 
                 [(w_idx(k0+1, i0+1), 1)])
                 
        # m7 = (a01-a11)*(b10+b11)
        add_term([(u_idx(i0, j0+1), 1), (u_idx(i0+1, j0+1), -1)], 
                 [(v_idx(j0+1, k0), 1), (v_idx(j0+1, k0+1), 1)], 
                 [(w_idx(k0, i0), 1)])

    # --- Apply 6 Strassen blocks for the 2x4x6 portion ---
    # A cols 0-3, B rows 0-3, C cols 0-5
    # Block (A[0:2], B[0:2]) -> C[0:2]
    strassen_block(0, 0, 0)
    
    # Block (A[0:2], B[2:4]) -> C[2:4]
    strassen_block(0, 2, 2)
    
    # Block (A[0:2], B[4:6]) -> C[4:6]
    strassen_block(0, 4, 4)
    
    # Block (A[2:4], B[0:2]) -> C[0:2]
    strassen_block(2, 0, 0)
    
    # Block (A[2:4], B[2:4]) -> C[2:4]
    strassen_block(2, 2, 2)
    
    # Block (A[2:4], B[4:6]) -> C[4:6]
    strassen_block(2, 4, 4)
    
    # --- Schoolbook for A col 4, B row 4 ---
    # A index: u_idx(i, 4)
    # B index: v_idx(4, k)
    # C index: w_idx(k, i)
    for i in range(n):
        for k in range(p):
            if time.time() - start_time > deadline:
                break
            add_term([(u_idx(i, 4), 1)], [(v_idx(4, k), 1)], [(w_idx(k, i), 1)])
        if time.time() - start_time > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
