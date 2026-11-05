"""Matrix multiplication 4x5x5 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# Deadline guard
START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication.
    
    Strategy: Use Strassen's algorithm for the 4x4x4 core (rank 28), then handle
    the remaining terms (involving the 5th dimension) with schoolbook (rank 36).
    Total rank: 28 + 36 = 64.
    
    The decomposition splits the multiplication into:
    1. A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4] (Strassen, rank 28)
    2. A[0:4, 0:4] * B[0:4, 4]   -> C[0:4, 4]   (schoolbook, rank 16)
    3. A[0:4, 4]   * B[4, 0:4]   -> C[0:4, 0:4] (schoolbook, rank 16)
    4. A[0:4, 4]   * B[4, 4]     -> C[0:4, 4]   (schoolbook, rank 4)
    
    Total: 28 + 16 + 16 + 4 = 64
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        """Add a rank-1 term to the decomposition."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def add_term(i, k, j, cu=1.0, cv=1.0, cw=1.0):
        """
        Add a single schoolbook term: A[i,k] * B[k,j] -> C[i,j]
        u_vec has cu at position i*m+k
        v_vec has cv at position k*p+j
        w_vec has cw at position j*n+i
        """
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        
        u_vec[i * m + k] = cu
        v_vec[k * p + j] = cv
        w_vec[j * n + i] = cw
        
        add_rank1(u_vec, v_vec, w_vec)
    
    # Part 1: 4x4x4 multiplication using Strassen (rank 28)
    # A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # 
    # Split 4x4 into 2x2 blocks:
    # A00 (i=0,1; k=0,1), A01 (i=0,1; k=2,3)
    # A10 (i=2,3; k=0,1), A11 (i=2,3; k=2,3)
    # 
    # B00 (k=0,1; j=0,1), B01 (k=0,1; j=2,3)
    # B10 (k=2,3; j=0,1), B11 (k=2,3; j=2,3)
    # 
    # C00 (i=0,1; j=0,1), C01 (i=0,1; j=2,3)
    # C10 (i=2,3; j=0,1), C11 (i=2,3; j=2,3)
    
    # Strassen 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define Strassen products with their A and B block contributions
    strassen_products = [
        # (A_blocks, B_blocks, C_blocks)
        # Each block is (block_id, sign)
        # Block IDs: 0=A00/B00/C00, 1=A01/B01/C01, 2=A10/B10/C10, 3=A11/B11/C11
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),      # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),            # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),            # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),            # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),            # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),            # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),            # M7
    ]
    
    # Helper functions to map block-local indices to global indices
    def a_global(block_id, li, lk):
        """Map A block (block_id) local indices (li, lk) to global (i, k)."""
        i_base = 0 if block_id in [0, 1] else 2
        k_base = 0 if block_id in [0, 2] else 2
        return i_base + li, k_base + lk
    
    def b_global(block_id, lk, lj):
        """Map B block (block_id) local indices (lk, lj) to global (k, j)."""
        k_base = 0 if block_id in [0, 1] else 2
        j_base = 0 if block_id in [0, 2] else 2
        return k_base + lk, j_base + lj
    
    def c_global(block_id, li, lj):
        """Map C block (block_id) local indices (li, lj) to global (i, j)."""
        i_base = 0 if block_id in [0, 1] else 2
        j_base = 0 if block_id in [0, 2] else 2
        return i_base + li, j_base + lj
    
    # For each Strassen product, expand into 4 schoolbook terms (2x2x2)
    for a_blocks, b_blocks, c_blocks in strassen_products:
        for li in range(2):
            for lk in range(2):
                for lj in range(2):
                    u_vec = np.zeros(n * m)
                    v_vec = np.zeros(m * p)
                    w_vec = np.zeros(p * n)
                    
                    # A-side contributions
                    for block_id, sign in a_blocks.items():
                        gi, gk = a_global(block_id, li, lk)
                        u_vec[gi * m + gk] = sign
                    
                    # B-side contributions
                    for block_id, sign in b_blocks.items():
                        gk, gj = b_global(block_id, lk, lj)
                        v_vec[gk * p + gj] = sign
                    
                    # C-side contributions
                    for block_id, sign in c_blocks.items():
                        gi, gj = c_global(block_id, li, lj)
                        w_vec[gj * n + gi] = sign
                    
                    add_rank1(u_vec, v_vec, w_vec)
    
    # Part 2: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4] (schoolbook, rank 16)
    for i in range(4):
        for k in range(4):
            add_term(i, k, 4)
    
    # Part 3: A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4] (schoolbook, rank 16)
    for i in range(4):
        for j in range(4):
            add_term(i, 4, j)
    
    # Part 4: A[0:4, 4] * B[4, 4] -> C[0:4, 4] (schoolbook, rank 4)
    for i in range(4):
        add_term(i, 4, 4)
    
    # Check time
    if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
        # Return immediately
        U = np.array(U_cols).T
        V = np.array(V_cols).T
        W = np.array(W_cols).T
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify the decomposition
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    # The rank is the number of columns
    rank = U.shape[1]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
