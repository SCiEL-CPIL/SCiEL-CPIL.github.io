"""Matrix multiplication 2x4 by 4x8 tensor decomposition.

We construct a rank-56 exact decomposition by applying Strassen's
rank-7 algorithm for 2x2x2 to each of the 8 independent (2 intermediate
rows x 4 output column blocks) sub-problems.  The output matrix W follows
the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    n, m, p = DIMENSIONS
    
    # Define Strassen's 7 products for 2x2 * 2x2
    # U: 2x2, V: 2x2, W: 2x2 (output is 2x2)
    # We represent coefficients as sparse dicts or dense arrays.
    # Here we use dense arrays for simplicity in generation, then convert.
    
    # Strassen products (u, v, w) where C = sum u_k * v_k * w_k
    # u_k, v_k, w_k are 2x2 matrices of coefficients.
    strassen = [
        # 0: (A00+A11)(B00+B11)
        (np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]])),
        # 1: (A10+A11)B00
        (np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]]), np.array([[0,1],[0,-1]])),
        # 2: A00(B01-B11)
        (np.array([[1,0],[0,0]]), np.array([[0,1],[0,-1]]), np.array([[1,0],[0,1]])),
        # 3: A11(B10-B00)
        (np.array([[0,0],[0,1]]), np.array([[0,-1],[1,0]]), np.array([[1,0],[0,1]])),
        # 4: (A00+A01)B11
        (np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]]), np.array([[-1,0],[0,1]])),
        # 5: (A10-A00)(B00+B01)
        (np.array([[-1,0],[1,0]]), np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]])),
        # 6: (A01-A11)(B10+B11)
        (np.array([[0,1],[0,-1]]), np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]])),
    ]
    
    rank = 56
    # Initialize U, V, W as numpy arrays for faster construction
    U_np = np.zeros((n*m, rank), dtype=int)
    V_np = np.zeros((m*p, rank), dtype=int)
    W_np = np.zeros((p*n, rank), dtype=int)
    
    col_idx = 0
    # 2 blocks of A rows (size 2)
    # 4 blocks of B cols (size 2)
    for ar in range(2):
        for qc in range(4):
            for u_mat, v_mat, w_mat in strassen:
                # u_mat is 2x2: indices (li, lj)
                # v_mat is 2x2: indices (lj, lk)
                # w_mat is 2x2: indices (lk, li) -> W[gk, gi] where gk=2*qc+lk, gi=li
                
                for li in range(2):
                    for lj in range(2):
                        c = u_mat[li, lj]
                        if c != 0:
                            gi = li
                            gj = 2*ar + lj
                            U_np[gi*m + gj, col_idx] += c
                            
                for lj in range(2):
                    for lk in range(2):
                        c = v_mat[lj, lk]
                        if c != 0:
                            gj = 2*ar + lj
                            gk = 2*qc + lk
                            V_np[gj*p + gk, col_idx] += c
                            
                for lk in range(2):
                    for li in range(2):
                        c = w_mat[lk, li]
                        if c != 0:
                            gk = 2*qc + lk
                            gi = li
                            W_np[gk*n + gi, col_idx] += c
                            
                col_idx += 1
                
    # Convert to lists of lists as required
    U = U_np.tolist()
    V = V_np.tolist()
    W = W_np.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
