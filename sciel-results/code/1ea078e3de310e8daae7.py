"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # We will build a decomposition of rank 71.
    # Strategy:
    # 1. Use Strassen (rank 7) for all 2x2x2 sub-blocks.
    # 2. Use schoolbook (rank 4) for 2x2x1 and 2x1x2 sub-blocks.
    # 3. Use schoolbook (rank 6) for 2x1x3 sub-blocks.
    # 4. Use schoolbook (rank 25) for the 1x5x5 slice.
    
    terms = []
    
    def add_term(u_sparse, v_sparse, w_sparse):
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        for idx, val in u_sparse.items():
            u[idx] = val
        for idx, val in v_sparse.items():
            v[idx] = val
        for idx, val in w_sparse.items():
            w[idx] = val
        terms.append((u, v, w))

    def strassen_2x2x2(i_base, j_base, k_base):
        """
        Adds 7 terms for a 2x2x2 block.
        i_base: starting row index for A/C
        j_base: starting col index for A / row index for B
        k_base: starting col index for B/C
        """
        # Local indices 0,1 map to global i_base+0, i_base+1 etc.
        i0, i1 = i_base, i_base + 1
        j0, j1 = j_base, j_base + 1
        k0, k1 = k_base, k_base + 1
        
        # M1 = (A11+A22)(B11+B22)
        u = {i0*m + j0: 1, i1*m + j1: 1}
        v = {j0*p + k0: 1, j1*p + k1: 1}
        w = {k0*n + i0: 1, k1*n + i0: 1, k0*n + i1: 1, k1*n + i1: 1}
        add_term(u, v, w)
        
        # M2 = (A21+A22)B11
        u = {i1*m + j0: 1, i1*m + j1: 1}
        v = {j0*p + k0: 1}
        w = {k0*n + i1: 1, k1*n + i1: 1}
        add_term(u, v, w)
        
        # M3 = A11(B12-B22)
        u = {i0*m + j0: 1}
        v = {j0*p + k1: 1, j1*p + k1: -1}
        w = {k1*n + i0: 1, k1*n + i1: 1}
        add_term(u, v, w)
        
        # M4 = A22(B21-B11)
        u = {i1*m + j1: 1}
        v = {j1*p + k0: 1, j0*p + k0: -1}
        w = {k0*n + i0: 1, k1*n + i1: 1}
        add_term(u, v, w)
        
        # M5 = (A11+A12)B22
        u = {i0*m + j0: 1, i0*m + j1: 1}
        v = {j1*p + k1: 1}
        w = {k0*n + i0: -1, k1*n + i0: 1}
        add_term(u, v, w)
        
        # M6 = (A21-A11)(B11+B12)
        u = {i1*m + j0: 1, i0*m + j0: -1}
        v = {j0*p + k0: 1, j0*p + k1: 1}
        w = {k1*n + i1: 1}
        add_term(u, v, w)
        
        # M7 = (A12-A22)(B21+B22)
        u = {i0*m + j1: 1, i1*m + j1: -1}
        v = {j1*p + k0: 1, j1*p + k1: 1}
        w = {k0*n + i0: 1, k0*n + i1: 1}
        add_term(u, v, w)

    def schoolbook_2x2x1(i_base, j_base, k_base):
        """
        Adds 4 terms for A[2x2] * B[2x1] -> C[2x1].
        i_base, j_base, k_base are global indices.
        """
        i0, i1 = i_base, i_base + 1
        j0, j1 = j_base, j_base + 1
        k0 = k_base
        
        # C[i0, k0] += A[i0, j0] * B[j0, k0]
        u = {i0*m + j0: 1}
        v = {j0*p + k0: 1}
        w = {k0*n + i0: 1}
        add_term(u, v, w)
        
        # C[i0, k0] += A[i0, j1] * B[j1, k0]
        u = {i0*m + j1: 1}
        v = {j1*p + k0: 1}
        w = {k0*n + i0: 1}
        add_term(u, v, w)
        
        # C[i1, k0] += A[i1, j0] * B[j0, k0]
        u = {i1*m + j0: 1}
        v = {j0*p + k0: 1}
        w = {k0*n + i1: 1}
        add_term(u, v, w)
        
        # C[i1, k0] += A[i1, j1] * B[j1, k0]
        u = {i1*m + j1: 1}
        v = {j1*p + k0: 1}
        w = {k0*n + i1: 1}
        add_term(u, v, w)

    def schoolbook_2x1x2(i_base, j_base, k_base):
        """
        Adds 4 terms for A[2x1] * B[1x2] -> C[2x2].
        i_base, j_base, k_base are global indices.
        """
        i0, i1 = i_base, i_base + 1
        j0 = j_base
        k0, k1 = k_base, k_base + 1
        
        # C[i0, k0] += A[i0, j0] * B[j0, k0]
        u = {i0*m + j0: 1}
        v = {j0*p + k0: 1}
        w = {k0*n + i0: 1}
        add_term(u, v, w)
        
        # C[i0, k1] += A[i0, j0] * B[j0, k1]
        u = {i0*m + j0: 1}
        v = {j0*p + k1: 1}
        w = {k1*n + i0: 1}
        add_term(u, v, w)
        
        # C[i1, k0] += A[i1, j0] * B[j0, k0]
        u = {i1*m + j0: 1}
        v = {j0*p + k0: 1}
        w = {k0*n + i1: 1}
        add_term(u, v, w)
        
        # C[i1, k1] += A[i1, j0] * B[j0, k1]
        u = {i1*m + j0: 1}
        v = {j0*p + k1: 1}
        w = {k1*n + i1: 1}
        add_term(u, v, w)

    def schoolbook_2x1x3(i_base, j_base, k_base):
        """
        Adds 6 terms for A[2x1] * B[1x3] -> C[2x3].
        """
        i0, i1 = i_base, i_base + 1
        j0 = j_base
        k0, k1, k2 = k_base, k_base + 1, k_base + 2
        
        for i in [i0, i1]:
            for k in [k0, k1, k2]:
                u = {i*m + j0: 1}
                v = {j0*p + k: 1}
                w = {k*n + i: 1}
                add_term(u, v, w)

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    strassen_2x2x2(0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # Sub 2a: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    strassen_2x2x2(0, 0, 2)
    # Sub 2b: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    schoolbook_2x2x1(0, 0, 4)
    
    # Block 3: A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
    # Sub 3a: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    strassen_2x2x2(0, 2, 0)
    # Sub 3b: A[0:2, 4:5] * B[4:5, 0:2] -> C[0:2, 0:2]
    schoolbook_2x1x2(0, 4, 0)
    
    # Block 4: A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5]
    # Sub 4a-i: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    strassen_2x2x2(0, 2, 2)
    # Sub 4a-ii: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
    schoolbook_2x2x1(0, 2, 4)
    # Sub 4b: A[0:2, 4:5] * B[4:5, 2:5] -> C[0:2, 2:5]
    schoolbook_2x1x3(0, 4, 2)
    
    # Block 5: A[2:3, 0:5] * B[0:5, 0:5] -> C[2:3, 0:5]
    # Schoolbook 1x5x5
    for j in range(5):
        for k in range(5):
            u = {2*m + j: 1}
            v = {j*p + k: 1}
            w = {k*n + 2: 1}
            add_term(u, v, w)
            
    rank = len(terms)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u, v, w) in enumerate(terms):
        U[:, c] = u
        V[:, c] = v
        W[:, c] = w
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
