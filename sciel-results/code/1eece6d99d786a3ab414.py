"""
Rewritten program to find a rank-minimal exact tensor decomposition for 2x3 by 3x5 matrix multiplication.
Uses Strassen blocks for 2x2x2 parts and schoolbook for the rest to achieve rank 28.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]
TIME_LIMIT = 150
SAFETY_MARGIN = 10

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    # U shape: (n*m, R) = (6, 28)
    # V shape: (m*p, R) = (15, 28)
    # W shape: (p*n, R) = (10, 28)
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_term(i, j, k, coeff):
        nonlocal col
        if col >= rank:
            return
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx, col] += coeff
        V[v_idx, col] += coeff
        W[w_idx, col] += coeff
        col += 1
        
    def add_strassen(i_off, j_off, k_off):
        # Adds 7 Strassen terms for the 2x2x2 block starting at offsets
        # i in {i_off, i_off+1}, j in {j_off, j_off+1}, k in {k_off, k_off+1}
        
        # M1 = (A00+A11)(B00+B11)
        # U: (0,0)+1, (1,1)+1
        # V: (0,0)+1, (1,1)+1
        # W: C00(+1), C11(+1) -> k=0,i=0 and k=1,i=1
        add_term(i_off, j_off, k_off, 1) # This is wrong. Strassen terms are not single i,j,k.
        # We need to add multiple entries to U, V, W for a single rank-1 term.
        # The `add_term` above assumes a single (i,j,k) tuple with coeff 1.
        # Strassen terms involve linear combinations.
        
        # Let's redefine the helper to handle linear combos.
        
    # Redefine helper to handle linear combinations for a single rank term
    def add_rank_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: list of (idx, coeff)
        v_coeffs: list of (idx, coeff)
        w_coeffs: list of (idx, coeff)
        """
        nonlocal col
        if col >= rank:
            return
        for idx, c in u_coeffs:
            U[idx, col] += c
        for idx, c in v_coeffs:
            V[idx, col] += c
        for idx, c in w_coeffs:
            W[idx, col] += c
        col += 1

    def strassen(i_off, j_off, k_off):
        # i_off, j_off, k_off are the starting indices for i, j, k in the 2x2x2 block
        # Local indices 0, 1 map to i_off, i_off+1, etc.
        
        i0, i1 = i_off, i_off + 1
        j0, j1 = j_off, j_off + 1
        k0, k1 = k_off, k_off + 1
        
        # 1. M1 = (A_{i0,j0} + A_{i1,j1}) (B_{j0,k0} + B_{j1,k1})
        u = [(i0*m+j0, 1), (i1*m+j1, 1)]
        v = [(j0*p+k0, 1), (j1*p+k1, 1)]
        w = [(k0*n+i0, 1), (k1*n+i1, 1)] # C_{i0,k0} += M1, C_{i1,k1} += M1
        add_rank_term(u, v, w)
        
        # 2. M2 = (A_{i1,j0} + A_{i1,j1}) B_{j0,k0}
        u = [(i1*m+j0, 1), (i1*m+j1, 1)]
        v = [(j0*p+k0, 1)]
        w = [(k0*n+i1, 1), (k1*n+i1, -1)] # C_{i1,k0} += M2, C_{i1,k1} -= M2
        add_rank_term(u, v, w)
        
        # 3. M3 = A_{i0,j0} (B_{j0,k1} - B_{j1,k1})
        u = [(i0*m+j0, 1)]
        v = [(j0*p+k1, 1), (j1*p+k1, -1)]
        w = [(k1*n+i0, 1), (k1*n+i1, 1)] # C_{i0,k1} += M3, C_{i1,k1} += M3
        add_rank_term(u, v, w)
        
        # 4. M4 = A_{i1,j1} (B_{j1,k0} - B_{j0,k0})
        u = [(i1*m+j1, 1)]
        v = [(j1*p+k0, 1), (j0*p+k0, -1)]
        w = [(k0*n+i0, -1), (k0*n+i1, 1)] # C_{i0,k0} -= M4, C_{i1,k0} += M4
        add_rank_term(u, v, w)
        
        # 5. M5 = (A_{i0,j0} + A_{i0,j1}) B_{j1,k1}
        u = [(i0*m+j0, 1), (i0*m+j1, 1)]
        v = [(j1*p+k1, 1)]
        w = [(k0*n+i0, -1), (k1*n+i0, 1)] # C_{i0,k0} -= M5, C_{i0,k1} += M5
        add_rank_term(u, v, w)
        
        # 6. M6 = (A_{i1,j0} - A_{i0,j0}) (B_{j0,k0} + B_{j1,k0})
        u = [(i1*m+j0, 1), (i0*m+j0, -1)]
        v = [(j0*p+k0, 1), (j1*p+k0, 1)]
        w = [(k1*n+i1, 1)] # C_{i1,k1} += M6
        add_rank_term(u, v, w)
        
        # 7. M7 = (A_{i0,j1} - A_{i1,j1}) (B_{j0,k1} + B_{j1,k1})
        u = [(i0*m+j1, 1), (i1*m+j1, -1)]
        v = [(j0*p+k1, 1), (j1*p+k1, 1)]
        w = [(k0*n+i0, 1)] # C_{i0,k0} += M7
        add_rank_term(u, v, w)

    def schoolbook(i_range, j_range, k_range):
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    u = [(i*m+j, 1)]
                    v = [(j*p+k, 1)]
                    w = [(k*n+i, 1)]
                    add_rank_term(u, v, w)

    # 1. Strassen for k in {0,1}, j in {0,1}
    strassen(0, 0, 0)
    
    # 2. Schoolbook for k in {0,1}, j=2
    schoolbook([0, 1], [2], [0, 1])
    
    # 3. Strassen for k in {2,3}, j in {0,1}
    strassen(0, 0, 2)
    
    # 4. Schoolbook for k=4, j in {0,1}
    schoolbook([0, 1], [0, 1], [4])
    
    # 5. Schoolbook for k in {2,3,4}, j=2
    schoolbook([0, 1], [2], [2, 3, 4])
    
    # Check if we used all 28 columns
    # If col < rank, we have unused space, but that's fine.
    # If col > rank, it's an error.
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    
    return result

# EVOLVE-BLOCK-END
