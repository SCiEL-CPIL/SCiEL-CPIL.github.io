import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def get_3x3_schoolbook():
        n3 = 3
        rank3 = 27
        U3 = np.zeros((9, rank3))
        V3 = np.zeros((9, rank3))
        W3 = np.zeros((9, rank3))
        for i in range(n3):
            for j in range(n3):
                for k in range(n3):
                    col = i*9 + j*3 + k
                    U3[i*3+j, col] = 1.0
                    V3[j*3+k, col] = 1.0
                    W3[k*3+i, col] = 1.0
        return U3, V3, W3

    U3, V3, W3 = get_3x3_schoolbook()
    r3 = 27
    
    total_rank = 7 * r3
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))
    
    # Strassen formulas with explicit block mappings
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # For A: block (bi,bj) -> rows bi*3..bi*3+2, cols bj*3..bj*3+2
    # Global U index for A[i,j] = i*6+j
    # For block (bi,bj) and local (li,lj): global i=bi*3+li, j=bj*3+lj
    # Global U index = (bi*3+li)*6 + (bj*3+lj)
    
    # For B: block (bi,bj) -> rows bi*3..bi*3+2, cols bj*3..bj*3+2
    # Global V index for B[j,k] = j*6+k
    # For block (bi,bj) and local (lj,lk): global j=bi*3+lj, k=bj*3+lk
    # Global V index = (bi*3+lj)*6 + (bj*3+lk)
    
    # For C: block (ci,cj) -> rows ci*3..ci*3+2, cols cj*3..cj*3+2
    # W index convention: W[k*n+i, r] where k=col, i=row
    # For block (ci,cj) and local (lk,li): global k=cj*3+lk, i=ci*3+li
    # Global W index = (cj*3+lk)*6 + (ci*3+li)
    
    def add_strassen_product(start_col, a_terms, b_terms, c_terms):
        """
        a_terms: list of (block_idx, coeff) for A
        b_terms: list of (block_idx, coeff) for B  
        c_terms: list of (block_idx, coeff) for C
        """
        for r in range(r3):
            # Get the r-th term of the 3x3 decomposition
            u_vec = U3[:, r]  # shape (9,), indexed by [i*3+j]
            v_vec = V3[:, r]  # shape (9,), indexed by [j*3+k]
            w_vec = W3[:, r]  # shape (9,), indexed by [k*3+i]
            
            # Process A terms
            for bi, coeff in a_terms:
                row_block = bi // 2
                col_block = bi % 2
                for li in range(3):
                    for lj in range(3):
                        val = coeff * u_vec[li*3 + lj]
                        if val != 0:
                            gi = row_block * 3 + li
                            gj = col_block * 3 + lj
                            U6[gi*6 + gj, start_col + r] += val
            
            # Process B terms
            for bi, coeff in b_terms:
                row_block = bi // 2
                col_block = bi % 2
                for lj in range(3):
                    for lk in range(3):
                        val = coeff * v_vec[lj*3 + lk]
                        if val != 0:
                            gj = row_block * 3 + lj
                            gk = col_block * 3 + lk
                            V6[gj*6 + gk, start_col + r] += val
            
            # Process C terms
            for ci, coeff in c_terms:
                row_block = ci // 2
                col_block = ci % 2
                for lk in range(3):
                    for li in range(3):
                        val = coeff * w_vec[lk*3 + li]
                        if val != 0:
                            gk = col_block * 3 + lk
                            gi = row_block * 3 + li
                            W6[gk*6 + gi, start_col + r] += val
    
    # P1 = (A11 + A22) * (B11 + B22)
    # Contributes to: C11, C12, C21, C22 with coefficients from Strassen
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5  
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # So P1 contributes: C11(+1), C22(+1)
    add_strassen_product(0, 
                         a_terms=[(0, 1), (3, 1)], 
                         b_terms=[(0, 1), (3, 1)], 
                         c_terms=[(0, 1), (3, 1)])
    
    # P2 = (A21 + A22) * B11
    # Contributes: C21(+1), C22(-1)
    add_strassen_product(r3, 
                         a_terms=[(2, 1), (3, 1)], 
                         b_terms=[(0, 1)], 
                         c_terms=[(2, 1), (3, -1)])
    
    # P3 = A11 * (B12 - B22)
    # Contributes: C12(+1), C22(+1)
    add_strassen_product(2*r3, 
                         a_terms=[(0, 1)], 
                         b_terms=[(1, 1), (3, -1)], 
                         c_terms=[(1, 1), (3, 1)])
    
    # P4 = A22 * (B21 - B11)
    # Contributes: C11(+1), C21(+1)
    add_strassen_product(3*r3, 
                         a_terms=[(3, 1)], 
                         b_terms=[(2, 1), (0, -1)], 
                         c_terms=[(0, 1), (2, 1)])
    
    # P5 = (A11 + A12) * B22
    # Contributes: C12(+1), C11(-1)
    add_strassen_product(4*r3, 
                         a_terms=[(0, 1), (1, 1)], 
                         b_terms=[(3, 1)], 
                         c_terms=[(1, 1), (0, -1)])
    
    # P6 = (A21 - A11) * (B11 + B12)
    # Contributes: C22(+1)
    add_strassen_product(5*r3, 
                         a_terms=[(2, 1), (0, -1)], 
                         b_terms=[(0, 1), (1, 1)], 
                         c_terms=[(3, 1)])
    
    # P7 = (A12 - A22) * (B21 + B22)
    # Contributes: C11(+1)
    add_strassen_product(6*r3, 
                         a_terms=[(1, 1), (3, -1)], 
                         b_terms=[(2, 1), (3, 1)], 
                         c_terms=[(0, 1)])
    
    # Verify
    if time.time() < deadline:
        try:
            U3g = U6.reshape(6, 6, -1)
            V3g = V6.reshape(6, 6, -1)
            W3g = W6.reshape(6, 6, -1)
            reconstructed = np.einsum('ijr,jkr,kir->ijk', U3g, V3g, W3g)
            target = np.ones((6, 6, 6))
            if np.allclose(reconstructed, target, atol=1e-9):
                return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
        except Exception:
            pass
    
    # Fallback
    rank = 216
    U_f = np.zeros((36, rank))
    V_f = np.zeros((36, rank))
    W_f = np.zeros((36, rank))
    for i in range(6):
        for j in range(6):
            for k in range(6):
                idx = i*6 + j
                jk = j*6 + k
                ki = k*6 + i
                col = i*36 + j*6 + k
                U_f[idx, col] = 1.0
                V_f[jk, col] = 1.0
                W_f[ki, col] = 1.0
    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

# EVOLVE-BLOCK-END
