"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time

def _strassen_2x2x2():
    """Return the 7 rank-1 terms for 2x2 by 2x2 matrix multiplication.
    
    Each term is (u_vec, v_vec, w_vec) where:
    - u_vec has length 4 (index i*2+j)
    - v_vec has length 4 (index j*2+k)
    - w_vec has length 4 (index k*2+i)
    """
    terms = []
    # M1 = (A00+A11)*(B00+B11)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2 = (A10+A11)*B00
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    # M3 = A00*(B10-B11)
    terms.append(([1,0,0,0], [0,1,0,-1], [1,0,-1,0]))
    # M4 = A11*(B01-B00)
    terms.append(([0,0,0,1], [1,-1,0,0], [0,1,0,-1]))
    # M5 = (A00+A01)*B11
    terms.append(([1,1,0,0], [0,0,0,1], [0,0,0,1]))
    # M6 = (A10-A00)*(B00+B01)
    terms.append(([0,0,1,-1], [1,1,0,0], [1,1,0,0]))
    # M7 = (A01-A11)*(B10+B11)
    terms.append(([0,1,0,-1], [0,0,1,1], [0,0,1,1]))
    return terms

def _pad_vector(vec, length):
    """Pad a vector with zeros to the specified length."""
    result = list(vec)
    while len(result) < length:
        result.append(0)
    return result

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 by 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 20
    W_dim = p * n  # 10
    
    all_terms = []
    
    # Use Strassen decomposition for 2x2 blocks
    strassen_terms = _strassen_2x2x2()
    
    # The 4x4 block (j=0..3, k=0..3) can be decomposed as four 2x2 blocks
    # For each (i_off, j_off, k_off) in {(0,0,0), (0,0,2), (0,2,0), (0,2,2)}
    # Use Strassen on the 2x2 sub-block
    
    for j_off in [0, 2]:
        for k_off in [0, 2]:
            for (u_local, v_local, w_local) in strassen_terms:
                # Build full U vector (length 8 = 2*4)
                # i is 0 or 1, j is j_off + (0 or 1)
                u_full = [0] * U_dim
                for i in range(2):
                    for j in range(2):
                        idx = i * m + (j_off + j)
                        u_full[idx] = u_local[i * 2 + j]
                
                # Build full V vector (length 20 = 4*5)
                # j is j_off + (0 or 1), k is k_off + (0 or 1)
                v_full = [0] * V_dim
                for j in range(2):
                    for k in range(2):
                        idx = (j_off + j) * p + (k_off + k)
                        v_full[idx] = v_local[j * 2 + k]
                
                # Build full W vector (length 10 = 5*2)
                # k is k_off + (0 or 1), i is 0 or 1
                w_full = [0] * W_dim
                for k in range(2):
                    for i in range(2):
                        idx = (k_off + k) * n + i
                        w_full[idx] = w_local[k * 2 + i]
                
                all_terms.append((u_full, v_full, w_full))
    
    # For the 4x1 block (j=0..3, k=4), use schoolbook: 4 multiplications
    # A[i,j] * B[j,4] for i in {0,1}, j in {0,1,2,3}
    for j in range(4):
        for i in range(2):
            u_full = [0] * U_dim
            u_full[i * m + j] = 1
            
            v_full = [0] * V_dim
            v_full[j * p + 4] = 1
            
            w_full = [0] * W_dim
            w_full[4 * n + i] = 1  # k=4, i
            
            all_terms.append((u_full, v_full, w_full))
    
    rank = len(all_terms)
    
    # Build U, V, W as lists of lists
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    
    for col, (u, v, w) in enumerate(all_terms):
        for i in range(U_dim):
            if u[i] != 0:
                U[i][col] = u[i]
        for i in range(V_dim):
            if v[i] != 0:
                V[i][col] = v[i]
        for i in range(W_dim):
            if w[i] != 0:
                W[i][col] = w[i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
