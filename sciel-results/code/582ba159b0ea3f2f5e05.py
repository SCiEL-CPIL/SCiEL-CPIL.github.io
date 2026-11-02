"""Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

def _build_subtensor(n, m, p):
    """Build the (n,m,p) matrix multiplication tensor with shapes (n*m, m*p, p*n)."""
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _strassen_2x2x2():
    """Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is (u_indices, v_indices, w_indices) where:
    u_indices: list of (index_in_A, coeff) for the 2x2 A block
    v_indices: list of (index_in_B, coeff) for the 2x2 B block
    w_indices: list of (index_in_C, coeff) for the 2x2 C block
    Indexing: A[i,j] -> i*2+j, B[i,j] -> i*2+j, C[i,j] -> i*2+j
    """
    # Strassen formulas for 2x2:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1+m4-m5+m7
    # c12 = m3+m5
    # c21 = m2+m4
    # c22 = m1-m2+m3+m6
    
    terms = []
    # m1 = (a11+a22)(b11+b22)
    terms.append(([ (0,1.0), (3,1.0) ], [ (0,1.0), (3,1.0) ], [ (0,1.0), (3,1.0) ]))
    # m2 = (a21+a22)b11
    terms.append(([ (2,1.0), (3,1.0) ], [ (0,1.0) ], [ (2,1.0), (3,-1.0) ]))
    # m3 = a11(b12-b22)
    terms.append(([ (0,1.0) ], [ (1,1.0), (3,-1.0) ], [ (1,1.0), (3,1.0) ]))
    # m4 = a22(b21-b11)
    terms.append(([ (3,1.0) ], [ (2,1.0), (0,-1.0) ], [ (0,-1.0), (2,1.0) ]))
    # m5 = (a11+a12)b22
    terms.append(([ (0,1.0), (1,1.0) ], [ (3,1.0) ], [ (0,-1.0), (1,1.0) ]))
    # m6 = (a21-a11)(b11+b12)
    terms.append(([ (2,1.0), (0,-1.0) ], [ (0,1.0), (1,1.0) ], [ (3,1.0) ]))
    # m7 = (a12-a22)(b21+b22)
    terms.append(([ (1,1.0), (3,-1.0) ], [ (2,1.0), (3,1.0) ], [ (0,1.0) ]))
    
    return terms

def construct():
    """Find a low-rank exact tensor decomposition for (3,3,8) matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    U shape: (n*m, rank) indexed by i*m+j
    V shape: (m*p, rank) indexed by j*p+k
    W shape: (p*n, rank) indexed by k*n+i
    """
    n, m, p = 3, 3, 8
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Build target tensor for verification
    T = _build_subtensor(n, m, p)
    
    # Strategy: Split A into top 2 rows and bottom 1 row.
    # For the top 2x2 block of A times each 2-column block of B: use Strassen (7 terms)
    # For the remaining rows/cols: use schoolbook
    # 
    # A decomposition:
    #   A[0:2, 0:2] is 2x2 -> Strassen with each 2-col B block
    #   A[0:2, 2] is 2x1 -> schoolbook (2 terms per col)
    #   A[2, 0:2] is 1x2 -> schoolbook (2 terms per col)
    #   A[2, 2] is 1x1 -> schoolbook (1 term per col)
    #
    # B is split into 4 blocks of 2 columns each: cols 0-1, 2-3, 4-5, 6-7
    #
    # Total terms:
    #   4 Strassen blocks (2x2 A x 2-col B): 4 * 7 = 28
    #   A[0:2,2] x each of 8 cols: 2 * 8 = 16
    #   A[2,0:2] x each of 8 cols: 2 * 8 = 16
    #   A[2,2] x each of 8 cols: 1 * 8 = 8
    #   Total = 28 + 16 + 16 + 8 = 68
    
    strassen_terms = _strassen_2x2x2()
    
    # Collect all terms as (U_sparse, V_sparse, W_sparse)
    # Each sparse representation is a dict: {index: coefficient}
    all_terms = []
    
    # Part 1: Strassen for A[0:2, 0:2] x B[:, k_block] for each 2-col block
    for block in range(4):
        k_start = block * 2
        for term in strassen_terms:
            u_sparse = {}
            v_sparse = {}
            w_sparse = {}
            
            # U: A[0:2, 0:2] indices are i*3+j for i=0,1 and j=0,1
            for idx_2x2, coeff in term[0]:
                # idx_2x2 is 0,1,2,3 corresponding to (0,0),(0,1),(1,0),(1,1)
                ii = idx_2x2 // 2
                jj = idx_2x2 % 2
                full_idx = ii * m + jj  # i*3+j
                u_sparse[full_idx] = u_sparse.get(full_idx, 0.0) + coeff
            
            # V: B[0:2, k_start:k_start+2] indices are j*p+k for j=0,1 and k=k_start,k_start+1
            for idx_2x2, coeff in term[1]:
                jj = idx_2x2 // 2
                kk = idx_2x2 % 2
                full_idx = jj * p + (k_start + kk)
                v_sparse[full_idx] = v_sparse.get(full_idx, 0.0) + coeff
            
            # W: C[0:2, k_start:k_start+2] indices are k*n+i for k=k_start,k_start+1 and i=0,1
            for idx_2x2, coeff in term[2]:
                kk = idx_2x2 // 2
                ii = idx_2x2 % 2
                full_idx = (k_start + kk) * n + ii
                w_sparse[full_idx] = w_sparse.get(full_idx, 0.0) + coeff
            
            all_terms.append((u_sparse, v_sparse, w_sparse))
    
    # Part 2: A[0:2, 2] x B[j, k] for j=0,1 and all k=0..7
    # A[i, 2] * B[j, k] contributes to C[i, k]
    # For i in {0,1}, j in {0,1}, k in {0..7}:
    #   U: A[i,2] -> index i*3+2
    #   V: B[j,k] -> index j*8+k
    #   W: C[i,k] -> index k*3+i
    for i in range(2):
        for j in range(2):
            for k in range(8):
                u_sparse = {i * m + 2: 1.0}
                v_sparse = {j * p + k: 1.0}
                w_sparse = {k * n + i: 1.0}
                all_terms.append((u_sparse, v_sparse, w_sparse))
    
    # Part 3: A[2, 0:2] x B[j, k] for j=0,1 and all k=0..7
    # A[2, j] * B[j, k] contributes to C[2, k]
    for j in range(2):
        for k in range(8):
            u_sparse = {2 * m + j: 1.0}
            v_sparse = {j * p + k: 1.0}
            w_sparse = {k * n + 2: 1.0}
            all_terms.append((u_sparse, v_sparse, w_sparse))
    
    # Part 4: A[2, 2] x B[2, k] for all k=0..7
    # A[2,2] * B[2,k] contributes to C[2,k]
    for k in range(8):
        u_sparse = {2 * m + 2: 1.0}
        v_sparse = {2 * p + k: 1.0}
        w_sparse = {k * n + 2: 1.0}
        all_terms.append((u_sparse, v_sparse, w_sparse))
    
    rank = len(all_terms)
    
    # Build dense matrices
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for c, (u_sp, v_sp, w_sp) in enumerate(all_terms):
        for idx, coeff in u_sp.items():
            U[idx, c] += coeff
        for idx, coeff in v_sp.items():
            V[idx, c] += coeff
        for idx, coeff in w_sp.items():
            W[idx, c] += coeff
    
    # Verify
    recon = np.zeros((nm, mp, pn))
    for c in range(rank):
        recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    
    if np.abs(recon - T).max() > 1e-8:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
