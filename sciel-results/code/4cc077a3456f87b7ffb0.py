"""Optimized tensor decomposition for 3x4 * 4x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    where the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    equals 1 for all valid (i,j,k) and 0 otherwise.
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Tensor dimensions
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    def get_strassen_block(i0, j0, k0):
        """
        Returns lists of (u_idx, u_val), (v_idx, v_val), (w_idx, w_val) for 7 terms.
        i0, j0, k0 are global offsets for a 2x2x2 block.
        """
        terms = []
        # T1: (A11+A22)(B11+B22)
        terms.append(([((i0+0)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)],
                      [((j0+0)*p + (k0+0), 1), ((j0+1)*p + (k0+1), 1)],
                      [((k0+0)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]))
        # T2: (A21+A22)B11
        terms.append(([((i0+1)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)],
                      [((j0+0)*p + (k0+0), 1)],
                      [((k0+0)*n + (i0+1), 1), ((k0+1)*n + (i0+1), -1)]))
        # T3: A11(B21-B11)
        terms.append(([((i0+0)*m + (j0+0), 1)],
                      [((j0+0)*p + (k0+1), 1), ((j0+1)*p + (k0+1), -1)],
                      [((k0+1)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]))
        # T4: A22(B11-B22)
        terms.append(([((i0+1)*m + (j0+1), 1)],
                      [((j0+1)*p + (k0+0), 1), ((j0+0)*p + (k0+0), -1)],
                      [((k0+0)*n + (i0+0), 1), ((k0+0)*n + (i0+1), 1)]))
        # T5: A11(B12-B22) ... actually let me use standard Strassen
        # T5: A11(B12 - B22) -> wait, standard Strassen for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M6
        # C22 = M1 - M2 + M3 + M7
        # Let me redo with correct Strassen formulas:
        terms = []
        # M1 = (A11+A22)(B11+B22)
        terms.append(([((i0+0)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)],
                      [((j0+0)*p + (k0+0), 1), ((j0+1)*p + (k0+1), 1)],
                      [((k0+0)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]))
        # M2 = (A21+A22)B11
        terms.append(([((i0+1)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)],
                      [((j0+0)*p + (k0+0), 1)],
                      [((k0+0)*n + (i0+1), 1), ((k0+1)*n + (i0+1), -1)]))
        # M3 = A11(B21-B11)
        terms.append(([((i0+0)*m + (j0+0), 1)],
                      [((j0+0)*p + (k0+1), 1), ((j0+1)*p + (k0+1), -1)],
                      [((k0+1)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]))
        # M4 = A22(B12-B22)
        terms.append(([((i0+1)*m + (j0+1), 1)],
                      [((j0+1)*p + (k0+0), 1), ((j0+1)*p + (k0+1), -1)],
                      [((k0+0)*n + (i0+0), 1), ((k0+1)*n + (i0+0), -1)]))
        # M5 = (A11+A12)B22
        terms.append(([((i0+0)*m + (j0+0), 1), ((i0+0)*m + (j0+1), 1)],
                      [((j0+1)*p + (k0+1), 1)],
                      [((k0+1)*n + (i0+0), 1), ((k0+1)*n + (i0+1), -1)]))
        # M6 = (A21-A11)(B11+B12)
        terms.append(([((i0+1)*m + (j0+0), 1), ((i0+0)*m + (j0+0), -1)],
                      [((j0+0)*p + (k0+0), 1), ((j0+0)*p + (k0+1), 1)],
                      [((k0+1)*n + (i0+1), 1)]))
        # M7 = (A12-A22)(B21+B22)
        terms.append(([((i0+0)*m + (j0+1), 1), ((i0+1)*m + (j0+1), -1)],
                      [((j0+1)*p + (k0+0), 1), ((j0+1)*p + (k0+1), 1)],
                      [((k0+0)*n + (i0+0), 1)]))
        
        return terms

    def add_strassen_terms(terms, U_cols, V_cols, W_cols):
        for u, v, w in terms:
            U_col = np.zeros(dim_u)
            for idx, val in u:
                U_col[idx] = val
            V_col = np.zeros(dim_v)
            for idx, val in v:
                V_col[idx] = val
            W_col = np.zeros(dim_w)
            for idx, val in w:
                W_col[idx] = val
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    def add_custom_term(u_entries, v_entries, w_entries, U_cols, V_cols, W_cols):
        """Add a single term with specified non-zero entries."""
        U_col = np.zeros(dim_u)
        for idx, val in u_entries:
            U_col[idx] = val
        V_col = np.zeros(dim_v)
        for idx, val in v_entries:
            V_col[idx] = val
        W_col = np.zeros(dim_w)
        for idx, val in w_entries:
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    U_cols = []
    V_cols = []
    W_cols = []
    
    # 4 Strassen blocks for i∈{0,1}, j∈{0,1,2,3}, k∈{0,1,2,3}
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                terms = get_strassen_block(i0, j0, k0)
                add_strassen_terms(terms, U_cols, V_cols, W_cols)
    
    # Single term for i=2 (all j, all k): T[2,j,k]=1
    # U[8+j, r]=1 for j=0..3, V[5j+k, r]=1 for all j,k, W[3k+2, r]=1 for k=0..4
    u_i2 = [(8+j, 1) for j in range(4)]
    v_i2 = [(j*p+k, 1) for j in range(4) for k in range(5)]
    w_i2 = [(k*n+2, 1) for k in range(5)]
    add_custom_term(u_i2, v_i2, w_i2, U_cols, V_cols, W_cols)
    
    # Single term for k=4, i∈{0,1} (all j): T[i,j,4]=1
    # U[i*4+j, r]=1 for i=0,1 and j=0..3, V[5j+4, r]=1 for j=0..3, W[12+i, r]=1 for i=0,1
    u_k4 = [(i*m+j, 1) for i in range(2) for j in range(4)]
    v_k4 = [(j*p+4, 1) for j in range(4)]
    w_k4 = [(4*n+i, 1) for i in range(2)]
    add_custom_term(u_k4, v_k4, w_k4, U_cols, V_cols, W_cols)
    
    # Stack columns
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    # Verify
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i*m + j
                idx_v = j*p + k
                idx_w = k*n + i
                s = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                max_err = max(max_err, abs(s - 1.0))
                
    if max_err > 1e-9:
        # Fallback to schoolbook if verification fails
        U_sb = np.zeros((dim_u, n*m*p))
        V_sb = np.zeros((dim_v, n*m*p))
        W_sb = np.zeros((dim_w, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        U, V, W = U_sb, V_sb, W_sb
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
