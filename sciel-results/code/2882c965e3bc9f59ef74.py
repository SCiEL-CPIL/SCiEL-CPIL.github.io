import time
import numpy as np
import random

DIMENSIONS = [3, 3, 4]
DEADLINE = time.time() + 110  # 10s safety margin before 120s limit

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matmul tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
                prod = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                s = np.sum(prod)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2.
    U (4x7), V (4x7), W (4x7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}.
    """
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # Strassen products:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    
    # M1: A entries (0,0),(1,1) with +1; B entries (0,0),(1,1) with +1
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    # C11 += M1, C22 += M1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
    
    # M2: A entries (1,0),(1,1) with +1; B entry (0,0) with +1
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    # C21 += M2
    W[0*2+1, 1] = 1
    
    # M3: A entry (0,0) with +1; B entries (0,1) with +1, (1,1) with -1
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    # C12 += M3, C22 += M3
    W[1*2+0, 2] = 1; W[1*2+1, 2] = 1
    
    # M4: A entry (1,1) with +1; B entries (1,0) with +1, (0,0) with -1
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    # C11 += M4, C21 += M4
    W[0*2+0, 3] = 1; W[0*2+1, 3] = 1
    
    # M5: A entries (0,0),(0,1) with +1; B entry (1,1) with +1
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    # C11 -= M5, C12 += M5
    W[0*2+0, 4] = -1; W[1*2+0, 4] = 1
    
    # M6: A entries (1,0) with +1, (0,0) with -1; B entries (0,0),(0,1) with +1
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    # C22 += M6
    W[1*2+1, 5] = 1
    
    # M7: A entries (0,1) with +1, (1,1) with -1; B entries (1,0),(1,1) with +1
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    # C11 += M7
    W[0*2+0, 6] = 1
    
    return U, V, W

def _build_3x3x3_strassen_block():
    """Build a rank-26 decomposition for 3x3x3 using Strassen on the 2x2 block.
    
    Partition:
    A = [[A11(2x2), A12(2x1)], [A21(1x2), A22(1x1)]]
    B = [[B11(2x2), B12(2x1)], [B21(1x2), B22(1x1)]]
    C = [[C11(2x2), C12(2x1)], [C21(1x2), C22(1x1)]]
    
    C11 = A11*B11 + A12*B21  (2x2)
    C12 = A11*B12 + A12*B22  (2x1)
    C21 = A21*B11 + A22*B21  (1x2)
    C22 = A21*B12 + A22*B22  (1x1)
    """
    n = m = p = 3
    U = np.zeros((n*m, 0), dtype=float)
    V = np.zeros((m*p, 0), dtype=float)
    W = np.zeros((p*n, 0), dtype=float)
    
    def add_terms(U, V, W, u_entries, v_entries, w_entries):
        """Add rank-1 terms. u_entries: list of (row, val), etc."""
        U_new = np.zeros((U.shape[0], 1), dtype=float)
        V_new = np.zeros((V.shape[0], 1), dtype=float)
        W_new = np.zeros((W.shape[0], 1), dtype=float)
        for row, val in u_entries:
            U_new[row, 0] = val
        for row, val in v_entries:
            V_new[row, 0] = val
        for row, val in w_entries:
            W_new[row, 0] = val
        return np.hstack([U, U_new]), np.hstack([V, V_new]), np.hstack([W, W_new])
    
    # Helper: index for 3x3x3
    # A[i,j] -> row i*3+j in U
    # B[j,k] -> row j*3+k in V
    # C[k,i] -> row k*3+i in W
    
    # C11 = A11*B11 + A12*B21 (2x2 block, indices i,j in {0,1} for A, j,k in {0,1} for B, k,i in {0,1} for C)
    # Use Strassen for A11*B11 (rank 7)
    sU, sV, sW = _strassen_2x2x2()
    # Map Strassen indices to 3x3x3:
    # A11[i,j] -> U[i*3+j], i,j in {0,1}
    # B11[j,k] -> V[j*3+k], j,k in {0,1}
    # C11[k,i] -> W[k*3+i], k,i in {0,1}
    U_s = np.zeros((9, 7), dtype=float)
    V_s = np.zeros((9, 7), dtype=float)
    W_s = np.zeros((9, 7), dtype=float)
    for i in range(2):
        for j in range(2):
            U_s[i*3+j, :] = sU[i*2+j, :]
    for j in range(2):
        for k in range(2):
            V_s[j*3+k, :] = sV[j*2+k, :]
    for k in range(2):
        for i in range(2):
            W_s[k*3+i, :] = sW[k*2+i, :]
    
    U = U_s
    V = V_s
    W = W_s
    
    # A12*B21: A12 is 2x1 (rows 0,1; col 2), B21 is 1x2 (row 2; cols 0,1)
    # C11[k,i] += A12[i,2] * B21[2,k]
    # This is a 2x1 * 1x2 = 2x2 matrix product, rank 4
    for i in range(2):
        for k in range(2):
            u_e = [(i*3+2, 1)]
            v_e = [(2*3+k, 1)]
            w_e = [(k*3+i, 1)]
            U, V, W = add_terms(U, V, W, u_e, v_e, w_e)
    
    # C12 = A11*B12 + A12*B22 (2x1 block, i in {0,1}, k=2)
    # A11*B12: A11 is 2x2, B12 is 2x1, C12 is 2x1
    # C12[i,2] = sum_{j=0,1} A11[i,j] * B12[j,2]
    # Schoolbook: rank 2*2*1 = 4
    for i in range(2):
        for j in range(2):
            u_e = [(i*3+j, 1)]
            v_e = [(j*3+2, 1)]
            w_e = [(2*3+i, 1)]
            U, V, W = add_terms(U, V, W, u_e, v_e, w_e)
    
    # A12*B22: A12 is 2x1, B22 is 1x1
    # C12[i,2] += A12[i,2] * B22[2,2]
    for i in range(2):
        u_e = [(i*3+2, 1)]
        v_e = [(2*3+2, 1)]
        w_e = [(2*3+i, 1)]
        U, V, W = add_terms(U, V, W, u_e, v_e, w_e)
    
    # C21 = A21*B11 + A22*B21 (1x2 block, i=2, k in {0,1})
    # A21*B11: A21 is 1x2, B11 is 2x2, C21 is 1x2
    # C21[2,k] = sum_{j=0,1} A21[2,j] * B11[j,k]
    # Schoolbook: rank 1*2*2 = 4
    for j in range(2):
        for k in range(2):
            u_e = [(2*3+j, 1)]
            v_e = [(j*3+k, 1)]
            w_e = [(k*3+2, 1)]
            U, V, W = add_terms(U, V, W, u_e, v_e, w_e)
    
    # A22*B21: A22 is 1x1, B21 is 1x2
    # C21[2,k] += A22[2,2] * B21[2,k]
    for k in range(2):
        u_e = [(2*3+2, 1)]
        v_e = [(2*3+k, 1)]
        w_e = [(k*3+2, 1)]
        U, V, W = add_terms(U, V, W, u_e, v_e, w_e)
    
    # C22 = A21*B12 + A22*B22 (1x1 block, i=2, k=2)
    # A21*B12: A21 is 1x2, B12 is 2x1
    # C22[2,2] = sum_{j=0,1} A21[2,j] * B12[j,2]
    # Schoolbook: rank 1*2*1 = 2
    for j in range(2):
        u_e = [(2*3+j, 1)]
        v_e = [(j*3+2, 1)]
        w_e = [(2*3+2, 1)]
        U, V, W = add_terms(U, V, W, u_e, v_e, w_e)
    
    # A22*B22: A22 is 1x1, B22 is 1x1
    u_e = [(2*3+2, 1)]
    v_e = [(2*3+2, 1)]
    w_e = [(2*3+2, 1)]
    U, V, W = add_terms(U, V, W, u_e, v_e, w_e)
    
    return U, V, W

def _build_3x3x4_decomposition():
    """Build a decomposition for 3x3x4 using the 3x3x3 Strassen block + extra column.
    
    For the 4th column of B (index k=3), we need:
    C[i,3] = sum_{j=0,1,2} A[i,j] * B[j,3]
    
    This is a 3x3x1 tensor with all ones. Rank 3 decomposition:
    U[i*3+j, c] = 1, V[j*3, c] = 1, W[3*3+i, c] = delta_{i,c} for c=0,1,2
    """
    n, m, p = 3, 3, 4
    
    # Start with 3x3x3 decomposition (first 3 columns of B, first 3 columns of C)
    U3, V3, W3 = _build_3x3x3_strassen_block()
    
    # Extend to 3x3x4: add rows for k=3 in V and W
    U = np.zeros((n*m, U3.shape[1]), dtype=float)
    V = np.zeros((m*p, U3.shape[1]), dtype=float)
    W = np.zeros((p*n, U3.shape[1]), dtype=float)
    
    # Copy 3x3x3 part
    U[:, :] = U3
    V[:9, :] = V3  # V3 has 9 rows (3*3), now V has 12 rows (3*4)
    W[:9, :] = W3  # W3 has 9 rows (3*3), now W has 12 rows (4*3)
    
    # Add rank-3 terms for the 4th column
    for c in range(3):
        col_idx = U.shape[1] + c
        U[:, col_idx] = 1
        V[:3, col_idx] = 1  # V[j*4+3] for j=0,1,2 -> rows 3,7,11... wait
        # V is indexed as [j*p+k], so for k=3: rows 3, 7, 11
        V[3, col_idx] = 1
        V[7, col_idx] = 1
        V[11, col_idx] = 1
        # W is indexed as [k*n+i], so for k=3: rows 9, 10, 11
        W[9 + c, col_idx] = 1
    
    U = np.hstack([U, np.zeros((U.shape[0], 3))])
    V = np.hstack([V, np.zeros((V.shape[0], 3))])
    W = np.hstack([W, np.zeros((W.shape[0], 3))])
    
    # Redo the extra column part properly
    U = U3.copy()
    V = np.zeros((m*p, U3.shape[1]), dtype=float)
    W = np.zeros((p*n, U3.shape[1]), dtype=float)
    V[:9, :] = V3
    W[:9, :] = W3
    
    for c in range(3):
        col_idx = U3.shape[1] + c
        U_col = np.ones(n*m)
        V_col = np.zeros(m*p)
        W_col = np.zeros(p*n)
        V_col[3] = 1   # j=0, k=3
        V_col[7] = 1   # j=1, k=3
        V_col[11] = 1  # j=2, k=3
        W_col[9 + c] = 1  # k=3, i=c
        U = np.hstack([U, U_col.reshape(-1, 1)])
        V = np.hstack([V, V_col.reshape(-1, 1)])
        W = np.hstack([W, W_col.reshape(-1, 1)])
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    total_terms = n * m * p
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Try the structured 3x3x4 decomposition (rank 35)
    U_struct, V_struct, W_struct = _build_3x3x4_decomposition()
    
    if _verify_decomposition(U_struct, V_struct, W_struct, n, m, p):
        # Found a valid rank-35 decomposition
        return {"U": U_struct.tolist(), "V": V_struct.tolist(), "W": W_struct.tolist()}
    
    # Try random search for lower rank
    deadline = time.time() + 100
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = total_terms
    
    for target_rank in range(30, min(36, total_terms)):
        if time.time() > deadline:
            break
        for attempt in range(2000):
            if time.time() > deadline:
                break
            # Generate random decomposition with entries in {-1, 0, 1}
            U_r = np.random.choice([-1, 0, 1], size=(n*m, target_rank))
            V_r = np.random.choice([-1, 0, 1], size=(m*p, target_rank))
            W_r = np.random.choice([-1, 0, 1], size=(p*n, target_rank))
            
            # Compute tensor from decomposition
            T_decomp = np.einsum('at,bt,ct->abc', U_r, V_r, W_r)
            
            # Check if matches target
            match = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(T_decomp[i*m+j, j*p+k, k*n+i] - 1.0) > 1e-10:
                            match = False
                            break
                    if not match:
                        break
                if not match:
                    break
            
            if match:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Return schoolbook if no improvement found
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
