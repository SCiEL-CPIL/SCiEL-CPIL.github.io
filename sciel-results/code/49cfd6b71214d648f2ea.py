import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We'll use the Kronecker product approach:
    # 1. Get rank-7 decomposition for 2x2x2 (Strassen)
    # 2. Construct 4x4x8 decomposition via Kronecker product (rank 49)
    # 3. Remove the k=7 slice (columns where k=7 in W)
    # 4. Remove any remaining zero columns
    
    # Strassen's algorithm for 2x2x2 matrix multiplication
    # U[i,j,r], V[j,k,r], W[k,i,r] for i,j,k in {0,1}, r in {0,...,6}
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U: shape (2*2, 7) = (4, 7) - indices (i,j) -> row, r -> col
    U_strassen = np.zeros((4, 7), dtype=float)
    # V: shape (2*2, 7) = (4, 7) - indices (j,k) -> row, r -> col
    V_strassen = np.zeros((4, 7), dtype=float)
    # W: shape (2*2, 7) = (4, 7) - indices (k,i) -> row, r -> col
    W_strassen = np.zeros((4, 7), dtype=float)
    
    # M1: (A11+A22)(B11+B22)
    # U: A11 (i=0,j=0) -> 1, A22 (i=1,j=1) -> 1
    U_strassen[0, 0] = 1  # i=0, j=0
    U_strassen[3, 0] = 1  # i=1, j=1
    # V: B11 (j=0,k=0) -> 1, B22 (j=1,k=1) -> 1
    V_strassen[0, 0] = 1  # j=0, k=0
    V_strassen[3, 0] = 1  # j=1, k=1
    # W: C11 += M1 (k=0,i=0), C22 += M1 (k=1,i=1)
    W_strassen[0, 0] = 1  # k=0, i=0
    W_strassen[3, 0] = 1  # k=1, i=1
    
    # M2: (A21+A22)B11
    # U: A21 (i=1,j=0) -> 1, A22 (i=1,j=1) -> 1
    U_strassen[2, 1] = 1  # i=1, j=0
    U_strassen[3, 1] = 1  # i=1, j=1
    # V: B11 (j=0,k=0) -> 1
    V_strassen[0, 1] = 1  # j=0, k=0
    # W: C21 += M2 (k=0,i=1)
    W_strassen[2, 1] = 1  # k=0, i=1
    
    # M3: A11(B12-B22)
    # U: A11 (i=0,j=0) -> 1
    U_strassen[0, 2] = 1  # i=0, j=0
    # V: B12 (j=0,k=1) -> 1, B22 (j=1,k=1) -> -1
    V_strassen[1, 2] = 1  # j=0, k=1
    V_strassen[3, 2] = -1  # j=1, k=1
    # W: C12 += M3 (k=1,i=0)
    W_strassen[1, 2] = 1  # k=1, i=0
    
    # M4: A22(B21-B11)
    # U: A22 (i=1,j=1) -> 1
    U_strassen[3, 3] = 1  # i=1, j=1
    # V: B21 (j=1,k=0) -> 1, B11 (j=0,k=0) -> -1
    V_strassen[3, 3] = 1  # j=1, k=0
    V_strassen[0, 3] = -1  # j=0, k=0
    # W: C21 += M4 (k=0,i=1)
    W_strassen[2, 3] = 1  # k=0, i=1
    
    # M5: (A11+A12)B22
    # U: A11 (i=0,j=0) -> 1, A12 (i=0,j=1) -> 1
    U_strassen[0, 4] = 1  # i=0, j=0
    U_strassen[1, 4] = 1  # i=0, j=1
    # V: B22 (j=1,k=1) -> 1
    V_strassen[3, 4] = 1  # j=1, k=1
    # W: C12 += M5 (k=1,i=0)
    W_strassen[1, 4] = 1  # k=1, i=0
    
    # M6: (A21-A11)(B11+B12)
    # U: A21 (i=1,j=0) -> 1, A11 (i=0,j=0) -> -1
    U_strassen[2, 5] = 1  # i=1, j=0
    U_strassen[0, 5] = -1  # i=0, j=0
    # V: B11 (j=0,k=0) -> 1, B12 (j=0,k=1) -> 1
    V_strassen[0, 5] = 1  # j=0, k=0
    V_strassen[1, 5] = 1  # j=0, k=1
    # W: C21 += M6 (k=0,i=1), C22 += M6 (k=1,i=1)
    W_strassen[2, 5] = 1  # k=0, i=1
    W_strassen[3, 5] = 1  # k=1, i=1
    
    # M7: (A12-A22)(B21+B22)
    # U: A12 (i=0,j=1) -> 1, A22 (i=1,j=1) -> -1
    U_strassen[1, 6] = 1  # i=0, j=1
    U_strassen[3, 6] = -1  # i=1, j=1
    # V: B21 (j=1,k=0) -> 1, B22 (j=1,k=1) -> 1
    V_strassen[3, 6] = 1  # j=1, k=0
    V_strassen[3, 6] += 1  # j=1, k=1 -> wait, let me redo this
    
    # Let me restart W for M7:
    W_strassen[2, 6] = 1  # k=0, i=1 (C21)
    W_strassen[3, 6] = 1  # k=1, i=1 (C22)
    
    # Fix V for M7: B21 (j=1,k=0) and B22 (j=1,k=1)
    V_strassen[2, 6] = 1  # j=1, k=0
    V_strassen[3, 6] = 1  # j=1, k=1
    
    # Now construct 4x4x8 decomposition using Kronecker product
    # We split indices: i = i1*2 + i2, j = j1*2 + j2, k = k1*2 + k2
    # where i1,j1,k1 in {0,1} and i2,j2,k2 in {0,1}
    # r = r1*7 + r2 where r1,r2 in {0,...,6}
    
    rank_448 = 7 * 7  # 49
    U_448 = np.zeros((16, rank_448), dtype=float)
    V_448 = np.zeros((16, rank_448), dtype=float)
    W_448 = np.zeros((16, rank_448), dtype=float)
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for r1 in range(7):
                    u1 = U_strassen[i1*2+j1, r1]
                    v1 = V_strassen[j1*2+k1, r1]
                    w1 = W_strassen[k1*2+i1, r1]
                    if u1 == 0 or v1 == 0 or w1 == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                for r2 in range(7):
                                    u2 = U_strassen[i2*2+j2, r2]
                                    v2 = V_strassen[j2*2+k2, r2]
                                    w2 = W_strassen[k2*2+i2, r2]
                                    if u2 == 0 or v2 == 0 or w2 == 0:
                                        continue
                                    i = i1*2 + i2
                                    j = j1*2 + j2
                                    k = k1*2 + k2
                                    r = r1*7 + r2
                                    U_448[i*4+j, r] += u1 * u2
                                    V_448[j*4+k, r] += v1 * v2
                                    W_448[k*4+i, r] += w1 * w2
    
    # Now we need 4x4x7, so we need to remove k=7
    # In W_448, the row index is k*4+i where k in {0,...,7}
    # We need to zero out all columns r where W_448[7*4+i, r] != 0 for any i
    # i.e., where k=7
    
    # Find columns to zero out
    cols_to_zero = set()
    for i in range(4):
        row = 7*4 + i  # k=7, i in {0,1,2,3}
        for r in range(rank_448):
            if W_448[row, r] != 0:
                cols_to_zero.add(r)
    
    # Zero out those columns
    for r in cols_to_zero:
        U_448[:, r] = 0
        V_448[:, r] = 0
        W_448[:, r] = 0
    
    # Remove zero columns
    nonzero_cols = []
    for r in range(rank_448):
        if np.any(U_448[:, r] != 0) or np.any(V_448[:, r] != 0) or np.any(W_448[:, r] != 0):
            nonzero_cols.append(r)
    
    U_final = U_448[:, nonzero_cols]
    V_final = V_448[:, nonzero_cols]
    W_final = W_448[:, nonzero_cols]
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Check if it's close to an integer or half-integer
                if abs(val - round(val)) < 1e-9:
                    row_list.append(int(round(val)))
                elif abs(val*2 - round(val*2)) < 1e-9:
                    row_list.append(round(val*2) / 2.0)
                else:
                    # Should not happen for this construction
                    row_list.append(val)
            result.append(row_list)
        return result
    
    U_out = to_list(U_final)
    V_out = to_list(V_final)
    W_out = to_list(W_final)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # This shouldn't happen, but just in case
        pass
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
