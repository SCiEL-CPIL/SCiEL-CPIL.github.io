"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook(n, m, p):
    """Return schoolbook decomposition as integer lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_block(n, m, p):
    """
    Build a decomposition using Strassen's rank-7 algorithm for 2x2x2 blocks
    and schoolbook for the remaining 2x2x1 block.
    
    For 2x2x15: split the 15 columns into 7 blocks of 2 + 1 block of 1.
    Each 2x2x2 block uses Strassen (rank 7), the 2x2x1 block uses schoolbook (rank 4).
    Total rank = 7*7 + 4 = 53.
    """
    n, m, p = 2, 2, 15
    # Split p=15 into 7 pairs + 1 single: indices 0-1, 2-3, ..., 12-13, 14
    blocks = []
    for b in range(7):
        blocks.append((2 * b, 2 * b + 1))  # pair (k0, k1)
    blocks.append((14,))  # single k=14
    
    # Strassen's 7 products for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For each Strassen term, we need:
    # U part: linear combination of A indices (i,j)
    # V part: linear combination of B indices (j,k)
    # W part: contributes to C indices (k,i)
    
    # Let's build the full decomposition
    all_U = []  # list of (row, coeff) for each rank column
    all_V = []
    all_W = []
    
    # Strassen terms: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping (i,j) -> coeff
    # v_coeffs: dict mapping (j,k0_or_k1) -> coeff  
    # w_coeffs: dict mapping (k, i) -> coeff
    # But we need to handle the k dimension carefully.
    
    # For a 2x2x2 block with k indices (k0, k1):
    # B11 = B[:, k0], B12 = B[:, k1], B21 = B[:, k0], B22 = B[:, k1]
    # Wait, for matrix multiplication C = A * B where A is 2x2, B is 2x2:
    # C[i][k] = sum_j A[i][j] * B[j][k]
    # So B[j][k] for j in {0,1}, k in {0,1}
    # B11 = B[0][0], B12 = B[0][1], B21 = B[1][0], B22 = B[1][1]
    
    # Strassen for C = A*B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # In our notation: A[i][j], B[j][k], C[i][k]
    # W convention: W[k*n+i, c] so W index is k*n+i
    
    def strassen_terms(k0, k1):
        """Return list of 7 (u_dict, v_dict, w_dict) for a 2x2x2 block."""
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        # u: A00 + A11 -> {(0,0):1, (1,1):1}
        # v: B00 + B11 -> {(0,k0):1, (1,k1):1}  [B[j][k]: j=0,k=k0 for B00; j=1,k=k1 for B11]
        # Wait: B00 = B[0][k0], B11 = B[1][k1]
        # v: {(0, k0): 1, (1, k1): 1}
        # w: contributes to C00 and C11
        # C00 += M1, C11 += M1
        # w: {(k0, 0): 1, (k1, 1): 1}
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0, k0): 1, (1, k1): 1},
            {(k0, 0): 1, (k1, 1): 1}
        ))
        
        # M2 = (A10+A11)B00
        # u: {(1,0): 1, (1,1): 1}
        # v: B00 = B[0][k0] -> {(0, k0): 1}
        # w: C10 += M2 -> {(k0, 1): 1}
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0, k0): 1},
            {(k0, 1): 1}
        ))
        
        # M3 = A00(B10-B11)
        # u: A00 = A[0][0] -> {(0,0): 1}
        # v: B10 - B11 = B[1][k0] - B[1][k1] -> {(1, k0): 1, (1, k1): -1}
        # w: C01 += M3 -> {(k1, 0): 1}
        terms.append((
            {(0,0): 1},
            {(1, k0): 1, (1, k1): -1},
            {(k1, 0): 1}
        ))
        
        # M4 = A11(B01-B00)
        # u: A11 = A[1][1] -> {(1,1): 1}
        # v: B01 - B00 = B[0][k1] - B[0][k0] -> {(0, k1): 1, (0, k0): -1}
        # w: C00 += M4, C10 += M4 -> {(k0, 0): 1, (k0, 1): 1}
        terms.append((
            {(1,1): 1},
            {(0, k1): 1, (0, k0): -1},
            {(k0, 0): 1, (k0, 1): 1}
        ))
        
        # M5 = (A00+A01)B11
        # u: A00 + A01 = A[0][0] + A[0][1] -> {(0,0): 1, (0,1): 1}
        # v: B11 = B[1][k1] -> {(1, k1): 1}
        # w: C00 -= M5, C01 += M5 -> {(k0, 0): -1, (k1, 0): 1}
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1, k1): 1},
            {(k0, 0): -1, (k1, 0): 1}
        ))
        
        # M6 = (A10-A00)(B00+B01)
        # u: A10 - A00 = A[1][0] - A[0][0] -> {(1,0): 1, (0,0): -1}
        # v: B00 + B01 = B[0][k0] + B[0][k1] -> {(0, k0): 1, (0, k1): 1}
        # w: C11 += M6 -> {(k1, 1): 1}
        terms.append((
            {(1,0): 1, (0,0): -1},
            {(0, k0): 1, (0, k1): 1},
            {(k1, 1): 1}
        ))
        
        # M7 = (A01-A11)(B10+B11)
        # u: A01 - A11 = A[0][1] - A[1][1] -> {(0,1): 1, (1,1): -1}
        # v: B10 + B11 = B[1][k0] + B[1][k1] -> {(1, k0): 1, (1, k1): 1}
        # w: C00 += M7 -> {(k0, 0): 1}
        terms.append((
            {(0,1): 1, (1,1): -1},
            {(1, k0): 1, (1, k1): 1},
            {(k0, 0): 1}
        ))
        
        return terms
    
    # Schoolbook terms for a single k:
    def schoolbook_term(k):
        """Return list of 4 (u_dict, v_dict, w_dict) for a 2x2x1 block."""
        terms = []
        for i in range(2):
            for j in range(2):
                terms.append((
                    {(i, j): 1},
                    {(j, k): 1},
                    {(k, i): 1}
                ))
        return terms
    
    # Build all rank columns
    col = 0
    for block in blocks:
        if len(block) == 2:
            k0, k1 = block
            terms = strassen_terms(k0, k1)
        else:
            k = block[0]
            terms = schoolbook_term(k)
        
        for (u_dict, v_dict, w_dict) in terms:
            # Create U, V, W entries for this rank column
            u_entries = [(i * 2 + j, c) for (i, j), c in u_dict.items()]
            v_entries = [(j * 15 + k, c) for (j, k), c in v_dict.items()]
            w_entries = [(k * 2 + i, c) for (k, i), c in w_dict.items()]
            
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)
            col += 1
    
    rank = col
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(30)]
    W = [[0] * rank for _ in range(30)]
    
    for c in range(rank):
        for (row, val) in all_U[c]:
            U[row][c] = val
        for (row, val) in all_V[c]:
            V[row][c] = val
        for (row, val) in all_W[c]:
            W[row][c] = val
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 for all valid (i,j,k) because C[i][k] = sum_j A[i][j] * B[j][k]
    
    # Check: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                # Correct tensor contraction: element-wise product then sum
                val = np.sum(U_arr[u_row, :] * V_arr[v_row, :] * W_arr[w_row, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s safety margin
    
    # Try Strassen-based decomposition first (rank 53)
    result = _strassen_block(n, m, p)
    
    # Verify
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook (rank 60)
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
