"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try rank 16 first, fall back to 17, then schoolbook
    for target_rank in [16, 17, n*m*p]:
        result = _try_decomposition(n, m, p, target_rank)
        if result is not None:
            return result
    return _schoolbook()

def _try_decomposition(n, m, p, target_rank):
    # Use Strassen for 2x2x2 sub-block (j in {0,1}, k in {0,1})
    # Then handle remaining terms with possible sharing
    
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 1],
        [0, 0, 0, 0, 1, 0, -1],
        [1, 1, 0, 0, 0, 1, 0],
        [1, 0, -1, 1, 0, 0, -1]
    ], dtype=float)
    
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, -1, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1]
    ], dtype=float)
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 0, 0, 0, 1, 0]
    ], dtype=float)
    
    U = np.zeros((n*m, target_rank), dtype=float)
    V = np.zeros((m*p, target_rank), dtype=float)
    W = np.zeros((p*n, target_rank), dtype=float)
    
    # Embed Strassen in j in {0,1}, k in {0,1}
    for i in range(2):
        for j in range(2):
            U[i*m + j, 0:7] = U_s[i*2 + j, :]
    for j in range(2):
        for k in range(2):
            V[j*p + k, 0:7] = V_s[j*2 + k, :]
    for i in range(2):
        for k in range(2):
            W[k*n + i, 0:7] = W_s[i*2 + k, :]
    
    # Remaining terms: j=2 or k=2 (10 terms)
    # For rank 16, we need to fit 10 terms in 9 columns (16-7=9)
    # Try to find a valid decomposition by combining terms
    
    if target_rank == 17:
        # Schoolbook for remaining 10 terms
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if j < 2 and k < 2:
                        continue
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        if col == target_rank and _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None
    
    if target_rank == 16:
        # Need to fit 10 cross terms in 9 columns
        # Try various combinations
        return _try_rank16(n, m, p, U, V, W)
    
    return None

def _try_rank16(n, m, p, U, V, W):
    # Remaining 10 terms need to fit in 9 columns (cols 7-15)
    # Terms: (i,j,k) where j=2 or k=2
    # (0,0,2), (0,1,2), (0,2,0), (0,2,1), (0,2,2)
    # (1,0,2), (1,1,2), (1,2,0), (1,2,1), (1,2,2)
    
    # Try to combine (0,2,2) and (1,2,2) using a Strassen-like trick
    # Or combine other pairs
    
    # One approach: use 2 terms to handle both i values for j=2,k=2
    # C[0,2] += A[0,2]*B[2,2]
    # C[1,2] += A[1,2]*B[2,2]
    # These are separate terms in schoolbook
    
    # Try: use column 7 for A[0,2]*B[2,2], column 8 for A[1,2]*B[2,2]
    # Then we have 7 more columns for 8 remaining terms - still need to save 1
    
    # Alternative: combine (0,0,2) and (1,0,2) or similar
    
    # Let me try a different Strassen-like combination
    # For the k=2 column: A[i,0]B[0,2] + A[i,1]B[1,2] + A[i,2]B[2,2]
    # For the j=2 row: A[i,2]B[2,0] + A[i,2]B[2,1] + A[i,2]B[2,2]
    
    # We can use Strassen on the 2x2 part of k=2 column (j=0,1) for each i
    # But that would add more terms
    
    # Try brute force: assign each of the 10 terms to one of 9 columns
    # with possible linear combinations
    
    # Actually, let's try a known rank-16 construction
    # Use 7 Strassen + 9 additional terms with clever sharing
    
    # Column 7: handle A[0,2]B[2,0] and A[1,2]B[2,0] together? No, they go to different C entries
    
    # Let me try: use 2 columns for the j=2 row (B[2,0], B[2,1], B[2,2])
    # and 2 columns for the k=2 column (A[0,0]B[0,2], A[0,1]B[1,2], A[1,0]B[0,2], A[1,1]B[1,2])
    # and 1 column for A[0,2]B[2,2] and A[1,2]B[2,2] - but these are separate
    
    # This is getting complex. Let me try a direct search for a valid rank-16 decomposition
    
    # For now, return None to fall back to rank 17
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p):
    # Check sum_c U_ij,c V_jk,c W_ik,c == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-END
