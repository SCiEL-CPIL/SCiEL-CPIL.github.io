import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Strategy: 
    # For i in {0,1}: use 4 non-overlapping 2x2x2 Strassen blocks in the j-k plane
    #   Blocks at (j0,k0) = (0,0), (2,0), (0,2), (2,2), each covering j in {j0,j0+1}, k in {k0,k0+1}
    #   4 blocks * rank 7 = 28, plus 9 schoolbook terms for uncovered (j,k) pairs
    # For i = 2: 25 schoolbook terms
    # Total rank: 28 + 9 + 25 = 62
    
    blocks_jk = [(0, 0), (2, 0), (0, 2), (2, 2)]
    
    # Determine which (j,k) pairs are covered by Strassen blocks for i in {0,1}
    covered_jk = set()
    for j0, k0 in blocks_jk:
        for dj in range(2):
            for dk in range(2):
                covered_jk.add((j0 + dj, k0 + dk))
    
    # Count schoolbook terms for i in {0,1}
    schoolbook_i01 = 0
    for j in range(m):
        for k in range(p):
            if (j, k) not in covered_jk:
                schoolbook_i01 += 1
    
    # Schoolbook terms for i=2
    schoolbook_i2 = m * p
    
    total_rank = len(blocks_jk) * 7 + schoolbook_i01 + schoolbook_i2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Add Strassen blocks for i in {0,1}
    for i in [0, 1]:
        for (j0, k0) in blocks_jk:
            i0 = i
            i1 = i + 1  # Wait, this is wrong. The Strassen block spans i in {0,1}, not {i, i+1}
            # Actually, all 4 blocks share the same i range {0,1}. Let me restructure.
            pass
    
    # Let me redo this properly. The 4 blocks all use i in {0,1}.
    # I need to add them all for the shared i range.
    
    # Reset
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    col = 0
    
    def add_strassen_block(i0, i1, j0, k0, col_offset):
        """Add a 2x2x2 Strassen block for i in {i0,i1}, j in {j0,j0+1}, k in {k0,k0+1}"""
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0, c] += 1
        U[i1*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1
        W[(k0+1)*n + i1, c] += 1
        
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[i1*m + j0, c] += 1
        U[i1*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        W[k0*n + i1, c] += 1
        W[(k0+1)*n + i1, c] += -1
        
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0, c] += 1
        V[j0*p + (k0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += -1
        W[(k0+1)*n + i0, c] += 1
        W[(k0+1)*n + i1, c] += 1
        
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[i1*m + (j0+1), c] += 1
        V[(j0+1)*p + k0, c] += 1
        V[j0*p + k0, c] += -1
        W[k0*n + i0, c] += 1
        W[k0*n + i1, c] += 1
        
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0, c] += 1
        U[i0*m + (j0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += -1
        W[(k0+1)*n + i0, c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[i1*m + j0, c] += 1
        U[i0*m + j0, c] += -1
        V[j0*p + k0, c] += 1
        V[j0*p + (k0+1), c] += 1
        W[(k0+1)*n + i1, c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1), c] += 1
        U[i1*m + (j0+1), c] += -1
        V[(j0+1)*p + k0, c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1
    
    # Add 4 Strassen blocks, all with i in {0,1}
    for j0, k0 in blocks_jk:
        add_strassen_block(0, 1, j0, k0, col)
        col += 7
    
    # Schoolbook for i in {0,1}, uncovered (j,k) pairs
    for i in [0, 1]:
        for j in range(m):
            for k in range(p):
                if (j, k) not in covered_jk:
                    U[i*m + j, col] += 1
                    V[j*p + k, col] += 1
                    W[k*n + i, col] += 1
                    col += 1
    
    # Schoolbook for i=2
    for j in range(m):
        for k in range(p):
            U[2*m + j, col] += 1
            V[j*p + k, col] += 1
            W[k*n + 2, col] += 1
            col += 1
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    
    # Verify: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    T_reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_reconstructed[i,j,k] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
    
    T_expected = np.ones((n, m, p))
    max_err = np.max(np.abs(T_reconstructed - T_expected))
    print(f"Max error: {max_err}")
    
    if max_err < 1e-10:
        print("Decomposition is correct!")
    else:
        print("Decomposition has errors!")
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T_reconstructed[i,j,k] - 1.0) > 1e-10:
                        print(f"  T[{i},{j},{k}] = {T_reconstructed[i,j,k]} (expected 1.0)")

