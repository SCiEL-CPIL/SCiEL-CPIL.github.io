import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return 7 factor triples for 2x2x2 Strassen decomposition."""
    # Each triple is (U_vec, V_vec, W_vec) of length 4
    # U indices: [a11, a12, a21, a22]
    # V indices: [b11, b12, b21, b22]
    # W indices: [c11, c12, c21, c22]
    return [
        ([1, 0, 0, 1],   [1, 0, 0, 1],   [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1],   [1, 0, 0, 0],   [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0],   [0, 1, 0, -1],  [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1],   [-1, 0, 1, 0],  [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0],   [0, 0, 0, 1],   [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0],  [1, 1, 0, 0],   [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1],  [0, 0, 1, 1],   [1, 0, 0, 0]),   # M7
    ]

def _strassen_4x4x4():
    """
    Construct 4x4x4 decomposition using recursive Strassen.
    Returns list of (U_vec, V_vec, W_vec) where each vec has length 16.
    U indices: [a00,a01,a02,a03,a10,a11,a12,a13,a20,a21,a22,a23,a30,a31,a32,a33]
    V indices: [b00,b01,b02,b03,b10,b11,b12,b13,b20,b21,b22,b23,b30,b31,b32,b33]
    W indices: [c00,c01,c02,c03,c10,c11,c12,c13,c20,c21,c22,c23,c30,c31,c32,c33]
    """
    s2 = _strassen_2x2x2()
    # 4x4x4 = 8 sub-blocks of 2x2x2, each using 7 Strassen terms
    # Total rank = 8 * 7 = 56... wait, that's not 49.
    # Actually, 4x4x4 via Strassen: split into 2x2 blocks, each 2x2x2 uses rank 7.
    # There are 2*2*2 = 8 sub-blocks, so rank = 8*7 = 56.
    # Hmm, but I said 49 earlier. Let me reconsider.
    # 4x4x4: if we use Strassen recursively, it's 7^2 = 49 only if we can share terms.
    # Actually, the standard recursive Strassen for n x n x n gives rank 7^log2(n).
    # For n=4: 7^2 = 49. But this requires that the 4x4x4 tensor can be decomposed
    # into 49 terms. Let me verify: 4x4x4 has 64 entries. The Strassen algorithm for
    # 4x4 matrix multiplication uses 49 multiplications (7^2). Yes, that's correct.
    # But how to construct it? We need to combine the 8 sub-blocks' Strassen terms
    # in a way that shares some multiplications.
    
    # Actually, the recursive Strassen works as follows:
    # Split 4x4 into four 2x2 quadrants.
    # C = A*B where A,B,C are 4x4.
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2.
    # Similarly for B and C.
    # Then C11 = A11*B11 + A12*B21, etc.
    # Using Strassen on the 2x2 block structure:
    # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
    # M2 = (A21+A22)B11 -> contributes to C21, C22
    # M3 = A11(B12-B22) -> contributes to C12, C22
    # M4 = A22(B21-B11) -> contributes to C11, C21
    # M5 = (A11+A12)B22 -> contributes to C11, C12
    # M6 = (A21-A11)(B11+B12) -> contributes to C22
    # M7 = (A12-A22)(B21+B22) -> contributes to C11
    
    # Each Mi is a 2x2 matrix product, which itself requires 7 scalar multiplications.
    # So total = 7 * 7 = 49 scalar multiplications.
    
    # To construct the full 4x4x4 decomposition, I need to expand each 2x2x2 Strassen
    # term into 7 scalar terms, and combine them.
    
    # This is complex. Let me build it step by step.
    
    # For each of the 7 Strassen terms at the top level, I need to decompose
    # the 2x2x2 sub-problem into 7 scalar multiplications.
    
    # Let me define the index mapping for 4x4:
    # A index: (i,j) -> i*4+j for i,j in 0..3
    # B index: (j,k) -> j*4+k
    # C index: (k,i) -> k*4+i
    
    factors = []
    
    # For each top-level Strassen term (t=0..6), and for each sub-Strassen term (s=0..6)
    for t in range(7):
        U_t, V_t, W_t = s2[t]
        # U_t, V_t, W_t are length-4 vectors for the 2x2 block structure
        # Now I need to expand each 2x2x2 term into 7 scalar terms
        
        # The 2x2 blocks are:
        # A: A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
        # B: B00 = B[0:2,0:2], B01 = B[0:2,2:4], B10 = B[2:4,0:2], B11 = B[2:4,2:4]
        # C: C00 = C[0:2,0:2], C01 = C[0:2,2:4], C10 = C[2:4,0:2], C11 = C[2:4,2:4]
        
        # For the t-th Strassen term, the effective 2x2 matrices are:
        # A_eff = U_t[0]*A00 + U_t[1]*A01 + U_t[2]*A10 + U_t[3]*A11
        # B_eff = V_t[0]*B00 + V_t[1]*B01 + V_t[2]*B10 + V_t[3]*B11
        # C_eff = W_t[0]*C00 + W_t[1]*C01 + W_t[2]*C10 + W_t[3]*C11
        
        # Now A_eff * B_eff = C_eff, where each is a 2x2 matrix.
        # This 2x2x2 product is decomposed using Strassen with 7 terms.
        
        for s in range(7):
            U_s, V_s, W_s = s2[s]
            # U_s, V_s, W_s are length-4 vectors for the scalar 2x2x2 decomposition
            
            # Build full 16-length vectors
            U_full = np.zeros(16)
            V_full = np.zeros(16)
            W_full = np.zeros(16)
            
            # For A_eff: it's a linear combination of A00, A01, A10, A11
            # Each of those is a 2x2 matrix, which maps to 4 scalar indices in the 4x4 A.
            # A00: indices 0,1,4,5 (rows 0-1, cols 0-1)
            # A01: indices 2,3,6,7 (rows 0-1, cols 2-3)
            # A10: indices 8,9,12,13 (rows 2-3, cols 0-1)
            # A11: indices 10,11,14,15 (rows 2-3, cols 2-3)
            
            # The 2x2x2 Strassen for A_eff*B_eff=C_eff:
            # U_s[0] corresponds to A_eff[0,0], U_s[1] to A_eff[0,1], etc.
            # A_eff[0,0] = U_t[0]*A00[0,0] + U_t[1]*A01[0,0] + U_t[2]*A10[0,0] + U_t[3]*A11[0,0]
            # A_eff[0,1] = U_t[0]*A00[0,1] + U_t[1]*A01[0,1] + U_t[2]*A10[0,1] + U_t[3]*A11[0,1]
            # A_eff[1,0] = U_t[0]*A00[1,0] + U_t[1]*A01[1,0] + U_t[2]*A10[1,0] + U_t[3]*A11[1,0]
            # A_eff[1,1] = U_t[0]*A00[1,1] + U_t[1]*A01[1,1] + U_t[2]*A10[1,1] + U_t[3]*A11[1,1]
            
            # So for each scalar component of A_eff, I add U_s[idx] * U_t[block] to the corresponding A indices.
            
            # A_eff[0,0] -> A indices: A00[0,0]=0, A01[0,0]=2, A10[0,0]=8, A11[0,0]=10
            # A_eff[0,1] -> A indices: A00[0,1]=1, A01[0,1]=3, A10[0,1]=9, A11[0,1]=11
            # A_eff[1,0] -> A indices: A00[1,0]=4, A01[1,0]=6, A10[1,0]=12, A11[1,0]=14
            # A_eff[1,1] -> A indices: A00[1,1]=5, A01[1,1]=7, A10[1,1]=13, A11[1,1]=15
            
            a_block_map = [
                [0, 2, 8, 10],   # A_eff[0,0]
                [1, 3, 9, 11],   # A_eff[0,1]
                [4, 6, 12, 14],  # A_eff[1,0]
                [5, 7, 13, 15],  # A_eff[1,1]
            ]
            b_block_map = [
                [0, 2, 8, 10],   # B_eff[0,0]
                [1, 3, 9, 11],   # B_eff[0,1]
                [4, 6, 12, 14],  # B_eff[1,0]
                [5, 7, 13, 15],  # B_eff[1,1]
            ]
            c_block_map = [
                [0, 2, 8, 10],   # C_eff[0,0]
                [1, 3, 9, 11],   # C_eff[0,1]
                [4, 6, 12, 14],  # C_eff[1,0]
                [5, 7, 13, 15],  # C_eff[1,1]
            ]
            
            for comp in range(4):
                coeff = U_s[comp]
                if coeff != 0:
                    for block_idx in range(4):
                        val = coeff * U_t[block_idx]
                        if val != 0:
                            U_full[a_block_map[comp][block_idx]] += val
                
                coeff = V_s[comp]
                if coeff != 0:
                    for block_idx in range(4):
                        val = coeff * V_t[block_idx]
                        if val != 0:
                            V_full[b_block_map[comp][block_idx]] += val
                
                coeff = W_s[comp]
                if coeff != 0:
                    for block_idx in range(4):
                        val = coeff * W_t[block_idx]
                        if val != 0:
                            W_full[c_block_map[comp][block_idx]] += val
            
            factors.append((U_full.tolist(), V_full.tolist(), W_full.tolist()))
    
    return factors

def _verify(n, m, p, U, V, W, tol=1e-9):
    """Verify the decomposition."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_5x5x5():
    """
    Construct 5x5x5 decomposition using 4+1 partition.
    4x4x4 core: rank 49 (recursive Strassen)
    Remaining blocks: schoolbook
    Total: 49 + 16+16+16+4+4+4+1 = 110
    """
    n, m, p = 5, 5, 5
    
    # Get 4x4x4 Strassen factors
    strassen_4x4x4 = _strassen_4x4x4()
    rank_4x4x4 = len(strassen_4x4x4)  # Should be 49
    
    # Calculate total rank
    # 4x4x4: 49
    # 4x4x1: 16
    # 4x1x4: 16
    # 1x4x4: 16
    # 4x1x1: 4
    # 1x4x1: 4
    # 1x1x4: 4
    # 1x1x1: 1
    total_rank = rank_4x4x4 + 16 + 16 + 16 + 4 + 4 + 4 + 1
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col = 0
    
    # 4x4x4 core block (indices 0-3 in each dimension)
    for (uf, vf, wf) in strassen_4x4x4:
        for idx in range(16):
            if uf[idx] != 0:
                U[idx, col] += uf[idx]
            if vf[idx] != 0:
                V[idx, col] += vf[idx]
            if wf[idx] != 0:
                W[idx, col] += wf[idx]
        col += 1
    
    # 4x4x1 block: i in 0-3, j in 0-3, k=4
    for i in range(4):
        for j in range(4):
            U_idx = i * m + j
            V_idx = j * p + 4
            W_idx = 4 * n + i
            U[U_idx, col] = 1.0
            V[V_idx, col] = 1.0
            W[W_idx, col] = 1.0
            col += 1
    
    # 4x1x4 block: i in 0-3, j=4, k in 0-3
    for i in range(4):
        for k in range(4):
            U_idx = i * m + 4
            V_idx = 4 * p + k
            W_idx = k * n + i
            U[U_idx, col] = 1.0
            V[V_idx, col] = 1.0
            W[W_idx, col] = 1.0
            col += 1
    
    # 1x4x4 block: i=4, j in 0-3, k in 0-3
    for j in range(4):
        for k in range(4):
            U_idx = 4 * m + j
            V_idx = j * p + k
            W_idx = k * n + 4
            U[U_idx, col] = 1.0
            V[V_idx, col] = 1.0
            W[W_idx, col] = 1.0
            col += 1
    
    # 4x1x1 block: i in 0-3, j=4, k=4
    for i in range(4):
        U_idx = i * m + 4
        V_idx = 4 * p + 4
        W_idx = 4 * n + i
        U[U_idx, col] = 1.0
        V[V_idx, col] = 1.0
        W[W_idx, col] = 1.0
        col += 1
    
    # 1x4x1 block: i=4, j in 0-3, k=4
    for j in range(4):
        U_idx = 4 * m + j
        V_idx = j * p + 4
        W_idx = 4 * n + 4
        U[U_idx, col] = 1.0
        V[V_idx, col] = 1.0
        W[W_idx, col] = 1.0
        col += 1
    
    # 1x1x4 block: i=4, j=4, k in 0-3
    for k in range(4):
        U_idx = 4 * m + 4
        V_idx = 4 * p + k
        W_idx = k * n + 4
        U[U_idx, col] = 1.0
        V[V_idx, col] = 1.0
        W[W_idx, col] = 1.0
        col += 1
    
    # 1x1x1 block: i=4, j=4, k=4
    U_idx = 4 * m + 4
    V_idx = 4 * p + 4
    W_idx = 4 * n + 4
    U[U_idx, col] = 1.0
    V[V_idx, col] = 1.0
    W[W_idx, col] = 1.0
    col += 1
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, total_rank

def construct():
    """Main entry point."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    result, rank = _construct_5x5x5()
    
    if _verify(n, m, p, result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: schoolbook rank 125
    U = np.zeros((n * m, 125), dtype=np.float64)
    V = np.zeros((m * p, 125), dtype=np.float64)
    W = np.zeros((p * n, 125), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
