import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 68
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    # A11 * B11 using Strassen-like decomposition (2x2 matrix times 2x8 matrix)
    # A11 = A[0:2, 0:2] (2x2)
    # B11 = B[0:2, 0:8] (2x8)
    # Result C11 = C[0:2, 0:8] (2x8)
    
    # M1 = (A11_11 + A11_22) * (B11_11 + B11_22)
    # A11_11 = A[0,0], A11_22 = A[1,1]
    # B11_11 = B[0,0:4], B11_22 = B[1,0:4]  (splitting 8 cols into 4+4)
    # Actually, Strassen works on 2x2 blocks. For 2x8, we need to split differently.
    # Let's use the standard approach: treat B11 as two 2x4 blocks.
    # Or better, just use the fact that rank(2x2x8) = 2*2*8 = 32 for schoolbook.
    # But we can do better. The rank of 2x2x8 is known to be 15 (or similar).
    # Actually, for 2x2xp, the minimal rank is 2p for p>=2? No, that's not right.
    # For 2x2x2, rank is 7 (Strassen).
    # For 2x2x8, we can view it as 2x2x2 repeated 4 times, giving rank 7*4 = 28.
    
    # Let's use the block approach:
    # Split B11 (2x8) into four 2x2 blocks: B11_0, B11_1, B11_2, B11_3
    # For each 2x2 block, use Strassen with the same 2x2 A11.
    # This gives 4 * 7 = 28 terms for A11 * B11.
    
    for b in range(4):
        # b-th block of B11: columns 2b to 2b+1
        col_start = 2 * b
        
        # M1 = (A[0,0] + A[1,1]) * (B[0, col_start:col_start+2] + B[1, col_start:col_start+2])
        # We need separate terms for each column
        for c in range(2):
            k = col_start + c  # global column index in B, 0..7
            # M1 term
            U[0 * 3 + 0, idx] = 1  # A[0,0]
            U[1 * 3 + 1, idx] = 1  # A[1,1]
            V[0 * 8 + k, idx] = 1  # B[0, k]
            V[1 * 8 + k, idx] = 1  # B[1, k]
            W[k * 3 + 0, idx] = 1  # C[0, k]
            W[k * 3 + 1, idx] = 1  # C[1, k]
            idx += 1
            # M2 = (A[1,0] + A[1,1]) * B[0, k]
            U[1 * 3 + 0, idx] = 1
            U[1 * 3 + 1, idx] = 1
            V[0 * 8 + k, idx] = 1
            W[k * 3 + 1, idx] = 1
            idx += 1
            # M3 = A[0,0] * (B[1, k] - B[0, k])
            U[0 * 3 + 0, idx] = 1
            V[1 * 8 + k, idx] = 1
            V[0 * 8 + k, idx] = -1
            W[k * 3 + 0, idx] = 1
            W[k * 3 + 1, idx] = 0
            idx += 1
            # M4 = A[1,1] * (B[0, k] - B[1, k])
            U[1 * 3 + 1, idx] = 1
            V[0 * 8 + k, idx] = 1
            V[1 * 8 + k, idx] = -1
            W[k * 3 + 1, idx] = 1
            idx += 1
            # M5 = (A[0,0] + A[0,1]) * B[1, k]
            U[0 * 3 + 0, idx] = 1
            U[0 * 3 + 1, idx] = 1
            V[1 * 8 + k, idx] = 1
            W[k * 3 + 0, idx] = 1
            idx += 1
            # M6 = (A[1,0] - A[0,0]) * (B[0, k] + B[1, k])
            U[1 * 3 + 0, idx] = 1
            U[0 * 3 + 0, idx] = -1
            V[0 * 8 + k, idx] = 1
            V[1 * 8 + k, idx] = 1
            W[k * 3 + 0, idx] = 1
            W[k * 3 + 1, idx] = 0
            idx += 1
            # M7 = (A[0,1] - A[1,1]) * (B[1, k] + B[0, k])
            U[0 * 3 + 1, idx] = 1
            U[1 * 3 + 1, idx] = -1
            V[1 * 8 + k, idx] = 1
            V[0 * 8 + k, idx] = 1
            W[k * 3 + 0, idx] = 1
            W[k * 3 + 1, idx] = 0
            idx += 1
    
    # Now we need to combine these to get the correct output.
    # This is getting complicated. Let's use a simpler approach.
    # Schoolbook for 3x3 @ 3x8 has rank 72.
    # We can try to reduce by noting that the last row/column has structure.
    
    # Reset and use schoolbook, but try to merge some terms.
    # Actually, the minimal rank for 3x3x8 is likely 24 or so.
    # Let's just use schoolbook to be safe, as it's guaranteed correct.
    # The task says "find a rank-minimal exact tensor decomposition" but the score is just rank.
    # Lower rank is better. Schoolbook gives 72.
    
    # Let me try a different approach: use the fact that we can share computations.
    # For now, return schoolbook which is valid.
    
    # Reset
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    
    # Convert to lists
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

if __name__ == "__main__":
    result = construct()
    print(f"Constructed rank {len(result['U'][0])}")
