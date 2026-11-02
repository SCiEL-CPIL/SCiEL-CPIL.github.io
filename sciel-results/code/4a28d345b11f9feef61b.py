import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    deadline = time.time() + 110
    n, m, p = 2, 3, 8

    def verify(U, V, W, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

    def build_strassen_44():
        """
        Rank-44 decomposition.
        Split m=3 into 1+2.
        Part 1: 2x1 * 1x8 -> 16 schoolbook mults.
        Part 2: 2x2 * 2x8 -> 4 blocks of 2x2 * 2x2 (Strassen, rank 7 each) -> 28 mults.
        Total: 16 + 28 = 44.
        """
        rank = 44
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        col = 0
        
        # Part 1: A[:,0] (2x1) * B[0,:] (1x8)
        # U indices: i*3 + 0
        # V indices: 0*8 + k
        # W indices: k*2 + i
        for i in range(n):
            for k in range(p):
                U[i*m + 0][col] = 1
                V[0*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
        
        # Part 2: A[:,1:3] (2x2) * B[1:3,:] (2x8)
        # Split B[1:3,:] into 4 blocks of 2 columns: B[1:3, 0:2], B[1:3, 2:4], B[1:3, 4:6], B[1:3, 6:8]
        # Each block is 2x2.
        # A[:,1:3] is 2x2.
        # Use Strassen for each 2x2 * 2x2 block.
        
        for block in range(4):
            # B block indices
            # Row 1 of B (local row 0 of block): global row 1
            # Row 2 of B (local row 1 of block): global row 2
            # Cols: block*2, block*2+1
            
            # U indices for A[:,1:3]
            # Local A[0,0] -> global U[1*3+1] = U[4]
            # Local A[0,1] -> global U[1*3+2] = U[5]
            # Local A[1,0] -> global U[0*3+1] = U[1]
            # Local A[1,1] -> global U[0*3+2] = U[2]
            
            # Wait, U index is i*m + j.
            # i=0 (row 0 of A), j=1 -> U[0*3+1] = U[1]
            # i=0, j=2 -> U[0*3+2] = U[2]
            # i=1, j=1 -> U[1*3+1] = U[4]
            # i=1, j=2 -> U[1*3+2] = U[5]
            
            # V indices for B[1:3, block*2 : block*2+2]
            # Local B[0,0] -> global V[1*8 + block*2]
            # Local B[0,1] -> global V[1*8 + block*2 + 1]
            # Local B[1,0] -> global V[2*8 + block*2]
            # Local B[1,1] -> global V[2*8 + block*2 + 1]
            
            # W indices for C[block*2 : block*2+2, :]
            # Local C[0,0] (col 0 of block, row 0 of A) -> global W[(block*2)*2 + 0]
            # Local C[0,1] (col 1 of block, row 0 of A) -> global W[(block*2+1)*2 + 0]
            # Local C[1,0] (col 0 of block, row 1 of A) -> global W[(block*2)*2 + 1]
            # Local C[1,1] (col 1 of block, row 1 of A) -> global W[(block*2+1)*2 + 1]
            
            # Strassen Multiplications
            # M1 = (a00 + a11)(b00 + b11)
            # M2 = (a10 + a11)b00
            # M3 = a00(b01 - b11)
            # M4 = a11(b10 - b00)
            # M5 = (a00 + a01)b11
            # M6 = (a10 - a00)(b00 + b01)
            # M7 = (a01 - a11)(b10 + b11)
            
            # Note: Standard Strassen formulas vary. I will use a verified set.
            # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-e)
            # M5 = (a+b)h
            # M6 = (c-a)(e+f)
            # M7 = (b-d)(g+h)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            a00_idx = 1  # U[1]
            a01_idx = 2  # U[2]
            a10_idx = 4  # U[4]
            a11_idx = 5  # U[5]
            
            b00_idx = 8 + block*2
            b01_idx = 8 + block*2 + 1
            b10_idx = 16 + block*2
            b11_idx = 16 + block*2 + 1
            
            c00_idx = (block*2)*2 + 0
            c01_idx = (block*2+1)*2 + 0
            c10_idx = (block*2)*2 + 1
            c11_idx = (block*2+1)*2 + 1
            
            # M1
            U[a00_idx][col] = 1; U[a11_idx][col] = 1
            V[b00_idx][col] = 1; V[b11_idx][col] = 1
            W[c00_idx][col] = 1; W[c11_idx][col] = 1
            col += 1
            
            # M2
            U[a10_idx][col] = 1; U[a11_idx][col] = 1
            V[b00_idx][col] = 1
            W[c10_idx][col] = 1; W[c11_idx][col] = -1
            col += 1
            
            # M3
            U[a00_idx][col] = 1
            V[b01_idx][col] = 1; V[b11_idx][col] = -1
            W[c01_idx][col] = 1; W[c11_idx][col] = 1
            col += 1
            
            # M4
            U[a11_idx][col] = 1
            V[b10_idx][col] = 1; V[b00_idx][col] = -1
            W[c00_idx][col] = 1; W[c10_idx][col] = 1
            col += 1
            
            # M5
            U[a00_idx][col] = 1; U[a01_idx][col] = 1
            V[b11_idx][col] = 1
            W[c00_idx][col] = -1; W[c01_idx][col] = 1
            col += 1
            
            # M6
            U[a10_idx][col] = 1; U[a00_idx][col] = -1
            V[b00_idx][col] = 1; V[b01_idx][col] = 1
            W[c11_idx][col] = 1
            col += 1
            
            # M7
            U[a01_idx][col] = 1; U[a11_idx][col] = -1
            V[b10_idx][col] = 1; V[b11_idx][col] = 1
            W[c00_idx][col] = 1
            col += 1
            
        return U, V, W

    # Try rank-44 construction
    U, V, W = build_strassen_44()
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook (should not happen)
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
