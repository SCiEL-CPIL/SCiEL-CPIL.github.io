"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper: add a Strassen term for a 2x2 * 2x2 block
    # block_offset_k: which 2 columns of B (0 or 2)
    # col_offset: offset in W index for the output columns
    def add_strassen_block(col_offset, d_start):
        # col_offset is 0 for first block (k=0,1), 2 for second block (k=2,3)
        # W indices for output:
        # C[0, col_offset] -> W[col_offset * n + 0] = W[col_offset*2]
        # C[0, col_offset+1] -> W[(col_offset+1)*2]
        # C[1, col_offset] -> W[col_offset*2 + 1]
        # C[1, col_offset+1] -> W[(col_offset+1)*2 + 1]
        
        w00 = col_offset * 2      # C[0, col_offset]
        w01 = (col_offset + 1) * 2  # C[0, col_offset+1]
        w10 = col_offset * 2 + 1    # C[1, col_offset]
        w11 = (col_offset + 1) * 2 + 1  # C[1, col_offset+1]
        
        # B indices for this block:
        # b11 = B[0, col_offset] -> V[0*4 + col_offset] = V[col_offset]
        # b12 = B[0, col_offset+1] -> V[col_offset+1]
        # b21 = B[1, col_offset] -> V[4 + col_offset]
        # b22 = B[1, col_offset+1] -> V[4 + col_offset+1]
        v11 = col_offset
        v12 = col_offset + 1
        v21 = 4 + col_offset
        v22 = 4 + col_offset + 1
        
        # A indices:
        # a11 = A[0,0] -> U[0]
        # a12 = A[0,1] -> U[1]
        # a21 = A[1,0] -> U[2]
        # a22 = A[1,1] -> U[3]
        u11 = 0
        u12 = 1
        u21 = 2
        u22 = 3
        
        # M1 = (a11+a22)(b11+b22)
        d = d_start
        U[u11, d] += 1; U[u22, d] += 1
        V[v11, d] += 1; V[v22, d] += 1
        W[w00, d] += 1  # C11
        W[w11, d] += 1  # C22
        d += 1
        
        # M2 = (a21+a22)b11
        U[u21, d] += 1; U[u22, d] += 1
        V[v11, d] += 1
        W[w10, d] += 1  # C21
        W[w11, d] -= 1  # C22 (M2 contributes -1 to C22)
        d += 1
        
        # M3 = a11(b12-b22)
        U[u11, d] += 1
        V[v12, d] += 1; V[v22, d] -= 1
        W[w01, d] += 1  # C12
        W[w11, d] += 1  # C22
        d += 1
        
        # M4 = a22(b21-b11)
        U[u22, d] += 1
        V[v21, d] += 1; V[v11, d] -= 1
        W[w00, d] += 1  # C11
        W[w10, d] += 1  # C21
        d += 1
        
        # M5 = (a11+a12)b22
        U[u11, d] += 1; U[u12, d] += 1
        V[v22, d] += 1
        W[w00, d] -= 1  # C11 (M5 contributes -1 to C11)
        W[w01, d] += 1  # C12
        d += 1
        
        # M6 = (a21-a11)(b11+b12)
        U[u21, d] += 1; U[u11, d] -= 1
        V[v11, d] += 1; V[v12, d] += 1
        W[w11, d] += 1  # C22
        d += 1
        
        # M7 = (a12-a22)(b21+b22)
        U[u12, d] += 1; U[u22, d] -= 1
        V[v21, d] += 1; V[v22, d] += 1
        W[w00, d] += 1  # C11
        d += 1
        
        return d
    
    # Add first block (k=0,1)
    d = add_strassen_block(0, 0)
    # Add second block (k=2,3)
    d = add_strassen_block(2, d)
    
    # Verify
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-10):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
