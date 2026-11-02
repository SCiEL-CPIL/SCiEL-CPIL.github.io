"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    # Use numpy for faster initialization and manipulation, then convert to lists
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_strassen_block(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        """
        Add a Strassen block for multiplying A[a_row_base:a_row_base+2, a_col_base:a_col_base+2]
        by B[b_row_base:b_row_base+2, b_col_base:b_col_base+2]
        contributing to C[c_row_base:c_row_base+2, c_col_base:c_col_base+2]
        
        U index for A element (i,j): i*m + j
        V index for B element (i,j): i*p + j
        W index for C element (i,j): j*n + i (trace convention: W[k*n+i, r] where k=col, i=row)
        """
        # U indices for A elements
        a00_u = a_row_base * m + a_col_base
        a01_u = a_row_base * m + a_col_base + 1
        a10_u = (a_row_base + 1) * m + a_col_base
        a11_u = (a_row_base + 1) * m + a_col_base + 1
        
        # V indices for B elements
        b00_v = b_row_base * p + b_col_base
        b01_v = b_row_base * p + b_col_base + 1
        b10_v = (b_row_base + 1) * p + b_col_base
        b11_v = (b_row_base + 1) * p + b_col_base + 1
        
        # W indices for C elements (trace convention: W[k*n+i, r] where k=col, i=row)
        # C element at row i, col j has W index = j*n + i
        c00_w = c_col_base * n + c_row_base
        c01_w = (c_col_base + 1) * n + c_row_base
        c10_w = c_col_base * n + (c_row_base + 1)
        c11_w = (c_col_base + 1) * n + (c_row_base + 1)
        
        # M1 = (a00+a11)(b00+b11)
        U[a00_u, offset + 0] += 1
        U[a11_u, offset + 0] += 1
        V[b00_v, offset + 0] += 1
        V[b11_v, offset + 0] += 1
        W[c00_w, offset + 0] += 1
        W[c11_w, offset + 0] += 1
        
        # M2 = (a10+a11)b00
        U[a10_u, offset + 1] += 1
        U[a11_u, offset + 1] += 1
        V[b00_v, offset + 1] += 1
        W[c10_w, offset + 1] += 1
        W[c11_w, offset + 1] -= 1
        
        # M3 = a00(b01-b11)
        U[a00_u, offset + 2] += 1
        V[b01_v, offset + 2] += 1
        V[b11_v, offset + 2] -= 1
        W[c01_w, offset + 2] += 1
        W[c11_w, offset + 2] += 1
        
        # M4 = a11(b10-b00)
        U[a11_u, offset + 3] += 1
        V[b10_v, offset + 3] += 1
        V[b00_v, offset + 3] -= 1
        W[c00_w, offset + 3] += 1
        W[c10_w, offset + 3] += 1
        
        # M5 = (a00+a01)b11
        U[a00_u, offset + 4] += 1
        U[a01_u, offset + 4] += 1
        V[b11_v, offset + 4] += 1
        W[c00_w, offset + 4] -= 1
        W[c01_w, offset + 4] += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10_u, offset + 5] += 1
        U[a00_u, offset + 5] -= 1
        V[b00_v, offset + 5] += 1
        V[b01_v, offset + 5] += 1
        W[c11_w, offset + 5] += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01_u, offset + 6] += 1
        U[a11_u, offset + 6] -= 1
        V[b10_v, offset + 6] += 1
        V[b11_v, offset + 6] += 1
        W[c00_w, offset + 6] += 1

    # Block 1: A_left (cols 0-1) * B_top_left (rows 0-1, cols 0-1) -> C_left (cols 0-1)
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A_right (cols 2-3) * B_bottom_left (rows 2-3, cols 0-1) -> C_left (cols 0-1)
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    
    # Block 3: A_left (cols 0-1) * B_top_right (rows 0-1, cols 2-3) -> C_right (cols 2-3)
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    
    # Block 4: A_right (cols 2-3) * B_bottom_right (rows 2-3, cols 2-3) -> C_right (cols 2-3)
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
            
    error = np.max(np.abs(C - Cf))
    print(f"Max Error: {error}")
    assert error < 1e-9, "Decomposition failed"
