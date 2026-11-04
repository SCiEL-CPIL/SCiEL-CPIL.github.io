import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145
    
    n, m, p = DIMENSIONS
    
    def build_rank50():
        rank = 50
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        col = 0
        
        def emit_strassen(i0, j0, k0):
            nonlocal col
            # Standard Strassen 7 products for 2x2x2 block
            # A block: rows i0, i0+1; cols j0, j0+1
            # B block: rows j0, j0+1; cols k0, k0+1
            # C block: rows i0, i0+1; cols k0, k0+1
            
            # M1 = (a+d)(e+h)
            U[i0*m + j0, col] = 1.0
            U[(i0+1)*m + (j0+1), col] = 1.0
            V[j0*p + k0, col] = 1.0
            V[(j0+1)*p + (k0+1), col] = 1.0
            W[k0*n + i0, col] = 1.0
            W[(k0+1)*n + (i0+1), col] = 1.0
            col += 1
            
            # M2 = (c+d)(g+h)
            U[(i0+1)*m + j0, col] = 1.0
            U[(i0+1)*m + (j0+1), col] = 1.0
            V[j0*p + k0, col] = 1.0
            W[k0*n + (i0+1), col] = 1.0
            W[(k0+1)*n + (i0+1), col] = -1.0
            col += 1
            
            # M3 = a(f-h)
            U[i0*m + j0, col] = 1.0
            V[j0*p + (k0+1), col] = 1.0
            V[(j0+1)*p + (k0+1), col] = -1.0
            W[(k0+1)*n + i0, col] = 1.0
            W[(k0+1)*n + (i0+1), col] = 1.0
            col += 1
            
            # M4 = d(b-c)
            U[(i0+1)*m + (j0+1), col] = 1.0
            V[(j0+1)*p + k0, col] = 1.0
            V[j0*p + k0, col] = -1.0
            W[k0*n + (i0+1), col] = 1.0
            W[k0*n + i0, col] = 1.0
            col += 1
            
            # M5 = (a+b)h
            U[i0*m + j0, col] = 1.0
            U[i0*m + (j0+1), col] = 1.0
            V[(j0+1)*p + (k0+1), col] = 1.0
            W[k0*n + i0, col] = -1.0
            W[(k0+1)*n + i0, col] = 1.0
            col += 1
            
            # M6 = (c+d)e
            U[(i0+1)*m + j0, col] = 1.0
            U[i0*m + j0, col] = -1.0
            V[j0*p + k0, col] = 1.0
            V[j0*p + (k0+1), col] = 1.0
            W[(k0+1)*n + (i0+1), col] = 1.0
            col += 1
            
            # M7 = (a-c)(g+d)
            U[i0*m + (j0+1), col] = 1.0
            U[(i0+1)*m + (j0+1), col] = -1.0
            V[(j0+1)*p + k0, col] = 1.0
            V[(j0+1)*p + (k0+1), col] = 1.0
            W[k0*n + i0, col] = 1.0
            col += 1

        def emit_schoolbook_2x2x1(i0, j0, k0):
            nonlocal col
            # Direct 4 products for 2x2x1 block
            U[i0*m + j0, col] = 1.0
            V[j0*p + k0, col] = 1.0
            W[k0*n + i0, col] = 1.0
            col += 1
            U[i0*m + (j0+1), col] = 1.0
            V[(j0+1)*p + k0, col] = 1.0
            W[k0*n + i0, col] = 1.0
            col += 1
            U[(i0+1)*m + j0, col] = 1.0
            V[j0*p + k0, col] = 1.0
            W[k0*n + (i0+1), col] = 1.0
            col += 1
            U[(i0+1)*m + (j0+1), col] = 1.0
            V[(j0+1)*p + k0, col] = 1.0
            W[k0*n + (i0+1), col] = 1.0
            col += 1

        # Apply to blocks
        # A is 2x4, B is 4x7
        # Split A into left (cols 0-1) and right (cols 2-3)
        # Split B into top (rows 0-1) and bottom (rows 2-3)
        # B cols 0-5 in pairs, col 6 single
        
        # Top part of B (rows 0-1)
        emit_strassen(0, 0, 0) # A[0:2, 0:2] * B[0:2, 0:2]
        emit_strassen(0, 0, 2) # A[0:2, 0:2] * B[0:2, 2:4]
        emit_strassen(0, 0, 4) # A[0:2, 0:2] * B[0:2, 4:6]
        emit_schoolbook_2x2x1(0, 0, 6) # A[0:2, 0:2] * B[0:2, 6:7]
        
        # Bottom part of B (rows 2-3)
        emit_strassen(0, 2, 0) # A[0:2, 2:4] * B[2:4, 0:2]
        emit_strassen(0, 2, 2) # A[0:2, 2:4] * B[2:4, 2:4]
        emit_strassen(0, 2, 4) # A[0:2, 2:4] * B[2:4, 4:6]
        emit_schoolbook_2x2x1(0, 2, 6) # A[0:2, 2:4] * B[2:4, 6:7]
        
        return U, V, W

    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    U, V, W = build_rank50()
    
    if not verify(U, V, W):
        return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
