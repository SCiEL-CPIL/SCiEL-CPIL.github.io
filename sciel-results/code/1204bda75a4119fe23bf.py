"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    
    # Initialize matrices
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (8, 28)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a Strassen block
    def add_strassen(i0, i1, j0, j1, k0, k1, col_offset):
        """
        Adds the rank-7 Strassen decomposition for the product of
        A[i0:i1+1, j0:j1+1] * B[j0:j1+1, k0:k1+1] -> C[i0:i1+1, k0:k1+1]
        into the global U, V, W matrices starting at column col_offset.
        
        i0, i1: row indices in A/C (0, 1)
        j0, j1: col indices in A / row indices in B (0..3)
        k0, k1: col indices in B/C (0..3)
        """
        
        # Map local 2x2 indices to global indices
        # A indices: i * m + j
        a_00 = i0 * m + j0
        a_01 = i0 * m + j1
        a_10 = i1 * m + j0
        a_11 = i1 * m + j1
        
        # B indices: j * p + k
        b_00 = j0 * p + k0
        b_01 = j0 * p + k1
        b_10 = j1 * p + k0
        b_11 = j1 * p + k1
        
        # C indices: k * n + i (Trace convention W[k*n+i, r])
        c_00 = k0 * n + i0
        c_01 = k1 * n + i0
        c_10 = k0 * n + i1
        c_11 = k1 * n + i1
        
        # Strassen coefficients
        # m1 = (a00+a11)(b00+b11)
        # U: a00+1, a11+1; V: b00+1, b11+1; W: c00+1, c11+1
        c = col_offset + 0
        U[a_00, c] += 1
        U[a_11, c] += 1
        V[b_00, c] += 1
        V[b_11, c] += 1
        W[c_00, c] += 1
        W[c_11, c] += 1
        
        # m2 = (a10+a11)b00
        # U: a10+1, a11+1; V: b00+1; W: c10+1
        c = col_offset + 1
        U[a_10, c] += 1
        U[a_11, c] += 1
        V[b_00, c] += 1
        W[c_10, c] += 1
        
        # m3 = a00(b01-b11)
        # U: a00+1; V: b01+1, b11-1; W: c01+1, c11+1
        c = col_offset + 2
        U[a_00, c] += 1
        V[b_01, c] += 1
        V[b_11, c] += -1
        W[c_01, c] += 1
        W[c_11, c] += 1
        
        # m4 = a11(b10-b00)
        # U: a11+1; V: b10+1, b00-1; W: c00+1, c10+1
        c = col_offset + 3
        U[a_11, c] += 1
        V[b_10, c] += 1
        V[b_00, c] += -1
        W[c_00, c] += 1
        W[c_10, c] += 1
        
        # m5 = (a00+a01)b11
        # U: a00+1, a01+1; V: b11+1; W: c01+1, c00-1
        c = col_offset + 4
        U[a_00, c] += 1
        U[a_01, c] += 1
        V[b_11, c] += 1
        W[c_01, c] += 1
        W[c_00, c] += -1
        
        # m6 = (a10-a00)(b00+b01)
        # U: a10+1, a00-1; V: b00+1, b01+1; W: c11+1
        c = col_offset + 5
        U[a_10, c] += 1
        U[a_00, c] += -1
        V[b_00, c] += 1
        V[b_01, c] += 1
        W[c_11, c] += 1
        
        # m7 = (a01-a11)(b10+b11)
        # U: a01+1, a11-1; V: b10+1, b11+1; W: c00+1, c11-1
        c = col_offset + 6
        U[a_01, c] += 1
        U[a_11, c] += -1
        V[b_10, c] += 1
        V[b_11, c] += 1
        W[c_00, c] += 1
        W[c_11, c] += -1

    # Block 1: A[:, 0:2] * B[0:2, :] -> C[:, 0:2] part?
    # Wait, let's stick to the 4 independent 2x2x2 blocks derived from:
    # C_left = A_00 B_00 + A_01 B_10
    # C_right = A_00 B_01 + A_01 B_11
    
    # Block 1.1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # i in 0,1; j in 0,1; k in 0,1
    add_strassen(0, 1, 0, 1, 0, 1, 0)
    
    # Block 1.2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # i in 0,1; j in 2,3; k in 0,1
    add_strassen(0, 1, 2, 3, 0, 1, 7)
    
    # Block 2.1: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # i in 0,1; j in 0,1; k in 2,3
    add_strassen(0, 1, 0, 1, 2, 3, 14)
    
    # Block 2.2: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # i in 0,1; j in 2,3; k in 2,3
    add_strassen(0, 1, 2, 3, 2, 3, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
