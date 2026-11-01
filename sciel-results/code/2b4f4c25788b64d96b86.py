"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    deadline = time.time() + 140
    
    # Build support tensor
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.append((a, b, c))
    
    all_tuples = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
    support_set = set(support)
    
    def compute_error(U, V, W):
        max_err = 0.0
        for (a, b, c) in support:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
        for (a, b, c) in all_tuples:
            if (a, b, c) not in support_set:
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                err = abs(val)
                if err > max_err:
                    max_err = err
        return max_err
    
    # Construct explicit rank-18 decomposition:
    # Block 1: Strassen (rank 7) for columns 0,1
    # Block 2: Strassen (rank 7) for columns 2,3
    # Block 3: Schoolbook (rank 4) for column 4
    # Total rank = 18
    
    rank = 18
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen_block(col_offset, base_idx):
        """Add Strassen decomposition for 2x2 x 2x2 -> 2x2 at columns col_offset, col_offset+1"""
        b00 = col_offset
        b01 = col_offset + 1
        b10 = 5 + col_offset
        b11 = 5 + col_offset + 1
        
        c00 = 2 * col_offset
        c01 = 2 * col_offset + 2
        c10 = 2 * col_offset + 1
        c11 = 2 * col_offset + 3
        
        # M1 = (A[0,0]+A[1,1])*(B[0,ck]+B[1,ck+1])
        U[0, base_idx] += 1; U[3, base_idx] += 1
        V[b00, base_idx] += 1; V[b11, base_idx] += 1
        W[c00, base_idx] += 1; W[c11, base_idx] += 1
        
        # M2 = (A[1,0]+A[1,1])*B[0,ck]
        U[2, base_idx+1] += 1; U[3, base_idx+1] += 1
        V[b00, base_idx+1] += 1
        W[c10, base_idx+1] += 1
        
        # M3 = A[0,0]*(B[0,ck+1]-B[1,ck+1])
        U[0, base_idx+2] += 1
        V[b01, base_idx+2] += 1; V[b11, base_idx+2] -= 1
        W[c01, base_idx+2] += 1
        
        # M4 = A[1,1]*(B[1,ck]-B[0,ck])
        U[3, base_idx+3] += 1
        V[b10, base_idx+3] += 1; V[b00, base_idx+3] -= 1
        W[c00, base_idx+3] += 1; W[c10, base_idx+3] += 1
        
        # M5 = (A[0,0]+A[0,1])*B[1,ck+1]
        U[0, base_idx+4] += 1; U[1, base_idx+4] += 1
        V[b11, base_idx+4] += 1
        W[c00, base_idx+4] -= 1; W[c01, base_idx+4] += 1
        
        # M6 = (A[0,1]-A[1,0])*(B[0,ck]+B[0,ck+1])
        U[1, base_idx+5] += 1; U[2, base_idx+5] -= 1
        V[b00, base_idx+5] += 1; V[b01, base_idx+5] += 1
        W[c11, base_idx+5] += 1
        
        # M7 = (A[1,0]-A[0,0])*(B[1,ck]+B[1,ck+1])
        U[2, base_idx+6] += 1; U[0, base_idx+6] -= 1
        V[b10, base_idx+6] += 1; V[b11, base_idx+6] += 1
        W[c00, base_idx+6] += 1
    
    # Block 1: Strassen for columns 0,1 (base_idx = 0)
    add_strassen_block(0, 0)
    
    # Block 2: Strassen for columns 2,3 (base_idx = 7)
    add_strassen_block(2, 7)
    
    # Block 3: Schoolbook for column 4 (base_idx = 14)
    U[0, 14] += 1
    V[4, 14] += 1
    W[8, 14] += 1
    
    U[1, 15] += 1
    V[9, 15] += 1
    W[8, 15] += 1
    
    U[2, 16] += 1
    V[4, 16] += 1
    W[9, 16] += 1
    
    U[3, 17] += 1
    V[9, 17] += 1
    W[9, 17] += 1
    
    # Verify rank-18 construction
    err = compute_error(U, V, W)
    if err < 1e-10:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook rank 20
    best_U = np.zeros((n*m, 20))
    best_V = np.zeros((m*p, 20))
    best_W = np.zeros((p*n, 20))
    for idx, (a, b, c) in enumerate(support):
        best_U[a, idx] = 1.0
        best_V[b, idx] = 1.0
        best_W[c, idx] = 1.0
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
