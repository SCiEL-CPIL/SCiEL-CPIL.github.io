"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    total_rank = num_blocks * 11
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # For each block b (k in {2b, 2b+1}), local k' in {0,1}:
    # Strassen terms (7): each defined by (list of (U_row, val), list of (V_row, val), list of (W_row, val))
    # U_row = i*m + j, V_row = j*p + k_global, W_row = k_global*n + i
    #
    # Local indices: j in {0,1,2}, k_local in {0,1}
    # A rows: i=0 -> U rows 0,1,2; i=1 -> U rows 3,4,5
    # B cols: for block b, k_global = 2b + k_local
    # C: W rows for (k_global, i)
    
    def make_block_terms(b):
        """Generate 11 terms for block b."""
        kg0 = 2 * b    # global k for local k'=0
        kg1 = 2 * b + 1  # global k for local k'=1
        
        # V row indices: j*p + kg0, j*p + kg1
        V_0_0 = 0 * p + kg0  # B[0, kg0]
        V_0_1 = 0 * p + kg1  # B[0, kg1]
        V_1_0 = 1 * p + kg0  # B[1, kg0]
        V_1_1 = 1 * p + kg1  # B[1, kg1]
        V_2_0 = 2 * p + kg0  # B[2, kg0]
        V_2_1 = 2 * p + kg1  # B[2, kg1]
        
        # W row indices: k_global*n + i
        W_0_0 = kg0 * n + 0  # C[0, kg0]
        W_0_1 = kg0 * n + 1  # C[1, kg0]
        W_1_0 = kg1 * n + 0  # C[0, kg1]
        W_1_1 = kg1 * n + 1  # C[1, kg1]
        
        # U row indices: i*m + j
        U_0_0 = 0 * m + 0  # A[0,0]
        U_0_1 = 0 * m + 1  # A[0,1]
        U_0_2 = 0 * m + 2  # A[0,2]
        U_1_0 = 1 * m + 0  # A[1,0]
        U_1_1 = 1 * m + 1  # A[1,1]
        U_1_2 = 1 * m + 2  # A[1,2]
        
        terms = []
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,kg0]+B[1,kg1])
        # C[0,kg0] += M1, C[1,kg1] += M1
        terms.append((
            [(U_0_0, 1), (U_1_1, 1)],
            [(V_0_0, 1), (V_1_1, 1)],
            [(W_0_0, 1), (W_1_1, 1)]
        ))
        
        # M2 = (A[1,0]+A[1,1]) * B[0,kg0]
        # C[1,kg0] += M2, C[0,kg1] += M2, C[1,kg1] -= M2
        terms.append((
            [(U_1_0, 1), (U_1_1, 1)],
            [(V_0_0, 1)],
            [(W_0_1, 1), (W_1_0, 1), (W_1_1, -1)]
        ))
        
        # M3 = A[0,0] * (B[0,kg1]-B[1,kg1])
        # C[0,kg1] += M3, C[1,kg0] += M3
        terms.append((
            [(U_0_0, 1)],
            [(V_0_1, 1), (V_1_1, -1)],
            [(W_1_0, 1), (W_0_1, 1)]
        ))
        
        # M4 = A[1,1] * (B[1,kg0]-B[0,kg0])
        # C[0,kg0] += M4, C[1,kg1] += M4... 
        # Wait let me redo. C11 = M1+M4-M5+M7 where C11=C[0,kg0]
        # C22 = M2+M3-M1+M6 where C22=C[1,kg1]
        # C12 = M1+M2 where C12=C[0,kg1]
        # C21 = M3+M5 where C21=C[1,kg0]
        
        # M4 = A[1,1]*(B[1,kg0]-B[0,kg0])
        # C[0,kg0] += M4
        terms.append((
            [(U_1_1, 1)],
            [(V_1_0, 1), (V_0_0, -1)],
            [(W_0_0, 1)]
        ))
        
        # M5 = (A[0,0]+A[0,1]) * B[1,kg1]
        # C[0,kg0] -= M5, C[1,kg0] += M5
        terms.append((
            [(U_0_0, 1), (U_0_1, 1)],
            [(V_1_1, 1)],
            [(W_0_0, -1), (W_0_1, 1)]
        ))
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,kg0]+B[0,kg1])
        # C[1,kg1] += M6
        terms.append((
            [(U_1_0, 1), (U_0_0, -1)],
            [(V_0_0, 1), (V_0_1, 1)],
            [(W_1_1, 1)]
        ))
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,kg0]+B[1,kg1])
        # C[0,kg0] += M7
        terms.append((
            [(U_0_1, 1), (U_1_1, -1)],
            [(V_1_0, 1), (V_1_1, 1)],
            [(W_0_0, 1)]
        ))
        
        # Outer terms: A[0,2]*B[2,kg0], A[0,2]*B[2,kg1], A[1,2]*B[2,kg0], A[1,2]*B[2,kg1]
        # C[0,kg0] += A[0,2]*B[2,kg0]
        terms.append((
            [(U_0_2, 1)],
            [(V_2_0, 1)],
            [(W_0_0, 1)]
        ))
        # C[0,kg1] += A[0,2]*B[2,kg1]
        terms.append((
            [(U_0_2, 1)],
            [(V_2_1, 1)],
            [(W_1_0, 1)]
        ))
        # C[1,kg0] += A[1,2]*B[2,kg0]
        terms.append((
            [(U_1_2, 1)],
            [(V_2_0, 1)],
            [(W_0_1, 1)]
        ))
        # C[1,kg1] += A[1,2]*B[2,kg1]
        terms.append((
            [(U_1_2, 1)],
            [(V_2_1, 1)],
            [(W_1_1, 1)]
        ))
        
        return terms
    
    col = 0
    for b in range(num_blocks):
        if time.time() > deadline:
            break
        for u_entries, v_entries, w_entries in make_block_terms(b):
            for row, val in u_entries:
                U[row, col] += val
            for row, val in v_entries:
                V[row, col] += val
            for row, val in w_entries:
                W[row, col] += val
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
