import numpy as np
import time

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Total rank calculation
    # 7 blocks of 2x2x2 (rank 7 each) = 49
    # 1 block of 2x2x1 (rank 4) = 4
    # Total rank = 53
    rank = 53
    
    # Initialize matrices with zeros
    U = np.zeros((n*m, rank), dtype=np.int8)
    V = np.zeros((m*p, rank), dtype=np.int8)
    W = np.zeros((p*n, rank), dtype=np.int8)
    
    # Helper to add a 2x2x2 Strassen block at k_offset
    def add_strassen_block(k_offset, r_offset):
        # U indices: 0..3 (A)
        # V indices: 0..3 (B local) -> global j*p + k_offset + k_loc
        # W indices: 0..3 (C local) -> global (k_offset + k_loc)*n + i
        
        # Strassen coefficients for U, V, W
        # r=0: m1
        U[0, r_offset] += 1
        U[3, r_offset] += 1
        V[0*p + k_offset + 0, r_offset] += 1 # e
        V[1*p + k_offset + 1, r_offset] += 1 # h
        W[(k_offset+0)*n + 0, r_offset] += 1 # C00
        W[(k_offset+1)*n + 1, r_offset] += 1 # C11
        
        # r=1: m2
        U[2, r_offset+1] += 1
        U[3, r_offset+1] += 1
        V[0*p + k_offset + 0, r_offset+1] += 1 # e
        W[(k_offset+0)*n + 1, r_offset+1] += 1 # C10
        W[(k_offset+1)*n + 1, r_offset+1] -= 1 # C11
        
        # r=2: m3
        U[0, r_offset+2] += 1
        V[0*p + k_offset + 1, r_offset+2] += 1 # f
        V[1*p + k_offset + 1, r_offset+2] -= 1 # h
        W[(k_offset+1)*n + 0, r_offset+2] += 1 # C01
        W[(k_offset+1)*n + 1, r_offset+2] += 1 # C11
        
        # r=3: m4
        U[3, r_offset+3] += 1
        V[1*p + k_offset + 0, r_offset+3] += 1 # g
        V[0*p + k_offset + 1, r_offset+3] -= 1 # f
        W[(k_offset+0)*n + 0, r_offset+3] += 1 # C00
        W[(k_offset+0)*n + 1, r_offset+3] += 1 # C10
        
        # r=4: m5
        U[0, r_offset+4] += 1
        U[1, r_offset+4] += 1
        V[1*p + k_offset + 1, r_offset+4] += 1 # h
        W[(k_offset+0)*n + 0, r_offset+4] -= 1 # C00
        W[(k_offset+1)*n + 0, r_offset+4] += 1 # C01
        
        # r=5: m6
        U[2, r_offset+5] += 1
        U[0, r_offset+5] -= 1
        V[0*p + k_offset + 0, r_offset+5] += 1 # e
        V[0*p + k_offset + 1, r_offset+5] += 1 # f
        W[(k_offset+1)*n + 1, r_offset+5] += 1 # C11
        
        # r=6: m7
        U[1, r_offset+6] += 1
        U[3, r_offset+6] -= 1
        V[1*p + k_offset + 0, r_offset+6] += 1 # g
        V[1*p + k_offset + 1, r_offset+6] += 1 # h
        W[(k_offset+0)*n + 0, r_offset+6] += 1 # C00
        W[(k_offset+1)*n + 1, r_offset+6] += 1 # C11

    # Helper to add a 2x2x1 Schoolbook block at k_offset
    def add_schoolbook_block(k_offset, r_offset):
        # U indices: 0..3
        # V indices: 0..1 (B local) -> global j*p + k_offset
        # W indices: 0..1 (C local) -> global (k_offset)*n + i
        
        # r=0: a*e -> C00
        U[0, r_offset] += 1
        V[0*p + k_offset, r_offset] += 1
        W[k_offset*n + 0, r_offset] += 1
        
        # r=1: b*f -> C00
        U[1, r_offset+1] += 1
        V[1*p + k_offset, r_offset+1] += 1
        W[k_offset*n + 0, r_offset+1] += 1
        
        # r=2: c*e -> C10
        U[2, r_offset+2] += 1
        V[0*p + k_offset, r_offset+2] += 1
        W[k_offset*n + 1, r_offset+2] += 1
        
        # r=3: d*f -> C10
        U[3, r_offset+3] += 1
        V[1*p + k_offset, r_offset+3] += 1
        W[k_offset*n + 1, r_offset+3] += 1

    r_offset = 0
    
    # 7 Strassen blocks for k=0..13
    for b in range(7):
        k_offset = 2 * b
        add_strassen_block(k_offset, r_offset)
        r_offset += 7
        
    # 1 Schoolbook block for k=14
    k_offset = 14
    add_schoolbook_block(k_offset, r_offset)
    r_offset += 4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
