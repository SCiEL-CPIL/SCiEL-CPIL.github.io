"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 52 decomposition: 2 * (7 Strassen + 19 Schoolbook)
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w, col):
        U[u_idx][col] += val_u
        V[v_idx][col] += val_v
        W[w_idx][col] += val_w

    def build_block(k_offset, col_offset):
        # Strassen terms for 2x2 block (indices 0,1)
        # M1 = (A00+A11)(B00+B11)
        add_term(0, k_offset, k_offset, 1, 1, 1, col_offset)
        add_term(4, k_offset, k_offset, 1, 1, 1, col_offset)
        add_term(0, k_offset+1, k_offset, 1, 1, -1, col_offset) # Wait, V index is j*p+k. j=0, k=0 -> 0. j=1, k=0 -> p.
        # Correction: V index is j*p + k.
        # For B00: j=0, k=0 -> 0.
        # For B11: j=1, k=1 -> p+1.
        # My add_term above was using k_offset for V index, which is wrong. V index depends on j and k.
        
        # Let's restart the block building with correct indices.
        pass

    # Correct Indexing:
    # U[i*m + j]
    # V[j*p + k]
    # W[k*n + i]
    
    col = 0
    for block in range(2):
        k_base = block * 3
        # Strassen terms
        # 1: (A00+A11)(B00+B11)
        U[0][col] += 1; V[0*p+k_base][col] += 1; W[k_base*n+0][col] += 1
        U[4][col] += 1; V[0*p+k_base][col] += 1; W[k_base*n+0][col] += 1
        U[0][col] += 1; V[1*p+k_base][col] += 1; W[(k_base+1)*n+0][col] += 1
        U[4][col] += 1; V[1*p+k_base][col] += 1; W[(k_base+1)*n+0][col] += 1
        col += 1
        
        # 2: (A10+A11)B00
        U[3][col] += 1; V[0*p+k_base][col] += 1; W[k_base*n+1][col] += 1
        U[4][col] += 1; V[0*p+k_base][col] += 1; W[k_base*n+1][col] += 1
        col += 1
        
        # 3: A00(B10-B11)
        U[0][col] += 1; V[1*p+k_base][col] += 1; W[(k_base+1)*n+0][col] += 1
        U[0][col] += 1; V[1*p+k_base+1][col] += -1; W[(k_base+1)*n+0][col] += -1
        col += 1
        
        # 4: A11(B01-B00)
        U[4][col] += 1; V[0*p+k_base+1][col] += 1; W[(k_base+1)*n+1][col] += 1
        U[4][col] += 1; V[0*p+k_base][col] += -1; W[(k_base+1)*n+1][col] += -1
        col += 1
        
        # 5: (A00+A01)B11
        U[0][col] += 1; V[1*p+k_base+1][col] += 1; W[(k_base+1)*n+0][col] += 1
        U[1][col] += 1; V[1*p+k_base+1][col] += 1; W[(k_base+1)*n+0][col] += 1
        col += 1
        
        # 6: (A10-A00)(B00+B01)
        U[3][col] += 1; V[0*p+k_base][col] += 1; W[k_base*n+1][col] += 1
        U[0][col] += -1; V[0*p+k_base][col] += 1; W[k_base*n+1][col] += 1
        U[3][col] += 1; V[0*p+k_base+1][col] += 1; W[(k_base+1)*n+1][col] += 1
        U[0][col] += -1; V[0*p+k_base+1][col] += 1; W[(k_base+1)*n+1][col] += 1
        col += 1
        
        # 7: (A01-A11)(B10+B11)
        U[1][col] += 1; V[1*p+k_base][col] += 1; W[k_base*n+0][col] += 1
        U[4][col] += -1; V[1*p+k_base][col] += 1; W[k_base*n+0][col] += 1
        U[1][col] += 1; V[1*p+k_base+1][col] += 1; W[(k_base+1)*n+0][col] += 1
        U[4][col] += -1; V[1*p+k_base+1][col] += 1; W[(k_base+1)*n+0][col] += 1
        col += 1
        
        # Schoolbook terms for remaining 19
        # C00: A02B20
        U[2][col] += 1; V[2*p+k_base][col] += 1; W[k_base*n+0][col] += 1; col += 1
        # C01: A02B21
        U[2][col] += 1; V[2*p+k_base+1][col] += 1; W[(k_base+1)*n+0][col] += 1; col += 1
        # C10: A12B20
        U[5][col] += 1; V[2*p+k_base][col] += 1; W[k_base*n+1][col] += 1; col += 1
        # C11: A12B21
        U[5][col] += 1; V[2*p+k_base+1][col] += 1; W[(k_base+1)*n+1][col] += 1; col += 1
        
        # C02: A00B02, A01B12, A02B22
        U[0][col] += 1; V[0*p+k_base+2][col] += 1; W[(k_base+2)*n+0][col] += 1; col += 1
        U[1][col] += 1; V[1*p+k_base+2][col] += 1; W[(k_base+2)*n+0][col] += 1; col += 1
        U[2][col] += 1; V[2*p+k_base+2][col] += 1; W[(k_base+2)*n+0][col] += 1; col += 1
        
        # C12: A10B02, A11B12, A12B22
        U[3][col] += 1; V[0*p+k_base+2][col] += 1; W[(k_base+2)*n+1][col] += 1; col += 1
        U[4][col] += 1; V[1*p+k_base+2][col] += 1; W[(k_base+2)*n+1][col] += 1; col += 1
        U[5][col] += 1; V[2*p+k_base+2][col] += 1; W[(k_base+2)*n+1][col] += 1; col += 1
        
        # C20: A20B00, A21B10, A22B20
        U[6][col] += 1; V[0*p+k_base][col] += 1; W[k_base*n+2][col] += 1; col += 1
        U[7][col] += 1; V[1*p+k_base][col] += 1; W[k_base*n+2][col] += 1; col += 1
        U[8][col] += 1; V[2*p+k_base][col] += 1; W[k_base*n+2][col] += 1; col += 1
        
        # C21: A20B01, A21B11, A22B21
        U[6][col] += 1; V[0*p+k_base+1][col] += 1; W[(k_base+1)*n+2][col] += 1; col += 1
        U[7][col] += 1; V[1*p+k_base+1][col] += 1; W[(k_base+1)*n+2][col] += 1; col += 1
        U[8][col] += 1; V[2*p+k_base+1][col] += 1; W[(k_base+1)*n+2][col] += 1; col += 1
        
        # C22: A20B02, A21B12, A22B22
        U[6][col] += 1; V[0*p+k_base+2][col] += 1; W[(k_base+2)*n+2][col] += 1; col += 1
        U[7][col] += 1; V[1*p+k_base+2][col] += 1; W[(k_base+2)*n+2][col] += 1; col += 1
        U[8][col] += 1; V[2*p+k_base+2][col] += 1; W[(k_base+2)*n+2][col] += 1; col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
