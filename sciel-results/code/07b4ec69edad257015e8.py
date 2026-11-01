"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(offset_v, offset_w, col_offset):
        # P1: (A00+A11)(B00+B01) -> C00+, C11+
        U[0][col_offset] += 1
        U[3][col_offset] += 1
        V[offset_v + 0][col_offset] += 1
        V[offset_v + 1][col_offset] += 1
        W[offset_w + 0][col_offset] += 1
        W[offset_w + 3][col_offset] += 1
        
        # P2: (A10+A11)B00 -> C10+, C11-
        U[2][col_offset+1] += 1
        U[3][col_offset+1] += 1
        V[offset_v + 0][col_offset+1] += 1
        W[offset_w + 1][col_offset+1] += 1
        W[offset_w + 3][col_offset+1] += -1
        
        # P3: A00(B01-B11) -> C01+, C11+
        U[0][col_offset+2] += 1
        V[offset_v + 1][col_offset+2] += 1
        V[offset_v + 5][col_offset+2] += -1
        W[offset_w + 2][col_offset+2] += 1
        W[offset_w + 3][col_offset+2] += 1
        
        # P4: A11(B10-B00) -> C00+, C10+
        U[3][col_offset+3] += 1
        V[offset_v + 4][col_offset+3] += 1
        V[offset_v + 0][col_offset+3] += -1
        W[offset_w + 0][col_offset+3] += 1
        W[offset_w + 1][col_offset+3] += 1
        
        # P5: (A00+A01)B11 -> C00-, C01+
        U[0][col_offset+4] += 1
        U[1][col_offset+4] += 1
        V[offset_v + 5][col_offset+4] += 1
        W[offset_w + 0][col_offset+4] += -1
        W[offset_w + 2][col_offset+4] += 1
        
        # P6: (A10-A00)(B00+B01) -> C11+
        U[2][col_offset+5] += 1
        U[0][col_offset+5] += -1
        V[offset_v + 0][col_offset+5] += 1
        V[offset_v + 1][col_offset+5] += 1
        W[offset_w + 3][col_offset+5] += 1
        
        # P7: (A01-A11)(B10+B11) -> C00+, C11-
        U[1][col_offset+6] += 1
        U[3][col_offset+6] += -1
        V[offset_v + 4][col_offset+6] += 1
        V[offset_v + 5][col_offset+6] += 1
        W[offset_w + 0][col_offset+6] += 1
        W[offset_w + 3][col_offset+6] += -1

    add_strassen_block(0, 0, 0)
    add_strassen_block(2, 4, 7)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
