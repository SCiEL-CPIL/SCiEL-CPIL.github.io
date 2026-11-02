"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Strassen's 7-term decomposition for 2x2 matrix multiplication
    # U_r: (n*m) x 1 = 4 x 1, V_r: (m*p) x 1 = 32 x 1, W_r: (p*n) x 1 = 32 x 1
    # We'll build the full decomposition as a tensor product:
    # Treat p=16 as 8 blocks of size 2. Each 2x2 block uses Strassen.
    # Total rank = 7 * 8 = 56
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen matrices for 2x2 multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U factors: extract linear combinations from A (2x2)
    # Row indices for U: i*m+j where i in {0,1}, j in {0,1}
    # A00 -> row 0, A01 -> row 1, A10 -> row 2, A11 -> row 3
    
    # V factors: extract linear combinations from B (2x2 per block)
    # For each block b (0..7), B has entries B_{j, 2b+s} for j in {0,1}, s in {0,1}
    # V row index: j*p + k where k = 2b+s
    # B00 for block b -> row 0*16 + 2b = 2b
    # B01 for block b -> row 0*16 + 2b+1 = 2b+1
    # B10 for block b -> row 1*16 + 2b = 16+2b
    # B11 for block b -> row 1*16 + 2b+1 = 17+2b
    
    # W factors: output C_{i, 2b+s}
    # W row index: k*n + i where k = 2b+s
    # C_{0, 2b} -> row (2b)*2 + 0 = 4b
    # C_{0, 2b+1} -> row (2b+1)*2 + 0 = 4b+2
    # C_{1, 2b} -> row (2b)*2 + 1 = 4b+1
    # C_{1, 2b+1} -> row (2b+1)*2 + 1 = 4b+3
    
    # Strassen U vectors (for the 2x2 A matrix, flattened row-major)
    U_strassen = np.array([
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [1, 0, -1, 0],  # M6: A00-A10  (wait, should be A10-A00)
        [0, 1, 0, -1],  # M7: A01-A11
    ], dtype=np.float64)
    
    # Fix M6: A10 - A00 = [-1, 0, 1, 0]
    U_strassen[5] = np.array([-1, 0, 1, 0], dtype=np.float64)
    
    # Strassen V vectors for each 2x2 B block (flattened as [B00, B01, B10, B11])
    V_strassen = np.array([
        [1, 1, 1, 1],   # M1: B00+B11 for both? No...
        # Let me re-derive carefully.
    ], dtype=np.float64)
    
    # Actually, let me define the Strassen decomposition more carefully.
    # C = A * B where A, B are 2x2
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Strassen:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B00 - B10)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U (A-side) vectors:
    U_s = np.array([
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ], dtype=np.float64)
    
    # V (B-side) vectors for each 2x2 block:
    V_s = np.array([
        [1, 1, 1, 1],   # M1: B00+B11 -> but need to map to 4 entries
        # For a single 2x2 block B = [[B00, B01], [B10, B11]]
        # M1 needs (B00 + B11): so V1 = [1, 0, 0, 1]
        # M2 needs B00: V2 = [1, 0, 0, 0]
        # M3 needs (B10 - B11): V3 = [0, 0, 1, -1]
        # M4 needs (B00 - B10): V4 = [1, 0, -1, 0]
        # M5 needs B11: V5 = [0, 0, 0, 1]
        # M6 needs (B00 + B01): V6 = [1, 1, 0, 0]
        # M7 needs (B10 + B11): V7 = [0, 0, 1, 1]
    ], dtype=np.float64)
    
    V_s = np.array([
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 0, 1, -1],  # M3
        [1, 0, -1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ], dtype=np.float64)
    
    # W (output-side) vectors: which M's contribute to each C entry
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # W_s[i, r] where i is C entry index (0=C00, 1=C01, 2=C10, 3=C11), r is M index
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ], dtype=np.float64)
    
    # Now build the full rank-56 decomposition
    for block in range(8):
        for r in range(7):
            col = block * 7 + r
            
            # U: same for all blocks
            U[0, col] = U_s[r, 0]
            U[1, col] = U_s[r, 1]
            U[2, col] = U_s[r, 2]
            U[3, col] = U_s[r, 3]
            
            # V: for this block
            # B00 at row 0*16 + 2*block = 2*block
            # B01 at row 0*16 + 2*block+1 = 2*block+1
            # B10 at row 1*16 + 2*block = 16+2*block
            # B11 at row 1*16 + 2*block+1 = 17+2*block
            V[2*block, col] = V_s[r, 0]
            V[2*block+1, col] = V_s[r, 1]
            V[16+2*block, col] = V_s[r, 2]
            V[17+2*block, col] = V_s[r, 3]
            
            # W: output entries for this block
            # C_{0, 2*block} at row (2*block)*2 + 0 = 4*block
            # C_{0, 2*block+1} at row (2*block+1)*2 + 0 = 4*block+2
            # C_{1, 2*block} at row (2*block)*2 + 1 = 4*block+1
            # C_{1, 2*block+1} at row (2*block+1)*2 + 1 = 4*block+3
            W[4*block, col] = W_s[0, r]
            W[4*block+2, col] = W_s[1, r]
            W[4*block+1, col] = W_s[2, r]
            W[4*block+3, col] = W_s[3, r]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
