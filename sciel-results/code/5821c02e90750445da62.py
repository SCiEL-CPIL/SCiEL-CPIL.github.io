import numpy as np
import time

DIMENSIONS = [2, 4, 4]
DEADLINE = 140.0

def _generate_strassen_decomposition():
    """
    Generates the rank-28 decomposition for 2x4x4 matrix multiplication.
    The decomposition splits the 4x4 matrix B into two 2x4 blocks (top/bottom) 
    or A into two 2x2 blocks? 
    Actually, the split used in the analysis:
    C = A B
    Split A into A0 (cols 0,1) and A1 (cols 2,3).
    Split B into B0 (rows 0,1) and B1 (rows 2,3).
    C = A0 B0 + A1 B1.
    Each term is 2x2 times 2x4 -> 2x4.
    This 2x2x4 multiplication can be split into two 2x2x2 multiplications.
    Total 4 Strassen multiplications.
    """
    # Strassen coefficients for 2x2x2
    # We define the 7 factors for one 2x2x2 block multiplication.
    # U, V, W are 4x7 matrices for one block.
    # Here we need to map these to the global 8x16, 16x16, 8x16 (or similar) sizes.
    
    # Global sizes:
    # A is 2x4 -> 8 elements. U is (8, rank)
    # B is 4x4 -> 16 elements. V is (16, rank)
    # C is 2x4 -> 8 elements. W is (8, rank)
    
    rank = 28
    U = np.zeros((8, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((8, rank), dtype=int)
    
    # Strassen 2x2x2 decomposition
    # Let X, Y be 2x2 matrices. Z = XY.
    # 7 products:
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y10 - Y11)
    # M4 = X11(Y01 - Y00)
    # M5 = (X00 + X10)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X11 - X10)(Y10 + Y11)
    
    # We have 4 such blocks.
    # Block 0: A0 (2x2) * B0 (2x2 part of 2x4) -> contributes to C0 (2x2 part of 2x4)
    # Wait, B is 2x4 in the sub-problem.
    # Let's stick to the 4 independent 2x2x2 multiplications.
    
    # The 4 multiplications are:
    # 1. A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2] part 1
    # 2. A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4] part 1
    # 3. A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2] part 2
    # 4. A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4] part 2
    
    # Let's define the Strassen factors for a single 2x2x2 multiplication.
    # Indices for A (2x2): 00, 01, 10, 11 -> flat 0,1,2,3
    # Indices for B (2x2): 00, 01, 10, 11 -> flat 0,1,2,3
    # Indices for C (2x2): 00, 01, 10, 11 -> flat 0,1,2,3
    
    # Strassen U, V, W for one block (4x7)
    # We represent the coefficients as lists of (index, coeff)
    
    def get_strassen_factors():
        # Returns lists of (local_idx, coeff) for U, V, W
        # Local indices 0..3 for 2x2 matrices
        
        # M1: (X00+X11)(Y00+Y11)
        u1 = [(0, 1), (3, 1)]
        v1 = [(0, 1), (3, 1)]
        w1 = [(0, 1), (1, 1), (2, 1), (3, 1)] # Z00+=M1, Z01+=M1, Z10+=M1, Z11+=M1? 
        # Actually Z00 = M1 - M4 + M7
        # Z01 = M5 + M7
        # Z10 = M2 + M4
        # Z11 = M1 - M3 + M6
        
        # Let's use the standard Strassen formulas for Z.
        # Z00 = M1 - M4 + M7
        # Z01 = M5 + M7
        # Z10 = M2 + M4
        # Z11 = M1 - M3 + M6
        
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y10-Y11)
        # M4 = X11(Y01-Y00)
        # M5 = (X00+X10)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X11-X10)(Y10+Y11)
        
        factors = []
        
        # 1
        factors.append(
            [(0, 1), (3, 1)], # U
            [(0, 1), (3, 1)], # V
            [(0, 1), (1, 1), (2, 1), (3, 1)] # W (M1 contributes to Z00, Z01, Z10, Z11? No.)
        )
        # Wait, the W matrix in the decomposition sums the products.
        # Z_k = sum_l U_{i,l} V_{j,l} W_{k,l}
        # So for each product M_l, we add to W the coefficients that map M_l to the output entries.
        
        # Let's redefine.
        # For each of the 7 products, we have:
        # U_l (vector in A space)
        # V_l (vector in B space)
        # W_l (vector in C space)
        
        # Product 1: M1 = (X00+X11)(Y00+Y11)
        # U1 = [1, 0, 0, 1] (X00+X11)
        # V1 = [1, 0, 0, 1] (Y00+Y11)
        # M1 contributes to:
        # Z00: +1
        # Z11: +1
        # (Other Z entries? No, standard Strassen:
        # Z00 = M1 - M4 + M7
        # Z01 = M5 + M7
        # Z10 = M2 + M4
        # Z11 = M1 - M3 + M6
        # So M1 contributes to Z00 and Z11.)
        # W1 = [1, 0, 0, 1] (Z00=1, Z11=1)
        
        f1 = ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
        
        # Product 2: M2 = (X10+X11)Y00
        # U2 = [0, 0, 1, 1]
        # V2 = [1, 0, 0, 0]
        # M2 contributes to Z10: +1
        # W2 = [0, 0, 1, 0]
        
        f2 = ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)])
        
        # Product 3: M3 = X00(Y10-Y11)
        # U3 = [1, 0, 0, 0]
        # V3 = [0, 0, 1, -1]
        # M3 contributes to Z11: -1
        # W3 = [0, 0, 0, -1]
        
        f3 = ([(0, 1)], [(2, 1), (3, -1)], [(3, -1)])
        
        # Product 4: M4 = X11(Y01-Y00)
        # U4 = [0, 0, 0, 1]
        # V4 = [0, 1, -1, 0]
        # M4 contributes to Z00: -1, Z10: +1
        # W4 = [-1, 0, 1, 0]
        
        f4 = ([(3, 1)], [(1, 1), (2, -1)], [(0, -1), (2, 1)])
        
        # Product 5: M5 = (X00+X10)Y11
        # U5 = [1, 0, 1, 0]
        # V5 = [0, 0, 0, 1]
        # M5 contributes to Z01: +1
        # W5 = [0, 1, 0, 0]
        
        f5 = ([(0, 1), (2, 1)], [(3, 1)], [(1, 1)])
        
        # Product 6: M6 = (X10-X00)(Y00+Y01)
        # U6 = [-1, 0, 1, 0]
        # V6 = [1, 1, 0, 0]
        # M6 contributes to Z11: +1
        # W6 = [0, 0, 0, 1]
        
        f6 = ([(0, -1), (2, 1)], [(0, 1), (1, 1)], [(3, 1)])
        
        # Product 7: M7 = (X11-X10)(Y10+Y11)
        # U7 = [0, 0, -1, 1]
        # V7 = [0, 0, 1, 1]
        # M7 contributes to Z00: +1, Z01: +1
        # W7 = [1, 1, 0, 0]
        
        f7 = ([(2, -1), (3, 1)], [(2, 1), (3, 1)], [(0, 1), (1, 1)])
        
        return [f1, f2, f3, f4, f5, f6, f7]

    strassen_factors = get_strassen_factors()
    
    # Now map these 4 blocks into the global matrices.
    # Block 0: A cols 0,1 (A0), B rows 0,1 & cols 0,1 (B00), C cols 0,1 (C0)
    # Block 1: A cols 0,1 (A0), B rows 0,1 & cols 2,3 (B01), C cols 2,3 (C1)
    # Block 2: A cols 2,3 (A1), B rows 2,3 & cols 0,1 (B10), C cols 0,1 (C0)
    # Block 3: A cols 2,3 (A1), B rows 2,3 & cols 2,3 (B11), C cols 2,3 (C1)
    
    # Global A indices: 0..7
    # A is 2x4. Row r, Col c. Index = r*4 + c.
    # A0 (cols 0,1):
    #   r=0, c=0 -> 0
    #   r=0, c=1 -> 1
    #   r=1, c=0 -> 4
    #   r=1, c=1 -> 5
    # A1 (cols 2,3):
    #   r=0, c=2 -> 2
    #   r=0, c=3 -> 3
    #   r=1, c=2 -> 6
    #   r=1, c=3 -> 7
    
    # Global B indices: 0..15
    # B is 4x4. Row r, Col c. Index = r*4 + c.
    # B00 (rows 0,1, cols 0,1):
    #   (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # B01 (rows 0,1, cols 2,3):
    #   (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    # B10 (rows 2,3, cols 0,1):
    #   (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13
    # B11 (rows 2,3, cols 2,3):
    #   (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15
    
    # Global C indices: 0..7
    # C is 2x4. Row r, Col c. Index = r*4 + c.
    # C0 (cols 0,1):
    #   (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # C1 (cols 2,3):
    #   (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    
    # Local to Global mapping functions
    
    def map_A(local_idx, block_col_off):
        # local_idx 0..3 for 2x2
        # r = local_idx // 2, c = local_idx % 2
        # global col = block_col_off*2 + c
        # global row = r
        # global idx = r*4 + (block_col_off*2 + c)
        r = local_idx // 2
        c = local_idx % 2
        return r * 4 + (block_col_off * 2 + c)
        
    def map_B(local_idx, block_row_off, block_col_off):
        # local_idx 0..3 for 2x2
        # r = local_idx // 2, c = local_idx % 2
        # global row = block_row_off*2 + r
        # global col = block_col_off*2 + c
        # global idx = (block_row_off*2 + r)*4 + (block_col_off*2 + c)
        r = local_idx // 2
        c = local_idx % 2
        return (block_row_off * 2 + r) * 4 + (block_col_off * 2 + c)
        
    def map_C(local_idx, block_col_off):
        # local_idx 0..3 for 2x2
        # r = local_idx // 2, c = local_idx % 2
        # global col = block_col_off*2 + c
        # global row = r
        # global idx = r*4 + (block_col_off*2 + c)
        r = local_idx // 2
        c = local_idx % 2
        return r * 4 + (block_col_off * 2 + c)
        
    # Blocks:
    # 0: A0, B00, C0
    # 1: A0, B01, C1
    # 2: A1, B10, C0
    # 3: A1, B11, C1
    
    blocks = [
        (0, 0, 0, 0), # A_col_off, B_row_off, B_col_off, C_col_off
        (0, 0, 1, 1),
        (1, 1, 0, 0),
        (1, 1, 1, 1)
    ]
    
    col_offset = 0
    for a_off, b_row_off, b_col_off, c_off in blocks:
        for u_local, v_local, w_local in strassen_factors:
            for loc_idx, coeff in u_local:
                U[map_A(loc_idx, a_off), col_offset] += coeff
            for loc_idx, coeff in v_local:
                V[map_B(loc_idx, b_row_off, b_col_off), col_offset] += coeff
            for loc_idx, coeff in w_local:
                W[map_C(loc_idx, c_off), col_offset] += coeff
            col_offset += 1
            
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    U, V, W = _generate_strassen_decomposition()
    
    if time.time() - start_time > DEADLINE:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
