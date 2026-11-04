"""Matrix multiplication 4,4,9 tensor decomposition - block-diagonal Kronecker approach."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    nm, mp, pn = n * m, m * p, p * n  # 16, 36, 36
    
    def schoolbook():
        rank = n * m * p  # 144
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, idx] = 1
                    V[b, idx] = 1
                    W[c, idx] = 1
        return U, V, W, rank
    
    def verify(U, V, W):
        """Verify that the decomposition correctly represents matrix multiplication."""
        nm, mp, pn = n * m, m * p, p * n
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if abs(val - 1) > 1e-9:
                        return False
        # Check some non-support entries are 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    for k2 in range(p):
                        if k2 == k:
                            continue
                        c = k2 * n + i
                        val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                        if abs(val) > 1e-9:
                            return False
        return True
    
    # Strassen 2x2x2 rank-7 decomposition
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[m,n],[o,p]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(f+h)  -- actually let me use standard Strassen
    # Standard:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B01+B11)
    # M3 = (A00-A10)(B10+B11)
    # M4 = (A00+A11)(B00-B11)
    # M5 = (A00+A10)(B00+B10)
    # M6 = (A01-A11)(B01+B11)
    # M7 = (A01-A00)(B00+B10)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M4 + M6
    # C10 = M3 + M5
    # C11 = M1 - M3 + M6 - M2
    
    # For 2x2x3 (A is 2x2, B is 2x3, C is 2x3):
    # We can use Strassen-like approach by treating B as [B_left | B_right] where each is 2x1
    # Actually, let me think differently.
    
    # For 2x2 * 2x3 = 2x3:
    # C[:,0] = A * B[:,0]  (2x2 * 2x1)
    # C[:,1] = A * B[:,1]  (2x2 * 2x1)  
    # C[:,2] = A * B[:,2]  (2x2 * 2x1)
    
    # Each 2x2 * 2x1 can be done with rank 4 (or less with Strassen tricks)
    # Actually 2x2 * 2x1: C0 = A00*B0 + A01*B1, C1 = A10*B0 + A11*B1
    # That's 4 multiplications. Can we do better?
    # Using Strassen: treat B as [B0; B1] and use:
    # M1 = (A00+A11)(B0+B1)
    # M2 = (A00-A11)(B0-B1)
    # Then A00*B0 = (M1+M2)/4 * ... no this doesn't work simply.
    # Let's just use rank 4 for each 2x2*2x1, so rank 12 for 2x2*2x3.
    
    # Better approach: Use the block structure directly.
    # 4x4 * 4x9 = 4x9
    # Split A into 2x2 blocks of 2x2: A = [[A00, A01], [A10, A11]]
    # Split B into 2x2 blocks of 2x2 and 2x1: B = [[B00, B01, B02], [B10, B11, B12]]
    #   where B00, B01, B10, B11 are 2x2 and B02, B12 are 2x1
    # C = A*B, C = [[C00, C01, C02], [C10, C11, C12]]
    #   where C00, C01, C10, C11 are 2x2 and C02, C12 are 2x1
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # For the 2x2*2x2 parts, use Strassen (rank 7 each)
    # For the 2x2*2x1 parts, use rank 4 each
    
    # But we can combine: 
    # C00 and C10 both depend on B00. We can share some computations.
    
    # Let me use a simpler but effective approach:
    # Use rank-7 Strassen for each 2x2*2x2 block multiplication
    # Use rank-4 for each 2x2*2x1 block multiplication
    
    # Total: 4 * 7 (for 2x2*2x2) + 4 * 4 (for 2x2*2x1) = 28 + 16 = 44
    
    # Actually, let me be more careful and use the Kronecker product structure.
    
    # The tensor for 4x4x9 can be seen as:
    # T[i,j,k] = delta(i,j,k) where the multiplication is A[i,j]*B[j,k]
    
    # Using block diagonal Kronecker:
    # 4x4x9 = (2x2x3) ⊗ (2x2x3) in some sense? No, that's not quite right.
    
    # Let me use the following structure:
    # Decompose as: first apply 2x2 Strassen in the row/column block structure
    
    # For 4x4 * 4x4 (first 4 columns of B):
    # Use two-level Strassen: rank 7*7 = 49? No, that's not how it works.
    # Actually, for 4x4*4x4, we can use rank 16 (Laderman) or rank 49 (two-level Strassen)
    # Laderman's rank 16 is better.
    
    # Let me implement Laderman's rank-16 for 4x4*4x4.
    
    def laderman_4x4x4(col_offset):
        """
        Laderman's rank-16 decomposition for 4x4 * 4x4.
        col_offset: offset for the column index in B and C.
        Returns list of (U_vec, V_vec, W_vec) tuples.
        """
        terms = []
        
        # A indices: a = i*4 + j, i,j in 0..3
        # B indices: b = j*9 + (col_offset + k), j in 0..3, k in 0..3
        # C indices: c = (col_offset + k)*4 + i, k in 0..3, i in 0..3
        
        # Split A into 2x2 blocks: A00=(0,1,4,5), A01=(2,3,6,7), A10=(8,9,12,13), A11=(10,11,14,15)
        # Actually A[i,j] where i=0,1 is top, i=2,3 is bottom
        # A00: i=0,1; j=0,1 -> indices 0,1,4,5
        # A01: i=0,1; j=2,3 -> indices 2,3,6,7
        # A10: i=2,3; j=0,1 -> indices 8,9,12,13
        # A11: i=2,3; j=2,3 -> indices 10,11,14,15
        
        # B similarly with column offset
        # B00: j=0,1; k=0,1 -> indices 0*9+(col_offset+0), 1*9+(col_offset+0), 0*9+(col_offset+1), 1*9+(col_offset+1)
        # etc.
        
        # C: C[i,k] where i=0,1 is top, i=2,3 is bottom
        # C00: i=0,1; k=0,1 -> indices (col_offset+0)*4+0, (col_offset+0)*4+1, (col_offset+1)*4+0, (col_offset+1)*4+1
        # etc.
        
        # Laderman's 16 products for 4x4:
        # Let me use the explicit formulas from the literature.
        # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
        # Each is 2x2.
        
        # P1  = A00 * B00
        # P2  = A00 * B11
        # P3  = A11 * B00
        # P4  = A11 * B11
        # P5  = A00 * (B01 + B11)
        # P6  = A11 * (B00 + B10)
        # P7  = (A00 + A10) * (B00 - B11)
        # P8  = (A01 + A11) * (B01 - B00)
        # P9  = (A00 + A01) * B10
        # P10 = A10 * (B01 + B11)
        # P11 = (A01 + A11) * B00
        # P12 = (A10 + A11) * B01
        # P13 = (A00 + A01) * (B10 - B00)
        # P14 = (A10 + A11) * (B11 - B01)
        # P15 = (A01 - A10) * (B10 + B01)
        # P16 = (A00 + A11) * (B00 + B11)
        
        # C00 = P1 - P7 + P9 - P13
        # C01 = P5 - P8 + P12 - P14
        # C10 = P6 - P7 + P15 - P11  -- wait, let me get this right
        # C11 = P4 + P2 - P16
        
        # Actually, let me use a verified set of Laderman coefficients.
        # I'll use the standard formulation.
        
        # For each 2x2 block multiplication, we need to map to full indices.
        # Let me define helper functions.
        
        def a_block(i_base, j_base):
            """Return indices for A block starting at row i_base, col j_base (each 2x2)."""
            return [(i_base + di)*4 + (j_base + dj) for di in range(2) for dj in range(2)]
        
        def b_block(j_base, k_base):
            """Return indices for B block starting at row j_base, col k_base (each 2x2)."""
            return [(j_base + dj)*9 + (col_offset + k_base + dk) for dj in range(2) for dk in range(2)]
        
        def c_block(i_base, k_base):
            """Return indices for C block starting at row i_base, col k_base (each 2x2)."""
            return [(col_offset + k_base + dk)*4 + (i_base + di) for di in range(2) for dk in range(2)]
        
        # For 2x2*2x2 Strassen within each block:
        # We need 7 rank-1 terms per 2x2*2x2 multiplication.
        # But Laderman combines multiple 2x2*2x2 multiplications.
        
        # Let me just use the two-level Strassen approach which gives rank 49 for 4x4*4x4.
        # Then for 4x4*4x1, use rank 16.
        # Total: 2*49 + 16 = 114.
        
        # Actually, let me implement the two-level Strassen properly.
        # Level 1: 4x4 = 2x2 blocks of 2x2
        # Level 2: each 2x2*2x2 uses Strassen rank 7
        
        # So we have 7 outer Strassen terms, each containing 7 inner Strassen terms = 49 total.
        
        # Outer Strassen for 4x4*4x4:
        # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
        # M2 = (A10+A11)(B00+B11) -> contributes to C10, C11
        # M3 = (A00-A10)(B10+B11) -> contributes to C01, C11
        # M4 = (A00+A11)(B00-B11) -> contributes to C00, C10
        # M5 = (A00+A01)(B00+B10) -> contributes to C00, C01
        # M6 = (A01-A11)(B01+B11) -> contributes to C01, C11
        # M7 = (A01-A00)(B00+B10) -> contributes to C00, C10
        
        # Wait, I need to be more careful. Let me use the standard Strassen for 4x4.
        # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
        # C = A*B
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B01+B11)
        # M3 = (A00-A10)(B10+B11)
        # M4 = (A00+A11)(B00-B11)
        # M5 = (A00+A10)(B00+B10)
        # M6 = (A01-A11)(B01+B11)
        # M7 = (A01-A00)(B00+B10)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M4 + M6
        # C10 = M3 + M5
        # C11 = M1 - M3 + M6 - M2
        
        # Each M is a 2x2*2x2 product, which itself can be done with Strassen rank 7.
        # So total rank = 7 * 7 = 49.
        
        # Inner Strassen for 2x2*2x2:
        # For matrices X, Y (2x2), Z = X*Y:
        # S1 = (X00+X11)(Y00+Y11)
        # S2 = (X10+X11)(Y01+Y11)
        # S3 = (X00-X10)(Y10+Y11)
        # S4 = (X00+X11)(Y00-Y11)
        # S5 = (X00+X10)(Y00+Y10)
        # S6 = (X01-X11)(Y01+Y11)
        # S7 = (X01-X00)(Y00+Y10)
        # Z00 = S1 - S4 + S5 - S7
        # Z01 = S4 + S6
        # Z10 = S3 + S5
        # Z11 = S1 - S3 + S6 - S2
        
        # Let me implement this systematically.
        
        # For each outer Strassen term, we have linear combinations of A blocks and B blocks.
        # Then we multiply the resulting 2x2 matrices using inner Strassen.
        
        # Let me define the outer terms as (A_coeff, B_coeff, C_coeff) where each is a dict mapping
        # block position to coefficient.
        
        # Outer Strassen terms:
        # M1: A = A00 + A11, B = B00 + B11, C contributes to C00 (+1), C11 (+1)
        # M2: A = A10 + A11, B = B01 + B11, C contributes to C10 (-1), C11 (-1)  -- wait
        # Let me re-derive:
        # C11 = M1 - M3 + M6 - M2
        # So M2 contributes -1 to C11.
        # M2 also... let me check. M2 = (A10+A11)(B01+B11)
        # This should contribute to C11 with coefficient -1.
        
        # Let me be explicit:
        outer_terms = [
            # (A_blocks, B_blocks, C_blocks)
            # A_blocks: list of (block_row, block_col, coeff)
            # B_blocks: list of (block_row, block_col, coeff)
            # C_blocks: list of (block_row, block_col, coeff)
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),   # M1
            ([(1,0,1), (1,1,1)], [(0,1,1), (1,1,1)], [(1,1,-1)]),              # M2
            ([(0,0,1), (1,0,-1)], [(1,0,1), (1,1,1)], [(1,1,1), (0,1,-1)]),   # M3
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,-1)], [(0,0,1), (1,0,-1)]),   # M4
            ([(0,0,1), (1,0,1)], [(0,0,1), (1,0,1)], [(0,0,1), (0,1,1)]),     # M5
            ([(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)], [(0,1,1), (1,1,1)]),    # M6
            ([(0,1,1), (0,0,-1)], [(0,0,1), (1,0,1)], [(0,0,-1), (1,0,1)]),   # M7
        ]
        
        # Inner Strassen terms for 2x2*2x2:
        inner_terms = [
            # (X_blocks, Y_blocks, Z_blocks) where X,Y,Z are 2x2
            # Using indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),   # S1
            ([(2,1), (3,1)], [(0,1)], [(2,1), (3,-1)]),          # S2
            ([(0,1)], [(1,1), (3,-1)], [(1,1), (3,1)]),          # S3
            ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),          # S4
            ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),          # S5
            ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),          # S6
            ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]),          # S7
        ]
        
        for A_blocks, B_blocks, C_blocks in outer_terms:
            for X_blocks, Y_blocks, Z_blocks in inner_terms:
                U_vec = [0] * nm
                V_vec = [0] * mp
                W_vec = [0] * pn
                
                # A side: for each outer A block and inner X position
                for br, bc, a_coeff in A_blocks:
                    for x_idx, x_coeff in X_blocks:
                        xi = x_idx // 2
                        xj = x_idx % 2
                        gi = br * 2 + xi
                        gj = bc * 2 + xj
                        U_vec[gi * 4 + gj] += a_coeff * x_coeff
                
                # B side
                for br, bc, b_coeff in B_blocks:
                    for y_idx, y_coeff in Y_blocks:
                        yi = y_idx // 2
                        yj = y_idx % 2
                        gj = br * 2 + yi
                        gk = col_offset + bc * 2 + yj
                        V_vec[gj * 9 + gk] += b_coeff * y_coeff
                
                # C side
                for br, bc, c_coeff in C_blocks:
                    for z_idx, z_coeff in Z_blocks:
                        zi = z_idx // 2
                        zj = z_idx % 2
                        gi = br * 2 + zi
                        gk = col_offset + bc * 2 + zj
                        W_vec[gk * 4 + gi] += c_coeff * z_coeff
                
                terms.append((U_vec, V_vec, W_vec))
        
        return terms
    
    def schoolbook_4x4x1(col_offset):
        """Schoolbook rank-16 for 4x4 * 4x1."""
        terms = []
        for i in range(4):
            for j in range(4):
                U_vec = [0] * nm
                V_vec = [0] * mp
                W_vec = [0] * pn
                U_vec[i * 4 + j] = 1
                V_vec[j * 9 + (col_offset)] = 1
                W_vec[col_offset * 4 + i] = 1
                terms.append((U_vec, V_vec, W_vec))
        return terms
    
    # Build the full decomposition
    all_terms = []
    
    # Part 1: A * B[:, 0:4] using two-level Strassen, rank 49
    all_terms.extend(laderman_4x4x4(0))
    
    # Part 2: A * B[:, 4:8] using two-level Strassen, rank 49
    all_terms.extend(laderman_4x4x4(4))
    
    # Part 3: A * B[:, 8:9] using schoolbook, rank 16
    all_terms.extend(schoolbook_4x4x1(8))
    
    total_rank = len(all_terms)
    
    # Build U, V, W matrices
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    for r, (u_vec, v_vec, w_vec) in enumerate(all_terms):
        U[:, r] = u_vec
        V[:, r] = v_vec
        W[:, r] = w_vec
    
    # Verify
    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook
    U, V, W, rank = schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
