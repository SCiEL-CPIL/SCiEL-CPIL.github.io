import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 6
    
    # We will implement a rank 77 decomposition:
    # 4x4x4 part uses rank 49 (Strassen-based)
    # 4x4x2 part uses rank 28 (Strassen-based)
    
    rank = 77
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_2x2x2_strassen(i0, j0, k0, col_start, scale=1.0):
        """
        Adds 7 columns for a 2x2x2 Strassen multiplication.
        A block is at [i0:i0+2, j0:j0+2] in the A matrix (size n x m)
        B block is at [j0:j0+2, k0:k0+2] in the B matrix (size m x p)
        C block is at [i0:i0+2, k0:k0+2] in the C matrix (size n x p)
        
        U indices: i * m + j
        V indices: j * p + k
        W indices: k * n + i
        """
        # M1 = (A00+A11)(B00+B11)
        # A: (i0,j0)=1, (i0+1,j0+1)=1
        # B: (j0,k0)=1, (j0+1,k0+1)=1
        # C: (i0,k0)=1, (i0+1,k0+1)=1
        c = col_start
        U[i0*m + j0, c] += scale
        U[(i0+1)*m + (j0+1), c] += scale
        V[j0*p + k0, c] += scale
        V[(j0+1)*p + (k0+1), c] += scale
        W[k0*n + i0, c] += scale
        W[(k0+1)*n + (i0+1), c] += scale
        
        # M2 = (A10+A11)B00
        # A: (i0+1,j0)=1, (i0+1,j0+1)=1
        # B: (j0,k0)=1
        # C: (i0+1,k0)=1
        c = col_start + 1
        U[(i0+1)*m + j0, c] += scale
        U[(i0+1)*m + (j0+1), c] += scale
        V[j0*p + k0, c] += scale
        W[k0*n + (i0+1), c] += scale
        
        # M3 = A00(B01-B11)
        # A: (i0,j0)=1
        # B: (j0,k0+1)=1, (j0+1,k0+1)=-1
        # C: (i0,k0+1)=1
        c = col_start + 2
        U[i0*m + j0, c] += scale
        V[j0*p + (k0+1), c] += scale
        V[(j0+1)*p + (k0+1), c] += -scale
        W[(k0+1)*n + i0, c] += scale
        
        # M4 = A11(B10-B00)
        # A: (i0+1,j0+1)=1
        # B: (j0+1,k0)=1, (j0,k0)=-1
        # C: (i0+1,k0)=1
        c = col_start + 3
        U[(i0+1)*m + (j0+1), c] += scale
        V[(j0+1)*p + k0, c] += scale
        V[j0*p + k0, c] += -scale
        W[k0*n + (i0+1), c] += scale
        
        # M5 = (A00+A01)B11
        # A: (i0,j0)=1, (i0,j0+1)=1
        # B: (j0+1,k0+1)=1
        # C: (i0,k0+1)=1
        c = col_start + 4
        U[i0*m + j0, c] += scale
        U[i0*m + (j0+1), c] += scale
        V[(j0+1)*p + (k0+1), c] += scale
        W[(k0+1)*n + i0, c] += scale
        
        # M6 = (A10-A00)(B00+B01)
        # A: (i0+1,j0)=1, (i0,j0)=-1
        # B: (j0,k0)=1, (j0,k0+1)=1
        # C: (i0+1,k0+1)=1
        c = col_start + 5
        U[(i0+1)*m + j0, c] += scale
        U[i0*m + j0, c] += -scale
        V[j0*p + k0, c] += scale
        V[j0*p + (k0+1), c] += scale
        W[(k0+1)*n + (i0+1), c] += scale
        
        # M7 = (A01-A11)(B10+B11)
        # A: (i0,j0+1)=1, (i0+1,j0+1)=-1
        # B: (j0+1,k0)=1, (j0+1,k0+1)=1
        # C: (i0,k0)=1
        c = col_start + 6
        U[i0*m + (j0+1), c] += scale
        U[(i0+1)*m + (j0+1), c] += -scale
        V[(j0+1)*p + k0, c] += scale
        V[(j0+1)*p + (k0+1), c] += scale
        W[k0*n + i0, c] += scale
        
        return col_start + 7

    # Part 1: 4x4x4 multiplication using rank 49
    # We split the 4x4 matrices into 2x2 blocks and use Strassen recursively.
    # The 4x4x4 tensor can be decomposed into 7 "Strassen steps" at the top level,
    # but each step involves 2x2 blocks which are themselves 2x2x2 multiplications.
    # Actually, a standard rank 49 decomposition for 4x4x4 uses 7 calls to a 
    # rank-7 decomposition for 2x2x2, but with linear combinations of the blocks.
    
    # Let's define the 7 top-level multipliers for 4x4x4 using 2x2 blocks.
    # Let A = [[A00, A01], [A10, A11]] where each is 2x2.
    # Let B = [[B00, B01], [B10, B11]] where each is 2x2.
    # C = [[C00, C01], [C10, C11]] where each is 2x2.
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi is a 2x2 matrix product. We need to decompose each 2x2x2 product
    # into 7 scalar multipliers. So total 7*7 = 49 multipliers.
    
    # However, we can't just call add_2x2x2_strassen directly because the inputs
    # to each 2x2x2 multiplication are linear combinations of the original 2x2 blocks.
    
    # Let's define a helper that adds a 2x2x2 multiplication where the inputs are
    # linear combinations of the 2x2 blocks at specific positions.
    
    def add_linear_2x2x2(A_comb, B_comb, C_comb, col_start, scale=1.0):
        """
        Adds 7 columns for a 2x2x2 Strassen multiplication where:
        A_comb: dict mapping (di, dj) to (i0, j0, coeff) for the A block
                 di, dj in {0,1} are the local 2x2 indices
                 (i0, j0) is the global 2x2 block position (0 or 1)
        B_comb: dict mapping (dj, dk) to (j0, k0, coeff) for the B block
        C_comb: dict mapping (di, dk) to (i0, k0, coeff) for the C block
        
        Actually, let's simplify. We know the 7 Strassen multipliers.
        For each multiplier, we have a specific pattern of A, B, C coefficients.
        We need to expand these into global indices.
        
        Let's define the 7 Strassen patterns as lists of (di, dj, coeff) for A,
        (dj, dk, coeff) for B, and (di, dk, coeff) for C.
        """
        strassen_patterns = [
            # M1: (A00+A11)(B00+B11)
            ([((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))]),
            # M2: (A10+A11)B00
            ([((1,0,1), (1,1,1))], [((0,0,1))], [((1,0,1))]),
            # M3: A00(B01-B11)
            ([((0,0,1))], [((0,1,1), (1,1,-1))], [((0,1,1))]),
            # M4: A11(B10-B00)
            ([((1,1,1))], [((1,0,1), (0,0,-1))], [((1,0,1))]),
            # M5: (A00+A01)B11
            ([((0,0,1), (0,1,1))], [((1,1,1))], [((0,1,1))]),
            # M6: (A10-A00)(B00+B01)
            ([((1,0,1), (0,0,-1))], [((0,0,1), (0,1,1))], [((1,1,1))]),
            # M7: (A01-A11)(B10+B11)
            ([((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))], [((0,0,1))]),
        ]
        
        for t in range(7):
            col = col_start + t
            A_parts, B_parts, C_parts = strassen_patterns[t]
            
            for (di, dj, coeff_a) in A_parts:
                # Global A block is at (A_comb[di,dj] if it exists)
                # But A_comb maps local (di,dj) to global block (gi, gj)
                # Let's assume A_comb is a dict: {(di,dj): (gi, gj, coeff)}
                if (di, dj) in A_comb:
                    gi, gj, coeff_global = A_comb[(di, dj)]
                    # The local element (di, dj) in the 2x2 block corresponds to
                    # global element (gi*2 + di, gj*2 + dj)
                    # But wait, the "A block" is already 2x2.
                    # So the local indices (di, dj) are within the 2x2 block.
                    # The global indices are (gi*2 + di, gj*2 + dj).
                    i_global = gi * 2 + di
                    j_global = gj * 2 + dj
                    U[i_global * m + j_global, col] += coeff_a * coeff_global * scale
            
            for (dj, dk, coeff_b) in B_parts:
                if (dj, dk) in B_comb:
                    gj, gk, coeff_global = B_comb[(dj, dk)]
                    j_global = gj * 2 + dj
                    k_global = gk * 2 + dk
                    V[j_global * p + k_global, col] += coeff_b * coeff_global * scale
            
            for (di, dk, coeff_c) in C_parts:
                if (di, dk) in C_comb:
                    gi, gk, coeff_global = C_comb[(di, dk)]
                    i_global = gi * 2 + di
                    k_global = gk * 2 + dk
                    W[k_global * n + i_global, col] += coeff_c * coeff_global * scale
        
        return col_start + 7

    # Now, let's define the 7 top-level Strassen multipliers for 4x4x4.
    # Each multiplier is a 2x2x2 multiplication of linear combinations of 2x2 blocks.
    
    # M1 = (A00+A11)(B00+B11)
    # A00 is at (0,0), A11 is at (1,1)
    # So the "effective A" has A00 and A11.
    # But the 2x2x2 multiplication expects a 2x2 matrix.
    # The local indices (di, dj) in the 2x2x2 multiplication correspond to
    # the 2x2 blocks. So:
    # Local (0,0) -> Global block (0,0) with coeff 1
    # Local (1,1) -> Global block (1,1) with coeff 1
    # Local (0,1) and (1,0) are not used (coeff 0)
    
    A_comb_M1 = {(0,0): (0,0,1), (1,1): (1,1,1)}
    B_comb_M1 = {(0,0): (0,0,1), (1,1): (1,1,1)}
    C_comb_M1 = {(0,0): (0,0,1), (1,1): (1,1,1)}
    
    col = 0
    col = add_linear_2x2x2(A_comb_M1, B_comb_M1, C_comb_M1, col)
    
    # M2 = (A10+A11)B00
    # A10 is at (1,0), A11 is at (1,1)
    # B00 is at (0,0)
    # C10 is at (1,0)
    A_comb_M2 = {(1,0): (1,0,1), (1,1): (1,1,1)}
    B_comb_M2 = {(0,0): (0,0,1)}
    C_comb_M2 = {(1,0): (1,0,1)}
    
    col = add_linear_2x2x2(A_comb_M2, B_comb_M2, C_comb_M2, col)
    
    # M3 = A00(B01-B11)
    # A00 is at (0,0)
    # B01 is at (0,1), B11 is at (1,1)
    # C01 is at (0,1)
    A_comb_M3 = {(0,0): (0,0,1)}
    B_comb_M3 = {(0,1): (0,1,1), (1,1): (1,1,-1)}
    C_comb_M3 = {(0,1): (0,1,1)}
    
    col = add_linear_2x2x2(A_comb_M3, B_comb_M3, C_comb_M3, col)
    
    # M4 = A11(B10-B00)
    # A11 is at (1,1)
    # B10 is at (1,0), B00 is at (0,0)
    # C10 is at (1,0)
    A_comb_M4 = {(1,1): (1,1,1)}
    B_comb_M4 = {(1,0): (1,0,1), (0,0): (0,0,-1)}
    C_comb_M4 = {(1,0): (1,0,1)}
    
    col = add_linear_2x2x2(A_comb_M4, B_comb_M4, C_comb_M4, col)
    
    # M5 = (A00+A01)B11
    # A00 is at (0,0), A01 is at (0,1)
    # B11 is at (1,1)
    # C01 is at (0,1)
    A_comb_M5 = {(0,0): (0,0,1), (0,1): (0,1,1)}
    B_comb_M5 = {(1,1): (1,1,1)}
    C_comb_M5 = {(0,1): (0,1,1)}
    
    col = add_linear_2x2x2(A_comb_M5, B_comb_M5, C_comb_M5, col)
    
    # M6 = (A10-A00)(B00+B01)
    # A10 is at (1,0), A00 is at (0,0)
    # B00 is at (0,0), B01 is at (0,1)
    # C11 is at (1,1)
    A_comb_M6 = {(1,0): (1,0,1), (0,0): (0,0,-1)}
    B_comb_M6 = {(0,0): (0,0,1), (0,1): (0,1,1)}
    C_comb_M6 = {(1,1): (1,1,1)}
    
    col = add_linear_2x2x2(A_comb_M6, B_comb_M6, C_comb_M6, col)
    
    # M7 = (A01-A11)(B10+B11)
    # A01 is at (0,1), A11 is at (1,1)
    # B10 is at (1,0), B11 is at (1,1)
    # C00 is at (0,0)
    A_comb_M7 = {(0,1): (0,1,1), (1,1): (1,1,-1)}
    B_comb_M7 = {(1,0): (1,0,1), (1,1): (1,1,1)}
    C_comb_M7 = {(0,0): (0,0,1)}
    
    col = add_linear_2x2x2(A_comb_M7, B_comb_M7, C_comb_M7, col)
    
    # Now Part 2: 4x4x2 multiplication using rank 28
    # We split the 4x4 matrix into 2x2 blocks and the 4x2 matrix into 2x1 blocks?
    # Or 4x4x2 can be seen as two 4x4x1 multiplications, each of rank 14?
    # Or we can use Strassen-like decomposition.
    
    # Let's use the fact that 4x4x2 = 2 * 4x4x1.
    # Rank of 4x4x1 is 14 (using Strassen for 2x2x1, rank 4, so 4x4x1 has rank 4*4/2? No.)
    # Actually, 2x2x1 has rank 4 (trivial) or 3? No, 2x2x1 is just 4 independent terms.
    # Wait, 2x2x1 tensor: C = A * B where A is 2x2, B is 2x1, C is 2x1.
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # This has rank 4 (no Strassen reduction for 1-column B).
    # So 4x4x1 has rank 4*4 = 16? No, we can use Strassen on the 2x2 blocks.
    # 4x4x1: Split A into 2x2 blocks, B into 2x1 blocks.
    # C = [[A00, A01], [A10, A11]] * [[B00], [B10]] where B00, B10 are 2x1.
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    # Each 2x2*2x1 = 2x1 has rank 4.
    # So total rank 4*4 = 16? But we can do better?
    # Actually, for 4x4x2, we can use rank 28.
    
    # Let's implement 4x4x2 using 7 Strassen multipliers for 2x2x2, but with B being 2x1.
    # Wait, B is 4x2, so it's two 2x1 blocks.
    # Let's just use the trivial decomposition for the remaining 2 columns of B.
    # But that would be rank 49 + 16 + 16 = 81. We want 77.
    
    # Let's use a known rank 28 decomposition for 4x4x2.
    # This can be achieved by applying Strassen to the 2x2 blocks with 2x1 B blocks.
    # Each 2x2x1 multiplication has rank 4.
    # But we have 4 such multiplications (2x2 blocks of A, 2 columns of B).
    # So 4*4 = 16? No, that's not right.
    
    # Let's just use the trivial decomposition for the 4x4x2 part.
    # Rank 4*4*2 = 32. Total 49+32 = 81.
    # But we want 77. So we need a better decomposition for 4x4x2.
    
    # Actually, 4x4x2 can be decomposed into 7 Strassen multipliers where each
    # multiplier is a 2x2x1 multiplication (rank 4). So 7*4 = 28.
    
    # Let's implement this.
    
    def add_2x2x1_strassen(i0, j0, k0, col_start, scale=1.0):
        """
        Adds 4 columns for a 2x2x1 trivial multiplication.
        A block is at [i0:i0+2, j0:j0+2]
        B block is at [j0:j0+2, k0:k0+1]
        C block is at [i0:i0+2, k0:k0+1]
        """
        for di in range(2):
            for dj in range(2):
                col = col_start + di*2 + dj
                i_global = i0 + di
                j_global = j0 + dj
                U[i_global * m + j_global, col] += scale
                V[j_global * p + k0, col] += scale
                W[k0 * n + i_global, col] += scale
        return col_start + 4
    
    # Now, apply Strassen to 4x4x2.
    # M1 = (A00+A11)(B00+B10) where B00, B10 are 2x1
    # But B00+B10 is a 2x1 matrix.
    # So we need to add 4 columns for each 2x2x1 multiplication.
    
    A_comb_M1_2 = {(0,0): (0,0,1), (1,1): (1,1,1)}
    B_comb_M1_2 = {(0,0): (0,0,1), (1,0): (1,0,1)}
    C_comb_M1_2 = {(0,0): (0,0,1), (1,0): (1,0,1)}
    
    # But add_linear_2x2x2 expects 2x2x2. We need a 2x2x1 version.
    # Let's write a separate function.
    
    def add_linear_2x2x1(A_comb, B_comb, C_comb, col_start, scale=1.0):
        """
        Adds 4 columns for a 2x2x1 trivial multiplication where:
        A_comb: dict mapping (di, dj) to (gi, gj, coeff)
        B_comb: dict mapping (dj, dk) to (gj, gk, coeff) where dk is always 0
        C_comb: dict mapping (di, dk) to (gi, gk, coeff) where dk is always 0
        """
        for di in range(2):
            for dj in range(2):
                col = col_start + di*2 + dj
                if (di, dj) in A_comb:
                    gi, gj, coeff_a = A_comb[(di, dj)]
                    i_global = gi * 2 + di
                    j_global = gj * 2 + dj
                    U[i_global * m + j_global, col] += coeff_a * scale
                
                if (dj, 0) in B_comb:
                    gj, gk, coeff_b = B_comb[(dj, 0)]
                    j_global = gj * 2 + dj
                    k_global = gk * 2 + 0
                    V[j_global * p + k_global, col] += coeff_b * scale
                
                if (di, 0) in C_comb:
                    gi, gk, coeff_c = C_comb[(di, 0)]
                    i_global = gi * 2 + di
                    k_global = gk * 2 + 0
                    W[k_global * n + i_global, col] += coeff_c * scale
        
        return col_start + 4
    
    # M1 = (A00+A11)(B00+B10)
    A_comb_M1_2 = {(0,0): (0,0,1), (1,1): (1,1,1)}
    B_comb_M1_2 = {(0,0): (0,0,1), (1,0): (1,0,1)}
    C_comb_M1_2 = {(0,0): (0,0,1), (1,0): (1,0,1)}
    
    col = add_linear_2x2x1(A_comb_M1_2, B_comb_M1_2, C_comb_M1_2, col)
    
    # M2 = (A10+A11)B00
    A_comb_M2_2 = {(1,0): (1,0,1), (1,1): (1,1,1)}
    B_comb_M2_2 = {(0,0): (0,0,1)}
    C_comb_M2_2 = {(1,0): (1,0,1)}
    
    col = add_linear_2x2x1(A_comb_M2_2, B_comb_M2_2, C_comb_M2_2, col)
    
    # M3 = A00(B01-B11) -> but B is 4x2, so B01 and B11 are 2x1?
    # Wait, B is 4x2, so it's [[B00, B01], [B10, B11]] where each is 2x1.
    # So B01 is at (0,1), B11 is at (1,1).
    # But in the 2x2x1 decomposition, we only have k0=0 or k0=1.
    # Let's be careful.
    
    # Actually, for 4x4x2, the B matrix is 4x2.
    # When we split into 2x2 blocks, B00 is 2x1, B01 is 2x1, etc.
    # So B00 is at (0,0), B01 is at (0,1), B10 is at (1,0), B11 is at (1,1).
    
    # M3 = A00(B01-B11)
    # A00 is at (0,0)
    # B01 is at (0,1), B11 is at (1,1)
    # C01 is at (0,1)
    A_comb_M3_2 = {(0,0): (0,0,1)}
    B_comb_M3_2 = {(0,1): (0,1,1), (1,1): (1,1,-1)}
    C_comb_M3_2 = {(0,1): (0,1,1)}
    
    col = add_linear_2x2x1(A_comb_M3_2, B_comb_M3_2, C_comb_M3_2, col)
    
    # M4 = A11(B10-B00)
    A_comb_M4_2 = {(1,1): (1,1,1)}
    B_comb_M4_2 = {(1,0): (1,0,1), (0,0): (0,0,-1)}
    C_comb_M4_2 = {(1,0): (1,0,1)}
    
    col = add_linear_2x2x1(A_comb_M4_2, B_comb_M4_2, C_comb_M4_2, col)
    
    # M5 = (A00+A01)B11
    A_comb_M5_2 = {(0,0): (0,0,1), (0,1): (0,1,1)}
    B_comb_M5_2 = {(1,1): (1,1,1)}
    C_comb_M5_2 = {(0,1): (0,1,1)}
    
    col = add_linear_2x2x1(A_comb_M5_2, B_comb_M5_2, C_comb_M5_2, col)
    
    # M6 = (A10-A00)(B00+B01)
    A_comb_M6_2 = {(1,0): (1,0,1), (0,0): (0,0,-1)}
    B_comb_M6_2 = {(0,0): (0,0,1), (0,1): (0,1,1)}
    C_comb_M6_2 = {(1,1): (1,1,1)}
    
    col = add_linear_2x2x1(A_comb_M6_2, B_comb_M6_2, C_comb_M6_2, col)
    
    # M7 = (A01-A11)(B10+B11)
    A_comb_M7_2 = {(0,1): (0,1,1), (1,1): (1,1,-1)}
    B_comb_M7_2 = {(1,0): (1,0,1), (1,1): (1,1,1)}
    C_comb_M7_2 = {(0,0): (0,0,1)}
    
    col = add_linear_2x2x1(A_comb_M7_2, B_comb_M7_2, C_comb_M7_2, col)
    
    # Total rank should be 49 + 28 = 77.
    # Let's verify the column count.
    assert col == 77, f"Expected 77 columns, got {col}"
    
    # Convert to lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
