import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with 10s margin
    
    n, m, p = 4, 4, 6
    
    # We'll build a rank-77 decomposition:
    # - 49 terms for the 4x4x4 part (using recursive Strassen)
    # # - 28 terms for the 4x4x2 part (using recursive Strassen)
    
    def add_strassen_term(A_terms, B_terms, C_terms, col):
        """
        Add one rank term where:
        A_terms: list of (i, j, coeff) for U
        B_terms: list of (j, k, coeff) for V  
        C_terms: list of (k, i, coeff) for W
        """
        for i, j, coeff in A_terms:
            U[i * m + j][col] += coeff
        for j, k, coeff in B_terms:
            V[j * p + k][col] += coeff
        for k, i, coeff in C_terms:
            W[k * n + i][col] += coeff
    
    def build_strassen_2x2x2(i0, j0, k0, col_start, U, V, W, m, p, n):
        """
        Build 7 rank terms for 2x2x2 Strassen multiplication.
        A block: rows i0:i0+2, cols j0:j0+2
        B block: rows j0:j0+2, cols k0:k0+2
        C block: rows i0:i0+2, cols k0:k0+2
        Returns col_start + 7
        """
        col = col_start
        
        # M1 = (A00+A11)(B00+B11)
        A1 = [(i0, j0, 1), (i0+1, j0+1, 1)]
        B1 = [(j0, k0, 1), (j0+1, k0+1, 1)]
        C1 = [(k0, i0, 1), (k0+1, i0+1, 1)]
        add_strassen_term(A1, B1, C1, col)
        col += 1
        
        # M2 = (A10+A11)B00
        A2 = [(i0+1, j0, 1), (i0+1, j0+1, 1)]
        B2 = [(j0, k0, 1)]
        C2 = [(k0, i0, 1), (k0+1, i0+1, 1)]
        add_strassen_term(A2, B2, C2, col)
        col += 1
        
        # M3 = A00(B01-B11)
        A3 = [(i0, j0, 1)]
        B3 = [(j0, k0+1, 1), (j0+1, k0+1, -1)]
        C3 = [(k0, i0, 1), (k0+1, i0+1, 1)]
        add_strassen_term(A3, B3, C3, col)
        col += 1
        
        # M4 = A11(B10-B00)
        A4 = [(i0+1, j0+1, 1)]
        B4 = [(j0+1, k0, 1), (j0, k0, -1)]
        C4 = [(k0, i0, 1), (k0+1, i0+1, 1)]
        add_strassen_term(A4, B4, C4, col)
        col += 1
        
        # M5 = (A00+A01)B11
        A5 = [(i0, j0, 1), (i0, j0+1, 1)]
        B5 = [(j0+1, k0+1, 1)]
        C5 = [(k0, i0, 1), (k0+1, i0+1, 1)]
        add_strassen_term(A5, B5, C5, col)
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        A6 = [(i0+1, j0, 1), (i0, j0, -1)]
        B6 = [(j0, k0, 1), (j0, k0+1, 1)]
        C6 = [(k0, i0, 1), (k0+1, i0+1, 1)]
        add_strassen_term(A6, B6, C6, col)
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        A7 = [(i0, j0+1, 1), (i0+1, j0+1, -1)]
        B7 = [(j0+1, k0, 1), (j0+1, k0+1, 1)]
        C7 = [(k0, i0, 1), (k0+1, i0+1, 1)]
        add_strassen_term(A7, B7, C7, col)
        col += 1
        
        return col
    
    def build_strassen_4x4x4(i0, j0, k0, col_start, U, V, W, m, p, n):
        """
        Build 49 rank terms for 4x4x4 multiplication using recursive Strassen.
        """
        # First, we need to decompose the 4x4x4 case.
        # We can use 7 calls to a 2x2x2 Strassen on linear combinations.
        # But we need to be careful: the "effective" A, B, C are 2x2 matrices
        # where each element is a 2x2 block of the original.
        
        # Let's define the 7 Strassen multipliers for 4x4x4:
        # M1 = (A00+A11)(B00+B11) where A00 is top-left 2x2, etc.
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each of these is a 2x2x2 multiplication of 2x2 blocks.
        # So each contributes 7 rank terms, total 7*7 = 49.
        
        col = col_start
        
        # Helper to get block offsets
        # A block (bi, bj) has i offset i0 + 2*bi, j offset j0 + 2*bj
        # B block (bj, bk) has j offset j0 + 2*bj, k offset k0 + 2*bk  
        # C block (bi, bk) has i offset i0 + 2*bi, k offset k0 + 2*bk
        
        # M1: (A00+A11)(B00+B11)
        # A part: A00 (top-left) + A11 (bottom-right)
        # B part: B00 (top-left) + B11 (bottom-right)
        # C part: C00 (top-left) + C11 (bottom-right)
        # This is a 2x2x2 Strassen where:
        # Effective A00 = A00 (block 0,0), A11 = A11 (block 1,1)
        # Effective B00 = B00 (block 0,0), B11 = B11 (block 1,1)
        # Effective C00 = C00 (block 0,0), C11 = C11 (block 1,1)
        
        # We need to call build_strassen_2x2x2 with appropriate offsets,
        # but the "effective" matrices are not contiguous in the same way.
        
        # Actually, for M1, the effective A matrix has:
        # A_eff[0,0] = A[i0:i0+2, j0:j0+2] (block 0,0)
        # A_eff[1,1] = A[i0+2:i0+4, j0+2:j0+4] (block 1,1)
        # A_eff[0,1] = 0, A_eff[1,0] = 0
        
        # This is not a standard 2x2x2 Strassen call because the blocks are not adjacent.
        
        # Instead, let's directly add the 7 terms for each multiplier.
        
        def add_2x2x2_with_offsets(A_offsets, B_offsets, C_offsets, col):
            """
            Add 7 rank terms for 2x2x2 where:
            A_offsets: dict mapping (bi,bj) to (i_off, j_off) for block (bi,bj)
            B_offsets: dict mapping (bj,bk) to (j_off, k_off) for block (bj,bk)
            C_offsets: dict mapping (bi,bk) to (i_off, k_off) for block (bi,bk)
            """
            col = col
            
            # M1 = (A00+A11)(B00+B11)
            A1 = []
            for (bi, bj), (i_off, j_off) in A_offsets.items():
                if (bi, bj) in [(0,0), (1,1)]:
                    # Add 2x2 block
                    for di in range(2):
                        for dj in range(2):
                            A1.append((i_off + di, j_off + dj, 1))
            B1 = []
            for (bj, bk), (j_off, k_off) in B_offsets.items():
                if (bj, bk) in [(0,0), (1,1)]:
                    for dj in range(2):
                        for dk in range(2):
                            B1.append((j_off + dj, k_off + dk, 1))
            C1 = []
            for (bi, bk), (i_off, k_off) in C_offsets.items():
                if (bi, bk) in [(0,0), (1,1)]:
                    for di in range(2):
                        for dk in range(2):
                            C1.append((k_off + dk, i_off + di, 1))
            add_strassen_term(A1, B1, C1, col)
            col += 1
            
            # M2 = (A10+A11)B00
            A2 = []
            for (bi, bj), (i_off, j_off) in A_offsets.items():
                if (bi, bj) in [(1,0), (1,1)]:
                    for di in range(2):
                        for dj in range(2):
                            A2.append((i_off + di, j_off + dj, 1))
            B2 = []
            for (bj, bk), (j_off, k_off) in B_offsets.items():
                if (bj, bk) == (0,0):
                    for dj in range(2):
                        for dk in range(2):
                            B2.append((j_off + dj, k_off + dk, 1))
            C2 = []
            for (bi, bk), (i_off, k_off) in C_offsets.items():
                if (bi, bk) in [(0,0), (1,1)]:
                    for di in range(2):
                        for dk in range(2):
                            C2.append((k_off + dk, i_off + di, 1))
            add_strassen_term(A2, B2, C2, col)
            col += 1
            
            # M3 = A00(B01-B11)
            A3 = []
            for (bi, bj), (i_off, j_off) in A_offsets.items():
                if (bi, bj) == (0,0):
                    for di in range(2):
                        for dj in range(2):
                            A3.append((i_off + di, j_off + dj, 1))
            B3 = []
            for (bj, bk), (j_off, k_off) in B_offsets.items():
                if (bj, bk) == (0,1):
                    for dj in range(2):
                        for dk in range(2):
                            B3.append((j_off + dj, k_off + dk, 1))
                if (bj, bk) == (1,1):
                    for dj in range(2):
                        for dk in range(2):
                            B3.append((j_off + dj, k_off + dk, -1))
            C3 = []
            for (bi, bk), (i_off, k_off) in C_offsets.items():
                if (bi, bk) in [(0,0), (1,1)]:
                    for di in range(2):
                        for dk in range(2):
                            C3.append((k_off + dk, i_off + di, 1))
            add_strassen_term(A3, B3, C3, col)
            col += 1
            
            # M4 = A11(B10-B00)
            A4 = []
            for (bi, bj), (i_off, j_off) in A_offsets.items():
                if (bi, bj) == (1,1):
                    for di in range(2):
                        for dj in range(2):
                            A4.append((i_off + di, j_off + dj, 1))
            B4 = []
            for (bj, bk), (j_off, k_off) in B_offsets.items():
                if (bj, bk) == (1,0):
                    for dj in range(2):
                        for dk in range(2):
                            B4.append((j_off + dj, k_off + dk, 1))
                if (bj, bk) == (0,0):
                    for dj in range(2):
                        for dk in range(2):
                            B4.append((j_off + dj, k_off + dk, -1))
            C4 = []
            for (bi, bk), (i_off, k_off) in C_offsets.items():
                if (bi, bk) in [(0,0), (1,1)]:
                    for di in range(2):
                        for dk in range(2):
                            C4.append((k_off + dk, i_off + di, 1))
            add_strassen_term(A4, B4, C4, col)
            col += 1
            
            # M5 = (A00+A01)B11
            A5 = []
            for (bi, bj), (i_off, j_off) in A_offsets.items():
                if (bi, bj) in [(0,0), (0,1)]:
                    for di in range(2):
                        for dj in range(2):
                            A5.append((i_off + di, j_off + dj, 1))
            B5 = []
            for (bj, bk), (j_off, k_off) in B_offsets.items():
                if (bj, bk) == (1,1):
                    for dj in range(2):
                        for dk in range(2):
                            B5.append((j_off + dj, k_off + dk, 1))
            C5 = []
            for (bi, bk), (i_off, k_off) in C_offsets.items():
                if (bi, bk) in [(0,0), (1,1)]:
                    for di in range(2):
                        for dk in range(2):
                            C5.append((k_off + dk, i_off + di, 1))
            add_strassen_term(A5, B5, C5, col)
            col += 1
            
            # M6 = (A10-A00)(B00+B01)
            A6 = []
            for (bi, bj), (i_off, j_off) in A_offsets.items():
                if (bi, bj) == (1,0):
                    for di in range(2):
                        for dj in range(2):
                            A6.append((i_off + di, j_off + dj, 1))
                if (bi, bj) == (0,0):
                    for di in range(2):
                        for dj in range(2):
                            A6.append((i_off + di, j_off + dj, -1))
            B6 = []
            for (bj, bk), (j_off, k_off) in B_offsets.items():
                if (bj, bk) in [(0,0), (0,1)]:
                    for dj in range(2):
                        for dk in range(2):
                            B6.append((j_off + dj, k_off + dk, 1))
            C6 = []
            for (bi, bk), (i_off, k_off) in C_offsets.items():
                if (bi, bk) in [(0,0), (1,1)]:
                    for di in range(2):
                        for dk in range(2):
                            C6.append((k_off + dk, i_off + di, 1))
            add_strassen_term(A6, B6, C6, col)
            col += 1
            
            # M7 = (A01-A11)(B10+B11)
            A7 = []
            for (bi, bj), (i_off, j_off) in A_offsets.items():
                if (bi, bj) == (0,1):
                    for di in range(2):
                        for dj in range(2):
                            A7.append((i_off + di, j_off + dj, 1))
                if (bi, bj) == (1,1):
                    for di in range(2):
                        for dj in range(2):
                            A7.append((i_off + di, j_off + dj, -1))
            B7 = []
            for (bj, bk), (j_off, k_off) in B_offsets.items():
                if (bj, bk) in [(1,0), (1,1)]:
                    for dj in range(2):
                        for dk in range(2):
                            B7.append((j_off + dj, k_off + dk, 1))
            C7 = []
            for (bi, bk), (i_off, k_off) in C_offsets.items():
                if (bi, bk) in [(0,0), (1,1)]:
                    for di in range(2):
                        for dk in range(2):
                            C7.append((k_off + dk, i_off + di, 1))
            add_strassen_term(A7, B7, C7, col)
            col += 1
            
            return col
        
        # For M1: A blocks (0,0) and (1,1), B blocks (0,0) and (1,1), C blocks (0,0) and (1,1)
        A_offsets_M1 = {(0,0): (i0, j0), (1,1): (i0+2, j0+2)}
        B_offsets_M1 = {(0,0): (j0, k0), (1,1): (j0+2, k0+2)}
        C_offsets_M1 = {(0,0): (i0, k0), (1,1): (i0+2, k0+2)}
        col = add_2x2x2_with_offsets(A_offsets_M1, B_offsets_M1, C_offsets_M1, col)
        
        # M2: A blocks (1,0) and (1,1), B block (0,0), C blocks (0,0) and (1,1)
        A_offsets_M2 = {(1,0): (i0+2, j0), (1,1): (i0+2, j0+2)}
        B_offsets_M2 = {(0,0): (j0, k0)}
        C_offsets_M2 = {(0,0): (i0, k0), (1,1): (i0+2, k0+2)}
        col = add_2x2x2_with_offsets(A_offsets_M2, B_offsets_M2, C_offsets_M2, col)
        
        # M3: A block (0,0), B blocks (0,1) and (1,1), C blocks (0,0) and (1,1)
        A_offsets_M3 = {(0,0): (i0, j0)}
        B_offsets_M3 = {(0,1): (j0, k0+2), (1,1): (j0+2, k0+2)}
        C_offsets_M3 = {(0,0): (i0, k0), (1,1): (i0+2, k0+2)}
        col = add_2x2x2_with_offsets(A_offsets_M3, B_offsets_M3, C_offsets_M3, col)
        
        # M4: A block (1,1), B blocks (1,0) and (0,0), C blocks (0,0) and (1,1)
        A_offsets_M4 = {(1,1): (i0+2, j0+2)}
        B_offsets_M4 = {(1,0): (j0+2, k0), (0,0): (j0, k0)}
        C_offsets_M4 = {(0,0): (i0, k0), (1,1): (i0+2, k0+2)}
        col = add_2x2x2_with_offsets(A_offsets_M4, B_offsets_M4, C_offsets_M4, col)
        
        # M5: A blocks (0,0) and (0,1), B block (1,1), C blocks (0,0) and (1,1)
        A_offsets_M5 = {(0,0): (i0, j0), (0,1): (i0, j0+2)}
        B_offsets_M5 = {(1,1): (j0+2, k0+2)}
        C_offsets_M5 = {(0,0): (i0, k0), (1,1): (i0+2, k0+2)}
        col = add_2x2x2_with_offsets(A_offsets_M5, B_offsets_M5, C_offsets_M5, col)
        
        # M6: A blocks (1,0) and (0,0), B blocks (0,0) and (0,1), C blocks (0,0) and (1,1)
        A_offsets_M6 = {(1,0): (i0+2, j0), (0,0): (i0, j0)}
        B_offsets_M6 = {(0,0): (j0, k0), (0,1): (j0, k0+2)}
        C_offsets_M6 = {(0,0): (i0, k0), (1,1): (i0+2, k0+2)}
        col = add_2x2x2_with_offsets(A_offsets_M6, B_offsets_M6, C_offsets_M6, col)
        
        # M7: A blocks (0,1) and (1,1), B blocks (1,0) and (1,1), C blocks (0,0) and (1,1)
        A_offsets_M7 = {(0,1): (i0, j0+2), (1,1): (i0+2, j0+2)}
        B_offsets_M7 = {(1,0): (j0+2, k0), (1,1): (j0+2, k0+2)}
        C_offsets_M7 = {(0,0): (i0, k0), (1,1): (i0+2, k0+2)}
        col = add_2x2x2_with_offsets(A_offsets_M7, B_offsets_M7, C_offsets_M7, col)
        
        return col
    
    # Build the 4x4x4 part (49 terms)
    U = [[0] * 77 for _ in range(n * m)]
    V = [[0] * 77 for _ in range(m * p)]
    W = [[0] * 77 for _ in range(p * n)]
    
    col = 0
    col = build_strassen_4x4x4(0, 0, 0, col, U, V, W, m, p, n)
    
    # Now build the 4x4x2 part (28 terms)
    # This is multiplying 4x4 by 4x2 to get 4x2
    # We can use a similar approach but with 2x2x1 or 2x1x1 blocks
    
    def add_2x2x1_term(A_terms, B_terms, C_terms, col):
        """
        Add one rank term for 2x2x1 multiplication.
        A: 2x2, B: 2x1, C: 2x1
        """
        for i, j, coeff in A_terms:
            U[i * m + j][col] += coeff
        for j, k, coeff in B_terms:
            V[j * p + k][col] += coeff
        for k, i, coeff in C_terms:
            W[k * n + i][col] += coeff
    
    def build_strassen_2x2x1(i0, j0, k0, col_start, U, V, W, m, p, n):
        """
        Build 5 rank terms for 2x2x1 multiplication using Strassen-like decomposition.
        A block: rows i0:i0+2, cols j0:j0+2
        B block: rows j0:j0+2, cols k0:k0+1
        C block: rows i0:i0+2, cols k0:k0+1
        """
        col = col_start
        
        # For 2x2x1, we can use 5 multiplications:
        # M1 = (A00+A11)B0
        # M2 = (A10+A11)B0
        # M3 = A00(B1-B0)
        # M4 = A11(B1-B0)
        # M5 = (A00+A01)B1
        
        # M1 = (A00+A11)B0
        A1 = [(i0, j0, 1), (i0+1, j0+1, 1)]
        B1 = [(j0, k0, 1)]
        C1 = [(k0, i0, 1), (k0, i0+1, 1)]
        add_2x2x1_term(A1, B1, C1, col)
        col += 1
        
        # M2 = (A10+A11)B0
        A2 = [(i0+1, j0, 1), (i0+1, j0+1, 1)]
        B2 = [(j0, k0, 1)]
        C2 = [(k0, i0, 1), (k0, i0+1, 1)]
        add_2x2x1_term(A2, B2, C2, col)
        col += 1
        
        # M3 = A00(B1-B0)
        A3 = [(i0, j0, 1)]
        B3 = [(j0+1, k0, 1), (j0, k0, -1)]
        C3 = [(k0, i0, 1), (k0, i0+1, 1)]
        add_2x2x1_term(A3, B3, C3, col)
        col += 1
        
        # M4 = A11(B1-B0)
        A4 = [(i0+1, j0+1, 1)]
        B4 = [(j0+1, k0, 1), (j0, k0, -1)]
        C4 = [(k0, i0, 1), (k0, i0+1, 1)]
        add_2x2x1_term(A4, B4, C4, col)
        col += 1
        
        # M5 = (A00+A01)B1
        A5 = [(i0, j0, 1), (i0, j0+1, 1)]
        B5 = [(j0+1, k0, 1)]
        C5 = [(k0, i0, 1), (k0, i0+1, 1)]
        add_2x2x1_term(A5, B5, C5, col)
        col += 1
        
        return col
    
    def build_strassen_4x4x2(i0, j0, k0, col_start, U, V, W, m, p, n):
        """
        Build 28 rank terms for 4x4x2 multiplication.
        """
        col = col_start
        
        # We'll use 7 calls to a 2x2x1-like decomposition on 2x2 blocks.
        # Each contributes 5 terms, but we need to be careful.
        
        # Actually, let's use a simpler approach: decompose into 7 2x2x1 multiplications
        # where each 2x2x1 uses 5 terms, total 35 terms. But we want 28.
        
        # Alternative: use 4 calls to 2x2x1 (each 5 terms = 20) plus 8 more?
        
        # Let me try a different approach: use the fact that 4x4x2 can be seen as
        # two 4x4x1 multiplications, each with 14 terms (total 28).
        
        # For 4x4x1: multiply 4x4 by 4x1 to get 4x1
        # This can be done with 14 terms using recursive Strassen.
        
        def build_4x4x1(i0, j0, k0, col_start, U, V, W, m, p, n):
            """
            Build 14 rank terms for 4x4x1 multiplication.
            """
            col = col_start
            
            # M1 = (A00+A11)B0 where B0 is 2x1
            # This is a 2x2x1 multiplication
            A1 = [(i0, j0, 1), (i0+2, j0+2, 1)]
            B1 = [(j0, k0, 1), (j0+2, k0, 1)]
            C1 = [(k0, i0, 1), (k0, i0+2, 1)]
            add_2x2x1_term(A1, B1, C1, col)
            col += 1
            
            # M2 = (A10+A11)B0
            A2 = [(i0+2, j0, 1), (i0+2, j0+2, 1)]
            B2 = [(j0, k0, 1), (j0+2, k0, 1)]
            C2 = [(k0, i0, 1), (k0, i0+2, 1)]
            add_2x2x1_term(A2, B2, C2, col)
            col += 1
            
            # M3 = A00(B1-B0)
            A3 = [(i0, j0, 1)]
            B3 = [(j0+2, k0, 1), (j0, k0, -1)]
            C3 = [(k0, i0, 1), (k0, i0+2, 1)]
            add_2x2x1_term(A3, B3, C3, col)
            col += 1
            
            # M4 = A11(B1-B0)
            A4 = [(i0+2, j0+2, 1)]
            B4 = [(j0+2, k0, 1), (j0, k0, -1)]
            C4 = [(k0, i0, 1), (k0, i0+2, 1)]
            add_2x2x1_term(A4, B4, C4, col)
            col += 1
            
            # M5 = (A00+A01)B1
            A5 = [(i0, j0, 1), (i0, j0+2, 1)]
            B5 = [(j0+2, k0, 1), (j0+2, k0, 1)]
            C5 = [(k0, i0, 1), (k0, i0+2, 1)]
            add_2x2x1_term(A5, B5, C5, col)
            col += 1
            
            # M6 = (A10-A00)(B0+B1)
            A6 = [(i0+2, j0, 1), (i0, j0, -1)]
            B6 = [(j0, k0, 1), (j0+2, k0, 1)]
            C6 = [(k0, i0, 1), (k0, i0+2, 1)]
            add_2x2x1_term(A6, B6, C6, col)
            col += 1
            
            # M7 = (A01-A11)(B0+B1)
            A7 = [(i0, j0+2, 1), (i0+2, j0+2, -1)]
            B7 = [(j0, k0, 1), (j0+2, k0, 1)]
            C7 = [(k0, i0, 1), (k0, i0+2, 1)]
            add_2x2x1_term(A7, B7, C7, col)
            col += 1
            
            return col
        
        # Build 4x4x1 for k=0 (14 terms)
        col = build_4x4x1(0, 0, 0, col, U, V, W, m, p, n)
        
        # Build 4x4x1 for k=1 (14 terms)
        col = build_4x4x1(0, 0, 1, col, U, V, W, m, p, n)
        
        return col
    
    col = build_strassen_4x4x2(0, 0, 0, col, U, V, W, m, p, n)
    
    # Check deadline
    if time.time() > deadline:
        # Return what we have so far
        pass
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 6)
    C_expected = A_test @ B_test
    
    # Compute using the decomposition
    C_computed = np.zeros((4, 6))
    for r in range(77):
        u = U_np[:, r]
        v = V_np[:, r]
        w = W_np[:, r]
        
        # Reshape to matrices
        u_mat = u.reshape(4, 4)
        v_mat = v.reshape(4, 6)
        w_mat = w.reshape(6, 4)
        
        # The rank term is: sum over i,j of u[i,j] * A[i,j] * sum over j,k of v[j,k] * B[j,k] * w[k,i]
        # Actually, the decomposition is: C[i,k] = sum_r (sum_ij U[i,j,r] A[i,j]) (sum_jk V[j,k,r] B[j,k]) W[k,i,r]
        
        # Let's compute it properly
        # For each r, we have:
        # alpha_r = sum_{i,j} U[i*m+j, r] * A[i,j]
        # beta_r = sum_{j,k} V[j*p+k, r] * B[j,k]
        # C[i,k] += alpha_r * beta_r * W[k*n+i, r]
        
        alpha_r = sum(U_np[i * 4 + j, r] * A_test[i, j] for i in range(4) for j in range(4))
        beta_r = sum(V_np[j * 6 + k, r] * B_test[j, k] for j in range(4) for k in range(6))
        
        for i in range(4):
            for k in range(6):
                C_computed[i, k] += alpha_r * beta_r * W_np[k * 4 + i, r]
    
    error = np.max(np.abs(C_computed - C_expected))
    
    if error > 1e-10:
        # Fallback to naive rank-96 if the Strassen decomposition is incorrect
        U = [[0] * 96 for _ in range(n * m)]
        V = [[0] * 96 for _ in range(m * p)]
        W = [[0] * 96 for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
