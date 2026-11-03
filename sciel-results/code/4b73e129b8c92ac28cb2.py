import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Block decomposition:
    # A (4x5) split into A_top (2x5) rows 0-1, A_bot (2x5) rows 2-3
    # B (5x5) split into B_left (5x2) cols 0-1, B_right (5x3) cols 2-4
    # C (4x5) = [[A_top*B_left, A_top*B_right],
    #            [A_bot*B_left, A_bot*B_right]]
    
    # Block sizes:
    # A_top: rows [0,1], cols [0,4] -> 2x5
    # A_bot: rows [2,3], cols [0,4] -> 2x5
    # B_left: rows [0,4], cols [0,1] -> 5x2
    # B_right: rows [0,4], cols [2,4] -> 5x3
    
    # Sub-blocks:
    # A_top * B_left: 2x2 * 2x2 -> Strassen rank 7
    # A_top * B_right: 2x2 * 2x3 -> schoolbook rank 6
    # A_bot * B_left: 2x2 * 2x2 -> Strassen rank 7
    # A_bot * B_right: 2x2 * 2x3 -> schoolbook rank 6
    
    # Total rank = 7 + 6 + 7 + 6 = 26
    
    rank = 26
    dim_u = n * m  # 20
    dim_v = m * p  # 25
    dim_w = p * n  # 20
    
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    
    col = 0
    
    # Helper: add a Strassen 2x2 block
    # A block: local rows a_r0..a_r1, local cols a_c0..a_c1 (2x2)
    # B block: local rows b_r0..b_r1, local cols b_c0..b_c1 (2x2)
    # C block: local rows c_r0..c_r1, local cols c_c0..c_c1 (2x2)
    def add_strassen_2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # A entries: (a_r0, a_c0), (a_r0, a_c0+1), (a_r0+1, a_c0), (a_r0+1, a_c0+1)
        # B entries: (b_r0, b_c0), (b_r0, b_c0+1), (b_r0+1, b_c0), (b_r0+1, b_c0+1)
        # C entries: (c_r0, c_c0), (c_r0, c_c0+1), (c_r0+1, c_c0), (c_r0+1, c_c0+1)
        
        # Strassen 7 products:
        # P1 = (a00+a11)(b00+b11)
        # P2 = (a10+a11)b00
        # P3 = a00(b10-b11)
        # P4 = a11(b01-b00)
        # P5 = (a00+a01)b11
        # P6 = (a10-a00)(b00+b10)
        # P7 = (a01-a11)(b01+b11)
        
        # P1
        U[a_r0*m + a_c0][col] += 1
        U[(a_r0+1)*m + (a_c0+1)][col] += 1
        V[b_r0*p + b_c0][col] += 1
        V[(b_r0+1)*p + (b_c0+1)][col] += 1
        W[c_c0*n + c_r0][col] += 1       # c00
        W[(c_c0+1)*n + (c_r0+1)][col] += 1  # c11
        col += 1
        
        # P2
        U[(a_r0+1)*m + a_c0][col] += 1
        U[(a_r0+1)*m + (a_c0+1)][col] += 1
        V[b_r0*p + b_c0][col] += 1
        W[(c_c0+1)*n + (c_r0+1)][col] += 1  # c10
        col += 1
        
        # P3
        U[a_r0*m + a_c0][col] += 1
        V[(b_r0+1)*p + b_c0][col] += 1
        V[(b_r0+1)*p + (b_c0+1)][col] -= 1
        W[(c_c0+1)*n + c_r0][col] += 1    # c01
        W[(c_c0+1)*n + (c_r0+1)][col] += 1  # c11
        col += 1
        
        # P4
        U[(a_r0+1)*m + (a_c0+1)][col] += 1
        V[b_r0*p + (b_c0+1)][col] += 1
        V[b_r0*p + b_c0][col] -= 1
        W[c_c0*n + c_r0][col] -= 1       # c00
        W[(c_c0+1)*n + c_r0][col] += 1    # c01
        col += 1
        
        # P5
        U[a_r0*m + a_c0][col] += 1
        U[a_r0*m + (a_c0+1)][col] += 1
        V[(b_r0+1)*p + (b_c0+1)][col] += 1
        W[c_c0*n + c_r0][col] -= 1       # c00
        W[(c_c0+1)*n + (c_r0+1)][col] += 1  # c11
        col += 1
        
        # P6
        U[(a_r0+1)*m + a_c0][col] += 1
        U[a_r0*m + a_c0][col] -= 1
        V[b_r0*p + b_c0][col] += 1
        V[(b_r0+1)*p + b_c0][col] += 1
        W[(c_c0+1)*n + (c_r0+1)][col] += 1  # c10
        col += 1
        
        # P7
        U[a_r0*m + (a_c0+1)][col] += 1
        U[(a_r0+1)*m + (a_c0+1)][col] -= 1
        V[b_r0*p + (b_c0+1)][col] += 1
        V[(b_r0+1)*p + (b_c0+1)][col] += 1
        W[c_c0*n + c_r0][col] -= 1       # c00
        W[(c_c0+1)*n + (c_r0+1)][col] += 1  # c11
        col += 1
    
    # Helper: add schoolbook 2x2 * 2x3 block
    # A block: 2x2, B block: 2x3, C block: 2x3
    def add_schoolbook_2x2x3(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # A: 2x2, B: 2x3, C: 2x3
        # C[i,j] = sum_k A[i,k] * B[k,j]
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U[(a_r0+i)*m + (a_c0+k)][col] += 1
                    V[(b_r0+k)*p + (b_c0+j)][col] += 1
                    W[(c_c0+j)*n + (c_r0+i)][col] += 1
                    col += 1
    
    # Block 1: A_top (rows 0-1) * B_left (cols 0-1) -> C top-left (2x2)
    # A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    add_strassen_2x2(a_r0=0, a_c0=0, b_r0=0, b_c0=0, c_r0=0, c_c0=0)
    
    # Block 2: A_top (rows 0-1) * B_right (cols 2-4) -> C top-right (2x3)
    # A rows 0-1, cols 0-1; B rows 0-1, cols 2-4; C rows 0-1, cols 2-4
    add_schoolbook_2x2x3(a_r0=0, a_c0=0, b_r0=0, b_c0=2, c_r0=0, c_c0=2)
    
    # Block 3: A_bot (rows 2-3) * B_left (cols 0-1) -> C bottom-left (2x2)
    # A rows 2-3, cols 0-1; B rows 0-1, cols 0-1; C rows 2-3, cols 0-1
    add_strassen_2x2(a_r0=2, a_c0=0, b_r0=0, b_c0=0, c_r0=2, c_c0=0)
    
    # Block 4: A_bot (rows 2-3) * B_right (cols 2-4) -> C bottom-right (2x3)
    # A rows 2-3, cols 0-1; B rows 0-1, cols 2-4; C rows 2-3, cols 2-4
    add_schoolbook_2x2x3(a_r0=2, a_c0=0, b_r0=0, b_c0=2, c_r0=2, c_c0=2)
    
    return {"U": U, "V": V, "W": W}
