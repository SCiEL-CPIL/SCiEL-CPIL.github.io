import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition of rank 52.
    # Split m=5 into m1=2 and m2=3.
    # A is 2x5, B is 5x6.
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x3.
    # B = [B1; B2] where B1 is 2x6, B2 is 3x6.
    # C = A1*B1 + A2*B2.
    
    # Subproblem 1: A1 (2x2) * B1 (2x6)
    # Split B1 into B1a (2x3) and B1b (2x3).
    # C1 = A1*B1a + A1*B1b.
    # Subproblem 1a: A1 (2x2) * B1a (2x3)
    # Split B1a into B1a1 (2x2) and B1a2 (2x1).
    # C1a = A1*B1a1 + A1*B1a2.
    # Subproblem 1a1: A1 (2x2) * B1a1 (2x2) -> Rank 7 (Strassen)
    # Subproblem 1a2: A1 (2x2) * B1a2 (2x1) -> Rank 1
    # Subproblem 1b: A1 (2x2) * B1b (2x3) -> Same as 1a, Rank 8.
    # Total Rank for A1*B1: 8 + 8 = 16.
    
    # Subproblem 2: A2 (2x3) * B2 (3x6)
    # Split B2 into B2a (3x3) and B2b (3x3).
    # C2 = A2*B2a + A2*B2b.
    # Subproblem 2a: A2 (2x3) * B2a (3x3) -> Rank 18 (Schoolbook)
    # Subproblem 2b: A2 (2x3) * B2b (3x3) -> Rank 18 (Schoolbook)
    # Total Rank for A2*B2: 18 + 18 = 36.
    
    # Total Rank: 16 + 36 = 52.
    
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen(u_offset, v_offset, w_offset, col_start):
        # U indices: 0,1 for row 0; 2,3 for row 1.
        # V indices: 0,1 for row 0; 2,3 for row 1.
        # W indices: 0 for col 0; 1 for col 1.
        
        # M1 = (A00+A11)(B00+B11)
        U[u_offset + 0][col_start] = 1
        U[u_offset + 3][col_start] = 1
        V[v_offset + 0][col_start] = 1
        V[v_offset + 1][col_start] = 1
        W[w_offset + 0][col_start] = 1
        W[w_offset + 1][col_start] = 1
        col_start += 1
        
        # M2 = (A10+A11)B00
        U[u_offset + 2][col_start] = 1
        U[u_offset + 3][col_start] = 1
        V[v_offset + 0][col_start] = 1
        W[w_offset + 1][col_start] = -1
        col_start += 1
        
        # M3 = A00(B01-B11)
        U[u_offset + 0][col_start] = 1
        V[v_offset + 1][col_start] = 1
        V[v_offset + 3][col_start] = -1
        W[w_offset + 1][col_start] = 1
        col_start += 1
        
        # M4 = A11(B10-B00)
        U[u_offset + 3][col_start] = 1
        V[v_offset + 2][col_start] = 1
        V[v_offset + 0][col_start] = -1
        W[w_offset + 1][col_start] = 1
        col_start += 1
        
        # M5 = (A00+A01)B11
        U[u_offset + 0][col_start] = 1
        U[u_offset + 1][col_start] = 1
        V[v_offset + 3][col_start] = 1
        W[w_offset + 0][col_start] = -1
        W[w_offset + 1][col_start] = -1
        col_start += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[u_offset + 2][col_start] = 1
        U[u_offset + 0][col_start] = -1
        V[v_offset + 0][col_start] = 1
        V[v_offset + 1][col_start] = 1
        W[w_offset + 1][col_start] = 1
        col_start += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[u_offset + 1][col_start] = 1
        U[u_offset + 3][col_start] = -1
        V[v_offset + 2][col_start] = 1
        V[v_offset + 3][col_start] = 1
        W[w_offset + 0][col_start] = 1
        W[w_offset + 1][col_start] = 1
        col_start += 1
        
        return col_start

    # Helper to add 2x2x1 (Rank 1)
    def add_2x2x1(u_offset, v_offset, w_offset, col_start):
        # C0 = A00*B00 + A01*B10
        # C1 = A10*B00 + A11*B10
        # This is not rank 1.
        # Wait, for 2x2x1, T_ij0 = 1.
        # Sum_c U_ic V_jc W_0c = 1.
        # U_i0=1, V_j0=1, W_00=1.
        U[u_offset + 0][col_start] = 1
        U[u_offset + 1][col_start] = 1
        U[u_offset + 2][col_start] = 1
        U[u_offset + 3][col_start] = 1
        V[v_offset + 0][col_start] = 1
        V[v_offset + 1][col_start] = 1
        W[w_offset + 0][col_start] = 1
        col_start += 1
        return col_start

    # Helper to add schoolbook 2x3x3 (Rank 18)
    def add_schoolbook_2x3x3(u_offset, v_offset, w_offset, col_start):
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U[u_offset + i*3 + j][col_start] = 1
                    V[v_offset + j*3 + k][col_start] = 1
                    W[w_offset + k*2 + i][col_start] = 1
                    col_start += 1
        return col_start

    # Subproblem 1a1: A1 (2x2) * B1a1 (2x2)
    # A1 is A[0:2, 0:2]. B1a1 is B[0:2, 0:2].
    # U offset 0, V offset 0, W offset 0.
    col = add_strassen(0, 0, 0, col)
    
    # Subproblem 1a2: A1 (2x2) * B1a2 (2x1)
    # B1a2 is B[0:2, 2:3].
    # U offset 0, V offset 6 (2*3), W offset 4 (2*2).
    # Wait, W index for k=2 is 2*2+0=4, 2*2+1=5.
    col = add_2x2x1(0, 6, 4, col)
    
    # Subproblem 1b: A1 (2x2) * B1b (2x3)
    # B1b is B[0:2, 3:6].
    # Split into B1b1 (2x2) and B1b2 (2x1).
    # B1b1 is B[0:2, 3:5].
    # U offset 0, V offset 12 (2*6), W offset 6 (3*2).
    col = add_strassen(0, 12, 6, col)
    
    # B1b2 is B[0:2, 5:6].
    # U offset 0, V offset 18 (3*6), W offset 10 (5*2).
    col = add_2x2x1(0, 18, 10, col)
    
    # Subproblem 2a: A2 (2x3) * B2a (3x3)
    # A2 is A[0:2, 2:5]. B2a is B[2:5, 0:3].
    # U offset 6 (2*3), V offset 18 (3*6), W offset 0.
    col = add_schoolbook_2x3x3(6, 18, 0, col)
    
    # Subproblem 2b: A2 (2x3) * B2b (3x3)
    # B2b is B[2:5, 3:6].
    # U offset 6, V offset 24 (4*6), W offset 6.
    col = add_schoolbook_2x3x3(6, 24, 6, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
