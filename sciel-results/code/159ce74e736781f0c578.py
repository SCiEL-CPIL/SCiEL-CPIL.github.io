import numpy as np
import time

def construct():
    n, m, p = 2, 4, 6
    dim_A = n * m  # 8
    dim_B = m * p  # 24
    dim_C = p * n  # 12
    
    # Rank 42 solution based on 2 * (3 * Strassen(2x2x2))
    rank = 42
    
    # Initialize matrices
    U = np.zeros((dim_A, rank))
    V = np.zeros((dim_B, rank))
    W = np.zeros((dim_C, rank))
    
    # Strassen basis for 2x2x2
    # U (A part): rows are indices of A (0,1,2,3 for 2x2 block)
    # V (B part): rows are indices of B (0,1,2,3 for 2x2 block)
    # W (C part): rows are indices of C (0,1,2,3 for 2x2 block)
    
    # Standard Strassen 7 terms for M = A*B (2x2)
    # m1 = (A00 + A11)(B00 + B11)
    # m2 = (A10 + A11)B00
    # m3 = A00(B01 - B11)
    # m4 = A11(B10 - B00)
    # m5 = (A00 + A01)B11
    # m6 = (A10 - A00)(B00 + B01)
    # m7 = (A01 - A11)(B10 + B11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Indices for a generic 2x2 block:
    # A indices: 0, 1, 2, 3
    # B indices: 0, 1, 2, 3
    # C indices: 0, 1, 2, 3
    
    # U entries (A):
    # m1: A00, A11 -> U[0]+=1, U[3]+=1
    # m2: A10, A11 -> U[2]+=1, U[3]+=1
    # m3: A00 -> U[0]+=1
    # m4: A11 -> U[3]+=1
    # m5: A00, A01 -> U[0]+=1, U[1]+=1
    # m6: A10, A00 -> U[2]+=1, U[0]-=1
    # m7: A01, A11 -> U[1]+=1, U[3]-=1
    
    # V entries (B):
    # m1: B00, B11 -> V[0]+=1, V[3]+=1
    # m2: B00 -> V[0]+=1
    # m3: B01, B11 -> V[1]+=1, V[3]-=1
    # m4: B10, B00 -> V[2]+=1, V[0]-=1
    # m5: B11 -> V[3]+=1
    # m6: B00, B01 -> V[0]+=1, V[1]+=1
    # m7: B10, B11 -> V[2]+=1, V[3]+=1
    
    # W entries (C):
    # m1: C00, C11 -> W[0]+=1, W[3]+=1
    # m2: C10, C11 -> W[2]+=1, W[3]-=1
    # m3: C01, C11 -> W[1]+=1, W[3]+=1
    # m4: C00, C10 -> W[0]+=1, W[2]+=1
    # m5: C00, C01 -> W[0]-=1, W[1]+=1
    # m6: C11 -> W[3]+=1
    # m7: C00 -> W[0]+=1
    
    # Wait, let's re-verify the W mapping.
    # C00 = m1 + m4 - m5 + m7
    # m1: W[0] += 1
    # m4: W[0] += 1
    # m5: W[0] -= 1
    # m7: W[0] += 1
    # C01 = m3 + m5
    # m3: W[1] += 1
    # m5: W[1] += 1
    # C10 = m2 + m4
    # m2: W[2] += 1
    # m4: W[2] += 1
    # C11 = m1 - m2 + m3 + m6
    # m1: W[3] += 1
    # m2: W[3] -= 1
    # m3: W[3] += 1
    # m6: W[3] += 1
    
    def add_strassen_block(a_offsets, b_offsets, c_offsets, start_col):
        # a_offsets: list of 4 indices in U corresponding to A00, A01, A10, A11
        # b_offsets: list of 4 indices in V corresponding to B00, B01, B10, B11
        # c_offsets: list of 4 indices in W corresponding to C00, C01, C10, C11
        
        A00, A01, A10, A11 = a_offsets
        B00, B01, B10, B11 = b_offsets
        C00, C01, C10, C11 = c_offsets
        
        # Term 0 (m1)
        U[A00, start_col+0] += 1
        U[A11, start_col+0] += 1
        V[B00, start_col+0] += 1
        V[B11, start_col+0] += 1
        W[C00, start_col+0] += 1
        W[C11, start_col+0] += 1
        
        # Term 1 (m2)
        U[A10, start_col+1] += 1
        U[A11, start_col+1] += 1
        V[B00, start_col+1] += 1
        W[C10, start_col+1] += 1
        W[C11, start_col+1] -= 1
        
        # Term 2 (m3)
        U[A00, start_col+2] += 1
        V[B01, start_col+2] += 1
        V[B11, start_col+2] -= 1
        W[C01, start_col+2] += 1
        W[C11, start_col+2] += 1
        
        # Term 3 (m4)
        U[A11, start_col+3] += 1
        V[B10, start_col+3] += 1
        V[B00, start_col+3] -= 1
        W[C00, start_col+3] += 1
        W[C10, start_col+3] += 1
        
        # Term 4 (m5)
        U[A00, start_col+4] += 1
        U[A01, start_col+4] += 1
        V[B11, start_col+4] += 1
        W[C00, start_col+4] -= 1
        W[C01, start_col+4] += 1
        
        # Term 5 (m6)
        U[A10, start_col+5] += 1
        U[A00, start_col+5] -= 1
        V[B00, start_col+5] += 1
        V[B01, start_col+5] += 1
        W[C11, start_col+5] += 1
        
        # Term 6 (m7)
        U[A01, start_col+6] += 1
        U[A11, start_col+6] -= 1
        V[B10, start_col+6] += 1
        V[B11, start_col+6] += 1
        W[C00, start_col+6] += 1

    # Define indices for the 2x2 blocks in A, B, C
    # A is 2x4. 
    # Left block (cols 0,1): indices 0,1,4,5 (A00, A01, A10, A11)
    # Right block (cols 2,3): indices 2,3,6,7
    A_L = [0, 1, 4, 5]
    A_R = [2, 3, 6, 7]
    
    # B is 4x6.
    # Top-left 2x2 (rows 0,1, cols 0,1): indices 0,1,6,7 (B00, B01, B10, B11)
    # Top-mid 2x2 (rows 0,1, cols 2,3): indices 2,3,8,9
    # Top-right 2x2 (rows 0,1, cols 4,5): indices 4,5,10,11
    # Bot-left 2x2 (rows 2,3, cols 0,1): indices 12,13,18,19
    # Bot-mid 2x2 (rows 2,3, cols 2,3): indices 14,15,20,21
    # Bot-right 2x2 (rows 2,3, cols 4,5): indices 16,17,22,23
    
    B_TL = [0, 1, 6, 7]
    B_TM = [2, 3, 8, 9]
    B_TR = [4, 5, 10, 11]
    B_BL = [12, 13, 18, 19]
    B_BM = [14, 15, 20, 21]
    B_BR = [16, 17, 22, 23]
    
    # C is 2x6.
    # Block 0 (cols 0,1): indices 0,1,2,3? 
    # W[k*n+i]. k is col, i is row.
    # Col 0: 0, 1. Col 1: 2, 3.
    # So C00=0, C01=1? No.
    # C[0,0] is k=0, i=0 -> 0.
    # C[0,1] is k=1, i=0 -> 2.
    # C[1,0] is k=0, i=1 -> 1.
    # C[1,1] is k=1, i=1 -> 3.
    # So for cols 0,1: C00=0, C10=1, C01=2, C11=3.
    # My add_strassen_block expects C00, C01, C10, C11.
    # So for Block 0: C00=0, C01=2, C10=1, C11=3.
    C_B0 = [0, 2, 1, 3]
    
    # Block 1 (cols 2,3):
    # C[0,2] is k=2, i=0 -> 4.
    # C[0,3] is k=3, i=0 -> 6.
    # C[1,2] is k=2, i=1 -> 5.
    # C[1,3] is k=3, i=1 -> 7.
    # C00=4, C01=6, C10=5, C11=7.
    C_B1 = [4, 6, 5, 7]
    
    # Block 2 (cols 4,5):
    # C[0,4] is k=4, i=0 -> 8.
    # C[0,5] is k=5, i=0 -> 10.
    # C[1,4] is k=4, i=1 -> 9.
    # C[1,5] is k=5, i=1 -> 11.
    # C00=8, C01=10, C10=9, C11=11.
    C_B2 = [8, 10, 9, 11]
    
    # Perform the decomposition
    # C_B0 = A_L B_TL + A_R B_BL
    add_strassen_block(A_L, B_TL, C_B0, 0)
    add_strassen_block(A_R, B_BL, C_B0, 7)
    
    # C_B1 = A_L B_TM + A_R B_BM
    add_strassen_block(A_L, B_TM, C_B1, 14)
    add_strassen_block(A_R, B_BM, C_B1, 21)
    
    # C_B2 = A_L B_TR + A_R B_BR
    add_strassen_block(A_L, B_TR, C_B2, 28)
    add_strassen_block(A_R, B_BR, C_B2, 35)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
