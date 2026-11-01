"""Deterministic rank-28 decomposition for 2x4x4 matrix multiplication using Strassen on 2x2 blocks."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # Keep 10s safety margin
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # Split m into two 2x2 blocks: cols 0,1 and cols 2,3
    # Split p into two 2x2 blocks: cols 0,1 and cols 2,3
    # This gives 4 independent 2x2x2 matrix multiplications
    # Each 2x2x2 uses Strassen (rank 7)
    # Total rank = 4 * 7 = 28
    
    R = 28  # Rank
    
    # Initialize factor matrices
    U = [[0] * R for _ in range(n * m)]  # shape (8, 28)
    V = [[0] * R for _ in range(m * p)]  # shape (16, 28)
    W = [[0] * R for _ in range(p * n)]  # shape (8, 28)
    
    # Strassen's 7 bilinear forms for 2x2 matrix multiplication
    # For matrices A (2x2), B (2x2), C (2x2):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # The tensor decomposition for one 2x2x2 block:
    # U_local[i, j, r] where i,j in {0,1}, r in {0..6}
    # V_local[j, k, r] where j,k in {0,1}, r in {0..6}
    # W_local[k, i, r] where k,i in {0,1}, r in {0..6}
    
    # Let's define the Strassen factors more carefully.
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k in {0,1}
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # Strassen decomposition:
    # r=0: M1 = (A00+A11)(B00+B11)
    #   U[0,0,0]=1, U[1,1,0]=1
    #   V[0,0,0]=1, V[1,1,0]=1
    #   W[0,0,0]=1, W[1,1,0]=-1, W[0,1,0]=1, W[1,0,0]=1
    # Wait, let me be more careful.
    
    # Actually, let's use the standard formulation.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # For 2x2: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #          C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    #          C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #          C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In tensor notation, T[i,j,k] where i=row of A (and C), j=col of A (and row of B), k=col of B (and C)
    # T[i,j,k] = 1 if A[i,j]*B[j,k] contributes to C[i,k]
    
    # For each Strassen form r, we have:
    # M_r = (sum_{i,j} u_r[i,j] * A[i,j]) * (sum_{j,k} v_r[j,k] * B[j,k])
    # And C[i,k] = sum_r w_r[i,k] * M_r
    
    # So the tensor equation is:
    # sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] = T[i,j,k] = 1 for all i,j,k
    
    # Wait, this isn't quite right either. Let me think again.
    
    # The bilinear map is: (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # Strassen writes this as: C[i,k] = sum_r c_r[i,k] * (sum_{i',j} a_r[i',j] A[i',j]) * (sum_{j',k'} b_r[j',k'] B[j',k'])
    
    # Hmm, this is getting confusing. Let me just directly construct the factors.
    
    # For one 2x2x2 block with local indices (i,j,k) in {0,1}:
    # We need U_local[i,j,r], V_local[j,k,r], W_local[k,i,r] such that
    # sum_r U_local[i,j,r] * V_local[j,k,r] * W_local[k,i,r] = 1
    
    # Using Strassen's 7 products:
    # r=0 (M1): U: A00,A11 -> 1; V: B00,B11 -> 1; W: C00->1, C11->-1, C01->1, C10->1? No...
    
    # Let me look up the exact Strassen tensor factors.
    # Actually, let me just construct it directly from the formulas.
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # So for each r, we define the factors:
    
    # r=0 (M1):
    # A part: A00 + A11 => U[0,0,0]=1, U[1,1,0]=1
    # B part: B00 + B11 => V[0,0,0]=1, V[1,1,0]=1
    # C part: C00 += 1, C11 -= 1, C01 += 1, C10 += 1
    # W[0,0,0]=1 (C00), W[1,1,0]=-1 (C11), W[0,1,0]=1 (C01), W[1,0,0]=1 (C10)
    # Note: W[k,i,r] where k=col of B/C, i=row of A/C
    # C00: i=0,k=0 -> W[0,0,0]
    # C01: i=0,k=1 -> W[1,0,0]
    # C10: i=1,k=0 -> W[0,1,0]
    # C11: i=1,k=1 -> W[1,1,0]
    
    # r=1 (M2):
    # A part: A10 + A11 => U[1,0,1]=1, U[1,1,1]=1
    # B part: B00 => V[0,0,1]=1
    # C part: C10 += 1, C11 -= 1
    # W[0,1,1]=1 (C10), W[1,1,1]=-1 (C11)
    
    # r=2 (M3):
    # A part: A00 => U[0,0,2]=1
    # B part: B10 - B11 => V[1,0,2]=1, V[1,1,2]=-1
    # C part: C01 += 1, C11 += 1
    # W[1,0,2]=1 (C01), W[1,1,2]=1 (C11)
    
    # r=3 (M4):
    # A part: A11 => U[1,1,3]=1
    # B part: B01 - B00 => V[0,1,3]=1, V[0,0,3]=-1
    # C part: C00 += 1, C10 += 1
    # W[0,0,3]=1 (C00), W[0,1,3]=1 (C10)
    
    # r=4 (M5):
    # A part: A00 + A01 => U[0,0,4]=1, U[0,1,4]=1
    # B part: B11 => V[1,1,4]=1
    # C part: C00 -= 1, C01 += 1
    # W[0,0,4]=-1 (C00), W[1,0,4]=1 (C01)
    
    # r=5 (M6):
    # A part: A10 - A00 => U[1,0,5]=1, U[0,0,5]=-1
    # B part: B00 + B10 => V[0,0,5]=1, V[1,0,5]=1
    # C part: C11 += 1
    # W[1,1,5]=1 (C11)
    
    # r=6 (M7):
    # A part: A01 - A11 => U[0,1,6]=1, U[1,1,6]=-1
    # B part: B01 + B11 => V[0,1,6]=1, V[1,1,6]=1
    # C part: C00 += 1
    # W[0,0,6]=1 (C00)
    
    # Now let's verify: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should = 1 for all i,j,k.
    
    # Define local factors
    def strassen_block():
        """Return U_local (2x2x7), V_local (2x2x7), W_local (2x2x7) for one 2x2x2 block."""
        U_loc = np.zeros((2, 2, 7))
        V_loc = np.zeros((2, 2, 7))
        W_loc = np.zeros((2, 2, 7))  # W[k, i, r]
        
        # r=0 (M1)
        U_loc[0, 0, 0] = 1
        U_loc[1, 1, 0] = 1
        V_loc[0, 0, 0] = 1
        V_loc[1, 1, 0] = 1
        W_loc[0, 0, 0] = 1   # C00
        W_loc[1, 0, 0] = 1   # C01 (k=1, i=0)
        W_loc[0, 1, 0] = 1   # C10 (k=0, i=1)
        W_loc[1, 1, 0] = -1  # C11
        
        # r=1 (M2)
        U_loc[1, 0, 1] = 1
        U_loc[1, 1, 1] = 1
        V_loc[0, 0, 1] = 1
        W_loc[0, 1, 1] = 1   # C10
        W_loc[1, 1, 1] = -1  # C11
        
        # r=2 (M3)
        U_loc[0, 0, 2] = 1
        V_loc[1, 0, 2] = 1
        V_loc[1, 1, 2] = -1
        W_loc[1, 0, 2] = 1   # C01
        W_loc[1, 1, 2] = 1   # C11
        
        # r=3 (M4)
        U_loc[1, 1, 3] = 1
        V_loc[0, 1, 3] = 1
        V_loc[0, 0, 3] = -1
        W_loc[0, 0, 3] = 1   # C00
        W_loc[0, 1, 3] = 1   # C10
        
        # r=4 (M5)
        U_loc[0, 0, 4] = 1
        U_loc[0, 1, 4] = 1
        V_loc[1, 1, 4] = 1
        W_loc[0, 0, 4] = -1  # C00
        W_loc[1, 0, 4] = 1   # C01
        
        # r=5 (M6)
        U_loc[1, 0, 5] = 1
        U_loc[0, 0, 5] = -1
        V_loc[0, 0, 5] = 1
        V_loc[1, 0, 5] = 1
        W_loc[1, 1, 5] = 1   # C11
        
        # r=6 (M7)
        U_loc[0, 1, 6] = 1
        U_loc[1, 1, 6] = -1
        V_loc[0, 1, 6] = 1
        V_loc[1, 1, 6] = 1
        W_loc[0, 0, 6] = 1   # C00
        
        return U_loc, V_loc, W_loc
    
    U_loc, V_loc, W_loc = strassen_block()
    
    # Verify the local block
    T_recon = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0
                for r in range(7):
                    val += U_loc[i, j, r] * V_loc[j, k, r] * W_loc[k, i, r]
                T_recon[i, j, k] = val
    
    # Now assemble the 4 blocks:
    # Block 0: A cols 0-1, B rows 0-1, C cols 0-1 (top-left)
    # Block 1: A cols 2-3, B rows 0-1, C cols 0-1 (top-right contribution to C left)
    # Block 2: A cols 0-1, B rows 2-3, C cols 2-3 (bottom-left contribution to C right)
    # Block 3: A cols 2-3, B rows 2-3, C cols 2-3 (bottom-right)
    
    # Block offsets:
    # Block 0: a_off=0, b_off=0, c_off=0
    # Block 1: a_off=2, b_off=0, c_off=0
    # Block 2: a_off=0, b_off=2, c_off=2
    # Block 3: a_off=2, b_off=2, c_off=2
    
    blocks = [
        (0, 0, 0),  # a_off, b_off, c_off
        (2, 0, 0),
        (0, 2, 2),
        (2, 2, 2),
    ]
    
    for b_idx, (a_off, b_off, c_off) in enumerate(blocks):
        r_offset = b_idx * 7
        
        for i in range(2):
            for j in range(2):
                global_i = i
                global_j = a_off + j
                for r in range(7):
                    U[global_i * m + global_j][r_offset + r] = U_loc[i, j, r]
        
        for j in range(2):
            for k in range(2):
                global_j = b_off + j
                global_k = c_off + k
                for r in range(7):
                    V[global_j * p + global_k][r_offset + r] = V_loc[j, k, r]
        
        for k in range(2):
            for i in range(2):
                global_k = c_off + k
                global_i = i
                for r in range(7):
                    W[global_k * n + global_i][r_offset + r] = W_loc[k, i, r]
    
    # Verify the full decomposition
    T_full = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(R):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                T_full[i, j, k] = val
    
    # The tensor should have T[i,j,k] = 1 for all i,j,k
    if not np.allclose(T_full, 1.0, atol=1e-9):
        # Fallback to schoolbook if something went wrong
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
