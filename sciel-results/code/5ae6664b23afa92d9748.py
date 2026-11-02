import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 4, 4  # A: n×m, B: m×p, C: p×n (but C is n×p in standard, trace convention)
    
    # Build the tensor T[i,j,k] where:
    # i indexes A: i = row_a * 4 + col_a (0..11)
    # j indexes B: j = row_b * 4 + col_b (0..15)  
    # k indexes C: k = col_c * 3 + row_c (0..11) [trace convention: W[k*n+i,r] means C[row_c, col_c] at index col_c*3+row_c]
    # T[i,j,k] = 1 if A[row_a,col_a] * B[row_b,col_b] contributes to C[row_c,col_c]
    # i.e., row_a = row_c, col_a = row_b, col_b = col_c
    
    T = np.zeros((12, 16, 12))
    for row_a in range(3):
        for col_a in range(4):
            for col_c in range(4):
                i = row_a * 4 + col_a
                j = col_a * 4 + col_c
                k = col_c * 3 + row_a
                T[i, j, k] = 1.0
    
    def verify(U, V, W, max_checks=200):
        rank = U.shape[1]
        # Check all nonzero entries of T
        nonzero = np.argwhere(T != 0)
        for idx in nonzero:
            i, j, k = idx
            computed = np.dot(U[i, :] * V[j, :], W[k, :])
            if abs(computed - 1.0) > 1e-9:
                return False
        # Check some zero entries
        np.random.seed(42)
        for _ in range(max_checks):
            i = np.random.randint(0, 12)
            j = np.random.randint(0, 16)
            k = np.random.randint(0, 12)
            if T[i, j, k] == 0:
                computed = np.dot(U[i, :] * V[j, :], W[k, :])
                if abs(computed) > 1e-9:
                    return False
        return True
    
    # Try rank 42: use Strassen on all 2x2 blocks more efficiently
    # Partition the 4 columns of A and B into two groups of 2
    # For each pair of rows of A (0,1) and (1,2), apply Strassen
    
    # Actually, let me try a different approach: 
    # Use the fact that we can share the Strassen computation across overlapping row pairs
    
    # Strategy: 
    # - For rows 0,1 of A: use Strassen on 2x2x2 blocks (rank 7 per 2x2 block)
    # - For rows 1,2 of A: use Strassen on 2x2x2 blocks (rank 7 per 2x2 block)  
    # - But row 1 is shared, so we might save some
    
    # Let me try to construct rank 42 by being clever
    
    # Alternative: try rank 43, 42, 41, 40...
    # Start with a known good solution and try to improve
    
    # Current best known: 44
    # Let me try to find if 42 or 43 is possible
    
    # One idea: instead of doing 4 separate 2x2x2 Strassen blocks (rank 7 each = 28) 
    # plus trivial for the bottom part, try to overlap
    
    # Let me try a systematic search for a lower rank solution
    # by trying to merge some of the trivial multiplications with Strassen
    
    # For now, let's stick with the known 44 solution but try to optimize it
    
    # Actually, let me try rank 42:
    # Use Strassen for the top-left 2x2, top-right 2x2, bottom-left 2x2, bottom-right 2x2
    # but share the computation for row 1 of A
    
    # Hmm, this is getting complex. Let me just return the known 44 solution 
    # but with a cleaner implementation, and add a search for improvement
    
    # Known rank-44 solution:
    U = np.zeros((12, 44))
    V = np.zeros((16, 44))
    W = np.zeros((12, 44))
    
    # Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5-M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M7
    
    # For our 3x4x4 problem, partition into 2x2 blocks:
    # A rows 0-1, cols 0-1 | A rows 0-1, cols 2-3
    # A rows 1-2, cols 0-1 | A rows 1-2, cols 2-3
    # B rows 0-1, cols 0-1 | B rows 0-1, cols 2-3
    # B rows 2-3, cols 0-1 | B rows 2-3, cols 2-3
    # C rows 0-1, cols 0-1 | C rows 0-1, cols 2-3
    # C rows 1-2, cols 0-1 | C rows 1-2, cols 2-3
    
    # This gives 4 Strassen blocks for the top 2 rows, and we need to handle row 2 separately
    
    # Let me just use the existing 44 solution and return it
    # The key insight from the current program:
    # - 4 Strassen 2x2x2 blocks (rank 7 each = 28) for the 2x2 sub-blocks
    # - Plus 4 trivial multiplications for the remaining part
    # - Total: 28 + 4*4 = 28 + 16 = 44
    
    # Let me reconstruct this cleanly
    col = 0
    for block_idx in range(4):
        # Each block handles a 2x2x2 multiplication
        # Determine which rows/cols of A, B, C this block covers
        a_row_start = [0, 0, 0, 0][block_idx]
        a_col_start = [0, 2, 0, 2][block_idx]
        b_row_start = [0, 2, 0, 2][block_idx]
        b_col_start = [0, 0, 2, 2][block_idx]
        c_row_start = [0, 0, 0, 0][block_idx]
        c_col_start = [0, 0, 2, 2][block_idx]
        
        # Strassen 2x2x2 (7 multiplications)
        # Local indices:
        # A: (0,0),(0,1),(1,0),(1,1) -> global A indices
        # B: (0,0),(0,1),(1,0),(1,1) -> global B indices  
        # C: (0,0),(0,1),(1,0),(1,1) -> global C indices
        
        # M1 = (A00+A11)(B00+B11)
        U[a_row_start*4 + a_col_start, col] = 1
        U[(a_row_start+1)*4 + (a_col_start+1), col] = 1
        V[b_row_start*4 + b_col_start, col] = 1
        V[(b_row_start+1)*4 + (b_col_start+1), col] = 1
        W[(c_col_start)*3 + c_row_start, col] = 1
        W[(c_col_start+1)*3 + (c_row_start+1), col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(a_row_start+1)*4 + a_col_start, col] = 1
        U[(a_row_start+1)*4 + (a_col_start+1), col] = 1
        V[b_row_start*4 + b_col_start, col] = 1
        W[(c_col_start)*3 + (c_row_start+1), col] = 1
        W[(c_col_start+1)*3 + (c_row_start+1), col] = -1
        col += 1
        
        # M3 = A00(B10-B11)
        U[a_row_start*4 + a_col_start, col] = 1
        V[(b_row_start+1)*4 + b_col_start, col] = 1
        V[(b_row_start+1)*4 + (b_col_start+1), col] = -1
        W[(c_col_start+1)*3 + c_row_start, col] = 1
        W[(c_col_start+1)*3 + (c_row_start+1), col] = 1
        col += 1
        
        # M4 = A11(B01-B10)
        U[(a_row_start+1)*4 + (a_col_start+1), col] = 1
        V[b_row_start*4 + (b_col_start+1), col] = 1
        V[b_row_start*4 + b_col_start, col] = -1
        W[(c_col_start)*3 + c_row_start, col] = 1
        W[(c_col_start+1)*3 + c_row_start, col] = 1
        col += 1
        
        # M5 = (A00+A10)B11
        U[a_row_start*4 + a_col_start, col] = 1
        U[(a_row_start+1)*4 + a_col_start, col] = 1
        V[(b_row_start+1)*4 + (b_col_start+1), col] = 1
        W[(c_col_start)*3 + c_row_start, col] = -1
        W[(c_col_start+1)*3 + c_row_start, col] = 1
        col += 1
        
        # M6 = (A11-A10)(B00+B10)
        U[(a_row_start+1)*4 + (a_col_start+1), col] = 1
        U[(a_row_start+1)*4 + a_col_start, col] = -1
        V[b_row_start*4 + b_col_start, col] = 1
        V[(b_row_start+1)*4 + b_col_start, col] = 1
        W[(c_col_start+1)*3 + (c_row_start+1), col] = 1
        col += 1
        
        # M7 = (A01-A11)(B01+B11)
        U[a_row_start*4 + (a_col_start+1), col] = 1
        U[(a_row_start+1)*4 + (a_col_start+1), col] = -1
        V[b_row_start*4 + (b_col_start+1), col] = 1
        V[(b_row_start+1)*4 + (b_col_start+1), col] = 1
        W[(c_col_start)*3 + c_row_start, col] = 1
        col += 1
    
    # Now we have 28 columns used for the top 2 rows (rows 0,1) of A
    # We need to handle row 2 of A (which contributes to row 2 of C)
    # Row 2 of A: A[2,0], A[2,1], A[2,2], A[2,3]
    # These multiply with B to give C[2,0], C[2,1], C[2,2], C[2,3]
    # C[2,j] = sum_k A[2,k] * B[k,j]
    
    # This is a 1x4x4 multiplication, which has rank 4 (trivial)
    # But we can potentially share with the Strassen computation above
    
    # For now, use trivial: 4 multiplications
    for k in range(4):
        for j in range(4):
            i = 2*4 + k  # A[2,k]
            j_idx = k*4 + j  # B[k,j]
            k_idx = j*3 + 2  # C[2,j] in trace convention
            U[i, col] = 1
            V[j_idx, col] = 1
            W[k_idx, col] = 1
            col += 1
    
    # Total rank: 28 + 16 = 44
    assert col == 44, f"Expected 44, got {col}"
    
    # Verify
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: trivial rank-48
    U = np.zeros((12, 48))
    V = np.zeros((16, 48))
    W = np.zeros((12, 48))
    for row_a in range(3):
        for col_a in range(4):
            for col_c in range(4):
                idx = (row_a * 4 + col_a) * 4 + col_c
                i = row_a * 4 + col_a
                j = col_a * 4 + col_c
                k = col_c * 3 + row_a
                U[i, idx] = 1
                V[j, idx] = 1
                W[k, idx] = 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
