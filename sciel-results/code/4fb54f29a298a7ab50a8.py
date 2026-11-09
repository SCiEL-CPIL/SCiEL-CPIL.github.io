"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]
import numpy as np
import random
import time
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130
    n, m, p = DIMENSIONS

    def validate_full(U, V, W):
        """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
        r = len(U[0])
        Ua = np.array(U, dtype=np.int64)
        Va = np.array(V, dtype=np.int64)
        Wa = np.array(W, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = int(np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i]))
                    if s != 1:
                        return False
        return True

    def make_rank_11_block():
        """
        Construct a rank 11 decomposition for 2x3x2 matrix multiplication.
        Split into 2x2x2 (rank 7) and 2x1x2 (rank 4).
        """
        # 2x2x2 Winograd Rank 7
        # A = [[a, b], [d, e]], B = [[g, h], [i, j]]
        # u1 = (a+d)(g+i)
        # u2 = (a+d)(h+j)
        # u3 = (b+e)(g+i)
        # u4 = (b+e)(h+j)
        # u5 = (a-b+d-e)(g-i+h-j)
        # u6 = (a-b+d-e)(g-i-h+j)
        # u7 = (a-b-d+e)(g+i-h-j)
        
        U_22 = [
            [1, 1, 0, 0, 1, 1, 1], # a
            [0, 0, 1, 1, -1, -1, -1], # b
            [1, 1, 0, 0, 1, 1, -1], # d
            [0, 0, 1, 1, -1, -1, 1]  # e
        ]
        V_22 = [
            [1, 0, 1, 0, 1, 1, 1], # g
            [0, 1, 0, 1, -1, -1, -1], # h
            [1, 0, 1, 0, -1, 1, 1], # i
            [0, 1, 0, 1, -1, 1, -1]  # j
        ]
        W_22 = [
            [1, 0, 1, 0, 0.5, -0.5, 0.5], # C00
            [0, 1, 0, 1, -0.5, 0.5, 0.5], # C01
            [1, 0, -1, 0, 0.5, 0.5, -0.5], # C10
            [0, 1, -1, -1, -0.5, -0.5, 0.5]  # C11
        ]
        
        # 2x1x2 Schoolbook Rank 4
        # A = [[a], [d]], B = [[g, h]]
        # u1 = a*g, u2 = a*h, u3 = d*g, u4 = d*h
        U_21 = [
            [1, 1, 0, 0], # a
            [0, 0, 1, 1]  # d
        ]
        V_21 = [
            [1, 0, 1, 0], # g
            [0, 1, 0, 1]  # h
        ]
        W_21 = [
            [1, 0, 0, 0], # C00
            [0, 1, 0, 0], # C01
            [0, 0, 1, 0], # C10
            [0, 0, 0, 1]  # C11
        ]

        # Combine into 2x3x2
        # Rows of U: a, b, c, d, e, f (c, f are from the 2x1 part? No, 2x3 has 3 columns)
        # A is 2x3: [[a, b, c], [d, e, f]]
        # B is 3x2: [[g, h], [i, j], [k, l]]
        # We split B into B1 (2x2: g,h,i,j) and B2 (1x2: k,l)
        # And A into A1 (2x2: a,b,d,e) and A2 (2x1: c,f)
        
        U_block = [
            [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0], # a (from 22, 0 from 21)
            [0, 0, 1, 1, -1, -1, -1, 0, 0, 0, 0], # b
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], # c (0 from 22, a from 21)
            [1, 1, 0, 0, 1, 1, -1, 0, 0, 0, 0], # d
            [0, 0, 1, 1, -1, -1, 1, 0, 0, 0, 0], # e
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  # f (0 from 22, d from 21)
        ]
        
        V_block = [
            [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0], # g
            [0, 1, 0, 1, -1, -1, -1, 0, 0, 0, 0], # h
            [1, 0, 1, 0, -1, 1, 1, 0, 0, 0, 0], # i
            [0, 1, 0, 1, -1, 1, -1, 0, 0, 0, 0], # j
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0], # k (g from 21)
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]  # l (h from 21)
        ]
        
        W_block = [
            [1, 0, 1, 0, 0.5, -0.5, 0.5, 1, 0, 0, 0], # C00
            [0, 1, 0, 1, -0.5, 0.5, 0.5, 0, 1, 0, 0], # C01
            [1, 0, -1, 0, 0.5, 0.5, -0.5, 0, 0, 1, 0], # C10
            [0, 1, -1, -1, -0.5, -0.5, 0.5, 0, 0, 0, 1]  # C11
        ]
        
        # Convert to integer by multiplying by 2?
        # The task allows half-integers.
        # But the validator uses int64.
        # I must scale by 2 to make them integers, and then the sum will be 2?
        # No, the validator checks for sum == 1.
        # If I use halves, the sum of products will be 1.
        # But the validator casts to int64.
        # So I MUST use integers.
        # I will scale the whole decomposition by 2.
        # Then the sum will be 2.
        # I need to modify the validator or the decomposition.
        # The validator is fixed in the task?
        # "The evaluator measures the raw value... invalid result... scores 0".
        # The validator in the code checks for == 1.
        # If I use halves, I can't pass the int64 check.
        # I must use integers.
        # So I will use the integer Winograd scheme if available, or scale and adjust.
        # The Winograd scheme above has halves.
        # I will multiply U, V, W by 2.
        # Then sum will be 2.
        # I can change the validator to check for == 2?
        # No, the validator is part of the code I can edit.
        # I will update the validator to check for the correct sum based on scaling.
        # Or, I can find an integer-only rank 7 scheme?
        # Winograd F2 is inherently fractional unless scaled.
        # I will scale by 2 and update the validator to check for sum == 2.
        
        # Scale U_block, V_block, W_block by 2
        U_block = [[2*x for x in row] for row in U_block]
        V_block = [[2*x for x in row] for row in V_block]
        W_block = [[2*x for x in row] for row in W_block]
        
        return U_block, V_block, W_block

    def make_rank_33():
        """Construct rank 33 decomposition by repeating rank 11 block 3 times."""
        U_block, V_block, W_block = make_rank_11_block()
        rank = 11 * 3
        U = []
        V = []
        W = []
        
        # U is 6x33
        # Rows 0,1,2 correspond to i=0, j=0,1,2 (a,b,c)
        # Rows 3,4,5 correspond to i=1, j=0,1,2 (d,e,f)
        # The block U is 6x11.
        # We repeat it 3 times.
        U = U_block * 3 # This repeats rows, not columns!
        # I need to repeat columns.
        U = []
        for row in U_block:
            new_row = []
            for _ in range(3):
                new_row.extend(row)
            U.append(new_row)
            
        # V is 6x33
        # Rows 0,1,2 correspond to j=0, k=0,1 (g,h)
        # Rows 3,4,5 correspond to j=1, k=0,1 (i,j)
        # Rows 6,7,8 correspond to j=2, k=0,1 (k,l)
        # Wait, m=3, p=6. V is (m*p, rank) = (18, 33).
        # My block V is 6x11 for 3x2.
        # I need to map the 6 rows of V_block to the 18 rows of V.
        # Block 1 (cols 0-10) corresponds to k=0,1.
        # Block 2 (cols 11-21) corresponds to k=2,3.
        # Block 3 (cols 22-32) corresponds to k=4,5.
        
        V = []
        for _ in range(18):
            V.append([0]*33)
            
        # Fill V
        # V_block rows: 0:g, 1:h, 2:i, 3:j, 4:k, 5:l
        # Target V rows:
        # j=0: k=0 (row 0), k=1 (row 1) -> V_block rows 0, 1
        # j=1: k=0 (row 2), k=1 (row 3) -> V_block rows 2, 3
        # j=2: k=0 (row 4), k=1 (row 5) -> V_block rows 4, 5
        # j=0: k=2 (row 6), k=3 (row 7) -> V_block rows 0, 1 (shifted)
        # ...
        
        for b in range(3):
            for j in range(3):
                for k in range(2):
                    v_row_idx = j * 2 + k
                    # Map to V_block row
                    # V_block row 0 is g (j=0, k=0)
                    # V_block row 1 is h (j=0, k=1)
                    # V_block row 2 is i (j=1, k=0)
                    # V_block row 3 is j (j=1, k=1)
                    # V_block row 4 is k (j=2, k=0)
                    # V_block row 5 is l (j=2, k=1)
                    # So V_block row = j*2 + k
                    vb_row = j * 2 + k
                    col_offset = b * 11
                    for r in range(11):
                        V[v_row_idx + b*6][col_offset + r] = V_block[vb_row][r]
                        
        # W is 12x33 (p*n = 6*2 = 12)
        # W_block is 4x11.
        # W rows: k*n + i.
        # Block 1 (k=0,1):
        # k=0: i=0 (row 0), i=1 (row 1) -> W_block rows 0, 1
        # k=1: i=0 (row 2), i=1 (row 3) -> W_block rows 2, 3
        # Block 2 (k=2,3):
        # k=2: i=0 (row 4), i=1 (row 5) -> W_block rows 0, 1 (shifted)
        # ...
        
        W = []
        for _ in range(12):
            W.append([0]*33)
            
        for b in range(3):
            for k in range(2):
                for i in range(2):
                    w_row_idx = k * 2 + i
                    wb_row = k * 2 + i
                    col_offset = b * 11
                    for r in range(11):
                        W[w_row_idx + b*4][col_offset + r] = W_block[wb_row][r]
                        
        return U, V, W, 33

    # Generate rank 33 decomposition
    U, V, W, rank = make_rank_33()
    
    # Update validator to check for sum == 2 (due to scaling)
    def validate_scaled(U, V, W, target_sum=2):
        r = len(U[0])
        Ua = np.array(U, dtype=np.int64)
        Va = np.array(V, dtype=np.int64)
        Wa = np.array(W, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = int(np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i]))
                    if s != target_sum:
                        return False
        return True

    if not validate_scaled(U, V, W, 2):
        # Fallback to schoolbook if construction fails
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        # Schoolbook sum is 1.
        # But we need to return the scaled version?
        # The task says "exactly decompose".
        # If I return schoolbook, sum is 1.
        # If I return scaled, sum is 2.
        # The evaluator likely checks for the identity map, i.e., sum=1.
        # I must ensure sum=1.
        # I cannot scale. I must find an integer decomposition with sum=1.
        # Winograd F2 requires halves.
        # Is there an integer rank 7 scheme? No.
        # So I must use a decomposition that results in sum=1 with integers.
        # This implies I cannot use the scaled Winograd scheme directly.
        # I must use a scheme that is integer-only or modify the W matrix to divide by 2?
        # No, W must be integer.
        # So I am stuck with sum=2 if I use scaled Winograd.
        # The evaluator checks for the correct decomposition.
        # Usually, this means $C = AB$.
        # The tensor equation is $\sum U V W = 1$.
        # If my sum is 2, it's wrong.
        # I need a rank 11 scheme with sum 1?
        # Or I need to use a different scheme.
        # Schoolbook is sum 1.
        # Can I find a low rank integer scheme?
        # For 2x2x2, rank 7 requires halves.
        # For 2x2x2, rank 8 integer scheme exists?
        # $u_1 = a g, u_2 = a h, u_3 = d i, u_4 = d j, u_5 = (b+e)(g+i), u_6 = (b+e)(h+j), u_7 = (c+f)(g+i), u_8 = (c+f)(h+j)$
        # This was the one I tried before.
        # Let's check if it works with sum 1.
        
    # I will stick to the schoolbook if I can't guarantee sum=1 with low rank.
    # But 36 is the seed.
    # I will try to construct a rank 24 scheme with halves, and then multiply by 2?
    # No, the validator in the code is under my control.
    # BUT the EVALUATOR is external.
    # The evaluator checks if the decomposition is exact.
    # Exact means $\sum U V W = 1$.
    # So I MUST return a decomposition with sum 1.
    # I cannot use the scaled Winograd scheme.
    # I must find an integer scheme with sum 1.
    # Or, I can use halves in U, V, W?
    # "integer/half-integer entries".
    # So I CAN use halves!
    # The validator in my code uses int64, which fails for halves.
    # I must update my validator to handle halves (floats or fractions).
    # And the evaluator likely handles halves correctly.
    
    def validate_half(U, V, W):
        r = len(U[0])
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Re-try rank 33 with halves (no scaling)
    U, V, W, rank = make_rank_33()
    # My make_rank_33 used scaled integers.
    # I need to make a version with halves.
    
    def make_rank_33_half():
        U_block, V_block, W_block = make_rank_11_block()
        # make_rank_11_block returns scaled integers (x2).
        # I need to divide by 2.
        U_block = [[x/2.0 for x in row] for row in U_block]
        V_block = [[x/2.0 for x in row] for row in V_block]
        W_block = [[x/2.0 for x in row] for row in W_block]
        
        rank = 33
        U = []
        for row in U_block:
            new_row = []
            for _ in range(3):
                new_row.extend(row)
            U.append(new_row)
            
        V = []
        for _ in range(18):
            V.append([0.0]*33)
        for b in range(3):
            for j in range(3):
                for k in range(2):
                    v_row_idx = j * 2 + k
                    vb_row = j * 2 + k
                    col_offset = b * 11
                    for r in range(11):
                        V[v_row_idx + b*6][col_offset + r] = V_block[vb_row][r]
                        
        W = []
        for _ in range(12):
            W.append([0.0]*33)
        for b in range(3):
            for k in range(2):
                for i in range(2):
                    w_row_idx = k * 2 + i
                    wb_row = k * 2 + i
                    col_offset = b * 11
                    for r in range(11):
                        W[w_row_idx + b*4][col_offset + r] = W_block[wb_row][r]
                        
        return U, V, W, 33

    U, V, W, rank = make_rank_33_half()
    if validate_half(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
