"""Improved program for 3x4 * 4x6 matrix multiplication tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# Known optimal or near-optimal ranks for small matrix multiplication
# We use block decomposition: split 3x4 into blocks and 4x6 into blocks,
# then use Strassen-like algorithms on the blocks.

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Strategy: Use block decomposition with Strassen for 2x2 blocks
    # Split the 3x4 matrix into blocks and 4x6 matrix into compatible blocks.
    # 
    # One approach: treat the problem as multiplying a (3x4) by (4x6).
    # We can partition rows of A (3 rows) and columns of B (6 cols) and the shared dimension (4).
    #
    # Let's try: partition the shared dimension m=4 into two parts of size 2.
    # A = [A1; A2] where A1 is 3x2, A2 is 3x2
    # B = [B1 B2] where B1 is 2x6, B2 is 2x6
    # C = A1*B1 + A2*B2
    #
    # Each product is a 3x2 * 2x6 multiplication.
    # The rank of 3x2 * 2x6 matrix multiplication:
    #   - Schoolbook: 3*2*6 = 36
    #   - Can we do better? The tensor rank of 3x2 * 2x6 is at most 36.
    #   - Actually, for small matrices, we can sometimes do better with Strassen-like methods.
    #
    # Alternative: Use a more refined block structure.
    # Let's partition n=3 as 1+2, m=4 as 2+2, p=6 as 2+2+2.
    #
    # A is 3x4: split rows into [1; 2], cols into [2 | 2]
    #   A = [[A11, A12], [A21, A22]] where A11 is 1x2, A12 is 1x2, A21 is 2x2, A22 is 2x2
    # B is 4x6: split rows into [2; 2], cols into [2 | 2 | 2]
    #   B = [[B11, B12, B13], [B21, B22, B23]] where Bij is 2x2
    #
    # C = A*B, C is 3x6:
    #   C[0,0:2]   = A11*B11
    #   C[0,2:4]   = A11*B12
    #   C[0,4:6]   = A11*B13
    #   C[1:3,0:2] = A21*B11
    #   C[1:3,2:4] = A21*B12
    #   C[1:3,4:6] = A21*B13
    #   C[1:3,0:2] += A22*B21  -> wait, this isn't right.
    #
    # Let me redo the block multiplication properly.
    # A (3x4) * B (4x6) = C (3x6)
    # Partition: n=3 -> [1, 2], m=4 -> [2, 2], p=6 -> [2, 2, 2]
    #
    # A = [A_top; A_bot] where A_top is 1x4, A_bot is 2x4
    #     A_top = [A11 | A12] where A11 is 1x2, A12 is 1x2
    #     A_bot = [A21 | A22] where A21 is 2x2, A22 is 2x2
    #
    # B = [B_top; B_bot] where B_top is 2x6, B_bot is 2x6
    #     B_top = [B11 | B12 | B13] where each Bij is 2x2
    #     B_bot = [B21 | B22 | B23] where each Bij is 2x2
    #
    # C = [C_top; C_bot] where C_top is 1x6, C_bot is 2x6
    #     C_top = [C11 | C12 | C13] where each is 1x2
    #     C_bot = [C21 | C22 | C23] where each is 2x2
    #
    # C11 = A11 * B11  (1x2 * 2x2 = 1x2)
    # C12 = A11 * B12  (1x2 * 2x2 = 1x2)
    # C13 = A11 * B13  (1x2 * 2x2 = 1x2)
    # C21 = A21 * B11 + A22 * B21  (2x2 * 2x2 + 2x2 * 2x2 = 2x2)
    # C22 = A21 * B12 + A22 * B22  (2x2 * 2x2 + 2x2 * 2x2 = 2x2)
    # C23 = A21 * B13 + A22 * B23  (2x2 * 2x2 + 2x2 * 2x2 = 2x2)
    #
    # So we have:
    # - 3 products of type 1x2 * 2x2 (rank 4 each? or less?)
    # - 6 products of type 2x2 * 2x2 (rank 7 each with Strassen)
    #
    # Total rank: 3 * rank(1x2,2x2) + 6 * rank(2x2,2x2)
    #
    # For 1x2 * 2x2: This is multiplying a 1x2 row vector by a 2x2 matrix to get a 1x2 row vector.
    #   The result is [a,b] * [[c,d],[e,f]] = [ac+be, ad+bf]
    #   This requires 4 multiplications in schoolbook. Can we do better?
    #   The tensor has entries indexed by (i,j,k) where i in {0}, j in {0,1}, k in {0,1}.
    #   Actually it's a 1x2 * 2x2 -> 1x2 multiplication.
    #   The number of scalar multiplications: 4 (schoolbook).
    #   Is there a rank-3 decomposition? The tensor is of size 1*2*2 = 4, so max rank is 4.
    #   Let's check if rank 3 is possible. The tensor T[i,j,k] = delta_{j,k} for the relevant indices.
    #   Actually for 1x2 * 2x2: T[a,b,c,d] = 1 if a matches and b=c, else 0. With a fixed.
    #   This is essentially the identity tensor for 2x2, which has rank 2? No...
    #   Wait, 1x2 * 2x2: the input is a 1x2 matrix and a 2x2 matrix, output is 1x2.
    #   T_{(i,j),(j,k)} = delta... no. Let me think in terms of tensor indices.
    #   A has shape (1,2), B has shape (2,2), C has shape (1,2).
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    #   The tensor T[i,j,k] = 1 if the index structure matches, i.e., T[i,j,k] = delta_{i,0} * 1 (for the contraction).
    #   Actually T is a 3-tensor with indices (i, j, k) where i in 0..0, j in 0..1, k in 0..1.
    #   T[i,j,k] = 1 for all valid (i,j,k) because the multiplication just contracts over j.
    #   Wait no. The matrix multiplication tensor M_{n,m,p} has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
    #   So T[i,j,k] = 1 for all i,j,k in valid ranges. The tensor is all ones? No, that's not right either.
    #   
    #   Let me be precise. The matrix multiplication tensor for (n x m) * (m x p) -> (n x p) is:
    #   T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    #   where e_{i,j} is the basis vector for the (i,j) entry of A, etc.
    #   So T[i,j,k] = 1 for all i in [n], j in [m], k in [p]. The tensor has all entries 1.
    #   Wait, that can't be right because then the rank would be... 
    #   
    #   Actually no. The tensor is T = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,n+i}... no.
    #   The standard definition: The matrix multiplication tensor M_{n,m,p} is the sum over all (i,j,k) of the basis tensor for (i,j) in A, (j,k) in B, and (i,k) in C.
    #   So as a 3-way tensor with dimensions (n*m, m*p, n*p), the entry at position ((i,j), (j',k), (i',k')) is 1 if i=i', j=j', k=k', else 0.
    #   So T[(i,j),(j',k),(i',k')] = 1 if i=i' and j=j' and k=k', else 0.
    #   This is a "diagonal" tensor in a sense, but with the constraint that the middle index matches.
    #
    #   OK so for 1x2 * 2x2 -> 1x2:
    #   A has 2 entries, B has 4 entries, C has 2 entries.
    #   T has shape (2, 4, 2). T[a_idx, b_idx, c_idx] = 1 if the indices are consistent.
    #   a_idx = j (0 or 1), b_idx = j'*2 + k, c_idx = k (0 or 1).
    #   Consistency: j = j' and k = k. So T[j, j'*2+k, k] = 1 if j=j'.
    #   So T[0, 0, 0] = 1, T[0, 1, 1] = 1, T[1, 2, 0] = 1, T[1, 3, 1] = 1. All others 0.
    #   This tensor has rank 4 (it's a permutation-like tensor). Actually it's a sum of 4 rank-1 tensors.
    #   Can we do it in rank 3? The 4 nonzero entries are in "general position" so likely rank 4.
    #   So rank(1x2, 2x2) = 4.
    #
    # For 2x2 * 2x2 -> 2x2: Strassen gives rank 7.
    #
    # Total: 3*4 + 6*7 = 12 + 42 = 54.
    # Schoolbook is 3*4*6 = 72. So 54 is an improvement.
    #
    # But can we do better? Let's think of other partitions.
    #
    # Alternative: n=3 -> [1,2], m=4 -> [1,3], p=6 -> [2,4]
    # This gets complicated. Let's stick with the 2+2 split for m and use Strassen.
    
    # Let's implement the decomposition with:
    # - 3 blocks of type (1x2) * (2x2) -> (1x2), each rank 4
    # - 6 blocks of type (2x2) * (2x2) -> (2x2), each rank 7 (Strassen)
    # Total rank = 3*4 + 6*7 = 54
    
    rank_total = 3 * 4 + 6 * 7  # 54
    
    U = [[0] * rank_total for _ in range(n * m)]  # 3*4 = 12 rows
    V = [[0] * rank_total for _ in range(m * p)]  # 4*6 = 24 rows
    W = [[0] * rank_total for _ in range(p * n)]  # 6*3 = 18 rows
    
    col = 0
    
    # Helper: add a decomposition block
    def add_block(u_rows, v_rows, w_rows, block_decomp):
        """block_decomp is a list of (u_vals, v_vals, w_vals) for each rank-1 component."""
        nonlocal col
        for u_vals, v_vals, w_vals in block_decomp:
            for r, val in zip(u_rows, u_vals):
                U[r][col] += val
            for r, val in zip(v_rows, v_vals):
                V[r][col] += val
            for r, val in zip(w_rows, w_vals):
                W[r][col] += val
            col += 1
    
    # Strassen decomposition for 2x2 * 2x2 -> 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def strassen_2x2(a_idx, b_idx, c_idx, d_idx, e_idx, f_idx, g_idx, h_idx,
                     c11_idx, c12_idx, c21_idx, c22_idx):
        """Return list of (u_vals, v_vals, w_vals) for Strassen 2x2."""
        decomps = []
        # M1 = (a+d)(e+h)
        decomps.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]))  # contributes to C11
        # Actually, let me be more careful. Each multiplication M_i is a rank-1 term.
        # M1 = (a+d) * (e+h) contributes to C11, C22
        # We need to track which output each M contributes to.
        
        # Let me restructure. For each Strassen product M_i, we have:
        # U values for the 4 entries of A block
        # V values for the 4 entries of B block  
        # W values for the 4 entries of C block (with signs)
        
        # M1 = (a+d)(e+h): U=[1,0,0,1], V=[1,0,0,1], W for C11: +1, C22: +1
        # But W is per-output-entry. The C block has 4 entries: c11, c12, c21, c22.
        # M1 contributes +1 to c11 and +1 to c22.
        decomps = []
        # M1
        decomps.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))  # c11:+1, c12:0, c21:0, c22:+1
        # M2 = (c+d)e: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]  # c21:+1
        decomps.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
        # M3 = a(f-h): U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]  # c12:+1
        decomps.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
        # M4 = d(g-e): U=[0,0,0,1], V=[0,0,1,-1], W=[1,0,1,0]  # c11:+1, c21:+1
        decomps.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]))
        # M5 = (a+b)h: U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0]  # c11:-1, c12:+1
        decomps.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        # M6 = (c-a)(e+f): U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]  # c22:+1
        decomps.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        # M7 = (b-d)(g+h): U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,-1]  # c11:+1, c22:-1
        decomps.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]))
        
        return decomps
    
    # Schoolbook decomposition for 1x2 * 2x2 -> 1x2
    # A = [a, b] (1x2), B = [[e, f], [g, h]] (2x2), C = [c1, c2] (1x2)
    # c1 = a*e + b*g, c2 = a*f + b*h
    # 4 multiplications: a*e, b*g, a*f, b*h
    
    def schoolbook_1x2_2x2(a_idx, b_idx, e_idx, f_idx, g_idx, h_idx, c1_idx, c2_idx):
        """Return list of (u_vals, v_vals, w_vals) for 1x2 * 2x2."""
        decomps = []
        # a*e -> c1
        decomps.append(([1, 0], [1, 0, 0, 0], [1, 0]))
        # b*g -> c1
        decomps.append(([0, 1], [0, 0, 1, 0], [1, 0]))
        # a*f -> c2
        decomps.append(([1, 0], [0, 1, 0, 0], [0, 1]))
        # b*h -> c2
        decomps.append(([0, 1], [0, 0, 0, 1], [0, 1]))
        return decomps
    
    # Now map the block indices to global indices.
    # A is 3x4, indexed as A[i,j] with i in 0..2, j in 0..3.
    # Global U index for A[i,j] = i*m + j = i*4 + j.
    # B is 4x6, indexed as B[j,k] with j in 0..3, k in 0..5.
    # Global V index for B[j,k] = j*p + k = j*6 + k.
    # C is 3x6, indexed as C[i,k] with i in 0..2, k in 0..5.
    # Global W index for C[i,k] = k*n + i = k*3 + i (trace convention W[k*n+i, r]).
    
    # Block partition:
    # A rows: [0] (top), [1,2] (bot)
    # A cols: [0,1] (left), [2,3] (right)
    # B rows: [0,1] (top), [2,3] (bot)
    # B cols: [0,1] (left), [2,3] (mid), [4,5] (right)
    
    # A11: A[0, 0:2] -> U indices [0*4+0, 0*4+1] = [0, 1]
    # A12: A[0, 2:4] -> U indices [0*4+2, 0*4+3] = [2, 3]
    # A21: A[1:3, 0:2] -> U indices [1*4+0, 1*4+1, 2*4+0, 2*4+1] = [4, 5, 8, 9]
    # A22: A[1:3, 2:4] -> U indices [1*4+2, 1*4+3, 2*4+2, 2*4+3] = [6, 7, 10, 11]
    
    # B11: B[0:2, 0:2] -> V indices [0*6+0, 0*6+1, 1*6+0, 1*6+1] = [0, 1, 6, 7]
    # B12: B[0:2, 2:4] -> V indices [0*6+2, 0*6+3, 1*6+2, 1*6+3] = [2, 3, 8, 9]
    # B13: B[0:2, 4:6] -> V indices [0*6+4, 0*6+5, 1*6+4, 1*6+5] = [4, 5, 10, 11]
    # B21: B[2:4, 0:2] -> V indices [2*6+0, 2*6+1, 3*6+0, 3*6+1] = [12, 13, 18, 19]
    # B22: B[2:4, 2:4] -> V indices [2*6+2, 2*6+3, 3*6+2, 3*6+3] = [14, 15, 20, 21]
    # B23: B[2:4, 4:6] -> V indices [2*6+4, 2*6+5, 3*6+4, 3*6+5] = [16, 17, 22, 23]
    
    # C11: C[0, 0:2] -> W indices [0*3+0, 1*3+0] = [0, 3]
    # C12: C[0, 2:4] -> W indices [2*3+0, 3*3+0] = [6, 9]
    # C13: C[0, 4:6] -> W indices [4*3+0, 5*3+0] = [12, 15]
    # C21: C[1:3, 0:2] -> W indices [0*3+1, 1*3+1, 0*3+2, 1*3+2] = [1, 4, 2, 5]
    # C22: C[1:3, 2:4] -> W indices [2*3+1, 3*3+1, 2*3+2, 3*3+2] = [7, 10, 8, 11]
    # C23: C[1:3, 4:6] -> W indices [4*3+1, 5*3+1, 4*3+2, 5*3+2] = [13, 16, 14, 17]
    
    # Process the 3 (1x2)*(2x2) blocks:
    # C11 = A11 * B11
    add_block([0, 1], [0, 1, 6, 7], [0, 3], schoolbook_1x2_2x2(0, 1, 0, 1, 2, 3, 0, 1))
    
    # C12 = A11 * B12
    add_block([0, 1], [2, 3, 8, 9], [6, 9], schoolbook_1x2_2x2(0, 1, 0, 1, 2, 3, 0, 1))
    
    # C13 = A11 * B13
    add_block([0, 1], [4, 5, 10, 11], [12, 15], schoolbook_1x2_2x2(0, 1, 0, 1, 2, 3, 0, 1))
    
    # Process the 6 (2x2)*(2x2) blocks:
    # C21 = A21*B11 + A22*B21
    # First term: A21 * B11
    add_block([4, 5, 8, 9], [0, 1, 6, 7], [1, 4, 2, 5], strassen_2x2(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3))
    # Second term: A22 * B21
    add_block([6, 7, 10, 11], [12, 13, 18, 19], [1, 4, 2, 5], strassen_2x2(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3))
    
    # C22 = A21*B12 + A22*B22
    add_block([4, 5, 8, 9], [2, 3, 8, 9], [7, 10, 8, 11], strassen_2x2(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3))
    add_block([6, 7, 10, 11], [14, 15, 20, 21], [7, 10, 8, 11], strassen_2x2(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3))
    
    # C23 = A21*B13 + A22*B23
    add_block([4, 5, 8, 9], [4, 5, 10, 11], [13, 16, 14, 17], strassen_2x2(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3))
    add_block([6, 7, 10, 11], [16, 17, 22, 23], [13, 16, 14, 17], strassen_2x2(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3))
    
    # Check deadline
    if time.time() > DEADLINE:
        pass  # We're done, return what we have
    
    # Convert to integer values if they're very close to integers
    for mat in [U, V, W]:
        for row in mat:
            for j in range(len(row)):
                if abs(row[j] - round(row[j])) < 1e-10:
                    row[j] = int(round(row[j]))
    
    return {"U": U, "V": V, "W": W}
