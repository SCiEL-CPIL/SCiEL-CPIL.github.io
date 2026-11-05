"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

import time

# EVOLVE-BLOCK-START
import time
_start_time = time.time()

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Split: C = A(4x4) * B(4x7) = A*B1(4x4) + A*B2(4x3)
    # Use Strassen for 4x4*4x4 (rank 28) + schoolbook for 4x4*4x3 (rank 48)
    # Total rank = 76
    
    rank_strassen = 28
    rank_schoolbook = 48
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 76
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x 76
    W = [[0] * total_rank for _ in range(p * n)]  # 28 x 76
    
    # Strassen part: columns 0..27
    # Block structure for 4x4:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # Similarly for B (first 4 cols) and C (first 4 cols)
    #
    # Strassen 7 terms:
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C21 += M1
    # M2 = (A21+A22)B11       -> C11 -= M2, C22 += M2
    # M3 = A11(B12-B22)       -> C12 += M3
    # M4 = A22(B21-B11)       -> C11 += M4, C21 += M4, C22 -= M4
    # M5 = (A11+A12)B22       -> C11 += M5, C22 += M5
    # M6 = (A21-A11)(B11+B12) -> C22 += M6
    # M7 = (A12-A22)(B21+B22) -> C12 += M7
    #
    # Each M_t is a 2x2*2x2 product (rank 4).
    # For each M_t, decompose into 4 rank-1 terms.
    #
    # A-side 2x2 matrices (as list of (row, col, coeff) for each entry):
    # Each entry is a linear combination of A[i,j]
    
    # Define A-side 2x2 matrices for each Strassen term
    # Format: list of 4 entries [a00, a01, a10, a11], each a list of (i, j, coeff)
    A_sides = [
        # M1: A11 + A22
        [[(0,0,1),(2,2,1)], [(0,1,1),(2,3,1)], [(1,0,1),(3,2,1)], [(1,1,1),(3,3,1)]],
        # M2: A21 + A22
        [[(2,0,1),(2,2,1)], [(2,1,1),(2,3,1)], [(3,0,1),(3,2,1)], [(3,1,1),(3,3,1)]],
        # M3: A11
        [[(0,0,1)], [(0,1,1)], [(1,0,1)], [(1,1,1)]],
        # M4: A22
        [[(2,2,1)], [(2,3,1)], [(3,2,1)], [(3,3,1)]],
        # M5: A11 + A12
        [[(0,0,1),(0,2,1)], [(0,1,1),(0,3,1)], [(1,0,1),(1,2,1)], [(1,1,1),(1,3,1)]],
        # M6: A21 - A11
        [[(2,0,1),(0,0,-1)], [(2,1,1),(0,1,-1)], [(3,0,1),(1,0,-1)], [(3,1,1),(1,1,-1)]],
        # M7: A12 - A22
        [[(0,2,1),(2,2,-1)], [(0,3,1),(2,3,-1)], [(1,2,1),(3,2,-1)], [(1,3,1),(3,3,-1)]],
    ]
    
    # Define B-side 2x2 matrices for each Strassen term
    # B1 = first 4 columns of B, so B1[j,k] = B[j,k] for k=0..3
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    B_sides = [
        # M1: B11 + B22
        [[(0,0,1),(2,2,1)], [(0,1,1),(2,3,1)], [(1,0,1),(3,2,1)], [(1,1,1),(3,3,1)]],
        # M2: B11
        [[(0,0,1)], [(0,1,1)], [(1,0,1)], [(1,1,1)]],
        # M3: B12 - B22
        [[(0,2,1),(2,2,-1)], [(0,3,1),(2,3,-1)], [(1,2,1),(3,2,-1)], [(1,3,1),(3,3,-1)]],
        # M4: B21 - B11
        [[(2,0,1),(0,0,-1)], [(2,1,1),(0,1,-1)], [(3,0,1),(1,0,-1)], [(3,1,1),(1,1,-1)]],
        # M5: B22
        [[(2,2,1)], [(2,3,1)], [(3,2,1)], [(3,3,1)]],
        # M6: B11 + B12
        [[(0,0,1),(0,2,1)], [(0,1,1),(0,3,1)], [(1,0,1),(1,2,1)], [(1,1,1),(1,3,1)]],
        # M7: B21 + B22
        [[(2,0,1),(2,2,1)], [(2,1,1),(2,3,1)], [(3,0,1),(3,2,1)], [(3,1,1),(3,3,1)]],
    ]
    
    # Output contributions: for each Strassen term, which C blocks it contributes to
    # C11 = rows 0-1, cols 0-3; C12 = rows 0-1, cols 4-6 (but we only have 4 cols in B1)
    # Actually C1 = A*B1 is 4x4, with blocks:
    # C11 = C[0:2, 0:2], C12 = C[0:2, 2:4], C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
    
    # Output block indices (block_row, block_col) where block_row, block_col in {0,1}
    # C block (br, bc) corresponds to rows br*2..br*2+1, cols bc*2..bc*2+1 in C1
    # In full C matrix (4x7), these are at rows br*2..br*2+1, cols bc*2..bc*2+1
    
    # Contributions: (block_row, block_col, sign)
    C_contribs = [
        # M1
        [(0,0,1), (1,0,1)],
        # M2
        [(0,0,-1), (1,1,1)],
        # M3
        [(0,1,1)],
        # M4
        [(0,0,1), (1,0,1), (1,1,-1)],
        # M5
        [(0,0,1), (1,1,1)],
        # M6
        [(1,1,1)],
        # M7
        [(0,1,1)],
    ]
    
    # For each Strassen term t (0..6), and for each of the 4 rank-1 components,
    # we add one rank-1 term to the overall decomposition.
    #
    # The 2x2*2x2 product R[i,j] = sum_k P[i,k]*Q[k,j]
    # Rank-4 decomposition: for each (i,j) in {0,1}x{0,1}, we have:
    # R[i,j] = P[i,0]*Q[0,j] + P[i,1]*Q[1,j]
    # This is NOT a rank-1 decomposition. We need to be more careful.
    #
    # The correct rank-4 decomposition of 2x2*2x2:
    # The tensor T[i,k,j] = delta[i,k]*delta[k,j] (identity in each index)
    # Actually, the multiplication tensor for 2x2*2x2 has entries:
    # T[i,j,k] = 1 if the product involves A[i,k]*B[k,j]
    # So T[i,j,k] = 1 for all i,j,k? No.
    #
    # The multiplication: C[i,j] = sum_k A[i,k]*B[k,j]
    # The tensor has 8 non-zero entries (one for each (i,k,j) triple).
    # But the rank is 4, not 8.
    #
    # The rank-4 decomposition:
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    # Term 1: A[0,1]*B[1,0] -> C[0,0]
    # Term 2: A[1,0]*B[0,1] -> C[1,0]
    # Term 3: A[1,1]*B[1,1] -> C[1,1]
    # But this misses C[0,1] and C[1,0] contributions from other terms.
    #
    # Actually, I think the correct approach is:
    # For each (i,j) in the 2x2 output, C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # We can write this as a sum of 2 rank-1 terms for each (i,j), giving 8 terms total.
    # But the rank of 2x2*2x2 is 4, so we can do better.
    #
    # The standard rank-4 decomposition of 2x2*2x2:
    # Let's use the fact that the multiplication tensor for n=2 has rank 4.
    # One decomposition:
    # M1 = A[0,0]*B[0,0] -> C[0,0]
    # M2 = A[0,1]*B[1,0] -> C[0,0]
    # M3 = A[1,0]*B[0,1] -> C[1,0]
    # M4 = A[1,1]*B[1,1] -> C[1,1]
    # But this doesn't give C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # or C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #
    # I think I need to reconsider. The rank of the 2x2*2x2 multiplication tensor IS 4.
    # Let me look up the correct decomposition.
    #
    # Actually, for 2x2*2x2, the schoolbook uses 4 multiplications:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # This uses 8 scalar multiplications, but the tensor rank is 4.
    # The rank-4 decomposition must share some multiplications.
    #
    # Actually, I think I'm confusing myself. The "rank" in tensor decomposition 
    # context means the number of rank-1 tensors needed. For 2x2*2x2, the 
    # multiplication tensor has rank 4, meaning we need 4 rank-1 terms.
    #
    # A rank-1 term for 2x2*2x2 is: u_i * v_k * w_j where u, v, w are vectors.
    # The 4 rank-1 terms are:
    # Term 0: u=(1,0), v=(1,0), w=(1,0) -> A[0,0]*B[0,0]*C[0,0]
    # Term 1: u=(0,1), v=(0,1), w=(1,0) -> A[1,0]*B[0,1]*C[1,0]
    # ... this isn't working.
    #
    # Let me just use 8 rank-1 terms per 2x2*2x2 (one for each (i,k,j) triple),
    # giving 7*8 = 56 for Strassen, + 48 schoolbook = 104. That's worse than 112.
    #
    # Alternatively, use 4 rank-1 terms per 2x2*2x2 (the actual rank), giving 28+48=76.
    #
    # Let me use the correct rank-4 decomposition. After more thought:
    # The 2x2*2x2 multiplication can be done with 4 rank-1 terms as follows:
    # Let the 4 terms correspond to the 4 "diagonal" elements in some sense.
    #
    # Actually, I'll just use 4 terms where each term handles one (i,j) output entry,
    # and for each, we combine the two k-contributions into a single rank-1 term 
    # by factoring. But that's not possible in general.
    #
    # Let me just use 8 terms per 2x2*2x2 for simplicity, giving rank 7*8+48 = 104.
    # That's still better than 112.
    
    # Actually, let me reconsider. I'll use the straightforward approach:
    # For each Strassen term, decompose the 2x2*2x2 into 4 rank-1 terms 
    # (the true rank). The correct decomposition:
    #
    # For 2x2*2x2, the rank-4 decomposition is:
    # Term 0: A[0,0]*B[0,0] contributes to C[0,0]
    # Term 1: A[0,1]*B[1,0] contributes to C[0,0]
    # Term 2: A[1,0]*B[0,1] contributes to C[1,0]
    # Term 3: A[1,1]*B[1,1] contributes to C[1,1]
    # But we also need C[0,1] and C[1,0] contributions...
    #
    # I think the issue is that I'm not getting the decomposition right.
    # Let me just use 4 terms where:
    # Term (i,j): for output entry (i,j), use A[i,0]*B[0,j] as one rank-1 term
    # and A[i,1]*B[1,j] as another. That's 8 terms.
    #
    # For simplicity and correctness, let me use 8 terms per 2x2*2x2.
    # Total Strassen rank = 7 * 8 = 56, plus schoolbook 48 = 104.
    
    # Actually, let me just implement it with 4 terms per 2x2*2x2 using the 
    # standard decomposition. I'll verify with numpy.
    
    # For now, let me use a simpler correct approach: 
    # Use Strassen at the block level but with schoolbook for inner 2x2*2x2.
    # This gives 7 * 4 = 28 rank-1 terms for the Strassen part.
    # Each "rank-1 term" here is actually a full 2x2*2x2 multiplication,
    # which itself has rank 4. So total is 7*4 = 28 scalar rank-1 terms.
    
    # Let me implement it as: for each of the 7 Strassen terms, and for each 
    # of the 4 scalar multiplications in the 2x2*2x2, create one rank-1 term.
    
    col = 0
    for t in range(7):
        # A-side 2x2 matrix entries: A_sides[t] = [a00, a01, a10, a11]
        # B-side 2x2 matrix entries: B_sides[t] = [b00, b01, b10, b11]
        # For each (i, k, j) in {0,1}^3, we have a scalar multiplication:
        # A_side[i,k] * B_side[k,j] contributes to R[i,j]
        # But we want to group by (i,j) output.
        #
        # Actually, for the rank-4 decomposition, we use 4 terms:
        # Term 0: (i=0, k=0, j=0) -> A_side[0,0]*B_side[0,0] -> R[0,0]
        # Term 1: (i=0, k=1, j=0) -> A_side[0,1]*B_side[1,0] -> R[0,0]
        # Term 2: (i=1, k=0, j=1) -> A_side[1,0]*B_side[0,1] -> R[1,0]
        # Term 3: (i=1, k=1, j=1) -> A_side[1,1]*B_side[1,1] -> R[1,1]
        # But this misses R[0,1] and R[1,0]...
        #
        # I'll just use 8 terms: one for each (i,k,j) triple.
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    # A_side entry (i,k) is at index i*2+k
                    # B_side entry (k,j) is at index k*2+j
                    a_entries = A_sides[t][i*2+k]
                    b_entries = B_sides[t][k*2+j]
                    
                    # For each (a_i, a_j, a_coeff) in a_entries and 
                    # (b_i, b_j, b_coeff) in b_entries, we need a rank-1 term.
                    # But this would explode the rank. Instead, we combine:
                    # The product is sum over a_entries and b_entries of 
                    # a_coeff * b_coeff * A[a_i,a_j] * B[b_i,b_j]
                    # This is not a single rank-1 term unless each list has 1 entry.
                    #
                    # For simplicity, let me just handle the case where each 
                    # 2x2 entry is a single A[i,j] or B[i,j] term.
                    # For Strassen terms with sums, this won't work directly.
                    #
                    # Let me take a different approach: directly enumerate all 
                    # scalar rank-1 terms for the full 4x4x4 Strassen decomposition.
                    
                    pass
    
    # This is getting too complex. Let me just use schoolbook for everything.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
